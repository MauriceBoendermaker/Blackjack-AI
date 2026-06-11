"""Worker-thread lifecycle around the DetectionEngine.

The GUI never blocks: start() returns immediately, model warm-up happens on
the worker thread, and the GUI consumes engine snapshots by polling
`engine.get_snapshot()` from a Tk `after()` loop.
"""

import threading
import time

from ..common import constants
from .engine import DetectionEngine
from .phase import BETTING_OPEN, MY_TURN


class DetectionController:
    def __init__(self, log=print):
        self.log = log
        self.engine = DetectionEngine(log=log)
        self._stop = threading.Event()
        self._thread = None
        self.state = "idle"  # idle | starting | running | stopped | error
        self._idle_refresh_failed = False  # gates the "Idle Ns" announcement

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def set_monitor(self, monitor):
        self.engine.set_monitor(monitor)

    def start(self):
        if self.running:
            return False
        # Each worker gets its OWN stop event. A stopped-but-still-finishing
        # old worker keeps watching its (already set) event, so a quick
        # Stop -> Start can never revive it alongside the new one.
        stop_event = threading.Event()
        self._stop = stop_event
        self.state = "starting"
        self._thread = threading.Thread(target=self._run, args=(stop_event,),
                                        daemon=True, name="detection")
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        self.state = "stopped"

    def _maybe_idle_refresh(self, prev_cycle_ts, now) -> bool:
        """Refresh model backends when the gap since the previous cycle says
        the machine slept (lid close, OS suspend) — hosted API sessions do
        not survive that. Returns False when a gap-triggered refresh failed
        (e.g. Wi-Fi still reassociating right after the wake): the caller
        then keeps its gap baseline so the still-open gap re-triggers the
        refresh every cycle until one succeeds."""
        gap = now - prev_cycle_ts
        if gap <= constants.IDLE_REFRESH_GAP_S:
            return True
        if not self._idle_refresh_failed:  # announce the gap once, not per retry
            self.log(f"Idle {gap:.0f}s — refreshing model backends")
        ok = self.engine.health_check_and_refresh() is not False
        self._idle_refresh_failed = not ok
        return ok

    def _run(self, stop_event):
        try:
            try:
                self.log("Initializing detection models...")
                self.engine.warm_up()
                self.log(f"Models ready ({self.engine.provider.backend_name}).")
            except Exception as e:
                self.state = "error"
                self.engine.last_error = str(e)
                self.engine.publish_snapshot()
                self.log(f"Model initialization failed: {e}")
                return

            self.state = "running"
            prev_cycle = time.monotonic()
            while not stop_event.is_set():
                refreshed = self._maybe_idle_refresh(prev_cycle, time.monotonic())
                activity = self.engine.run_cycle()
                if refreshed:  # failed refresh keeps the gap open -> retry
                    prev_cycle = time.monotonic()
                if activity == "dealing":
                    sleep_s = constants.CYCLE_SLEEP_DEALING
                elif activity == "complete":
                    sleep_s = constants.CYCLE_SLEEP_COMPLETE
                else:
                    sleep_s = constants.CYCLE_SLEEP_WAITING
                if self.engine.current_phase in (BETTING_OPEN, MY_TURN):
                    # Hard deadlines (~12-15 s betting window, ~10-13 s
                    # decision timer): sample fast while one is running.
                    sleep_s = min(sleep_s, constants.CYCLE_SLEEP_ACTION)
                stop_event.wait(sleep_s)
            self.state = "stopped"
        finally:
            # Release this thread's GDI capture resources.
            self.engine.capture.close_local()
