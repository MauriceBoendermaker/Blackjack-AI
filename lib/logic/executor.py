"""Ghost-mode executor & one-key assisted execute (V4 Feature 2).

Implements AUTONOMY_PLAN.md stages 1-2 and deliberately stops there:

  * GHOST — computes the exact click it WOULD make (button template
    location from the phase detector, chip sequence for the bet ramp),
    surfaces it (HUD marker + sidebar), clicks NOTHING, and logs every
    plan + confidence into SessionStore.executor_log. This is the honest
    benchmark of whether detection could ever be trusted with money.
  * ASSIST — ghost + a single global confirm key per action. The human
    stays the actor on every irreversible money move; the executor removes
    the aiming, not the deciding. Every fire passes the full §5 guard
    stack and is verified afterwards; an unverified click DISARMS and is
    NEVER retried (a blind re-click is a double-bet).
  * Stage 3 (unattended autonomy) is intentionally not implemented.

Scope guard for v1: assisted execution covers DECISION clicks only
(Hit / Stand / Double / Split). Bet placement stays ghost-only — clicking
the spot with whatever chip happens to be selected is a wrong-denomination
money bug, and chip-tray calibration is future work.

Architecture: a SECOND CONSUMER of the engine snapshot (never touches
engine internals). The GUI poll drives step() on the Tk thread (pure dict
math); the actual click runs on a short-lived dispatch thread. Audit
writes go through the engine's io pool (single SQLite writer contract).

Click delivery: OS SendInput via ctypes (stdlib, physical-pixel exact in
this per-monitor-DPI-aware process) with an optional CDP path (playwright
`connect_over_cdp`) that frees the physical cursor; CDP failures fall back
to OS input. A corner FAILSAFE aborts any dispatch when the mouse sits in
a screen corner.
"""

import threading
import time

from ..common import constants
from . import phase

MODES = ("off", "ghost", "assist")
_ACTION_TO_CONTROL = {"H": "hit", "S": "stand", "D": "double", "P": "split"}
ACTION_NAMES = {"S": "Stand", "H": "Hit", "D": "Double", "P": "Split",
                "R": "Surrender"}


def _default_log(message, level=None):
    print(message)


class ExecutorAbort(RuntimeError):
    """Raised by the dispatcher when the FAILSAFE blocks a click."""


def chip_sequence(amount, chips) -> list | None:
    """Greedy chip decomposition of a bet, largest first. None when the
    amount is not exactly representable — an executor that rounds a bet is
    a money bug, so it must abstain instead."""
    if amount is None or amount <= 0:
        return None
    remaining = round(float(amount), 2)
    seq = []
    for chip in sorted((float(c) for c in chips), reverse=True):
        if chip <= 0:
            continue
        while remaining + 1e-9 >= chip:
            seq.append(chip)
            remaining = round(remaining - chip, 2)
    return seq if remaining < 0.005 else None


# --------------------------------------------------------------- dispatch

class ClickDispatcher:
    """Delivers one left click at FRAME coordinates of the watched monitor.

    OS path: SendInput absolute move + click over the virtual desktop —
    hardware-real events the canvas accepts unconditionally; coordinates
    are exact because the process is per-monitor-DPI-aware and frames are
    physical pixels. CDP path (optional, `use_cdp` + playwright installed):
    Input.dispatchMouseEvent through the connected tab, which frees the
    physical cursor; the screen→viewport mapping assumes 100% page zoom
    and is verified post-click like everything else.
    """

    def __init__(self, monitor_fn, log=_default_log):
        self.monitor_fn = monitor_fn
        self.log = log
        self._cdp_browser = None
        self._cdp_page = None
        self._pw = None
        self._pool = None

    def submit(self, fn, *args):
        """Run a dispatch job on the ONE persistent click thread. The
        playwright sync API is greenlet-bound to its creating thread, so
        every CDP call must happen on the same thread for the connection
        cache to survive past the first click."""
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="executor-click")
        return self._pool.submit(fn, *args)

    # ------------------------------------------------------------ helpers

    @staticmethod
    def virtual_screen():
        """(left, top, width, height) of the Windows virtual desktop."""
        import ctypes
        m = ctypes.windll.user32.GetSystemMetrics
        return m(76), m(77), m(78), m(79)  # SM_X/Y/CX/CYVIRTUALSCREEN

    @staticmethod
    def to_absolute(virtual, x, y):
        """Physical desktop px -> SendInput's normalized 0..65535 space."""
        left, top, width, height = virtual
        nx = round((x - left) * 65535 / max(1, width - 1))
        ny = round((y - top) * 65535 / max(1, height - 1))
        return max(0, min(65535, nx)), max(0, min(65535, ny))

    def _failsafe(self):
        """Abort when the mouse is parked in a screen corner — the dead-man
        gesture (slam the mouse to a corner = hard stop)."""
        import ctypes

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        left, top, width, height = self.virtual_screen()
        margin = 10
        near_x = pt.x <= left + margin or pt.x >= left + width - margin
        near_y = pt.y <= top + margin or pt.y >= top + height - margin
        if near_x and near_y:
            raise ExecutorAbort("FAILSAFE: mouse parked in a screen corner")

    # ------------------------------------------------------------- click

    def click(self, frame_x, frame_y) -> str:
        """One left click at frame coords; returns the backend used.
        Raises ExecutorAbort (failsafe) or RuntimeError (delivery failed)."""
        monitor = self.monitor_fn()
        if monitor is None:
            raise RuntimeError("no monitor selected")
        ax, ay = monitor.x + int(frame_x), monitor.y + int(frame_y)
        self._failsafe()
        if constants.EXECUTOR.get("use_cdp"):
            try:
                self._click_cdp(ax, ay)
                return "cdp"
            except Exception as e:
                self.log(f"CDP click failed ({e}); falling back to OS input.",
                         level="WARNING")
        self._click_os(ax, ay)
        return "os"

    def _click_os(self, ax, ay):
        import ctypes

        ULONG_PTR = ctypes.WPARAM

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                        ("mouseData", ctypes.c_ulong),
                        ("dwFlags", ctypes.c_ulong),
                        ("time", ctypes.c_ulong),
                        ("dwExtraInfo", ULONG_PTR)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]

        MOVE, ABSOLUTE, VIRTUALDESK = 0x0001, 0x8000, 0x4000
        LEFTDOWN, LEFTUP = 0x0002, 0x0004
        nx, ny = self.to_absolute(self.virtual_screen(), ax, ay)

        def send(flags, dx=0, dy=0):
            inp = INPUT(type=0, mi=MOUSEINPUT(dx, dy, 0, flags, 0, 0))
            if ctypes.windll.user32.SendInput(
                    1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
                raise RuntimeError("SendInput rejected the event")

        send(MOVE | ABSOLUTE | VIRTUALDESK, nx, ny)
        time.sleep(0.04)
        self._failsafe()  # the human can still yank the mouse away
        send(LEFTDOWN)
        time.sleep(0.05)
        send(LEFTUP)

    @staticmethod
    def _monitor_scale(ax, ay):
        """The OS scale factor (1.0, 1.5, ...) of the monitor containing an
        absolute desktop point; None when unavailable."""
        import ctypes

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        try:
            hmon = ctypes.windll.user32.MonitorFromPoint(
                POINT(int(ax), int(ay)), 2)  # MONITOR_DEFAULTTONEAREST
            dpi_x, dpi_y = ctypes.c_uint(), ctypes.c_uint()
            if ctypes.windll.shcore.GetDpiForMonitor(
                    hmon, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y)) == 0:
                return dpi_x.value / 96.0
        except Exception:
            pass
        return None

    def _click_cdp(self, ax, ay):
        page = self._cdp_target()
        probe = page.evaluate(
            "() => [window.screenX, window.screenY, window.devicePixelRatio,"
            " window.outerWidth, window.innerWidth, window.outerHeight,"
            " window.innerHeight]")
        sx, sy, dpr, ow, iw, oh, ih = probe
        dpr = dpr or 1.0
        # The mapping divides physical px by the page's devicePixelRatio
        # and treats window.screenX/Y as same-scale DIPs. On a mixed-DPI
        # setup that assumption silently breaks (browser zoom, cross-scale
        # window placement), so refuse — the OS-input fallback is exact.
        scale = self._monitor_scale(ax, ay)
        if scale is not None and abs(scale - dpr) > 0.05:
            raise RuntimeError(
                f"DPI mismatch (monitor {scale:.2f}x vs page {dpr:.2f}x) — "
                "screen→viewport mapping untrustworthy")
        border = (ow - iw) / 2
        chrome_top = oh - ih - border
        css_x = ax / dpr - sx - border
        css_y = ay / dpr - sy - chrome_top
        if not (0 <= css_x <= iw and 0 <= css_y <= ih):
            raise RuntimeError(
                f"target maps outside the viewport ({css_x:.0f},{css_y:.0f})")
        page.mouse.click(css_x, css_y)

    def _cdp_target(self):
        if self._cdp_page is not None:
            try:
                if not self._cdp_page.is_closed():
                    return self._cdp_page
            except Exception:
                pass
            self._cdp_page = None
        from playwright.sync_api import sync_playwright  # optional dep
        if self._cdp_browser is None:
            if self._pw is None:
                self._pw = sync_playwright().start()
            port = int(constants.EXECUTOR.get("cdp_port", 9222))
            self._cdp_browser = self._pw.chromium.connect_over_cdp(
                f"http://localhost:{port}")
        match = str(constants.EXECUTOR.get("cdp_url_match") or "").lower()
        for context in self._cdp_browser.contexts:
            for page in context.pages:
                if match in (page.url or "").lower():
                    self._cdp_page = page
                    return page
        raise RuntimeError(f"no open tab matches '{match}'")


# --------------------------------------------------------------- executor

class Executor:
    """Plans (and in assist mode fires, once confirmed) the click the
    advice implies. Thread model: step()/confirm() run on the Tk thread,
    dispatch on a worker thread, audit writes on the engine io pool —
    everything shared sits behind self._lock."""

    def __init__(self, snapshot_fn, monitor_fn, store=None, io_submit=None,
                 log=_default_log, dispatcher=None, running_fn=None):
        self.snapshot_fn = snapshot_fn
        self.monitor_fn = monitor_fn
        self.running_fn = running_fn   # detection liveness; gates arming
        self.store = store
        self.io_submit = io_submit or (lambda fn, *a: fn(*a))
        self.log = log
        self.dispatcher = dispatcher or ClickDispatcher(monitor_fn, log)

        self._lock = threading.RLock()
        self.mode = (constants.EXECUTOR.get("mode")
                     if constants.EXECUTOR.get("mode") in MODES else "off")
        self.armed = False           # NEVER persists; off every session
        self.disarm_reason = None
        self.session = {"plans": 0, "fired": 0, "verified": 0,
                        "mismatches": 0}
        self._plan = None
        self._logged_keys = set()    # ghost-audit dedupe
        self._fired_keys = set()     # never-reclick latches
        self._round_actions = {}     # round -> fired count
        self._verify = None          # pending post-click verification
        self._firing = False

    # ------------------------------------------------------ mode & arming

    def set_mode(self, mode: str):
        with self._lock:
            if mode not in MODES:
                return
            if mode != self.mode:
                self.mode = mode
                constants.EXECUTOR["mode"] = mode
                self._disarm_locked(None)
                self.log(f"Executor mode: {mode}.")

    def arm(self) -> bool:
        with self._lock:
            if self.mode != "assist":
                return False
            if self.running_fn is not None and not self.running_fn():
                # Without live snapshots every guard would evaluate a frozen
                # frame — arming then is a blind click waiting to happen.
                self.disarm_reason = "detection is not running"
                self.log("Executor cannot arm — detection is not running.",
                         level="WARNING")
                return False
            self.armed = True
            self.disarm_reason = None
            self.log("Executor ARMED — the confirm key fires real clicks.",
                     level="WARNING")
            return True

    def disarm(self, reason=None):
        with self._lock:
            self._disarm_locked(reason)

    def _disarm_locked(self, reason):
        # Any disarm voids in-flight verification — judging a frozen spec
        # minutes later against unrelated state would corrupt the audit
        # telemetry the ARM gate quotes.
        self._verify = None
        if self.armed:
            self.armed = False
            self.disarm_reason = reason
            if reason:
                self.log(f"Executor DISARMED — {reason}", level="WARNING")
        elif reason:
            self.disarm_reason = reason

    def kill(self, source="kill switch"):
        """Hard stop: disarm + drop any pending verification state."""
        with self._lock:
            self._verify = None
            self._disarm_locked(source)

    # ------------------------------------------------------------- planning

    def _build_plan(self, snap):
        """The click the advice implies right now, or (None, reason)."""
        state = snap.get("phase") or {}
        name = state.get("phase")
        if name == phase.MY_TURN:
            return self._plan_action(snap, state)
        if name == phase.BETTING_OPEN:
            return self._plan_bet(snap, state)
        return None, None

    def _plan_action(self, snap, state):
        seat_idx = state.get("my_seat")
        if seat_idx is None:
            return None, "no owned seat to act for"
        if not state.get("seat_confident"):
            return None, "seat attribution ambiguous — abstaining"
        seats = snap.get("seats", [])
        if not (0 <= seat_idx < len(seats)) or not seats[seat_idx].get("mine"):
            return None, "active seat is not yours"
        seat = seats[seat_idx]
        code = seat.get("optimal_action")
        if isinstance(code, list):
            return None, "split hands need manual play (v1)"
        if not code:
            return None, "exact advice still computing — abstaining"
        if code == "R":
            return None, "surrender has no captured control"
        control = _ACTION_TO_CONTROL.get(code)
        btn = (state.get("buttons") or {}).get(control)
        if not btn or not btn.get("found"):
            return None, f"{control} button not located — abstaining"
        if not btn.get("enabled"):
            return None, f"{control} button looks disabled — abstaining"
        n_cards = len(seat.get("cards", []))
        return {
            "kind": "action", "action": code, "control": control,
            "seat": seat_idx, "target": tuple(btn["center"]),
            "confidence": float(btn.get("score") or 0.0),
            "round": snap["round"], "phase": phase.MY_TURN,
            "n_cards": n_cards, "split": bool(seat.get("split")),
            "key": (snap["round"], "turn", seat_idx, n_cards),
            "label": f"{ACTION_NAMES[code]} (P{seat_idx + 1})",
        }, None

    def _plan_bet(self, snap, state):
        amount = snap.get("bet_suggested")
        if amount is None:
            amount = snap.get("bet_placed")
        if snap.get("bet_sit_out"):
            return None, "sit out — no bet this round"
        if not amount or amount <= 0:
            return None, "no bet amount to place"
        ocr_vals = snap.get("ocr") or {}
        seen_bet = ocr_vals.get("bet")
        if seen_bet and time.time() - (ocr_vals.get("ts") or 0) < 10 \
                and seen_bet >= amount * 0.99:
            return None, "bet already on the table"
        spot = state.get("bet_spot")
        if spot and spot.get("changed"):
            return None, "chips already sit on the spot"
        seq = chip_sequence(amount, constants.EXECUTOR.get("chips", []))
        if seq is None:
            return None, (f"€{amount:g} not representable with the "
                          "configured chips — abstaining")
        if spot is None:
            return None, "bet spot not calibrated"
        chips_txt = " + ".join(f"{c:g}" for c in seq)
        return {
            "kind": "bet", "action": "BET", "control": "bet_spot",
            "seat": None, "target": tuple(spot["center"]),
            "confidence": 1.0, "round": snap["round"],
            "phase": phase.BETTING_OPEN, "amount": float(amount),
            # The amount is part of the key: a suggestion that moves during
            # the betting window is a NEW plan and re-audits.
            "chips": seq, "key": (snap["round"], "bet", float(amount)),
            "label": f"Bet €{amount:g} ({chips_txt})",
        }, None

    # ---------------------------------------------------------------- step

    def step(self, snap) -> dict:
        """Per-snapshot tick (Tk thread): refresh the plan, run the passive
        guard checks, advance verification. Returns the display state."""
        with self._lock:
            if self.mode == "off" or snap is None:
                self._plan = None
                return self._display(None, None)
            plan, reason = self._build_plan(snap)
            if plan and plan["key"] not in self._logged_keys:
                self._logged_keys.add(plan["key"])
                self.session["plans"] += 1
                self._audit({**plan, "mode": self.mode, "fired": False,
                             "reason": f"planned: {plan['label']}"})
            self._plan = plan
            self._passive_guards(snap)
            self._tick_verify(snap)
            return self._display(plan, reason)

    def _passive_guards(self, snap):
        """Auto-disarm conditions that hold regardless of firing. Runs on
        every GUI poll tick (not just new snapshots) so a stalled worker
        still disarms."""
        if not self.armed:
            return
        cfg = constants.EXECUTOR
        snap_ts = float(snap.get("ts") or 0)
        if snap_ts and (time.time() - snap_ts
                        > 2 * float(cfg.get("max_snapshot_age_s", 2.5))):
            self._disarm_locked("snapshots stalled — detection hung?")
            return
        state = snap.get("phase") or {}
        if state.get("phase") == phase.UNKNOWN:
            self._disarm_locked("unidentified screen (modal/overlay?)")
            return
        pnl = (snap.get("session_pnl") or {}).get("eur", 0.0)
        stop_loss = float(cfg.get("stop_loss_eur") or 0)
        if stop_loss and pnl <= -stop_loss:
            self._disarm_locked(f"stop-loss hit (€{pnl:+.2f})")
            return
        stop_win = float(cfg.get("stop_win_eur") or 0)
        if stop_win and pnl >= stop_win:
            self._disarm_locked(f"stop-win reached (€{pnl:+.2f})")
            return
        ocr_vals = snap.get("ocr") or {}
        balance = ocr_vals.get("balance")
        bankroll = snap.get("bankroll")
        tolerance = float(cfg.get("balance_tolerance_eur") or 0)
        # Explicit None checks: a €0.00 balance (busted account) is exactly
        # when this guard matters most, and `or balance` made the
        # comparison self-cancelling for a falsy bankroll.
        if (balance is not None and bankroll is not None and tolerance
                and time.time() - (ocr_vals.get("ts") or 0) < 10
                and abs(balance - bankroll) > tolerance):
            self._disarm_locked(
                f"balance mismatch (screen €{balance:g} vs model "
                f"€{bankroll:g})")

    # -------------------------------------------------------------- confirm

    def confirm(self) -> str:
        """The one-key fire path (assist mode). Re-plans against a FRESH
        snapshot, runs the full guard stack, requires the fresh plan to BE
        the plan the human was shown, then dispatches on the persistent
        click thread. Returns a status string for the GUI/log."""
        with self._lock:
            if self.mode != "assist":
                return "confirm ignored — executor is not in assist mode"
            if not self.armed:
                return "confirm ignored — not armed"
            if self._firing:
                return "confirm ignored — a click is already in flight"
            if self._verify is not None:
                return "confirm refused — previous click still verifying"
            snap = self.snapshot_fn()
            if snap is None:
                return "confirm refused — no snapshot"
            self._passive_guards(snap)
            if not self.armed:
                return f"confirm refused — {self.disarm_reason}"
            plan, reason = self._build_plan(snap)
            if plan is None:
                return f"confirm refused — {reason or 'nothing to do'}"
            ok, why = self._fire_guards(snap, plan)
            if not ok:
                return f"confirm refused — {why}"
            shown = self._plan
            if (shown is None or plan["key"] != shown["key"]
                    or plan["action"] != shown["action"]
                    or abs(plan["target"][0] - shown["target"][0]) > 8
                    or abs(plan["target"][1] - shown["target"][1]) > 8):
                # The human approves what the MARKER shows. If the state
                # rolled between their glance and the key press (advice
                # flipped, seat advanced, new decision), firing the new plan
                # would stake money on an action they never saw.
                return ("confirm refused — the plan changed; check the "
                        "marker and press again")
            self._fired_keys.add(plan["key"])
            self._round_actions[plan["round"]] = \
                self._round_actions.get(plan["round"], 0) + 1
            self.session["fired"] += 1
            self._firing = True
            self._audit({**plan, "mode": self.mode, "fired": True,
                         "reason": f"confirmed: {plan['label']}"})
            spec = self._verify_spec(snap, plan)
            self.dispatcher.submit(self._dispatch_job, plan, spec)
            return f"firing: {plan['label']}"

    def _fire_guards(self, snap, plan):
        """The §5 per-click precondition stack. Abstaining is always safe;
        any money/desync category also disarms."""
        cfg = constants.EXECUTOR
        state = snap.get("phase") or {}
        age = time.time() - float(snap.get("ts") or 0)
        if age > float(cfg.get("max_snapshot_age_s", 2.5)):
            return False, (f"snapshot is {age:.1f}s old — detection "
                           "stalled? never click a dead frame")
        if plan["kind"] != "action":
            return False, "bet execution is ghost-only in v1 — place chips manually"
        if plan["key"] in self._fired_keys:
            return False, "already clicked for this decision (never re-click)"
        if state.get("frames", 0) < int(cfg.get("confirm_frames", 3)):
            return False, (f"phase only confirmed {state.get('frames', 0)} "
                           f"frame(s) — need {cfg.get('confirm_frames', 3)}")
        if not state.get("seat_confident"):
            return False, "seat attribution ambiguous"
        if plan["confidence"] < float(cfg.get("min_button_score", 0.8)):
            return False, (f"button match {plan['confidence']:.2f} below "
                           f"{cfg.get('min_button_score', 0.8)}")
        if self._round_actions.get(plan["round"], 0) >= int(
                cfg.get("max_actions_per_round", 6)):
            self._disarm_locked("max actions per round exceeded")
            return False, "max actions per round exceeded — disarmed"
        monitor = self.monitor_fn()
        if monitor is None:
            return False, "no monitor selected"
        x, y = plan["target"]
        if not (0 <= x < monitor.width and 0 <= y < monitor.height):
            return False, "target outside the monitor — geometry desync"
        if plan["action"] in ("D", "P"):
            # The stake about to be doubled: the model's bet_placed,
            # cross-checked against the fresher of the OCR'd screen bet.
            bet = float(snap.get("bet_placed") or 0.0)
            ocr_vals = snap.get("ocr") or {}
            if (ocr_vals.get("bet")
                    and time.time() - (ocr_vals.get("ts") or 0) < 10):
                bet = max(bet, float(ocr_vals["bet"]))
            if bet <= 0:
                return False, ("stake unknown (bet placed is €0) — set it "
                               "before doubling/splitting")
            max_bet = float(cfg.get("max_bet_eur") or 0)
            if max_bet and bet * 2 > max_bet:
                self._disarm_locked(
                    f"{ACTION_NAMES[plan['action']]} would stake "
                    f"€{bet * 2:g} > max €{max_bet:g}")
                return False, "money limit — disarmed"
        return True, ""

    def _dispatch_job(self, plan, spec):
        with self._lock:
            if not self.armed:
                # Killed/disarmed between confirm and dispatch: abort.
                self._firing = False
                self.log("Click aborted — disarmed before dispatch.",
                         level="WARNING")
                return
        try:
            backend = self.dispatcher.click(*plan["target"])
            # Verification arms only AFTER the click is delivered —
            # otherwise a pre-click snapshot could be judged (and stamped)
            # as the click's outcome.
            with self._lock:
                spec["deadline"] = time.monotonic() + float(
                    constants.EXECUTOR.get("verify_timeout_s", 4.0))
                if self.armed:
                    self._verify = spec
            self.log(f"Executor clicked {plan['label']} via {backend}.")
        except ExecutorAbort as e:
            with self._lock:
                self._disarm_locked(str(e))
        except Exception as e:
            with self._lock:
                self._disarm_locked(f"click delivery failed: {e}")
        finally:
            with self._lock:
                self._firing = False

    # ------------------------------------------------------- verification

    @staticmethod
    def _verify_spec(snap, plan):
        return {
            "plan": plan,
            "deadline": time.monotonic() + float(
                constants.EXECUTOR.get("verify_timeout_s", 4.0)),
            "round": snap["round"],
            "n_cards": plan.get("n_cards"),
            "split": plan.get("split"),
        }

    def _tick_verify(self, snap):
        v = self._verify
        if v is None:
            return
        plan = v["plan"]
        outcome, observed = self._verify_outcome(snap, v, plan)
        if outcome is None and time.monotonic() <= v["deadline"]:
            return
        self._verify = None
        verified = bool(outcome)
        if verified:
            self.session["verified"] += 1
            self.log(f"Executor verified: {plan['label']} "
                     f"({observed}).")
        else:
            self.session["mismatches"] += 1
            observed = observed or "no observable effect"
            self._disarm_locked(
                f"post-click verification failed for {plan['label']} — "
                "never re-clicking")
        if self.store is not None:
            self.io_submit(self.store.update_executor_verify, plan["round"],
                           plan["phase"], plan["action"], verified, observed,
                           plan.get("seat"))

    def _verify_outcome(self, snap, v, plan):
        """(True/None, observed) — None means keep waiting."""
        state = snap.get("phase") or {}
        seats = snap.get("seats", [])
        seat = seats[plan["seat"]] if plan.get("seat") is not None and \
            plan["seat"] < len(seats) else None
        if snap["round"] != v["round"]:
            # The round rolled — for Stand that's a completed playout.
            return (True, "round advanced") if plan["action"] == "S" \
                else (False, "round advanced before the effect was seen")
        if plan["action"] == "P":
            if seat is not None and seat.get("split"):
                return True, "seat split into two hands"
            return None, None
        if plan["action"] in ("H", "D"):
            if seat is not None and len(seat.get("cards", [])) > v["n_cards"]:
                return True, "card count grew"
            return None, None
        if plan["action"] == "S":
            if state.get("phase") != phase.MY_TURN:
                return True, "turn ended"
            return None, None
        return None, None

    # ------------------------------------------------------------- display

    def _display(self, plan, reason):
        state = {
            "mode": self.mode,
            "armed": self.armed,
            "disarm_reason": self.disarm_reason,
            "session": dict(self.session),
            "plan": plan,
            "status": "",
            "marker": None,
        }
        if self.mode == "off":
            return state
        if plan is not None:
            verb = "WOULD" if self.mode == "ghost" or not self.armed else \
                "READY (confirm key fires)"
            state["status"] = f"{verb}: {plan['label']} " \
                              f"@({plan['target'][0]},{plan['target'][1]}) " \
                              f"{plan['confidence'] * 100:.0f}%"
            color = "#4dabf7" if self.mode == "ghost" or not self.armed \
                else "#dc3545"
            state["marker"] = {"x": plan["target"][0], "y": plan["target"][1],
                               "text": plan["label"], "color": color}
        elif reason:
            state["status"] = reason
        elif self.disarm_reason:
            state["status"] = f"disarmed: {self.disarm_reason}"
        return state

    # --------------------------------------------------------------- audit

    def _audit(self, entry):
        if self.store is None:
            return
        try:
            self.io_submit(self.store.record_executor, entry)
        except Exception:
            pass  # the audit trail must never break the planner

    def stats(self):
        if self.store is None:
            return None
        try:
            return self.store.executor_stats(session_only=True)
        except Exception:
            return None
