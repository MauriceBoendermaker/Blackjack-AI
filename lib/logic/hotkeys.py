"""Global hotkey watcher (V4 Feature 2) — confirm key and kill switch.

Tk key bindings only fire while the app has focus, but during assisted
execution the BROWSER has focus. This polls GetAsyncKeyState on a daemon
thread (stdlib ctypes, no extra dependency, no keyboard hook to trip AV
heuristics) and fires the callback on the key's DOWN edge.

Callbacks run on the watcher thread — callers marshal to the Tk thread
themselves (the established `widget.after(0, ...)` pattern).
"""

import sys
import threading


class GlobalHotkeys:
    def __init__(self, bindings: dict, poll_s: float = 0.04):
        """bindings: {virtual_key_code: callback}. Windows-only; start() is
        a no-op elsewhere."""
        self.bindings = dict(bindings)
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running or sys.platform != "win32":
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="hotkeys")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        import ctypes
        get_state = ctypes.windll.user32.GetAsyncKeyState
        down = {vk: False for vk in self.bindings}
        while not self._stop.is_set():
            for vk, callback in self.bindings.items():
                pressed = bool(get_state(vk) & 0x8000)
                if pressed and not down[vk]:
                    try:
                        callback()
                    except Exception:
                        pass  # a callback error must never kill the watcher
                down[vk] = pressed
            self._stop.wait(self.poll_s)
