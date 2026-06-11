"""Ghost-click marker (V4 Feature 2) — draws WHERE the executor would
click, on top of the casino stream.

A tiny frameless always-on-top Toplevel positioned at the target's screen
coordinates: a ring + action label. Reuses the OverlayHUD Win32 recipe
(WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW so it never steals focus or shows in
alt-tab) plus WS_EX_TRANSPARENT so it is CLICK-THROUGH — an OS-input click
at the marked pixel must hit the casino button underneath, not the marker.
The background color is keyed transparent via -transparentcolor.

Tk-thread only, like every interface module.
"""

import ctypes
import sys
import tkinter as tk

from ..common import constants

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_TRANSPARENT = 0x00000020

_KEY = "#010203"  # transparency key — never used by the drawing
SIZE = 56         # ring canvas, px
LABEL_H = 22


class GhostMarker(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        try:
            self.attributes("-transparentcolor", _KEY)
        except tk.TclError:
            pass  # non-Windows: an opaque marker is still useful
        self.configure(bg=_KEY)
        self.canvas = tk.Canvas(self, width=SIZE, height=SIZE + LABEL_H,
                                bg=_KEY, highlightthickness=0)
        self.canvas.pack()
        self._ring = self.canvas.create_oval(
            6, 6, SIZE - 6, SIZE - 6, outline="#4dabf7", width=3)
        self._dot = self.canvas.create_oval(
            SIZE / 2 - 3, SIZE / 2 - 3, SIZE / 2 + 3, SIZE / 2 + 3,
            fill="#4dabf7", outline="")
        self._label = self.canvas.create_text(
            SIZE / 2, SIZE + LABEL_H / 2, text="",
            font=(constants.FONT_FAMILY, 9, "bold"), fill="#4dabf7")
        self.withdraw()
        self.after(80, self._apply_win_styles)

    def _apply_win_styles(self):
        if sys.platform != "win32":
            return
        try:
            self.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE,
                style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT)
        except Exception:
            pass

    def show(self, screen_x, screen_y, text, color):
        """Center the ring on the absolute (virtual-desktop) screen point."""
        if not self.winfo_exists():
            return
        for item in (self._ring,):
            self.canvas.itemconfigure(item, outline=color)
        self.canvas.itemconfigure(self._dot, fill=color)
        self.canvas.itemconfigure(self._label, text=text, fill=color)
        self.geometry(f"+{int(screen_x - SIZE / 2)}+{int(screen_y - SIZE / 2)}")
        self.deiconify()
        self.attributes("-topmost", True)

    def hide(self):
        if self.winfo_exists():
            self.withdraw()
