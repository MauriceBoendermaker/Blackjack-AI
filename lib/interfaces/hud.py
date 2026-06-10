"""Overlay HUD (V2 Feature 4 — visual only, no audio).

A compact frameless always-on-top panel to park next to the casino stream:
true count, exact edge + bet call, insurance alert, advice for your seats,
+EV side-bet flags, session P&L. Verified Windows recipe: overrideredirect +
-topmost (reasserted on a timer — other topmost windows can climb above),
WS_EX_NOACTIVATE via ctypes so it never steals focus from the browser, and
WS_EX_TOOLWINDOW to keep it out of alt-tab. Drag anywhere to move; ✕ closes.

Works over browsers incl. F11/HTML5 fullscreen (borderless, DWM-composited).
Updates arrive from the main GUI's snapshot poll — main thread only.
"""

import ctypes
import sys
import tkinter as tk

from ..common import constants

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080

BG = "#10151b"
FG = "#e8f4ec"
DIM = "#8aa399"
ALERT = "#ffc107"


def format_hud_lines(snap) -> dict:
    """Snapshot -> display strings. Pure (testable without Tk).

    Returns {"count", "bet", "insurance", "seats": [str], "sidebets", "pnl"}.
    """
    count = snap["count"]
    lines = {
        "count": f"TC {count['true']:+.1f}   RC {count['running']:+d}   "
                 f"{count['decks_remaining']:.1f} decks",
        "bet": snap.get("bet", ""),
        "insurance": "",
        "seats": [],
        "sidebets": "",
        "pnl": "",
    }
    ins = snap.get("insurance")
    if ins:
        lines["insurance"] = ins["text"]

    seats = snap.get("seats", [])
    chosen = [s for s in seats if s.get("mine")]
    if not chosen:
        chosen = [s for s in seats if s.get("cards")][:3]
    for s in chosen:
        if not s.get("cards"):
            continue
        star = " ★" if s.get("mine") else ""
        advice = s.get("optimal") or s.get("advice") or ""
        advice = advice.replace("Optimal: ", "").replace("\n", " · ")
        if advice:
            lines["seats"].append(f"P{s['index'] + 1}{star}: {advice}")

    plus = [f"{b['label']} {b['ev'] * 100:+.1f}%"
            for b in snap.get("side_bets", [])
            if b.get("ev") is not None and b["ev"] > 0]
    if plus:
        lines["sidebets"] = "● BET  " + "  ·  ".join(plus)

    pnl = snap.get("session_pnl") or {}
    if pnl.get("rounds"):
        lines["pnl"] = f"Session €{pnl['eur']:+.2f} ({pnl['units']:+g}u)"
    return lines


class OverlayHUD(tk.Toplevel):
    def __init__(self, parent, on_close=None):
        super().__init__(parent)
        self.on_close = on_close
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(bg=BG, padx=12, pady=8,
                       highlightbackground="#2c3a44", highlightthickness=1)
        self.geometry("+60+60")

        header = tk.Frame(self, bg=BG)
        header.pack(fill=tk.X)
        tk.Label(header, text="BLACKJACK AI", font=(constants.FONT_FAMILY, 8, "bold"),
                 bg=BG, fg=DIM).pack(side=tk.LEFT)
        close = tk.Label(header, text="✕", font=(constants.FONT_FAMILY, 9, "bold"),
                         bg=BG, fg=DIM, cursor="hand2")
        close.pack(side=tk.RIGHT)
        close.bind("<Button-1>", lambda e: self.close())

        self._vars = {}
        self._labels = {}
        specs = [
            ("count", (constants.FONT_FAMILY, 14, "bold"), FG),
            ("bet", (constants.FONT_FAMILY, 10, "bold"), FG),
            ("insurance", (constants.FONT_FAMILY, 11, "bold"), ALERT),
            ("seats", (constants.FONT_FAMILY, 10), FG),
            ("sidebets", (constants.FONT_FAMILY, 9, "bold"), "#2fbf71"),
            ("pnl", (constants.FONT_FAMILY, 9), DIM),
        ]
        for key, font, fg in specs:
            var = tk.StringVar(value="")
            lbl = tk.Label(self, textvariable=var, font=font, bg=BG, fg=fg,
                           justify="left", anchor="w")
            self._vars[key] = var
            self._labels[key] = lbl

        # Drag anywhere on the panel to move it.
        for widget in (self, header):
            widget.bind("<Button-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_move)
        self._drag_origin = (0, 0)

        self.after(80, self._apply_win_styles)
        self._reassert_topmost()

    # ----------------------------------------------------------- behavior

    def _apply_win_styles(self):
        """Never steal focus from the casino tab; stay out of alt-tab."""
        if sys.platform != "win32":
            return
        try:
            self.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        except Exception:
            pass

    def _reassert_topmost(self):
        if not self.winfo_exists():
            return
        try:
            self.attributes("-topmost", True)
            self.lift()
        except tk.TclError:
            return
        self.after(2000, self._reassert_topmost)

    def _drag_start(self, event):
        self._drag_origin = (event.x_root - self.winfo_x(),
                             event.y_root - self.winfo_y())

    def _drag_move(self, event):
        ox, oy = self._drag_origin
        self.geometry(f"+{event.x_root - ox}+{event.y_root - oy}")

    def close(self):
        if self.on_close:
            self.on_close()
        self.destroy()

    # ------------------------------------------------------------- update

    def update_from_snapshot(self, snap):
        if snap is None or not self.winfo_exists():
            return
        lines = format_hud_lines(snap)
        for key, var in self._vars.items():
            text = lines[key]
            if key == "seats":
                text = "\n".join(lines["seats"])
            if var.get() != text:
                var.set(text)
            lbl = self._labels[key]
            if text and not lbl.winfo_manager():
                lbl.pack(fill=tk.X, pady=1)
            elif not text and lbl.winfo_manager():
                lbl.pack_forget()
