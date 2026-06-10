"""Session statistics window (Feature 8): round history summary + CSV export."""

import time
import tkinter as tk
from tkinter import filedialog, messagebox

from ..common import constants

C = constants.COLORS

_ROWS = [
    ("net_eur", "Net P&L (€, owned seats)"),
    ("net_units", "Net P&L (units)"),
    ("win_rate", "Hand win rate (excl. pushes)"),
    ("settled_rounds", "Rounds settled"),
    ("rounds", "Rounds recorded"),
    ("avg_tc", "Average true count"),
    ("max_tc", "Best true count"),
    ("min_tc", "Worst true count"),
    ("rounds_tc_2_plus", "Rounds at TC ≥ +2 (raise spots)"),
    ("dealer_ace_rounds", "Dealer-ace rounds"),
    ("insurance_take_rounds", "Rounds where insurance was +EV"),
    ("plus_ev_sidebet_rounds", "Rounds with a +EV side bet"),
]


class StatsWindow(tk.Toplevel):
    def __init__(self, parent, store, engine=None):
        super().__init__(parent)
        self.title("Session Statistics")
        self.configure(bg=C["bg_secondary"], padx=20, pady=16)
        self.resizable(False, False)
        self.store = store
        self.engine = engine  # live snapshot source for session-only counters
        self._vars = {}

        tk.Label(self, text="Session Statistics", font=constants.FONT_TITLE,
                 bg=C["bg_secondary"], fg=C["text_primary"]).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self.scope = tk.StringVar(value="session")
        scope_row = tk.Frame(self, bg=C["bg_secondary"])
        scope_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))
        for text, value in (("This session", "session"), ("All recorded", "all")):
            tk.Radiobutton(scope_row, text=text, value=value, variable=self.scope,
                           command=self.refresh, bg=C["bg_secondary"],
                           fg=C["text_primary"], font=constants.FONT_BODY,
                           activebackground=C["bg_secondary"]).pack(side=tk.LEFT, padx=(0, 10))

        for i, (key, label) in enumerate(_ROWS, start=2):
            tk.Label(self, text=label, font=constants.FONT_BODY, anchor="w",
                     bg=C["bg_secondary"], fg=C["text_secondary"], width=30
                     ).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="—")
            tk.Label(self, textvariable=var, font=constants.FONT_BODY_BOLD,
                     bg=C["bg_secondary"], fg=C["text_primary"]
                     ).grid(row=i, column=1, sticky="w", padx=(12, 0))
            self._vars[key] = var

        row = 2 + len(_ROWS)
        # Bet-cap telemetry lives in the engine (session-scoped), not the DB.
        tk.Label(self, text="Bets capped at table max", font=constants.FONT_BODY,
                 anchor="w", bg=C["bg_secondary"], fg=C["text_secondary"],
                 width=30).grid(row=row, column=0, sticky="w", pady=2)
        self._bet_capped_var = tk.StringVar(value="—")
        tk.Label(self, textvariable=self._bet_capped_var,
                 font=constants.FONT_BODY_BOLD, bg=C["bg_secondary"],
                 fg=C["text_primary"]).grid(row=row, column=1, sticky="w",
                                            padx=(12, 0))
        row += 1

        tk.Label(self, text="Per-seat accuracy", font=constants.FONT_SECTION,
                 bg=C["bg_secondary"], fg=C["text_primary"]).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(12, 4))
        row += 1
        self.seat_frame = tk.Frame(self, bg=C["bg_secondary"])
        self.seat_frame.grid(row=row, column=0, columnspan=2, sticky="ew")
        row += 1

        buttons = tk.Frame(self, bg=C["bg_secondary"])
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew",
                     pady=(14, 0))
        tk.Button(buttons, text="Export CSV", command=self._export,
                  bg=C["accent"], fg="white", relief="flat", padx=14, pady=6,
                  font=constants.FONT_BODY, cursor="hand2").pack(side=tk.RIGHT, padx=4)
        tk.Button(buttons, text="Refresh", command=self.refresh,
                  bg=C["text_secondary"], fg="white", relief="flat", padx=14, pady=6,
                  font=constants.FONT_BODY, cursor="hand2").pack(side=tk.RIGHT, padx=4)

        self.refresh()

    def refresh(self):
        session_only = self.scope.get() == "session"
        try:
            stats = self.store.stats(session_only=session_only)
            per_seat = self.store.seat_stats(session_only=session_only)
        except Exception as e:
            messagebox.showerror("Session Statistics", f"Could not read stats: {e}",
                                 parent=self)
            return
        for key, var in self._vars.items():
            value = stats.get(key, 0)
            if key == "win_rate":
                var.set(f"{value * 100:.1f}%")
            elif isinstance(value, float):
                var.set(f"{value:+.2f}")
            else:
                var.set(str(value))
        if self.engine is not None:
            snap = self.engine.get_snapshot() or {}
            self._bet_capped_var.set(
                f"{snap.get('bet_capped_rounds', 0)} (this session)")
        self._render_seat_stats(per_seat)

    def _render_seat_stats(self, per_seat):
        """Rebuild the per-seat accuracy rows (seat set varies per refresh)."""
        for child in self.seat_frame.winfo_children():
            child.destroy()
        if not per_seat:
            tk.Label(self.seat_frame, text="No recorded hands yet.",
                     font=constants.FONT_BODY, bg=C["bg_secondary"],
                     fg=C["text_secondary"]).grid(row=0, column=0, sticky="w",
                                                  pady=2)
            return
        for i, (idx, s) in enumerate(per_seat.items()):
            tk.Label(self.seat_frame, text=f"Seat {idx + 1}",
                     font=constants.FONT_BODY, anchor="w", bg=C["bg_secondary"],
                     fg=C["text_secondary"], width=30
                     ).grid(row=i, column=0, sticky="w", pady=2)
            book = (f"{s['book_pct'] * 100:.0f}% book-played"
                    if s["book_pct"] is not None else "book n/a")
            units = (f"{s['avg_units']:+.2f} u/hand"
                     if s["avg_units"] is not None else "— u/hand")
            tk.Label(self.seat_frame,
                     text=(f"{s['hands']} hands · {book} · {units}"
                           f" · {s['divergences']} EV≠book"),
                     font=constants.FONT_BODY_BOLD, bg=C["bg_secondary"],
                     fg=C["text_primary"]).grid(row=i, column=1, sticky="w",
                                                padx=(12, 0))

    def _export(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".csv",
            initialfile=f"blackjack_session_{time.strftime('%Y%m%d_%H%M')}.csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            n = self.store.export_csv(path)
        except Exception as e:
            messagebox.showerror("Export failed", str(e), parent=self)
            return
        messagebox.showinfo("Export complete", f"{n} rounds written to\n{path}",
                            parent=self)
