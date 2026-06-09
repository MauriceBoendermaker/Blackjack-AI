"""Session statistics window (Feature 8): round history summary + CSV export."""

import time
import tkinter as tk
from tkinter import filedialog, messagebox

from ..common import constants

C = constants.COLORS

_ROWS = [
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
    def __init__(self, parent, store):
        super().__init__(parent)
        self.title("Session Statistics")
        self.configure(bg=C["bg_secondary"], padx=20, pady=16)
        self.resizable(False, False)
        self.store = store
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

        buttons = tk.Frame(self, bg=C["bg_secondary"])
        buttons.grid(row=2 + len(_ROWS), column=0, columnspan=2, sticky="ew",
                     pady=(14, 0))
        tk.Button(buttons, text="Export CSV", command=self._export,
                  bg=C["accent"], fg="white", relief="flat", padx=14, pady=6,
                  font=constants.FONT_BODY, cursor="hand2").pack(side=tk.RIGHT, padx=4)
        tk.Button(buttons, text="Refresh", command=self.refresh,
                  bg=C["text_secondary"], fg="white", relief="flat", padx=14, pady=6,
                  font=constants.FONT_BODY, cursor="hand2").pack(side=tk.RIGHT, padx=4)

        self.refresh()

    def refresh(self):
        try:
            stats = self.store.stats(session_only=self.scope.get() == "session")
        except Exception as e:
            messagebox.showerror("Session Statistics", f"Could not read stats: {e}",
                                 parent=self)
            return
        for key, var in self._vars.items():
            value = stats.get(key, 0)
            var.set(f"{value:+.2f}" if isinstance(value, float) else str(value))

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
