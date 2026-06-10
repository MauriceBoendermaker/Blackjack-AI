"""Bankroll & Risk window (V2 Feature 2).

Shows the risk numbers for the configured ramp and bankroll: lifetime and
trip risk of ruin, the Kelly-fraction risk table, DI/SCORE/CE, the bankroll
needed for target risk levels — and a Monte Carlo simulation that resamples
the user's own recorded rounds when enough exist (falling back to the
TC-frequency model otherwise).
"""

import tkinter as tk
from tkinter import messagebox

from ..common import constants
from ..logic import bankroll as br
from .validation import attach_numeric_entry

C = constants.COLORS


class BankrollWindow(tk.Toplevel):
    def __init__(self, parent, store=None):
        super().__init__(parent)
        self.title("Bankroll & Risk")
        self.configure(bg=C["bg_secondary"], padx=20, pady=16)
        self.resizable(False, False)
        self.store = store
        self._vars = {}

        tk.Label(self, text="Bankroll & Risk", font=constants.FONT_TITLE,
                 bg=C["bg_secondary"], fg=C["text_primary"]).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.source_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.source_var, font=constants.FONT_SMALL,
                 bg=C["bg_secondary"], fg=C["text_secondary"], wraplength=380,
                 justify="left").grid(row=1, column=0, columnspan=2, sticky="w",
                                      pady=(0, 8))

        rows = [
            ("mu_sigma", "Per round (μ / σ)"),
            ("ror_life", "Lifetime risk of ruin"),
            ("ror_trip", "Trip risk of ruin"),
            ("kelly", "Ruin at ¼ / ½ / full Kelly"),
            ("need", "Bankroll for 5% / 1% ruin"),
            ("di", "DI / SCORE"),
            ("ce", "Certainty equivalent / round"),
        ]
        for i, (key, label) in enumerate(rows, start=2):
            tk.Label(self, text=label, font=constants.FONT_BODY, anchor="w",
                     bg=C["bg_secondary"], fg=C["text_secondary"], width=26
                     ).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="—")
            tk.Label(self, textvariable=var, font=constants.FONT_BODY_BOLD,
                     bg=C["bg_secondary"], fg=C["text_primary"]
                     ).grid(row=i, column=1, sticky="w", padx=(12, 0))
            self._vars[key] = var

        row = 2 + len(rows)
        sim = tk.Frame(self, bg=C["bg_secondary"])
        sim.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 2))
        tk.Label(sim, text="Simulate", font=constants.FONT_SECTION,
                 bg=C["bg_secondary"], fg=C["text_primary"]).pack(side=tk.LEFT)
        self.rounds_var = tk.IntVar(value=1000)
        rounds_spin = tk.Spinbox(sim, from_=100, to=100_000, increment=100,
                                 textvariable=self.rounds_var, width=8)
        rounds_spin.pack(side=tk.LEFT, padx=8)
        attach_numeric_entry(rounds_spin, integer=True)
        tk.Label(sim, text="rounds ×10k futures", font=constants.FONT_BODY,
                 bg=C["bg_secondary"], fg=C["text_secondary"]).pack(side=tk.LEFT)
        tk.Button(sim, text="Run Monte Carlo", command=self._simulate,
                  bg=C["accent"], fg="white", relief="flat", padx=14, pady=6,
                  font=constants.FONT_BODY, cursor="hand2").pack(side=tk.RIGHT)

        self.mc_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.mc_var, font=constants.FONT_BODY,
                 bg=C["bg_secondary"], fg=C["text_primary"], justify="left",
                 wraplength=400).grid(row=row + 1, column=0, columnspan=2,
                                      sticky="w", pady=(6, 0))
        self.refresh()

    # ------------------------------------------------------------- data

    def _round_stats(self):
        """((mu, sigma), outcomes, source_label) — empirical when possible."""
        outcomes = []
        if self.store is not None:
            try:
                outcomes = self.store.settled_pnl(session_only=False)
            except Exception:
                outcomes = []
        stats = br.empirical_round_stats(outcomes)
        if stats is not None:
            return stats, outcomes, (f"Source: {len(outcomes)} recorded settled "
                                     "rounds (your actual results).")
        mu_sigma = br.model_round_stats()
        return mu_sigma, outcomes, (
            "Source: TC-frequency model × your configured ramp "
            f"({len(outcomes)} settled rounds recorded — needs 30+ to switch "
            "to your real data).")

    def refresh(self):
        (mu, sigma), _, source = self._round_stats()
        bank = constants.BETTING["bankroll"]
        self.source_var.set(source)
        self._vars["mu_sigma"].set(f"€{mu:+.2f} / €{sigma:.2f}")
        self._vars["ror_life"].set(f"{br.lifetime_ror(mu, sigma, bank) * 100:.2f}%")
        self._vars["ror_trip"].set(
            f"{br.trip_ror(mu, sigma, bank, 1000) * 100:.2f}% over 1000 rounds")
        self._vars["kelly"].set("  /  ".join(
            f"{br.kelly_fixed_ror(f) * 100:.2f}%" for f in (0.25, 0.5, 1.0)))
        need5 = br.bankroll_for_ror(0.05, mu, sigma)
        need1 = br.bankroll_for_ror(0.01, mu, sigma)
        if need5 is None:
            self._vars["need"].set("n/a — edge is not positive")
        else:
            self._vars["need"].set(f"€{need5:,.0f} / €{need1:,.0f}")
        di, score = br.di_score(mu, sigma)
        self._vars["di"].set(f"{di:.2f} / {score:.1f}")
        self._vars["ce"].set(f"€{br.certainty_equivalent(mu, sigma, bank):+.3f}")

    def _simulate(self):
        (mu, sigma), outcomes, _ = self._round_stats()
        try:
            n_rounds = int(self.rounds_var.get())
        except tk.TclError:  # field left empty mid-edit
            n_rounds = 1000
            self.rounds_var.set(n_rounds)
        if len(outcomes) < 30:
            # Model fallback: synthesize outcomes ~ N(mu, sigma) via numpy.
            import numpy as np
            rng = np.random.default_rng(7)
            outcomes = list(rng.normal(mu, sigma, size=5000))
        result = br.monte_carlo(outcomes, constants.BETTING["bankroll"],
                                n_rounds)
        if result is None:
            messagebox.showerror("Monte Carlo", "Not enough data to simulate.",
                                 parent=self)
            return
        self.mc_var.set(
            f"Ruin: {result['ruin'] * 100:.2f}%   ·   "
            f"P(profit): {result['p_profit'] * 100:.1f}%\n"
            f"Final after {result['n_rounds']} rounds: "
            f"€{result['final_p10']:+,.0f} (p10) / €{result['final_p50']:+,.0f} (median) / "
            f"€{result['final_p90']:+,.0f} (p90)\n"
            f"Max drawdown: €{result['drawdown_p50']:,.0f} (median) / "
            f"€{result['drawdown_p90']:,.0f} (p90)")
