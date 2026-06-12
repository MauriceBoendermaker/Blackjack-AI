"""Bankroll & Risk window (V2 Feature 2) + Ramp designer (V3 Feature 5).

Risk tab: the risk numbers for the configured ramp and bankroll — lifetime
and trip risk of ruin, the Kelly-fraction risk table, DI/SCORE/CE, the
bankroll needed for target risk levels — and a Monte Carlo simulation that
resamples the user's own recorded rounds when enough exist (falling back to
the TC-frequency model otherwise).

Ramp designer tab: solves a personal integer per-TC bet table for a target
risk of ruin from the TC distribution measured in session.db
(lib/logic/ramp_optimizer.py), shows current vs optimal side by side, and
installs the winner into BETTING["bet_table"] — the live ramp follows it
from the next snapshot on.
"""

import tkinter as tk
from tkinter import messagebox, ttk

from ..common import constants, settings
from ..logic import bankroll as br
from ..logic import ramp_optimizer
from .settings_dialog import notebook_style
from .validation import attach_numeric_entry

C = constants.COLORS

#: Nominal Evolution pace when too few rounds are recorded to measure one.
FALLBACK_ROUNDS_PER_HOUR = 60.0


class BankrollWindow(tk.Toplevel):
    def __init__(self, parent, store=None, engine=None):
        super().__init__(parent)
        self.title("Bankroll & Risk")
        self.configure(bg=C["bg_secondary"], padx=12, pady=10)
        self.resizable(False, False)
        self.store = store
        self.engine = engine
        self._vars = {}
        self._opt_result = None
        self._opt_freqs = None

        notebook = ttk.Notebook(self, style=notebook_style(self))
        notebook.pack(fill=tk.BOTH, expand=True)
        risk = tk.Frame(notebook, bg=C["bg_secondary"], padx=12, pady=8)
        ramp = tk.Frame(notebook, bg=C["bg_secondary"], padx=12, pady=8)
        notebook.add(risk, text="Risk")
        notebook.add(ramp, text="Ramp designer")
        self._build_risk_tab(risk)
        self._build_ramp_tab(ramp)
        self.refresh()

    # ------------------------------------------------------------ risk tab

    def _build_risk_tab(self, tab):
        tk.Label(tab, text="Bankroll & Risk", font=constants.FONT_TITLE,
                 bg=C["bg_secondary"], fg=C["text_primary"]).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.source_var = tk.StringVar(value="")
        tk.Label(tab, textvariable=self.source_var, font=constants.FONT_SMALL,
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
            tk.Label(tab, text=label, font=constants.FONT_BODY, anchor="w",
                     bg=C["bg_secondary"], fg=C["text_secondary"], width=26
                     ).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="—")
            tk.Label(tab, textvariable=var, font=constants.FONT_BODY_BOLD,
                     bg=C["bg_secondary"], fg=C["text_primary"]
                     ).grid(row=i, column=1, sticky="w", padx=(12, 0))
            self._vars[key] = var

        row = 2 + len(rows)
        sim = tk.Frame(tab, bg=C["bg_secondary"])
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
        tk.Label(tab, textvariable=self.mc_var, font=constants.FONT_BODY,
                 bg=C["bg_secondary"], fg=C["text_primary"], justify="left",
                 wraplength=400).grid(row=row + 1, column=0, columnspan=2,
                                      sticky="w", pady=(6, 0))

    # ------------------------------------------------------------- data

    def _my_seat_count(self) -> int:
        """Starred seats right now — the model's k for covariance-aware
        multi-seat sizing (V3 E5). The empirical sample needs no k: the
        recorded per-round EUR already contains the correlation."""
        if self.engine is None:
            return 1
        try:
            snap = self.engine.get_snapshot() or {}
        except Exception:
            return 1
        return max(1, sum(1 for s in snap.get("seats", [])
                          if s.get("mine")))

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
        seats = self._my_seat_count()
        mu_sigma = br.model_round_stats(seats=seats)
        note = (f" Sizing for {seats} simultaneous seats (covariance-aware)."
                if seats > 1 else "")
        return mu_sigma, outcomes, (
            "Source: TC-frequency model × your configured ramp "
            f"({len(outcomes)} settled rounds recorded — needs 30+ to switch "
            f"to your real data).{note}")

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
        self._refresh_ramp_status()

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

    # ------------------------------------------------------- ramp designer

    def _build_ramp_tab(self, tab):
        tk.Label(tab, text="Ramp designer", font=constants.FONT_TITLE,
                 bg=C["bg_secondary"], fg=C["text_primary"]).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))
        self.ramp_status_var = tk.StringVar(value="")
        tk.Label(tab, textvariable=self.ramp_status_var,
                 font=constants.FONT_SMALL, bg=C["bg_secondary"],
                 fg=C["text_secondary"], wraplength=420, justify="left").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(0, 8))

        def spin(col, label, var, lo, hi, step, integer=False, fmt=None):
            tk.Label(tab, text=label, font=constants.FONT_BODY,
                     bg=C["bg_secondary"], fg=C["text_secondary"]).grid(
                row=2, column=col, sticky="w", padx=(0 if col == 0 else 10, 0))
            box = tk.Spinbox(tab, from_=lo, to=hi, increment=step,
                             textvariable=var, width=6,
                             **({"format": fmt} if fmt else {}))
            box.grid(row=3, column=col, sticky="w",
                     padx=(0 if col == 0 else 10, 0))
            attach_numeric_entry(box, integer=integer)

        self.target_ror_var = tk.DoubleVar(value=5.0)
        self.chip_var = tk.DoubleVar(value=5.0)
        self.spread_var = tk.IntVar(value=20)
        spin(0, "Target RoR %", self.target_ror_var, 0.1, 50.0, 0.1,
             fmt="%.1f")
        spin(1, "Chip step €", self.chip_var, 0.5, 1000.0, 0.5, fmt="%.1f")
        spin(2, "Max spread 1:n", self.spread_var, 1, 200, 1, integer=True)
        self.sit_out_var = tk.BooleanVar(value=True)
        tk.Checkbutton(tab, text="Sit out at ≤0 edge", variable=self.sit_out_var,
                       bg=C["bg_secondary"], fg=C["text_secondary"],
                       selectcolor=C["bg_primary"], font=constants.FONT_BODY,
                       activebackground=C["bg_secondary"]).grid(
            row=3, column=3, sticky="w", padx=(12, 0))

        tk.Button(tab, text="Optimize", command=self._optimize,
                  bg=C["accent"], fg="white", relief="flat", padx=14, pady=6,
                  font=constants.FONT_BODY, cursor="hand2").grid(
            row=4, column=0, sticky="w", pady=(10, 6))
        self.freq_source_var = tk.StringVar(value="")
        tk.Label(tab, textvariable=self.freq_source_var,
                 font=constants.FONT_SMALL, bg=C["bg_secondary"],
                 fg=C["text_secondary"], wraplength=300, justify="left").grid(
            row=4, column=1, columnspan=3, sticky="w", pady=(10, 6),
            padx=(10, 0))

        self.result_frame = tk.Frame(tab, bg=C["bg_secondary"])
        self.result_frame.grid(row=5, column=0, columnspan=4, sticky="nsew")

        buttons = tk.Frame(tab, bg=C["bg_secondary"])
        buttons.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        self.apply_btn = tk.Button(buttons, text="Apply optimal ramp",
                                   command=self._apply_ramp, state=tk.DISABLED,
                                   bg=C["accent"], fg="white", relief="flat",
                                   padx=14, pady=6, font=constants.FONT_BODY,
                                   cursor="hand2")
        self.apply_btn.pack(side=tk.LEFT)
        tk.Button(buttons, text="Clear installed ramp",
                  command=self._clear_ramp, bg=C["bg_primary"],
                  fg=C["text_primary"], relief="flat", padx=14, pady=6,
                  font=constants.FONT_BODY, cursor="hand2").pack(
            side=tk.LEFT, padx=(8, 0))
        self._refresh_ramp_status()

    def _refresh_ramp_status(self):
        table = ramp_optimizer.from_bet_table(
            constants.BETTING.get("bet_table"))
        if table:
            lo, hi = min(table), max(table)
            self.ramp_status_var.set(
                f"Live ramp: per-TC bet table installed "
                f"({len(table)} buckets, TC {lo:+d}…{hi:+d}). The formula "
                "is overridden until cleared.")
        else:
            self.ramp_status_var.set(
                "Live ramp: fractional-Kelly formula (no bet table "
                "installed). Optimize and apply to follow a solved integer "
                "ramp instead.")

    def _rounds_per_hour(self):
        if self.store is not None:
            try:
                measured = self.store.rounds_per_hour()
            except Exception:
                measured = None
            if measured:
                return measured, f"{measured:.0f} rounds/h measured"
        return (FALLBACK_ROUNDS_PER_HOUR,
                f"{FALLBACK_ROUNDS_PER_HOUR:.0f} rounds/h nominal")

    def _optimize(self):
        try:
            target = max(0.001, min(0.5, float(self.target_ror_var.get()) / 100.0))
            chip = max(0.01, float(self.chip_var.get()))
            spread = max(1, int(self.spread_var.get()))
        except tk.TclError:
            messagebox.showerror("Ramp designer",
                                 "Fill in target RoR, chip step and spread.",
                                 parent=self)
            return
        freqs, freq_label = ramp_optimizer.frequency_source(self.store)
        rph, rph_label = self._rounds_per_hour()
        self.freq_source_var.set(f"TC distribution {freq_label} · {rph_label}")
        result = ramp_optimizer.optimize(
            constants.BETTING, freqs, target_ror=target, chip_step=chip,
            max_spread=spread, sit_out_negative=bool(self.sit_out_var.get()),
            rounds_per_hour=rph)
        self._opt_result = result
        self._opt_freqs = freqs
        self._render_result(result, freqs, rph)
        self.apply_btn.config(state=tk.NORMAL if result["feasible"]
                              else tk.DISABLED)

    def _render_result(self, result, freqs, rph):
        for child in self.result_frame.winfo_children():
            child.destroy()
        current = ramp_optimizer.formula_ramp()
        cur_m = ramp_optimizer.ramp_metrics(current, freqs,
                                            rounds_per_hour=rph)
        opt_m = result["metrics"]

        def cell(row, col, text, bold=False, fg=None, pad=(0, 8)):
            tk.Label(self.result_frame, text=text,
                     font=constants.FONT_BODY_BOLD if bold else constants.FONT_SMALL,
                     bg=C["bg_secondary"],
                     fg=fg or (C["text_primary"] if bold else C["text_secondary"])
                     ).grid(row=row, column=col, sticky="e", padx=pad)

        cell(0, 0, "TC", bold=True)
        cell(0, 1, "freq", bold=True)
        cell(0, 2, "current €", bold=True)
        cell(0, 3, "optimal €", bold=True)
        for i, tc in enumerate(sorted(result["ramp"]), start=1):
            opt = result["ramp"][tc]
            cell(i, 0, f"{tc:+d}")
            cell(i, 1, f"{freqs.get(tc, 0.0) * 100:.1f}%")
            cell(i, 2, f"{current.get(tc, 0.0):g}")
            cell(i, 3, "sit out" if opt <= 0 else f"{opt:g}", bold=True,
                 fg=C["accent"])

        def metric_rows(col_base):
            rows = [
                ("EV / round", lambda m: f"€{m['mu']:+.3f}"),
                ("EV / hour", lambda m: f"€{m['ev_hr']:+.2f}"),
                ("Lifetime RoR", lambda m: f"{m['ror'] * 100:.2f}%"),
                ("N0 (rounds)", lambda m: ("—" if m["n0"] is None
                                           else f"{m['n0']:,.0f}")),
                ("DI / SCORE", lambda m: f"{m['di']:.2f} / {m['score']:.1f}"),
                ("CE / round", lambda m: f"€{m['ce']:+.3f}"),
            ]
            cell(0, col_base + 1, "current", bold=True)
            cell(0, col_base + 2, "optimal", bold=True)
            for i, (label, fmt) in enumerate(rows, start=1):
                cell(i, col_base, label)
                cell(i, col_base + 1, fmt(cur_m))
                cell(i, col_base + 2, fmt(opt_m), bold=True, fg=C["accent"])
            return len(rows)

        n = metric_rows(5)
        verdict_row = max(n, len(result["ramp"])) + 1
        if not result["feasible"]:
            verdict = ("No ramp meets that RoR target at this table minimum "
                       "and bankroll — showing the lowest-ruin candidate "
                       f"({opt_m['ror'] * 100:.1f}%). Lower the target, the "
                       "table min, or grow the bankroll.")
        else:
            mc = br.monte_carlo(
                ramp_optimizer.synth_outcomes(result["ramp"], freqs),
                constants.BETTING["bankroll"], n_rounds=5000, trials=4000)
            verdict = (f"Scanned {result['evaluated']} candidates · MC check "
                       f"(5k rounds): ruin {mc['ruin'] * 100:.1f}%, "
                       f"P(profit) {mc['p_profit'] * 100:.0f}%, median "
                       f"€{mc['final_p50']:+,.0f}" if mc else
                       f"Scanned {result['evaluated']} candidates")
        tk.Label(self.result_frame, text=verdict, font=constants.FONT_SMALL,
                 bg=C["bg_secondary"], fg=C["text_secondary"], wraplength=430,
                 justify="left").grid(row=verdict_row, column=0, columnspan=8,
                                      sticky="w", pady=(8, 0))

    def _persist_betting(self):
        """Write BETTING (with the new table) to settings.json — through the
        engine's io thread when available, inline otherwise."""
        if self.engine is not None:
            try:
                self.engine.persist_settings_async()
                self.engine.publish_snapshot()
                return
            except Exception:
                pass
        settings.save()

    def _apply_ramp(self):
        if not self._opt_result or not self._opt_result["feasible"]:
            return
        constants.BETTING["bet_table"] = ramp_optimizer.to_bet_table(
            self._opt_result["ramp"])
        self._persist_betting()
        self.refresh()
        messagebox.showinfo(
            "Ramp designer",
            "Ramp installed — the live bet call now follows the per-TC "
            "table. Clear it here to return to the formula.", parent=self)

    def _clear_ramp(self):
        if not ramp_optimizer.from_bet_table(constants.BETTING.get("bet_table")):
            self._refresh_ramp_status()
            return
        constants.BETTING["bet_table"] = {}
        self._persist_betting()
        self.refresh()
