"""Leak finder window (V3 Feature 6): ranked coaching report over the
recorded rounds — play errors costed by the exact engine, bet-discipline
and side-bet leaks in EUR — with one-click "Drill my leaks" into the
trainer and an exportable HTML report.

Mining (SQL/JSON) is cheap and runs on the Tk thread like the stats
window; the exact-EV costing pass blocks on ev_offload subprocesses, so it
runs on a 1-worker thread and marshals back through a queue + after()
drain (the trainer-window recipe).
"""

import queue
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import messagebox

from ..common import constants
from ..logic import leaks

C = constants.COLORS


class LeaksWindow(tk.Toplevel):
    def __init__(self, parent, store, open_trainer=None):
        super().__init__(parent)
        self.title("Leak Finder")
        self.configure(bg=C["bg_secondary"], padx=20, pady=16)
        self.resizable(False, False)
        self.store = store
        self.open_trainer = open_trainer  # callable(deck) -> trainer window
        self._result = None
        self._closing = False
        self._queue = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="leaks-ev")
        self.bind("<Destroy>", self._on_destroy, add="+")
        self.after(100, self._drain_queue)

        tk.Label(self, text="Leak Finder", font=constants.FONT_TITLE,
                 bg=C["bg_secondary"], fg=C["text_primary"]).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        tk.Label(self, text=("What your recorded mistakes actually cost — "
                             "play errors priced by the exact engine."),
                 font=constants.FONT_SMALL, bg=C["bg_secondary"],
                 fg=C["text_secondary"], wraplength=460, justify="left").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.scope = tk.StringVar(value="all")
        scope_row = tk.Frame(self, bg=C["bg_secondary"])
        scope_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))
        for text, value in (("This session", "session"), ("All recorded", "all")):
            tk.Radiobutton(scope_row, text=text, value=value,
                           variable=self.scope, command=self.refresh,
                           bg=C["bg_secondary"], fg=C["text_primary"],
                           font=constants.FONT_BODY,
                           activebackground=C["bg_secondary"]).pack(
                side=tk.LEFT, padx=(0, 10))

        self.status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.status_var, font=constants.FONT_SMALL,
                 bg=C["bg_secondary"], fg=C["text_secondary"]).grid(
            row=3, column=0, columnspan=2, sticky="w")

        self.result_frame = tk.Frame(self, bg=C["bg_secondary"])
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="nsew",
                               pady=(6, 0))

        buttons = tk.Frame(self, bg=C["bg_secondary"])
        buttons.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        self.drill_btn = tk.Button(buttons, text="Drill my leaks",
                                   command=self._drill, state=tk.DISABLED,
                                   bg=C["accent"], fg="white", relief="flat",
                                   padx=14, pady=6, font=constants.FONT_BODY,
                                   cursor="hand2")
        self.drill_btn.pack(side=tk.LEFT)
        tk.Button(buttons, text="Export HTML report", command=self._export,
                  bg=C["text_secondary"], fg="white", relief="flat",
                  padx=14, pady=6, font=constants.FONT_BODY,
                  cursor="hand2").pack(side=tk.RIGHT, padx=4)
        tk.Button(buttons, text="Refresh", command=self.refresh,
                  bg=C["text_secondary"], fg="white", relief="flat",
                  padx=14, pady=6, font=constants.FONT_BODY,
                  cursor="hand2").pack(side=tk.RIGHT, padx=4)

        self.refresh()

    # ------------------------------------------------------------ lifecycle

    def _on_destroy(self, event):
        if event.widget is self:
            # Stop the costing pass: cancel anything queued AND flip the
            # abort flag the running pass checks between EV jobs — a
            # non-daemon worker grinding orphaned subprocess work would
            # otherwise survive the window and delay app exit.
            self._closing = True
            self._pool.shutdown(wait=False, cancel_futures=True)

    def _drain_queue(self):
        if not self.winfo_exists():
            return
        try:
            while True:
                callback = self._queue.get_nowait()
                callback()
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    # -------------------------------------------------------------- mining

    def refresh(self):
        session_only = self.scope.get() == "session"
        try:
            result = leaks.find_leaks(self.store, session_only=session_only)
        except Exception as e:
            messagebox.showerror("Leak Finder", f"Could not mine rounds: {e}",
                                 parent=self)
            return
        self._result = result
        self._render(result, costing=bool(result["play"]))
        self.drill_btn.config(state=tk.DISABLED)
        if result["play"]:
            self.status_var.set("Pricing play errors with the exact engine…")
            shown = result  # identity guard: a re-refresh drops stale costs
            self._pool.submit(self._cost_job, shown)
        else:
            self.status_var.set("")

    def _cost_job(self, result):
        # Price a COPY off-thread: the Tk thread renders/exports the live
        # result object and must never see a half-priced, mid-sort list.
        import copy
        play = copy.deepcopy(result["play"])
        try:
            leaks.cost_play_leaks(
                play, abort=lambda: self._closing or self._result is not result)
        except Exception:
            pass
        if self._closing:
            return

        def deliver():
            if self._result is not result:
                return  # user changed scope while we were pricing
            result["play"] = play
            self.status_var.set("")
            self._render(result, costing=False)
        self._queue.put(deliver)

    # ------------------------------------------------------------- render

    def _render(self, result, costing):
        for child in self.result_frame.winfo_children():
            child.destroy()
        owned = result["owned_rounds"]
        per100 = 100.0 / owned if owned else 0.0
        row = 0

        def line(text, bold=False, fg=None, indent=0, pad=2):
            nonlocal row
            tk.Label(self.result_frame, text=text,
                     font=constants.FONT_BODY_BOLD if bold else constants.FONT_BODY,
                     bg=C["bg_secondary"],
                     fg=fg or (C["text_primary"] if bold else C["text_secondary"]),
                     wraplength=470, justify="left").grid(
                row=row, column=0, sticky="w", padx=(indent, 0), pady=pad)
            row += 1

        line(f"{result['rounds']} rounds mined · {owned} with your money",
             bold=True)

        line("Play errors (your seats)", bold=True, pad=(8, 2))
        if not result["play"]:
            line("none found 🎉", indent=12)
        for g in result["play"][:10]:
            cost = g.get("cost_units")
            cost_txt = ("pricing…" if costing else
                        "cost n/a" if cost is None else
                        f"−{cost:.2f}u total · −{cost * per100:.2f}u/100"
                        if owned else f"−{cost:.2f}u total")
            line(f"{g['count']}× {g['pattern']} — {cost_txt}", indent=12)

        b = result["bets"]
        line("Bet discipline", bold=True, pad=(8, 2))
        any_bet = False
        if b["missed_sit_outs"]:
            line(f"{b['missed_sit_outs']} rounds played at a sit-out call — "
                 f"expected loss €{b['sit_out_cost']:.2f}", indent=12)
            any_bet = True
        if b["underbet_rounds"]:
            line(f"{b['underbet_rounds']} underbet raise spots — "
                 f"€{b['underbet_ev']:.2f} EV given up", indent=12)
            any_bet = True
        if b["overbet_rounds"]:
            line(f"{b['overbet_rounds']} overbet rounds — risk cost "
                 f"€{b['overbet_risk_ce']:.2f} (CE)"
                 + (f" + €{b['overbet_neg_ev']:.2f} at -EV"
                    if b["overbet_neg_ev"] else ""), indent=12)
            any_bet = True
        if not any_bet:
            line(f"none found across {b['compared']} compared rounds 🎉",
                 indent=12)
        if b["reconstructed"]:
            line(f"({b['reconstructed']} comparisons reconstructed with "
                 "today's ramp — older rounds predate the persisted "
                 "suggestion)", indent=12)

        line("-EV side-bet habit", bold=True, pad=(8, 2))
        if not result["side_bets"]:
            line("none found 🎉", indent=12)
        for agg in result["side_bets"].values():
            line(f"{agg['label']}: {agg['count']} stakes (€{agg['staked']:.2f})"
                 f" at ≤ 0 EV — expected loss €{agg['cost_eur']:.2f}",
                 indent=12)

        if not costing:
            deck = leaks.drill_items(result)
            self.drill_btn.config(
                state=tk.NORMAL if deck and self.open_trainer else tk.DISABLED)

    # ------------------------------------------------------------- actions

    def _drill(self):
        if self._result is None or self.open_trainer is None:
            return
        deck = leaks.drill_items(self._result)
        if not deck:
            return
        self.open_trainer(deck)

    def _export(self):
        if self._result is None:
            return
        label = ("this session" if self.scope.get() == "session"
                 else "all sessions")
        html_text = leaks.report_html(self._result, session_label=label)
        out_dir = constants.OUTPUT_DIR / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"leaks_{time.strftime('%Y%m%d_%H%M%S')}.html"
        try:
            path.write_text(html_text, encoding="utf-8")
        except OSError as e:
            messagebox.showerror("Export failed", str(e), parent=self)
            return
        try:
            import os
            os.startfile(path)  # noqa: S606 — local report, user-initiated
        except OSError:
            messagebox.showinfo("Report written", str(path), parent=self)
