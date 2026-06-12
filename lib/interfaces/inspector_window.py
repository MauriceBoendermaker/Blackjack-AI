"""EV explainability inspector — "why this play?" (V3 Feature 8).

Opened by clicking any advice line on the table felt. Shows the WHOLE
decision instead of the conclusion: per-action exact EV bars, per-action
P(win/push/lose), the dealer final-total distribution (memoized inside the
engine on every advice call, surfaced here for the first time), and the
composition drivers — which ranks' depletion moved the call, with a
fresh-shoe counterfactual flip indicator. A what-if sandbox lets the hand,
up-card, or remaining-shoe composition be edited and re-solved by the same
oracle-validated engine — an interactive exact-EV calculator.

Computation runs in the dedicated ev_offload "analysis" subprocess —
inspect jobs are several times an advise() and must never queue a live
seat's advice behind them — via a 1-worker thread + queue + after()
drain (the trainer-window recipe).
"""

import queue
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk

from ..common import constants
from ..logic import ev_engine, ev_offload
from . import scaling

C = constants.COLORS

ACTION_NAMES = {"S": "Stand", "H": "Hit", "D": "Double", "P": "Split",
                "R": "Surrender"}
ACTION_ORDER = ("S", "H", "D", "P", "R")
DEALER_SLOTS = (("17", "17"), ("18", "18"), ("19", "19"), ("20", "20"),
                ("21", "21*"), ("bj", "BJ"), ("bust", "Bust"))


class InspectorWindow(tk.Toplevel):
    def __init__(self, parent, seat, dealer, per_rank, deck_count):
        super().__init__(parent)
        self.title("Why this play?")
        self.configure(bg=C["bg_secondary"], padx=16, pady=12)
        self.resizable(False, False)
        self.deck_count = deck_count
        self.dealer = dealer
        self._queue = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="inspector-ev")
        self._job = None  # identity guard for stale results
        self.bind("<Destroy>", self._on_destroy, add="+")
        self.after(100, self._drain_queue)

        self.hands = self._split_hands(seat)
        self.live_comp = ev_engine.comp_from_per_rank(per_rank, deck_count)

        header = tk.Frame(self, bg=C["bg_secondary"])
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.title_var = tk.StringVar(value="")
        tk.Label(header, textvariable=self.title_var, font=constants.FONT_TITLE,
                 bg=C["bg_secondary"], fg=C["text_primary"]).pack(side=tk.LEFT)
        if len(self.hands) > 1:
            self.hand_var = tk.StringVar(value=self.hands[0][0])
            picker = ttk.Combobox(header, state="readonly", width=5,
                                  values=[h[0] for h in self.hands],
                                  textvariable=self.hand_var)
            picker.pack(side=tk.RIGHT)
            picker.bind("<<ComboboxSelected>>", lambda e: self._compute_live())
        else:
            self.hand_var = None

        self.status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.status_var, font=constants.FONT_SMALL,
                 bg=C["bg_secondary"], fg=C["text_secondary"]).grid(
            row=1, column=0, columnspan=2, sticky="w")

        self.result_frame = tk.Frame(self, bg=C["bg_secondary"])
        self.result_frame.grid(row=2, column=0, columnspan=2, sticky="nsew",
                               pady=(4, 0))
        self._build_sandbox()
        self._compute_live()

    # ----------------------------------------------------------- lifecycle

    @staticmethod
    def _split_hands(seat):
        """[(label, cards, post_split)] — split seats inspect per hand."""
        cards = [c for c in seat["cards"] if c and c != "-"]
        if not seat.get("split"):
            return [("", cards, False)]
        tags = seat.get("hand_of") or [0] * len(seat["cards"])
        hands = []
        for hand_no in (0, 1):
            hand = [c for c, t in zip(seat["cards"], tags)
                    if c and c != "-" and t == hand_no]
            if len(hand) >= 2:
                hands.append((f"H{hand_no + 1}", hand, True))
        return hands or [("", cards, False)]

    def _on_destroy(self, event):
        if event.widget is self:
            self._pool.shutdown(wait=False, cancel_futures=True)

    def _drain_queue(self):
        if not self.winfo_exists():
            return
        try:
            while True:
                fn = self._queue.get_nowait()
                try:
                    fn()
                except tk.TclError:
                    pass
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    # ------------------------------------------------------------- compute

    def _current_hand(self):
        if self.hand_var is None:
            return self.hands[0]
        label = self.hand_var.get()
        return next(h for h in self.hands if h[0] == label)

    def _compute_live(self):
        label, cards, post_split = self._current_hand()
        self._submit(cards, self.dealer, self.live_comp, post_split,
                     note=f"{label} " if label else "")

    def _submit(self, cards, dealer, comp, post_split, note=""):
        short = " ".join(_short(c) for c in cards)
        self.title_var.set(f"{note}{short}  vs  {_short(dealer)}")
        self.status_var.set("Solving with the exact engine…")
        rules = ev_engine.current_rules()
        job = object()
        self._job = job

        def work():
            try:
                # Dedicated "analysis" worker: an inspect job is several
                # times an advise() and must never queue a live seat's
                # advice behind it (the advice pool is one process).
                result = ev_offload.run("analysis", ev_engine.inspect_hand,
                                        cards, dealer, comp, self.deck_count,
                                        rules, post_split)
                error = None
            except Exception as e:
                result, error = None, str(e)

            def deliver():
                if self._job is not job:
                    return
                self.status_var.set(error or "")
                if result is None and not error:
                    self.status_var.set("No decision to inspect for this "
                                        "hand (blackjack/bust/too few cards).")
                if result is not None:
                    self._render(result)
            self._queue.put(deliver)
        self._pool.submit(work)

    # -------------------------------------------------------------- render

    def _render(self, result):
        for child in self.result_frame.winfo_children():
            child.destroy()
        evs = result["evs"]
        outcomes = result["outcomes"]
        best = result["best"]
        row = 0

        def section(text):
            nonlocal row
            tk.Label(self.result_frame, text=text, font=constants.FONT_SECTION,
                     bg=C["bg_secondary"], fg=C["text_primary"]).grid(
                row=row, column=0, sticky="w", pady=(8, 2))
            row += 1

        section("Action EVs (per unit bet)")
        canvas = self._bars(
            [(f"{'★ ' if code == best else ''}{ACTION_NAMES[code]}",
              evs[code], code == best)
             for code in ACTION_ORDER if code in evs],
            fmt=lambda v: f"{v:+.4f}", signed=True)
        canvas.grid(row=row, column=0, sticky="w")
        row += 1

        section("Outcome odds (best play after this action)")
        for code in ACTION_ORDER:
            if code not in outcomes:
                continue
            w, p, l = outcomes[code]
            suffix = " (net of both hands)" if code == "P" else ""
            tk.Label(self.result_frame,
                     text=(f"{ACTION_NAMES[code]:<9}  win {w * 100:5.1f}%  ·  "
                           f"push {p * 100:4.1f}%  ·  lose {l * 100:5.1f}%"
                           f"{suffix}"),
                     font=constants.FONT_SMALL, bg=C["bg_secondary"],
                     fg=C["text_primary" if code == best else "text_secondary"]
                     ).grid(row=row, column=0, sticky="w")
            row += 1

        section("Dealer final total")
        dist = result["dealer_dist"]
        canvas = self._bars([(label, dist[key], False)
                             for key, label in DEALER_SLOTS],
                            fmt=lambda v: f"{v * 100:.1f}%", signed=False)
        canvas.grid(row=row, column=0, sticky="w")
        row += 1
        tk.Label(self.result_frame, text="21* = multi-card 21 (not a natural)",
                 font=constants.FONT_SMALL, bg=C["bg_secondary"],
                 fg=C["text_secondary"]).grid(row=row, column=0, sticky="w")
        row += 1

        section("Composition drivers")
        movers = sorted(result["drivers"], key=lambda d: -abs(d["delta"]))[:4]
        for d in movers:
            arrow = "▲" if d["delta"] > 0 else "▼"
            tk.Label(self.result_frame,
                     text=(f"{d['label']:>8}: {d['live'] * 100:.1f}% of shoe "
                           f"vs {d['baseline'] * 100:.1f}% baseline {arrow}"),
                     font=constants.FONT_SMALL, bg=C["bg_secondary"],
                     fg=C["text_secondary"]).grid(row=row, column=0, sticky="w")
            row += 1
        if result["baseline_best"] and result["baseline_best"] != best:
            flip = (f"Fresh shoe says {ACTION_NAMES[result['baseline_best']]} "
                    f"— this composition flips the call to {ACTION_NAMES[best]}.")
            color = C["accent"]
        elif result["baseline_best"]:
            flip = "The fresh-shoe verdict agrees — totals, not composition."
            color = C["text_secondary"]
        else:
            flip = ""
            color = C["text_secondary"]
        if flip:
            tk.Label(self.result_frame, text=flip, font=constants.FONT_BODY_BOLD,
                     bg=C["bg_secondary"], fg=color, wraplength=420,
                     justify="left").grid(row=row, column=0, sticky="w",
                                          pady=(4, 0))
            row += 1

    def _bars(self, rows, fmt, signed):
        """Tiny horizontal bar chart on a Canvas; DPI-scaled."""
        width, row_h = scaling.px(420), scaling.px(20)
        label_w, value_w = scaling.px(86), scaling.px(64)
        canvas = tk.Canvas(self.result_frame, width=width,
                           height=row_h * len(rows), bg=C["bg_secondary"],
                           highlightthickness=0)
        span = max((abs(v) for _, v, _ in rows), default=1.0) or 1.0
        bar_max = width - label_w - value_w - scaling.px(10)
        x0 = label_w + (bar_max // 2 if signed else 0)
        for i, (label, value, emphasize) in enumerate(rows):
            y = i * row_h + row_h // 2
            canvas.create_text(scaling.px(4), y, anchor="w", text=label,
                               fill=C["text_primary" if emphasize
                                      else "text_secondary"],
                               font=constants.FONT_SMALL)
            length = int((abs(value) / span) * (bar_max // (2 if signed else 1)))
            if signed:
                x1 = x0 + (length if value >= 0 else -length)
                color = C["success"] if value >= 0 else C["danger"]
                canvas.create_line(x0, 0, x0, row_h * len(rows),
                                   fill=C["border"])
            else:
                x1 = x0 + length
                color = C["accent"]
            canvas.create_rectangle(min(x0, x1), y - scaling.px(6),
                                    max(x0, x1), y + scaling.px(6),
                                    fill=color, width=0)
            canvas.create_text(width - scaling.px(4), y, anchor="e",
                               text=fmt(value),
                               fill=C["text_primary" if emphasize
                                      else "text_secondary"],
                               font=constants.FONT_SMALL)
        return canvas

    # ------------------------------------------------------------- sandbox

    def _build_sandbox(self):
        box = tk.Frame(self, bg=C["bg_secondary"],
                       highlightbackground=C["border"], highlightthickness=1)
        box.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        inner = tk.Frame(box, bg=C["bg_secondary"], padx=8, pady=6)
        inner.pack(fill=tk.X)
        tk.Label(inner, text="What-if sandbox", font=constants.FONT_SECTION,
                 bg=C["bg_secondary"], fg=C["text_primary"]).grid(
            row=0, column=0, columnspan=6, sticky="w", pady=(0, 4))

        tk.Label(inner, text="Hand (ranks):", font=constants.FONT_SMALL,
                 bg=C["bg_secondary"], fg=C["text_secondary"]).grid(
            row=1, column=0, sticky="w")
        label, cards, _ = self._current_hand()
        self.hand_entry = tk.Entry(inner, width=14, font=constants.FONT_SMALL)
        self.hand_entry.insert(0, ", ".join(_short(c) for c in cards))
        self.hand_entry.grid(row=1, column=1, sticky="w", padx=(4, 10))
        tk.Label(inner, text="Dealer:", font=constants.FONT_SMALL,
                 bg=C["bg_secondary"], fg=C["text_secondary"]).grid(
            row=1, column=2, sticky="w")
        self.dealer_var = tk.StringVar(value=_short(self.dealer))
        ttk.Combobox(inner, state="readonly", width=4,
                     values=["A"] + [str(v) for v in range(2, 11)],
                     textvariable=self.dealer_var).grid(row=1, column=3,
                                                        sticky="w",
                                                        padx=(4, 10))
        tk.Button(inner, text="Recompute", command=self._recompute,
                  bg=C["accent"], fg="white", relief="flat", padx=10, pady=3,
                  font=constants.FONT_SMALL, cursor="hand2").grid(
            row=1, column=4, sticky="w")

        tk.Label(inner, text="Unseen cards left:", font=constants.FONT_SMALL,
                 bg=C["bg_secondary"], fg=C["text_secondary"]).grid(
            row=2, column=0, sticky="w", pady=(6, 0))
        grid = tk.Frame(inner, bg=C["bg_secondary"])
        grid.grid(row=3, column=0, columnspan=6, sticky="w")
        self.comp_vars = []
        for i, rank_label in enumerate(ev_engine.RANK_LABELS):
            tk.Label(grid, text=rank_label, font=constants.FONT_SMALL,
                     bg=C["bg_secondary"], fg=C["text_secondary"]).grid(
                row=0, column=i, padx=2)
            var = tk.IntVar(value=self.live_comp[i])
            tk.Spinbox(grid, from_=0, to=16 * self.deck_count, width=4,
                       textvariable=var, font=constants.FONT_SMALL).grid(
                row=1, column=i, padx=2)
            self.comp_vars.append(var)

    def _recompute(self):
        try:
            comp = tuple(max(0, int(v.get())) for v in self.comp_vars)
        except tk.TclError:
            self.status_var.set("Composition fields must be whole numbers.")
            return
        ranks = [tok.strip() for tok in self.hand_entry.get().split(",")
                 if tok.strip()]
        if len(ranks) < 2:
            self.status_var.set("Enter at least two hand ranks (e.g. 10, 6).")
            return
        try:
            for r in ranks:
                ev_engine.card_index(r)
        except (KeyError, ValueError):
            self.status_var.set(f"Unknown rank in hand: {ranks}")
            return
        dealer = self.dealer_var.get()
        self._submit(ranks, dealer, comp, post_split=False, note="what-if ")


def _short(card_name):
    """'King of Hearts' -> 'K'; '10 of Clubs' -> '10'; 'Ace' -> 'A'."""
    rank = str(card_name).split(" ")[0]
    return {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}.get(rank, rank)
