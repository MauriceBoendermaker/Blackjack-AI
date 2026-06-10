"""Practice & replay trainer window (V2 Feature 7).

Three drills: deck countdown (speed-configurable, benchmark <30 s/deck),
deviation flashcards generated from the active rules profile (graded with
EV-lost-per-error via the exact engine), and replays of your own recorded
rounds. EV grading runs on a small worker thread so the UI never blocks.
"""

import queue
import random
import threading
import tkinter as tk

from ..common import constants
from ..logic import trainer
from ..logic.strategy import StrategyAdvisor

C = constants.COLORS
SUITS = {"Hearts": "♥", "Diamonds": "♦", "Spades": "♠", "Clubs": "♣"}
RED = ("Hearts", "Diamonds")


def card_text(name):
    rank, _, suit = name.partition(" of ")
    short = {"10": "10", "Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}.get(rank, rank)
    return f"{short}{SUITS.get(suit, '')}", ("#d33" if suit in RED else "#222")


class TrainerWindow(tk.Toplevel):
    def __init__(self, parent, store=None):
        super().__init__(parent)
        self.title("Trainer")
        self.configure(bg=C["bg_secondary"], padx=18, pady=14)
        self.resizable(False, False)
        self.store = store
        self.advisor = StrategyAdvisor()
        self._queue = queue.Queue()
        self.after(100, self._drain_queue)

        tabs = tk.Frame(self, bg=C["bg_secondary"])
        tabs.pack(fill=tk.X, pady=(0, 10))
        self.frames = {}
        for key, label in (("countdown", "Deck countdown"),
                           ("flash", "Deviation flashcards"),
                           ("replay", "Replay my rounds")):
            tk.Button(tabs, text=label, command=lambda k=key: self._show(k),
                      bg=C["accent"], fg="white", relief="flat", padx=12, pady=6,
                      font=constants.FONT_BODY, cursor="hand2"
                      ).pack(side=tk.LEFT, padx=3)
            self.frames[key] = tk.Frame(self, bg=C["bg_secondary"],
                                        width=460, height=300)
        self._build_countdown(self.frames["countdown"])
        self._build_flashcards(self.frames["flash"])
        self._build_replay(self.frames["replay"])
        self._show("countdown")

    def _show(self, key):
        for frame in self.frames.values():
            frame.pack_forget()
        self.frames[key].pack(fill=tk.BOTH, expand=True)

    def _drain_queue(self):
        try:
            while True:
                fn = self._queue.get_nowait()
                fn()
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._drain_queue)

    # ----------------------------------------------------------- countdown

    def _build_countdown(self, f):
        top = tk.Frame(f, bg=C["bg_secondary"])
        top.pack(fill=tk.X)
        tk.Label(top, text="Cards/sec", bg=C["bg_secondary"], fg=C["text_secondary"],
                 font=constants.FONT_BODY).pack(side=tk.LEFT)
        self.cd_speed = tk.IntVar(value=2)
        tk.Spinbox(top, from_=1, to=10, textvariable=self.cd_speed, width=4
                   ).pack(side=tk.LEFT, padx=6)
        self.cd_btn = tk.Button(top, text="Deal one deck", command=self._cd_start,
                                bg=C["success"], fg="white", relief="flat",
                                padx=12, pady=5, font=constants.FONT_BODY,
                                cursor="hand2")
        self.cd_btn.pack(side=tk.LEFT, padx=8)
        self.cd_card = tk.Label(f, text="—", font=(constants.FONT_FAMILY, 56, "bold"),
                                bg="white", fg="#222", width=5, pady=14)
        self.cd_card.pack(pady=14)
        entry_row = tk.Frame(f, bg=C["bg_secondary"])
        entry_row.pack()
        tk.Label(entry_row, text="Final running count:", bg=C["bg_secondary"],
                 fg=C["text_primary"], font=constants.FONT_BODY).pack(side=tk.LEFT)
        self.cd_answer = tk.Entry(entry_row, width=6, font=constants.FONT_BODY_BOLD)
        self.cd_answer.pack(side=tk.LEFT, padx=6)
        self.cd_answer.bind("<Return>", lambda e: self._cd_grade())
        self.cd_result = tk.Label(f, text="Goal: one deck under 30 s, perfect, "
                                          "five times in a row.",
                                  bg=C["bg_secondary"], fg=C["text_secondary"],
                                  font=constants.FONT_BODY, wraplength=430)
        self.cd_result.pack(pady=8)
        self._cd_deck = []
        self._cd_started = None
        self._cd_streak = 0

    def _cd_start(self):
        import time
        self._cd_deck = trainer.fresh_deck(random)
        self._cd_started = time.monotonic()
        self.cd_btn.config(state="disabled")
        self.cd_result.config(text="Counting…")
        self._cd_show(0)

    def _cd_show(self, i):
        import time
        if i >= len(self._cd_deck):
            self.cd_card.config(text="?")
            self._cd_elapsed = time.monotonic() - self._cd_started
            self.cd_btn.config(state="normal")
            self.cd_result.config(text=f"Deck done in {self._cd_elapsed:.1f} s — "
                                       "enter the running count.")
            self.cd_answer.focus_set()
            return
        text, color = card_text(self._cd_deck[i])
        self.cd_card.config(text=text, fg=color)
        self.after(int(1000 / max(1, self.cd_speed.get())),
                   lambda: self._cd_show(i + 1))

    def _cd_grade(self):
        if not self._cd_deck:
            return
        try:
            answer = int(self.cd_answer.get())
        except ValueError:
            return
        result = trainer.grade_countdown(self._cd_deck, answer)
        self._cd_streak = self._cd_streak + 1 if result["right"] else 0
        verdict = "✓ correct" if result["right"] else \
            f"✗ it was {result['correct_count']:+d}"
        self.cd_result.config(
            text=f"{verdict} · {self._cd_elapsed:.1f} s · streak {self._cd_streak}/5"
                 + (" — casino-ready pace!" if result["right"]
                    and self._cd_elapsed < 30 else ""))
        self.cd_answer.delete(0, tk.END)
        self._cd_deck = []

    # ---------------------------------------------------------- flashcards

    def _build_flashcards(self, f):
        self.fc_question = tk.Label(f, text="", font=(constants.FONT_FAMILY, 16, "bold"),
                                    bg=C["bg_secondary"], fg=C["text_primary"])
        self.fc_question.pack(pady=(10, 14))
        buttons = tk.Frame(f, bg=C["bg_secondary"])
        buttons.pack()
        for code in ("H", "S", "D", "P", "R"):
            tk.Button(buttons, text=trainer.ACTION_NAMES[code], width=9,
                      command=lambda c=code: self._fc_answer(c),
                      bg=C["accent"], fg="white", relief="flat", pady=6,
                      font=constants.FONT_BODY, cursor="hand2"
                      ).pack(side=tk.LEFT, padx=3)
        self.fc_result = tk.Label(f, text="", bg=C["bg_secondary"],
                                  fg=C["text_primary"], font=constants.FONT_BODY,
                                  wraplength=440, justify="left")
        self.fc_result.pack(pady=10)
        self.fc_score = tk.Label(f, text="", bg=C["bg_secondary"],
                                 fg=C["text_secondary"], font=constants.FONT_SMALL)
        self.fc_score.pack()
        self._fc_right = self._fc_total = 0
        self._fc_card = None
        self._fc_next()

    def _fc_next(self):
        self._fc_card = trainer.make_flashcard(random)
        self.fc_question.config(text=self._fc_card["question"])
        self.fc_result.config(text="")

    def _fc_answer(self, code):
        card = self._fc_card
        if card is None:
            return
        right = trainer.grade_flashcard(card, code)
        self._fc_total += 1
        self._fc_right += right
        verdict = "✓ correct" if right else \
            f"✗ book: {trainer.ACTION_NAMES[card['correct']]}"
        self.fc_result.config(text=f"{verdict}\n{card['rule']}\nEV cost: computing…")
        self.fc_score.config(text=f"Score {self._fc_right}/{self._fc_total}")
        threading.Thread(target=self._fc_ev_cost, args=(card, code),
                         daemon=True).start()
        self.after(2600, self._fc_next)

    def _fc_ev_cost(self, card, code):
        try:
            cost = trainer.ev_cost(card, code)
        except Exception:
            cost = None
        text = ("n/a" if cost is None else f"{cost:.4f} units of bet")
        verdict = "✓ correct" if trainer.grade_flashcard(card, code) else \
            f"✗ book: {trainer.ACTION_NAMES[card['correct']]}"
        self._queue.put(lambda: self.fc_result.config(
            text=f"{verdict}\n{card['rule']}\nEV cost: {text}"))

    # -------------------------------------------------------------- replay

    def _build_replay(self, f):
        self.rp_question = tk.Label(f, text="", font=(constants.FONT_FAMILY, 15, "bold"),
                                    bg=C["bg_secondary"], fg=C["text_primary"],
                                    wraplength=440)
        self.rp_question.pack(pady=(10, 14))
        buttons = tk.Frame(f, bg=C["bg_secondary"])
        buttons.pack()
        for code in ("H", "S", "D", "P"):
            tk.Button(buttons, text=trainer.ACTION_NAMES[code], width=9,
                      command=lambda c=code: self._rp_answer(c),
                      bg=C["accent"], fg="white", relief="flat", pady=6,
                      font=constants.FONT_BODY, cursor="hand2"
                      ).pack(side=tk.LEFT, padx=3)
        self.rp_result = tk.Label(f, text="", bg=C["bg_secondary"],
                                  fg=C["text_primary"], font=constants.FONT_BODY,
                                  wraplength=440, justify="left")
        self.rp_result.pack(pady=10)
        self._rp_items = trainer.replay_items(self.store)
        random.shuffle(self._rp_items)
        self._rp_item = None
        self._rp_next()

    def _rp_next(self):
        if not self._rp_items:
            self.rp_question.config(text="No recorded rounds yet — play a "
                                         "session first, then come back.")
            return
        self._rp_item = self._rp_items.pop()
        item = self._rp_item
        hand = "  ".join(card_text(c)[0] for c in item["cards"])
        self.rp_question.config(
            text=f"Your hand: {hand}   vs dealer {item['dealer']}\n"
                 f"(true count was {item['tc']:+.1f})")
        self.rp_result.config(text="")

    def _rp_answer(self, code):
        item = self._rp_item
        if item is None:
            return
        result = trainer.grade_replay(item, code, self.advisor)
        if result["right"] is None:
            self.rp_result.config(text="No book line for this hand — skipped.")
        else:
            verdict = "✓ correct" if result["right"] else \
                f"✗ book: {result['book_name']}"
            extra = f"\nAt the table the app said: {item['optimal_text']}" \
                if item.get("optimal_text") else ""
            self.rp_result.config(text=verdict + extra)
        self.after(2400, self._rp_next)
