"""Modal card-selection dialog: 4 suit rows x 13 rank columns, sized to fit."""

import tkinter as tk

from ..common import constants
from . import scaling

C = constants.COLORS

_SUIT_SYMBOLS = {"Spades": "♠", "Hearts": "♥", "Diamonds": "♦", "Clubs": "♣"}


def _compact(name):
    """'King of Hearts' -> 'K♥' for the hand-selector labels."""
    rank, _, suit = str(name).partition(" of ")
    short = rank if rank.isdigit() else rank[:1]
    return short + _SUIT_SYMBOLS.get(suit, "")


class CardPicker(tk.Toplevel):
    """Calls on_select(card_name) — or on_select(None) for 'remove card'.

    With `split_info` ({"is_split": True, "hands": [[h0 names], [h1 names]]})
    a Hand 1 / Hand 2 selector appears and the callback becomes
    on_select(card_name, hand_idx) instead; without it the single-argument
    callback is preserved."""

    def __init__(self, parent, title, image_lookup, on_select, allow_remove=True,
                 split_info=None):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=C["bg_primary"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self._on_select = on_select
        self._hand_var = None

        header = tk.Label(self, text=title, font=constants.FONT_SECTION,
                          bg=C["bg_primary"], fg=C["text_primary"])
        header.pack(pady=(14, 8))

        if split_info and split_info.get("is_split"):
            hands = (split_info.get("hands") or [[], []])[:2]
            while len(hands) < 2:
                hands.append([])
            # Default to the shorter hand — the one most likely owed a card.
            self._hand_var = tk.IntVar(
                value=0 if len(hands[0]) <= len(hands[1]) else 1)
            hand_row = tk.Frame(self, bg=C["bg_primary"])
            hand_row.pack(fill=tk.X, padx=14, pady=(0, 6))
            tk.Label(hand_row, text="Add to:", font=constants.FONT_BODY,
                     bg=C["bg_primary"], fg=C["text_secondary"]
                     ).pack(side=tk.LEFT, padx=(0, 8))
            for h, hand in enumerate(hands):
                cards_text = " ".join(_compact(n) for n in hand) or "empty"
                tk.Radiobutton(hand_row, text=f"Hand {h + 1} — {cards_text}",
                               variable=self._hand_var, value=h,
                               font=constants.FONT_BODY, bg=C["bg_primary"],
                               fg=C["text_primary"], selectcolor=C["bg_secondary"],
                               activebackground=C["bg_primary"],
                               activeforeground=C["text_primary"]
                               ).pack(side=tk.LEFT, padx=4)

        grid = tk.Frame(self, bg=C["bg_primary"])
        grid.pack(padx=14, pady=4)

        suit_symbols = _SUIT_SYMBOLS
        for row, suit in enumerate(constants.CARD_SUITS):
            color = C["danger"] if suit in ("Hearts", "Diamonds") else C["text_primary"]
            tk.Label(grid, text=f"{suit_symbols[suit]}", font=(constants.FONT_FAMILY, 14, "bold"),
                     bg=C["bg_primary"], fg=color, width=2).grid(row=row, column=0, padx=(0, 6))
            for col, rank in enumerate(constants.CARD_RANKS, start=1):
                name = f"{rank} of {suit}"
                photo = image_lookup(name, scaling.size(constants.PICKER_CARD_SIZE))
                btn = tk.Button(grid, image=photo, relief="flat", bd=1,
                                bg=C["bg_secondary"], activebackground=C["border"],
                                cursor="hand2", command=lambda n=name: self._choose(n))
                btn.image = photo
                btn.grid(row=row, column=col, padx=1, pady=2)

        footer = tk.Frame(self, bg=C["bg_primary"])
        footer.pack(fill=tk.X, padx=14, pady=(8, 14))
        tk.Button(footer, text="Cancel", command=self.destroy,
                  bg=C["text_secondary"], fg="white", relief="flat", cursor="hand2",
                  font=constants.FONT_BODY, padx=18, pady=7).pack(side=tk.LEFT)
        if allow_remove:
            tk.Button(footer, text="Remove card", command=lambda: self._choose(None),
                      bg=C["danger"], fg="white", relief="flat", cursor="hand2",
                      font=constants.FONT_BODY, padx=18, pady=7).pack(side=tk.RIGHT)

        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _choose(self, name):
        # Read the selector before destroy so the variable is still live.
        hand_idx = self._hand_var.get() if self._hand_var is not None else None
        self.destroy()
        if hand_idx is not None:
            self._on_select(name, hand_idx)
        else:
            self._on_select(name)
