"""Modal card-selection dialog: 4 suit rows x 13 rank columns, sized to fit."""

import tkinter as tk

from ..common import constants

C = constants.COLORS


class CardPicker(tk.Toplevel):
    """Calls on_select(card_name) — or on_select(None) for 'remove card'."""

    def __init__(self, parent, title, image_lookup, on_select, allow_remove=True):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=C["bg_primary"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self._on_select = on_select

        header = tk.Label(self, text=title, font=constants.FONT_SECTION,
                          bg=C["bg_primary"], fg=C["text_primary"])
        header.pack(pady=(14, 8))

        grid = tk.Frame(self, bg=C["bg_primary"])
        grid.pack(padx=14, pady=4)

        suit_symbols = {"Spades": "♠", "Hearts": "♥", "Diamonds": "♦", "Clubs": "♣"}
        for row, suit in enumerate(constants.CARD_SUITS):
            color = C["danger"] if suit in ("Hearts", "Diamonds") else C["text_primary"]
            tk.Label(grid, text=f"{suit_symbols[suit]}", font=(constants.FONT_FAMILY, 14, "bold"),
                     bg=C["bg_primary"], fg=color, width=2).grid(row=row, column=0, padx=(0, 6))
            for col, rank in enumerate(constants.CARD_RANKS, start=1):
                name = f"{rank} of {suit}"
                photo = image_lookup(name, constants.PICKER_CARD_SIZE)
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
        self.destroy()
        self._on_select(name)
