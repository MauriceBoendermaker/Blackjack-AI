"""Table-rules / game / side-bet settings dialog (Feature 9).

Every EV in the app is conditional on these values, so they are editable per
table and persisted to output/settings.json. Saving applies the profile to
the live constants and tells the engine to drop its advice caches.
"""

import tkinter as tk
from tkinter import ttk

from ..common import constants, settings

C = constants.COLORS

_RULE_FIELDS = [
    # (key, label, kind, options/None, tooltip-ish hint shown after the field)
    ("s17", "Dealer soft 17", "choice", [("Stands (S17)", True), ("Hits (H17)", False)]),
    ("peek", "Hole card", "choice", [("No hole card (ENHC)", False), ("US peek", True)]),
    ("dealer_bj_takes", "Dealer BJ takes", "choice", [("All bets", "all"), ("Original bet only", "obo")]),
    ("das", "Double after split", "bool", None),
    ("double_on", "Double allowed on", "choice", [("Any two cards", "any"), ("9–11 only", "9-11"), ("10–11 only", "10-11")]),
    ("hit_split_aces", "Hit split aces", "bool", None),
    ("surrender", "Late surrender", "bool", None),
    ("bj_pays", "Blackjack pays", "choice", [("3:2", 1.5), ("6:5", 1.2)]),
]


class SettingsDialog(tk.Toplevel):
    """Modal editor for the active table profile."""

    def __init__(self, parent, on_apply=None):
        super().__init__(parent)
        self.title("Table Settings")
        self.configure(bg=C["bg_secondary"], padx=18, pady=14)
        self.resizable(False, False)
        self.transient(parent)
        self.on_apply = on_apply
        self._vars = {}

        row = 0
        row = self._heading(row, "Table rules — match the game help EXACTLY; they change every EV")
        for key, label, kind, options in _RULE_FIELDS:
            row = self._field(row, key, label, kind, options, constants.RULES[key])

        row = self._heading(row, "Game")
        row = self._spin(row, "deck_count", "Decks in shoe", 1, 8, constants.DECK_COUNT)
        row = self._spin(row, "base_bet", "Base bet (€)", 1, 10_000, int(constants.BASE_BET))

        row = self._heading(row, "Bet sizing (fractional Kelly)")
        row = self._field(row, "betting:kelly_fraction", "Kelly fraction", "choice",
                          [("1/4 Kelly (safest)", 0.25), ("1/2 Kelly", 0.5),
                           ("Full Kelly", 1.0)],
                          constants.BETTING["kelly_fraction"])
        row = self._spin(row, "betting:table_min", "Table minimum (€)", 1, 100_000,
                         int(constants.BETTING["table_min"]))
        row = self._spin(row, "betting:table_max", "Table maximum (€, 0 = none)", 0,
                         1_000_000, int(constants.BETTING["table_max"]))

        row = self._heading(row, "Side bets offered")
        for key, cfg in constants.SIDE_BETS.items():
            var = tk.BooleanVar(value=bool(cfg.get("enabled")))
            self._vars[f"sidebet:{key}"] = var
            tk.Checkbutton(self, text=cfg.get("label", key), variable=var,
                           bg=C["bg_secondary"], fg=C["text_primary"],
                           font=constants.FONT_BODY, anchor="w",
                           activebackground=C["bg_secondary"]
                           ).grid(row=row, column=0, columnspan=2, sticky="w")
            row += 1

        note = tk.Label(self, text="Deck count applies to new EV calculations immediately;\n"
                                   "reset the shoe after changing it mid-session.",
                        font=constants.FONT_SMALL, bg=C["bg_secondary"],
                        fg=C["text_secondary"], justify="left")
        note.grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 4))
        row += 1

        buttons = tk.Frame(self, bg=C["bg_secondary"])
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tk.Button(buttons, text="Save & Apply", command=self._save,
                  bg=C["success"], fg="white", relief="flat", padx=16, pady=7,
                  font=constants.FONT_BODY, cursor="hand2").pack(side=tk.RIGHT, padx=4)
        tk.Button(buttons, text="Cancel", command=self.destroy,
                  bg=C["text_secondary"], fg="white", relief="flat", padx=16, pady=7,
                  font=constants.FONT_BODY, cursor="hand2").pack(side=tk.RIGHT, padx=4)

        self.grab_set()

    # ------------------------------------------------------------ widgets

    def _heading(self, row, text):
        pad = (12, 4) if row else (0, 4)
        tk.Label(self, text=text, font=constants.FONT_SECTION, bg=C["bg_secondary"],
                 fg=C["text_primary"], wraplength=360, justify="left"
                 ).grid(row=row, column=0, columnspan=2, sticky="w", pady=pad)
        return row + 1

    def _field(self, row, key, label, kind, options, current):
        if kind == "bool":
            var = tk.BooleanVar(value=bool(current))
            self._vars[key] = ("bool", var, None)
            tk.Checkbutton(self, text=label, variable=var, bg=C["bg_secondary"],
                           fg=C["text_primary"], font=constants.FONT_BODY,
                           activebackground=C["bg_secondary"], anchor="w"
                           ).grid(row=row, column=0, columnspan=2, sticky="w")
            return row + 1
        labels = [text for text, _ in options]
        values = [value for _, value in options]
        var = tk.StringVar(value=labels[values.index(current)] if current in values else labels[0])
        self._vars[key] = ("choice", var, dict(zip(labels, values)))
        tk.Label(self, text=label, font=constants.FONT_BODY, bg=C["bg_secondary"],
                 fg=C["text_secondary"], anchor="w").grid(row=row, column=0, sticky="w")
        ttk.Combobox(self, textvariable=var, values=labels, state="readonly",
                     width=22).grid(row=row, column=1, sticky="w", padx=(10, 0), pady=2)
        return row + 1

    def _spin(self, row, key, label, lo, hi, current):
        var = tk.IntVar(value=current)
        self._vars[key] = ("int", var, None)
        tk.Label(self, text=label, font=constants.FONT_BODY, bg=C["bg_secondary"],
                 fg=C["text_secondary"], anchor="w").grid(row=row, column=0, sticky="w")
        tk.Spinbox(self, from_=lo, to=hi, textvariable=var, width=8
                   ).grid(row=row, column=1, sticky="w", padx=(10, 0), pady=2)
        return row + 1

    # -------------------------------------------------------------- save

    def _save(self):
        data = {"rules": {}, "side_bets": {}, "betting": {}}
        for key, spec in self._vars.items():
            if key.startswith("sidebet:"):
                data["side_bets"][key.split(":", 1)[1]] = {"enabled": bool(spec.get())}
                continue
            kind, var, mapping = spec
            value = mapping[var.get()] if kind == "choice" else var.get()
            if key.startswith("betting:"):
                data["betting"][key.split(":", 1)[1]] = value
            elif key in ("deck_count", "base_bet"):
                data[key] = value
            else:
                data["rules"][key] = value
        settings.apply(data)
        settings.save()
        if self.on_apply:
            self.on_apply()
        self.destroy()
