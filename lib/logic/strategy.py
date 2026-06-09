"""Basic-strategy advice from the cheat-sheet CSV, plus the count-based bet hint."""

import csv

from ..common import constants
from . import cards


class StrategyAdvisor:
    """Loads assets/strategy.csv (dealer;hand;action) and answers lookups."""

    def __init__(self):
        self._table = {}
        self._load()

    def _load(self):
        with open(constants.STRATEGY_CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter=";"):
                if len(row) != 3:
                    continue
                dealer, hand, action = (x.strip().upper() for x in row)
                self._table[(dealer, hand)] = action

    def advice(self, player_cards, dealer_card):
        """Returns (action_code, display_text, color).

        action_code is the raw CSV code ('H', 'S', 'D/H', ...) or None when no
        advice applies (no cards / unknown dealer / hand not in table).
        """
        hand = [c for c in player_cards if c and c != "-"]
        if len(hand) < 2:
            return None, "", constants.ACTION_COLORS["-"]

        total = cards.hand_value(hand)
        if total > 21:
            return None, "Bust", constants.ACTION_COLORS["R/H"]
        if total == 21 and len(hand) == 2:
            return None, "Blackjack!", constants.ACTION_COLORS["H"]

        dealer_rank = cards.dealer_strategy_rank(dealer_card)
        if dealer_rank is None:
            return None, "Waiting for dealer card", constants.ACTION_COLORS["-"]

        key = (dealer_rank.upper(), cards.hand_key(hand).upper())
        action = self._table.get(key)
        if action is None:
            return None, "-", constants.ACTION_COLORS["-"]
        text = constants.ACTION_MAPPING.get(action, action)
        color = constants.ACTION_COLORS.get(action, constants.ACTION_COLORS["-"])
        return action, text, color

    @staticmethod
    def bet_suggestion(true_count, base_bet=constants.BASE_BET):
        if true_count > 4:
            return f"2x base bet (€{2 * base_bet:g})"
        if true_count >= 2:
            return f"1.5x base bet (€{1.5 * base_bet:g})"
        if true_count < 0:
            return f"0.5x base bet (€{0.5 * base_bet:g}) — consider sitting out"
        return f"Base bet (€{base_bet:g})"
