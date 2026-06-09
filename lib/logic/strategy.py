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

    def advice(self, player_cards, dealer_card, post_split=False):
        """Returns (action_code, display_text, color).

        action_code is the raw CSV code ('H', 'S', 'D/H', ...) or None when no
        advice applies (no cards / unknown dealer / hand not in table).
        `post_split` hands can't resplit, so a new pair uses its total row,
        and a two-card 21 is just 21 — not a blackjack.
        """
        hand = [c for c in player_cards if c and c != "-"]
        if len(hand) < 2:
            return None, "", constants.ACTION_COLORS["-"]

        total = cards.hand_value(hand)
        if total > 21:
            return None, "Bust", constants.ACTION_COLORS["R/H"]
        if total == 21 and len(hand) == 2:
            if not post_split:
                return None, "Blackjack!", constants.ACTION_COLORS["H"]
            return "S", "Stand", constants.ACTION_COLORS["S"]

        dealer_rank = cards.dealer_strategy_rank(dealer_card)
        if dealer_rank is None:
            return None, "Waiting for dealer card", constants.ACTION_COLORS["-"]

        hand_key = cards.hand_key(hand)
        if post_split and cards.is_pair(hand):
            if cards.rank_of(hand[0]) == "Ace":
                hand_key = "A,1"  # soft 12 — usually absent from the CSV: '-'
            else:
                hand_key = str(total)
        key = (dealer_rank.upper(), hand_key.upper())
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
