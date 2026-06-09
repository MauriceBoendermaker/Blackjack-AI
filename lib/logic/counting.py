"""Hi-Lo card counting across a shoe. Thread-safe; no Tkinter."""

import threading

from ..common import constants
from . import cards

# Counter rows shown in the UI; J/Q/K are folded into "10".
COUNTER_KEYS = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


def counter_key(card_name: str) -> str:
    rank = cards.rank_of(card_name)
    return "10" if rank in ("Jack", "Queen", "King") else rank


class CardCounter:
    """Tracks running count, per-rank seen counts, and shoe penetration.

    Counts accumulate across rounds (a shoe spans many rounds) and reset only
    on `reset_shoe`. Every mutation is identity-based: the detection engine
    counts each physical card exactly once and can retract it when the user
    manually corrects a misread card.
    """

    def __init__(self, deck_count=constants.DECK_COUNT):
        self._lock = threading.Lock()
        self.deck_count = deck_count
        self.running_count = 0
        self.cards_seen = 0
        self.per_rank = {k: 0 for k in COUNTER_KEYS}

    def count_card(self, card_name: str):
        with self._lock:
            self.running_count += cards.hilo_delta(card_name)
            self.cards_seen += 1
            self.per_rank[counter_key(card_name)] += 1

    def uncount_card(self, card_name: str):
        """Retract a previously counted card (manual correction of a misread)."""
        with self._lock:
            self.running_count -= cards.hilo_delta(card_name)
            self.cards_seen = max(0, self.cards_seen - 1)
            key = counter_key(card_name)
            self.per_rank[key] = max(0, self.per_rank[key] - 1)

    def adjust_manual(self, rank_key: str, delta: int):
        """Manual +/- from the UI; keeps running count and shoe totals consistent."""
        if rank_key not in self.per_rank:
            return
        with self._lock:
            if delta < 0 and self.per_rank[rank_key] <= 0:
                return
            sample_card = "Ace" if rank_key == "Ace" else rank_key
            self.per_rank[rank_key] += delta
            self.cards_seen = max(0, self.cards_seen + delta)
            self.running_count += cards.hilo_delta(sample_card) * delta

    def reset_shoe(self):
        with self._lock:
            self.running_count = 0
            self.cards_seen = 0
            self.per_rank = {k: 0 for k in COUNTER_KEYS}

    @property
    def decks_remaining(self) -> float:
        return max(0.5, self.deck_count - self.cards_seen / 52.0)

    @property
    def true_count(self) -> float:
        with self._lock:
            return self.running_count / self.decks_remaining

    def snapshot(self) -> dict:
        with self._lock:
            decks = max(0.5, self.deck_count - self.cards_seen / 52.0)
            return {
                "running": self.running_count,
                "true": self.running_count / decks,
                "decks_remaining": decks,
                "cards_seen": self.cards_seen,
                "per_rank": dict(self.per_rank),
            }
