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
        # Suit-level refinements of per_rank (Feature 4). Player detections
        # carry full names ("8 of Diamonds") -> suit_seen; dealer detections
        # carry only a rank ("King") -> rank_seen_nosuit; manual +/- stays at
        # the 10-bucket level and lives only in per_rank.
        self.suit_seen = {}        # "8 of Diamonds" -> count
        self.rank_seen_nosuit = {} # "King" -> count

    def _track_identity(self, card_name: str, delta: int):
        if " of " in card_name:
            store, key = self.suit_seen, card_name
        else:
            store, key = self.rank_seen_nosuit, cards.rank_of(card_name)
        store[key] = max(0, store.get(key, 0) + delta)

    def count_card(self, card_name: str):
        with self._lock:
            self.running_count += cards.hilo_delta(card_name)
            self.cards_seen += 1
            self.per_rank[counter_key(card_name)] += 1
            self._track_identity(card_name, +1)

    def uncount_card(self, card_name: str):
        """Retract a previously counted card (manual correction of a misread)."""
        with self._lock:
            self.running_count -= cards.hilo_delta(card_name)
            self.cards_seen = max(0, self.cards_seen - 1)
            key = counter_key(card_name)
            self.per_rank[key] = max(0, self.per_rank[key] - 1)
            self._track_identity(card_name, -1)

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
            self.suit_seen = {}
            self.rank_seen_nosuit = {}

    def get_state(self) -> dict:
        """Serializable shoe state for persistence (Feature 8)."""
        with self._lock:
            return {
                "deck_count": self.deck_count,
                "running_count": self.running_count,
                "cards_seen": self.cards_seen,
                "per_rank": dict(self.per_rank),
                "suit_seen": dict(self.suit_seen),
                "rank_seen_nosuit": dict(self.rank_seen_nosuit),
            }

    def apply_state(self, state: dict):
        """Restore a persisted shoe state (inverse of get_state)."""
        with self._lock:
            self.deck_count = int(state.get("deck_count", self.deck_count))
            self.running_count = int(state.get("running_count", 0))
            self.cards_seen = int(state.get("cards_seen", 0))
            per_rank = state.get("per_rank", {})
            self.per_rank = {k: int(per_rank.get(k, 0)) for k in COUNTER_KEYS}
            self.suit_seen = dict(state.get("suit_seen", {}))
            self.rank_seen_nosuit = dict(state.get("rank_seen_nosuit", {}))

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
                "suit_seen": dict(self.suit_seen),
                "rank_seen_nosuit": dict(self.rank_seen_nosuit),
            }
