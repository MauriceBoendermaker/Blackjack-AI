"""Expected 52-cell shoe composition — the suit-aware shoe model (Feature 4).

The main-game EV engine only needs the 10-bucket per-rank counts, but side
bets (21+3 flushes, Perfect Pairs suit tiers, Hot 3 suited 21) need the full
rank x suit composition. The counter tracks three levels of card identity:

  * exact      — player detections: "8 of Diamonds"      (suit_seen)
  * rank only  — dealer detections: "King"               (rank_seen_nosuit)
  * bucket only — manual +/- on the 10-bucket counters    (per_rank residual)

`composition52` turns those into the EXPECTED remaining count per (rank, suit)
cell: exact removals subtract from their cell, rank-only removals spread 1/4
across the rank's suits, and the bucket residual spreads uniformly across the
bucket's cells (16 for the ten bucket, 4 otherwise). Fractional counts are
fine — every consumer treats the composition as probabilities. Evaluating EV
at the expected composition instead of the exact (unknown) one is the standard
treatment for unseen-suit removals; the error is small because rank-only
sightings are a minority of removals (the dealer's cards).
"""

from ..common import constants
from . import cards
from .counting import COUNTER_KEYS

RANKS = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
SUITS = list(constants.CARD_SUITS)

_BUCKET_RANKS = {key: [r for r in RANKS
                       if ("10" if r in ("10", "Jack", "Queen", "King") else r) == key]
                 for key in COUNTER_KEYS}


def composition52(per_rank: dict, suit_seen: dict, rank_seen_nosuit: dict,
                  deck_count: int = constants.DECK_COUNT) -> dict:
    """Expected remaining count per (rank, suit) cell, as a dict
    {(rank, suit): float} over all 52 cells. Cells clamp at >= 0."""
    removed = {(r, s): 0.0 for r in RANKS for s in SUITS}

    known_in_bucket = {key: 0 for key in COUNTER_KEYS}
    for name, n in suit_seen.items():
        if not n:
            continue
        rank, _, suit = name.partition(" of ")
        if rank not in RANKS or suit not in SUITS:
            continue
        removed[(rank, suit)] += n
        known_in_bucket[("10" if rank in ("10", "Jack", "Queen", "King") else rank)] += n

    for rank, n in rank_seen_nosuit.items():
        if not n or rank not in RANKS:
            continue
        for suit in SUITS:
            removed[(rank, suit)] += n / 4.0
        known_in_bucket[("10" if rank in ("10", "Jack", "Queen", "King") else rank)] += n

    for key in COUNTER_KEYS:
        residual = per_rank.get(key, 0) - known_in_bucket[key]
        if residual <= 0:
            continue  # manual minus can briefly under-run the refinements
        bucket_cells = [(r, s) for r in _BUCKET_RANKS[key] for s in SUITS]
        share = residual / len(bucket_cells)
        for cell in bucket_cells:
            removed[cell] += share

    return {cell: max(0.0, deck_count - removed[cell]) for cell in removed}


def total_remaining(comp52: dict) -> float:
    return sum(comp52.values())


def from_counter_snapshot(count: dict, deck_count: int = constants.DECK_COUNT) -> dict:
    """Convenience: build the 52-cell composition from CardCounter.snapshot()."""
    return composition52(count["per_rank"], count.get("suit_seen", {}),
                         count.get("rank_seen_nosuit", {}), deck_count)
