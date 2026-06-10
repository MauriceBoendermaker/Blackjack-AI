"""Item 11: per-seat accuracy drilldown computed from stored rounds.

Run:  .venv\\Scripts\\python -m unittest tests.test_seat_stats -v
"""

import tempfile
import unittest
from pathlib import Path

from lib.logic.session_store import SessionStore


def snapshot(round_number, seats, settlement=None, dealer="King"):
    return {
        "round": round_number,
        "dealer": {"card": dealer, "locked": True, "extras": []},
        "seats": seats,
        "count": {"running": 0, "true": 0.0, "decks_remaining": 6.0,
                  "cards_seen": 10},
        "insurance": None,
        "side_bets": [],
        "settlement": settlement,
        "bet_placed": 10.0,
    }


def seat(index, cards, book_action="H", optimal_action="H", split=False):
    entry = {"index": index, "cards": cards, "total": "", "advice": "",
             "optimal": "", "split": split, "mine": False}
    if book_action is not None:
        entry["book_action"] = book_action
    if optimal_action is not None:
        entry["optimal_action"] = optimal_action
    return entry


def settled(index, hands, dealer_total=20):
    """Settlement dict for one seat; hands = [(cards, outcome, units), ...]."""
    return {"dealer_total": dealer_total, "dealer_bj": False, "seats": [
        {"index": index,
         "hands": [{"cards": c, "outcome": o, "units": u} for c, o, u in hands],
         "net_units": sum(u for _, _, u in hands)}]}


class SeatStats(unittest.TestCase):
    def setUp(self):
        self.store = SessionStore(Path(tempfile.mkdtemp()) / "session.db")

    def test_book_pct_and_avg_units(self):
        # 16 vs King hit to 21 = book play, won.
        hit_hand = ["10 of Hearts", "6 of Clubs", "5 of Hearts"]
        self.store.record_round(snapshot(
            1, [seat(0, hit_hand)],
            settlement=settled(0, [(hit_hand, "win", 1.0)])))
        # 16 vs King left at two cards = stood where the book hits, lost.
        stand_hand = ["10 of Hearts", "6 of Clubs"]
        self.store.record_round(snapshot(
            2, [seat(0, stand_hand)],
            settlement=settled(0, [(stand_hand, "lose", -1.0)])))
        stats = self.store.seat_stats()
        self.assertEqual(list(stats), [0])
        self.assertEqual(stats[0]["hands"], 2)
        self.assertAlmostEqual(stats[0]["book_pct"], 0.5)
        self.assertAlmostEqual(stats[0]["avg_units"], 0.0)
        self.assertEqual(stats[0]["divergences"], 0)

    def test_divergence_needs_both_codes_and_accepts_any_alternative(self):
        hand = ["10 of Hearts", "6 of Clubs"]
        rounds = [
            seat(0, hand, book_action="R/H", optimal_action="S"),  # S not offered
            seat(0, hand, book_action="D/H", optimal_action="D"),  # primary match
            seat(0, hand, book_action="R/H", optimal_action="H"),  # no surrender:
            seat(0, hand, book_action="D/H", optimal_action="H"),  # fallback letter
            seat(0, hand, book_action="H", optimal_action=None),   # one missing
            seat(0, hand, book_action=None, optimal_action=None),  # pre-upgrade row
        ]
        for i, s in enumerate(rounds, start=1):
            self.store.record_round(snapshot(i, [s]))
        stats = self.store.seat_stats()
        self.assertEqual(stats[0]["divergences"], 1)
        # No settlements recorded: nothing settled, nothing judgeable.
        self.assertEqual(stats[0]["hands"], 0)
        self.assertIsNone(stats[0]["book_pct"])
        self.assertIsNone(stats[0]["avg_units"])

    def test_split_seat_scores_per_hand(self):
        flat = ["8 of Hearts", "8 of Spades", "3 of Clubs", "10 of Clubs"]
        h1 = ["8 of Hearts", "3 of Clubs"]    # 11 vs K: book doubles -> 2 cards = not book
        h2 = ["8 of Spades", "10 of Clubs"]   # 18 vs K: stand at 2 cards = book
        self.store.record_round(snapshot(
            1, [seat(0, flat, book_action=["H", "S"],
                     optimal_action=["S", "S"], split=True)],
            settlement=settled(0, [(h1, "lose", -1.0), (h2, "lose", -1.0)])))
        stats = self.store.seat_stats()
        self.assertEqual(stats[0]["hands"], 2)
        self.assertAlmostEqual(stats[0]["book_pct"], 0.5)
        self.assertAlmostEqual(stats[0]["avg_units"], -1.0)
        # One hand's codes disagree -> the round counts once.
        self.assertEqual(stats[0]["divergences"], 1)

    def test_session_scoping(self):
        hand = ["10 of Hearts", "King of Spades"]
        self.store.record_round(snapshot(
            1, [seat(0, hand)], settlement=settled(0, [(hand, "push", 0.0)])))
        other = SessionStore(self.store.path)  # same DB, new session id
        other.record_round(snapshot(
            1, [seat(0, hand)], settlement=settled(0, [(hand, "push", 0.0)])))
        self.assertEqual(other.seat_stats(session_only=True)[0]["hands"], 1)
        self.assertEqual(other.seat_stats(session_only=False)[0]["hands"], 2)

    def test_empty_store(self):
        self.assertEqual(self.store.seat_stats(), {})


if __name__ == "__main__":
    unittest.main()
