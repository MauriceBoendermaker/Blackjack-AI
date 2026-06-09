"""Feature 4: 52-cell suit-aware shoe model.

Run:  .venv\\Scripts\\python -m unittest tests.test_shoe -v
"""

import unittest

from lib.logic.counting import CardCounter
from lib.logic.shoe import RANKS, SUITS, composition52, from_counter_snapshot, total_remaining


class Composition52(unittest.TestCase):
    def fresh(self):
        return CardCounter(deck_count=8)

    def test_full_shoe(self):
        c = self.fresh()
        comp = from_counter_snapshot(c.snapshot(), 8)
        self.assertEqual(len(comp), 52)
        self.assertTrue(all(v == 8.0 for v in comp.values()))
        self.assertAlmostEqual(total_remaining(comp), 416.0)

    def test_exact_suit_removal(self):
        c = self.fresh()
        c.count_card("8 of Diamonds")
        comp = from_counter_snapshot(c.snapshot(), 8)
        self.assertAlmostEqual(comp[("8", "Diamonds")], 7.0)
        self.assertAlmostEqual(comp[("8", "Hearts")], 8.0)
        self.assertAlmostEqual(total_remaining(comp), 415.0)

    def test_rank_only_removal_spreads_over_suits(self):
        c = self.fresh()
        c.count_card("King")  # dealer detection, suit unknown
        comp = from_counter_snapshot(c.snapshot(), 8)
        for suit in SUITS:
            self.assertAlmostEqual(comp[("King", suit)], 8.0 - 0.25)
        # Other ten-bucket ranks untouched: rank IS known.
        self.assertAlmostEqual(comp[("Queen", "Hearts")], 8.0)
        self.assertAlmostEqual(total_remaining(comp), 415.0)

    def test_manual_bucket_adjust_spreads_over_bucket(self):
        c = self.fresh()
        c.adjust_manual("10", +4)  # four unspecified ten-bucket cards
        comp = from_counter_snapshot(c.snapshot(), 8)
        for rank in ("10", "Jack", "Queen", "King"):
            for suit in SUITS:
                self.assertAlmostEqual(comp[(rank, suit)], 8.0 - 4 / 16)
        c.adjust_manual("5", +2)
        comp = from_counter_snapshot(c.snapshot(), 8)
        for suit in SUITS:
            self.assertAlmostEqual(comp[("5", suit)], 8.0 - 2 / 4)

    def test_mixed_levels_in_one_bucket(self):
        c = self.fresh()
        c.count_card("King of Spades")   # exact
        c.count_card("King")             # rank only
        c.adjust_manual("10", +1)        # bucket only
        comp = from_counter_snapshot(c.snapshot(), 8)
        # King of Spades: 1 exact + 1/4 rank-spread + 1/16 bucket-spread
        self.assertAlmostEqual(comp[("King", "Spades")], 8 - 1 - 0.25 - 1 / 16)
        self.assertAlmostEqual(comp[("King", "Hearts")], 8 - 0.25 - 1 / 16)
        self.assertAlmostEqual(comp[("10", "Clubs")], 8 - 1 / 16)
        self.assertAlmostEqual(total_remaining(comp), 416 - 3)

    def test_uncount_restores(self):
        c = self.fresh()
        c.count_card("8 of Diamonds")
        c.count_card("King")
        c.uncount_card("8 of Diamonds")
        c.uncount_card("King")
        comp = from_counter_snapshot(c.snapshot(), 8)
        self.assertTrue(all(abs(v - 8.0) < 1e-12 for v in comp.values()))

    def test_reset_clears_refinements(self):
        c = self.fresh()
        c.count_card("8 of Diamonds")
        c.count_card("King")
        c.reset_shoe()
        snap = c.snapshot()
        self.assertEqual(snap["suit_seen"], {})
        self.assertEqual(snap["rank_seen_nosuit"], {})
        comp = from_counter_snapshot(snap, 8)
        self.assertAlmostEqual(total_remaining(comp), 416.0)

    def test_cells_never_negative(self):
        c = self.fresh()
        for _ in range(9):  # more 8-of-Diamonds than exist in 8 decks
            c.count_card("8 of Diamonds")
        comp = from_counter_snapshot(c.snapshot(), 8)
        self.assertGreaterEqual(comp[("8", "Diamonds")], 0.0)

    def test_consistency_with_per_rank_totals(self):
        c = self.fresh()
        for name in ("Ace of Spades", "King", "10 of Hearts", "3 of Clubs"):
            c.count_card(name)
        c.adjust_manual("6", +2)
        snap = c.snapshot()
        comp = from_counter_snapshot(snap, 8)
        self.assertAlmostEqual(total_remaining(comp), 416 - snap["cards_seen"])
        # Per-bucket totals in the 52-cell view match per_rank exactly.
        for key, seen in snap["per_rank"].items():
            ranks = [r for r in RANKS
                     if ("10" if r in ("10", "Jack", "Queen", "King") else r) == key]
            cells = sum(comp[(r, s)] for r in ranks for s in SUITS)
            full = 8.0 * 4 * len(ranks)
            self.assertAlmostEqual(cells, full - seen, places=10)


if __name__ == "__main__":
    unittest.main()
