"""Feature 3: side-bet EV engines, validated against published house edges.

All full-shoe baselines below are 8-deck values from wizardofodds.com (the
21+3 and Bust It figures were re-verified to 6 decimals during research;
others are published to 2 decimals).

Run:  .venv\\Scripts\\python -m unittest tests.test_sidebets -v
"""

import unittest

from lib.common import constants
from lib.logic import sidebets
from lib.logic.counting import CardCounter
from lib.logic.ev_engine import full_shoe
from lib.logic.shoe import RANKS, SUITS, from_counter_snapshot


def full_comp52(decks=8):
    return {(r, s): float(decks) for r in RANKS for s in SUITS}


PT = {k: v["paytable"] for k, v in constants.SIDE_BETS.items()}


class FullShoeHouseEdges(unittest.TestCase):
    """The engines must reproduce the published house edges off the top."""

    def test_perfect_pairs(self):
        ev = sidebets.ev_perfect_pairs(full_comp52(8), PT["perfect_pairs"])
        self.assertAlmostEqual(ev, -0.0410, delta=5e-4)

    def test_21_plus_3(self):
        ev = sidebets.ev_three_card(full_comp52(8), "21+3", PT["21+3"])
        self.assertAlmostEqual(ev, -0.037039, delta=5e-6)

    def test_hot3(self):
        ev = sidebets.ev_three_card(full_comp52(8), "hot3", PT["hot3"])
        self.assertAlmostEqual(ev, -0.0540, delta=5e-4)

    def test_bust_it(self):
        ev = sidebets.ev_bust_it(full_shoe(8), PT["bust_it"], s17=True)
        self.assertAlmostEqual(ev, -0.061842, delta=5e-6)

    def test_lucky_lucky(self):
        ev = sidebets.ev_three_card(full_comp52(8), "lucky_lucky", PT["lucky_lucky"])
        self.assertAlmostEqual(ev, -0.0263, delta=5e-4)

    def test_lucky_ladies(self):
        ev = sidebets.ev_lucky_ladies(full_comp52(8), PT["lucky_ladies"])
        self.assertAlmostEqual(ev, -0.2405, delta=1.5e-3)


class CompositionResponse(unittest.TestCase):
    """Side-bet EVs must move the right way as the shoe composition shifts."""

    def test_bust_it_loves_low_cards(self):
        # Strip tens/aces -> low-card-rich shoe -> dealer busts longer/more.
        comp = list(full_shoe(8))
        base = sidebets.ev_bust_it(tuple(comp), PT["bust_it"], s17=True)
        comp[9] -= 60   # tens
        comp[0] -= 15   # aces
        rich = sidebets.ev_bust_it(tuple(comp), PT["bust_it"], s17=True)
        self.assertGreater(rich, base)

    def test_21p3_loves_suit_imbalance(self):
        comp = full_comp52(8)
        base = sidebets.ev_three_card(comp, "21+3", PT["21+3"])
        for r in RANKS:  # half of two suits gone -> flush-rich remainder
            comp[(r, "Hearts")] = 4.0
            comp[(r, "Diamonds")] = 4.0
        skewed = sidebets.ev_three_card(comp, "21+3", PT["21+3"])
        self.assertGreater(skewed, base)

    def test_perfect_pairs_loves_rank_concentration(self):
        comp = full_comp52(8)
        base = sidebets.ev_perfect_pairs(comp, PT["perfect_pairs"])
        for r in RANKS:  # deplete 9 ranks, concentrate the rest
            if r not in ("Ace", "King", "Queen", "Jack"):
                for s in SUITS:
                    comp[(r, s)] = 2.0
        conc = sidebets.ev_perfect_pairs(comp, PT["perfect_pairs"])
        self.assertGreater(conc, base)

    def test_hot3_loves_sevens_and_mid_cards(self):
        comp = full_comp52(8)
        base = sidebets.ev_three_card(comp, "hot3", PT["hot3"])
        for r in ("2", "3", "4"):
            for s in SUITS:
                comp[(r, s)] = 1.0
        rich = sidebets.ev_three_card(comp, "hot3", PT["hot3"])
        self.assertGreater(rich, base)


class EvaluateAll(unittest.TestCase):
    def test_evaluate_all_from_counter(self):
        c = CardCounter(deck_count=8)
        c.count_card("8 of Diamonds")
        c.count_card("King")
        snap = c.snapshot()
        comp52 = from_counter_snapshot(snap, 8)
        from lib.logic.ev_engine import comp_from_per_rank
        comp10 = comp_from_per_rank(snap["per_rank"], 8)
        results = sidebets.evaluate_all(comp52, comp10)
        keys = {r["key"] for r in results}
        self.assertEqual(keys, {"perfect_pairs", "21+3", "hot3", "bust_it"})
        for r in results:
            self.assertIsNotNone(r["ev"], r["key"])
            self.assertLess(r["ev"], 0)  # near-full shoe: all -EV

    def test_depleted_shoe_returns_none(self):
        tiny = {(r, s): 0.0 for r in RANKS for s in SUITS}
        tiny[("Ace", "Spades")] = 1.0
        self.assertIsNone(sidebets.ev_perfect_pairs(tiny, PT["perfect_pairs"]))
        tiny[("Ace", "Spades")] = 2.0
        self.assertIsNone(sidebets.ev_three_card(tiny, "hot3", PT["hot3"]))
        # Two identical cards left IS a guaranteed perfect pair.
        self.assertAlmostEqual(
            sidebets.ev_perfect_pairs(tiny, PT["perfect_pairs"]), 25.0)


if __name__ == "__main__":
    unittest.main()
