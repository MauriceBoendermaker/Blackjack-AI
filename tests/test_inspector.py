"""V3 Feature 8: EV explainability inspector — outcome probabilities,
dealer distribution, composition drivers, fresh-shoe flip indicator.

The outcome triples must be consistent with the validated EV recursion by
arithmetic identity (stand/hit pay ±1 so EV = P(win) − P(lose); a double
pays ±2 under ENHC take-all so EV = 2(P(win) − P(lose))) — that makes the
WoO-golden oracle indirectly validate the probabilities too.

Run:  .venv\\Scripts\\python -m unittest tests.test_inspector -v
"""

import json
import unittest
from pathlib import Path

from lib.logic import ev_engine
from lib.logic.ev_engine import (ACE, TEN, Rules, action_outcomes,
                                 composition_drivers, dealer_distribution,
                                 evaluate, full_shoe, inspect_from_per_rank,
                                 inspect_hand)

GOLDENS = json.loads((Path(__file__).parent / "goldens.json").read_text("utf-8"))
RICH_16VT = tuple(next(g for g in GOLDENS
                       if g["case"] == "16vT_rich_enhc")["comp"])


def fresh_minus(indices, deck_count=8):
    comp = list(full_shoe(deck_count))
    for idx in indices:
        comp[idx] -= 1
    return tuple(comp)


class OutcomeProbabilities(unittest.TestCase):
    HAND = (5, TEN)  # T,6 = hard 16
    UP = TEN

    def setUp(self):
        self.comp = fresh_minus([5, TEN, TEN])
        self.rules = Rules()  # ENHC take-all defaults

    def test_triples_are_distributions(self):
        out = action_outcomes(self.HAND, self.UP, self.comp, self.rules)
        for code, (w, p, l) in out.items():
            with self.subTest(action=code):
                self.assertAlmostEqual(w + p + l, 1.0, places=9)
                for x in (w, p, l):
                    self.assertGreaterEqual(x, -1e-12)

    def test_stand_and_hit_match_ev_identity(self):
        evs = evaluate(self.HAND, self.UP, self.comp, self.rules)["evs"]
        out = action_outcomes(self.HAND, self.UP, self.comp, self.rules)
        for code in ("S", "H"):
            w, _, l = out[code]
            self.assertAlmostEqual(evs[code], w - l, places=9,
                                   msg=f"EV({code}) != P(win)-P(lose)")

    def test_double_matches_ev_identity(self):
        hand = (4, 5)  # 5,6 = hard 11
        comp = fresh_minus([4, 5, TEN])
        evs = evaluate(hand, TEN, comp, self.rules)["evs"]
        out = action_outcomes(hand, TEN, comp, self.rules)
        w, _, l = out["D"]
        self.assertAlmostEqual(evs["D"], 2.0 * (w - l), places=9)

    def test_peek_rules_skip_bj_mixing(self):
        rules = Rules(peek=True)
        evs = evaluate(self.HAND, self.UP, self.comp, rules)["evs"]
        out = action_outcomes(self.HAND, self.UP, self.comp, rules)
        w, _, l = out["S"]
        self.assertAlmostEqual(evs["S"], w - l, places=9)

    def test_split_net_is_a_distribution(self):
        hand = (7, 7)  # 8,8
        comp = fresh_minus([7, 7, TEN])
        out = action_outcomes(hand, TEN, comp, self.rules)
        w, p, l = out["P"]
        self.assertAlmostEqual(w + p + l, 1.0, places=9)


class DealerDistribution(unittest.TestCase):
    def test_sums_to_one_and_carries_bj(self):
        comp = fresh_minus([5, TEN, TEN])
        dist = dealer_distribution(comp, TEN, Rules())
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)
        # Ten up: BJ chance = unseen aces / unseen cards.
        self.assertAlmostEqual(dist["bj"], comp[ACE] / sum(comp), places=12)

    def test_small_card_up_has_no_bj(self):
        comp = fresh_minus([5, TEN, 4])
        dist = dealer_distribution(comp, 4, Rules())  # dealer 5
        self.assertEqual(dist["bj"], 0.0)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)


class InspectorJob(unittest.TestCase):
    def test_flip_indicator_on_the_rich_shoe(self):
        result = inspect_hand(["10 of Hearts", "6 of Clubs"], "King",
                              RICH_16VT, deck_count=8, rules=Rules())
        self.assertEqual(result["best"], "S")
        self.assertEqual(result["baseline_best"], "H")

    def test_drivers_show_the_ten_excess(self):
        drivers = composition_drivers(RICH_16VT, 8)
        tens = next(d for d in drivers if d["label"] == "10/J/Q/K")
        self.assertGreater(tens["delta"], 0.01)
        self.assertAlmostEqual(tens["baseline"], 16 / 52, places=12)

    def test_from_per_rank_matches_advise(self):
        per_rank = {"10": 8, "6": 3}  # CardCounter-style seen counts
        result = inspect_from_per_rank(["10 of Hearts", "6 of Clubs"],
                                       "King", per_rank, deck_count=8)
        advice = ev_engine.advise(["10 of Hearts", "6 of Clubs"], "King",
                                  per_rank, deck_count=8)
        self.assertEqual(result["best"], advice["best"])
        self.assertEqual(result["evs"], advice["evs"])

    def test_no_decision_returns_none(self):
        comp = fresh_minus([TEN, TEN, 9])
        self.assertIsNone(inspect_hand(["10 of Hearts"], "King", comp))
        self.assertIsNone(inspect_hand(
            ["10 of Hearts", "Ace of Spades"], "King",
            fresh_minus([TEN, ACE, TEN])))  # blackjack: nothing to decide


if __name__ == "__main__":
    unittest.main()
