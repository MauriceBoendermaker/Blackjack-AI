"""EV-engine tests against Wizard of Odds golden values (tests/goldens.json).

The goldens were fetched once from the WoO hand-calculator backend by
tests/fetch_wizard_goldens.py — an exact composition-dependent calculator
(verified: it returns e.g. insurance EV of exactly -35/413 for TT vs A from a
full 8-deck shoe). Matching it to ~1e-9 validates the whole recursion:
dealer distribution, no-blackjack conditioning, ENHC mixing, H17, draw-prob
Bayes correction.

Run:  .venv\\Scripts\\python -m unittest tests.test_ev_engine -v
"""

import json
import time
import unittest
from pathlib import Path

from lib.logic import ev_engine
from lib.logic.ev_engine import (ACE, TEN, Rules, comp_from_per_rank, evaluate,
                                 full_shoe, hand_state, insurance_ev)

GOLDENS = json.loads((Path(__file__).parent / "goldens.json").read_text("utf-8"))

_IDX = {"A": ACE, "T": TEN, **{str(v): v - 1 for v in range(2, 10)}}


def _rules(kw):
    return Rules(s17=not kw.get("h17"), peek=bool(kw.get("peek")),
                 dealer_bj_takes="all", das=True, double_on="any",
                 hit_split_aces=False, surrender=False)


class GoldenValues(unittest.TestCase):
    """Every case must match the WoO oracle: S/H/D to 1e-9, split loosely
    (WoO's split model differs from our split-once independence model)."""

    def test_against_oracle(self):
        for g in GOLDENS:
            with self.subTest(case=g["case"]):
                comp = tuple(g["comp"])
                hand = tuple(sorted(_IDX[c] for c in g["hand"]))
                up = _IDX[g["upcard"]]
                result = evaluate(hand, up, comp, _rules(g["kw"]))
                resp = g["response"]
                self.assertFalse(resp["Error"])
                self.assertAlmostEqual(result["evs"]["S"], resp["Stand"], places=9)
                self.assertAlmostEqual(result["evs"]["H"], resp["Hit"], places=9)
                if resp["HasDouble"]:
                    self.assertAlmostEqual(result["evs"]["D"], resp["Double"], places=9)
                if resp["HasSplit"]:
                    self.assertAlmostEqual(result["evs"]["P"], resp["Split"], delta=0.03)
                if resp["HasInsurance"]:
                    _, ins = insurance_ev(comp)
                    self.assertAlmostEqual(ins, resp["Insurance"], places=9)

    def test_composition_flips_decision(self):
        """The whole point: identical hands, different shoes, different advice."""
        full = next(g for g in GOLDENS if g["case"] == "16vT_full_enhc")
        rich = next(g for g in GOLDENS if g["case"] == "16vT_rich_enhc")
        hand = (5, TEN)  # T,6
        r = Rules()
        self.assertEqual(evaluate(hand, TEN, tuple(full["comp"]), r)["best"], "H")
        self.assertEqual(evaluate(hand, TEN, tuple(rich["comp"]), r)["best"], "S")

        poor = next(g for g in GOLDENS if g["case"] == "12v4_poor_enhc")
        self.assertEqual(evaluate((1, TEN), 3, tuple(poor["comp"]), r)["best"], "H")


class EngineBasics(unittest.TestCase):
    def test_full_shoe_and_per_rank(self):
        self.assertEqual(sum(full_shoe(8)), 416)
        per_rank = {"Ace": 2, "5": 1, "10": 4}
        comp = comp_from_per_rank(per_rank, 8)
        self.assertEqual(comp[ACE], 30)
        self.assertEqual(comp[4], 31)
        self.assertEqual(comp[TEN], 124)
        self.assertEqual(sum(comp), 416 - 7)

    def test_hand_state(self):
        self.assertEqual(hand_state((ACE, 6)), (18, True))        # A,7 soft 18
        self.assertEqual(hand_state((ACE, ACE)), (12, True))      # A,A soft 12
        self.assertEqual(hand_state((TEN, 5, ACE)), (17, False))  # T,6,A hard 17
        self.assertEqual(hand_state((TEN, TEN)), (20, False))

    def test_dealer_dist_sums_to_one(self):
        comp = full_shoe(8)
        for up in (ACE, 4, TEN):
            for excl in (None, ACE if up == TEN else TEN if up == ACE else None):
                dist = ev_engine._dealer_dist(comp, up, excl, True)
                self.assertAlmostEqual(sum(dist), 1.0, places=12)

    def test_insurance_threshold(self):
        # Tens fraction > 1/3 -> +EV. 140 tens of 416 unseen is just over.
        comp = list(full_shoe(8))
        comp[TEN] = 142
        rest = (416 - 142) // 9
        for i in range(9):
            comp[i] = rest
        p, ev = insurance_ev(tuple(comp))
        self.assertGreater(p, 1 / 3)
        self.assertGreater(ev, 0)

    def test_advise_wrapper(self):
        per_rank = {k: 0 for k in ("Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10")}
        per_rank["10"] = 2  # player T + dealer T seen
        per_rank["6"] = 1
        result = ev_engine.advise(["10 of Hearts", "6 of Clubs"], "King", per_rank)
        self.assertIsNotNone(result)
        golden = next(g for g in GOLDENS if g["case"] == "16vT_full_enhc")
        self.assertAlmostEqual(result["evs"]["S"], golden["response"]["Stand"], places=9)
        # No advice for blackjack or one card.
        self.assertIsNone(ev_engine.advise(["Ace of Spades", "King of Hearts"],
                                           "5", per_rank))
        self.assertIsNone(ev_engine.advise(["9 of Hearts"], "5", per_rank))

    def test_performance(self):
        """Cold decisions (all caches empty) within the async-advice budget.
        Ace up-cards are the deep case — they run on the advice thread, so
        seconds are tolerable, but they must stay bounded."""
        def cold(hand, up, budget):
            evaluate.cache_clear()
            ev_engine._dealer_dist.cache_clear()
            ev_engine.clear_thread_caches()
            comp = full_shoe(8)
            for idx in hand + (up,):
                comp = ev_engine._minus(comp, idx)
            t0 = time.perf_counter()
            evaluate(hand, up, comp, Rules())
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, budget,
                            f"cold {hand} vs {up} took {elapsed:.3f}s")

        cold((1, 5), TEN, 1.5)   # 2,6 vs T — deep hit tree, common case
        cold((2, TEN), ACE, 8.0) # 13 vs A — the conditioned deep case
        ev_engine.clear_thread_caches()  # don't leave 100s of MB behind


if __name__ == "__main__":
    unittest.main()
