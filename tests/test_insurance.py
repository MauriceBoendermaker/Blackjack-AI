"""Feature 5: insurance / even-money advisor.

Covers the exact-threshold math (take iff unseen tens fraction > 1/3, with
the 6:5 even-money variant) and the full snapshot integration on a headless
DetectionEngine (models and capture are lazy, so construction is offline).

Run:  .venv\\Scripts\\python -m unittest tests.test_insurance -v
"""

import unittest

from lib.logic import ev_engine
from lib.logic.ev_engine import Rules, insurance_advice

ALL_RANKS = ("Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10")


def per_rank(**seen):
    pr = {k: 0 for k in ALL_RANKS}
    pr.update(seen)
    return pr


class InsuranceMath(unittest.TestCase):
    def test_full_shoe_declines(self):
        # 128/416 tens = 30.8% < 1/3 — insurance is always -EV off the top.
        info = insurance_advice(per_rank())
        self.assertFalse(info["take"])
        self.assertLess(info["ev"], 0)
        self.assertLess(info["even_money_edge"], 0)

    def test_ten_rich_shoe_takes(self):
        # Strip 90 non-ten cards: 128 tens of 326 unseen = 39.3% > 1/3.
        seen = {r: 10 for r in ALL_RANKS if r != "10"}
        info = insurance_advice(per_rank(**seen))
        self.assertTrue(info["take"])
        self.assertGreater(info["ev"], 0)
        self.assertGreater(info["even_money_edge"], 0)

    def test_threshold_is_exactly_one_third(self):
        # Engineer p just above and below 1/3 and check the flip.
        # 104 tens / 312 unseen = exactly 1/3 -> EV 0, decline (not >).
        seen = {"10": 24, "2": 20, "3": 20, "4": 20, "5": 20}  # 416-104 = 312 unseen
        info = insurance_advice(per_rank(**seen))
        self.assertAlmostEqual(info["p_ten"], 1 / 3, places=12)
        self.assertAlmostEqual(info["ev"], 0.0, places=12)
        self.assertFalse(info["take"])
        # One more low card gone: 104/311 > 1/3.
        seen["6"] = 1
        self.assertTrue(insurance_advice(per_rank(**seen))["take"])

    def test_even_money_six_to_five(self):
        # On a 6:5 table even money is right already at p > 1/6.
        info = insurance_advice(per_rank(), rules=Rules(bj_pays=1.2))
        self.assertGreater(info["even_money_edge"], 0)   # 1 - 1.2*(1-0.3077) > 0
        self.assertFalse(info["take"])                   # insurance itself still -EV

    def test_matches_oracle_value(self):
        # TT vs A from a full 8-deck shoe: WoO returned exactly -35/413.
        info = insurance_advice(per_rank(**{"10": 2, "Ace": 1}))
        self.assertAlmostEqual(info["ev"], -35 / 413, places=12)


class EngineSnapshotIntegration(unittest.TestCase):
    def _engine(self):
        from lib.logic.engine import DetectionEngine
        return DetectionEngine(log=lambda *a, **k: None)

    def test_insurance_and_even_money_in_snapshot(self):
        eng = self._engine()
        self.assertIsNone(eng.get_snapshot()["insurance"])

        eng.replace_card(0, 0, "Ace of Spades")
        eng.replace_card(0, 1, "King of Hearts")   # natural blackjack
        eng.replace_card(1, 0, "10 of Hearts")
        eng.replace_card(1, 1, "6 of Clubs")       # ordinary hand
        eng.replace_dealer("Ace")

        snap = eng.get_snapshot()
        ins = snap["insurance"]
        self.assertIsNotNone(ins)
        self.assertFalse(ins["take"])  # near-full shoe
        self.assertTrue(ins["text"].startswith("Insurance: Decline"))

        self.assertTrue(snap["seats"][0]["optimal"].startswith("Even money: Decline"))
        self.assertTrue(snap["seats"][1]["optimal"].startswith("Optimal:"))

        eng.replace_dealer("9")  # no ace, no insurance
        self.assertIsNone(eng.get_snapshot()["insurance"])
        self.assertFalse(eng.get_snapshot()["seats"][0]["optimal"].startswith("Even money"))


if __name__ == "__main__":
    unittest.main()
