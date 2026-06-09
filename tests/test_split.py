"""Feature 6: split-hand support.

Run:  .venv\\Scripts\\python -m unittest tests.test_split -v
"""

import unittest

from lib.logic import ev_engine
from lib.logic.ev_engine import TEN, Rules, evaluate, full_shoe
from lib.logic.strategy import StrategyAdvisor


def _engine():
    from lib.logic.engine import DetectionEngine
    return DetectionEngine(log=lambda *a, **k: None)


class PostSplitEv(unittest.TestCase):
    def test_no_resplit_or_surrender_after_split(self):
        comp = full_shoe(8)
        rules = Rules(surrender=True, das=True)
        normal = evaluate((7, 7), TEN, comp, rules)
        post = evaluate((7, 7), TEN, comp, rules, post_split=True)
        self.assertIn("P", normal["evs"])
        self.assertIn("R", normal["evs"])
        self.assertNotIn("P", post["evs"])
        self.assertNotIn("R", post["evs"])

    def test_double_after_split_gated_by_das(self):
        comp = full_shoe(8)
        post_das = evaluate((4, 5), 4, comp, Rules(das=True), post_split=True)
        post_nodas = evaluate((4, 5), 4, comp, Rules(das=False), post_split=True)
        self.assertIn("D", post_das["evs"])
        self.assertNotIn("D", post_nodas["evs"])


class PostSplitCsv(unittest.TestCase):
    def test_pair_uses_total_row_after_split(self):
        adv = StrategyAdvisor()
        # 8,8 vs 10: book says Split — but after a split it's just hard 16.
        action, _, _ = adv.advice(["8 of Hearts", "8 of Spades"], "King")
        self.assertEqual(action, "P")
        action, _, _ = adv.advice(["8 of Hearts", "8 of Spades"], "King",
                                  post_split=True)
        self.assertNotEqual(action, "P")

    def test_post_split_21_is_not_blackjack(self):
        adv = StrategyAdvisor()
        _, text, _ = adv.advice(["Ace of Spades", "King of Hearts"], "5")
        self.assertEqual(text, "Blackjack!")
        action, text, _ = adv.advice(["Ace of Spades", "King of Hearts"], "5",
                                     post_split=True)
        self.assertEqual(action, "S")


class EngineSplitFlow(unittest.TestCase):
    def test_split_lifecycle(self):
        eng = _engine()
        eng.replace_card(0, 0, "8 of Hearts")
        eng.replace_card(0, 1, "8 of Spades")
        eng.replace_dealer("6")
        eng.flush_advice(timeout=60)
        snap = eng.get_snapshot()
        self.assertTrue(snap["seats"][0]["can_split"])
        self.assertFalse(snap["seats"][0]["split"])

        eng.set_split(0)
        eng.flush_advice(timeout=60)
        snap = eng.get_snapshot()
        seat = snap["seats"][0]
        self.assertTrue(seat["split"])
        self.assertEqual(seat["hand_of"], [0, 1])
        self.assertIn("H1:", seat["total"])
        self.assertIn("H2:", seat["total"])
        # Per-hand advice for a lone 8 needs a second card first -> no crash,
        # and adding hit cards routes to hands.
        eng.replace_card(0, 99, "3 of Clubs")   # appended -> shorter hand (h0)
        eng.replace_card(0, 99, "10 of Clubs")  # -> other hand
        eng.flush_advice(timeout=60)
        seat = eng.get_snapshot()["seats"][0]
        self.assertEqual(sorted(seat["hand_of"]), [0, 0, 1, 1])
        self.assertIn("H1:", seat["advice"])
        self.assertIn("H2:", seat["advice"])

        eng.set_split(0, on=False)
        seat = eng.get_snapshot()["seats"][0]
        self.assertFalse(seat["split"])
        self.assertEqual(seat["hand_of"], [0, 0, 0, 0])

    def test_split_requires_a_pair(self):
        eng = _engine()
        eng.replace_card(1, 0, "8 of Hearts")
        eng.replace_card(1, 1, "9 of Spades")
        eng.set_split(1)
        self.assertFalse(eng.get_snapshot()["seats"][1]["split"])

    def test_split_aces_stand(self):
        eng = _engine()
        eng.replace_card(0, 0, "Ace of Hearts")
        eng.replace_card(0, 1, "Ace of Spades")
        eng.replace_dealer("6")
        eng.set_split(0)
        eng.replace_card(0, 99, "5 of Clubs")
        eng.replace_card(0, 99, "9 of Clubs")
        eng.flush_advice(timeout=60)
        seat = eng.get_snapshot()["seats"][0]
        self.assertIn("Stand (one card)", seat["advice"])

    def test_round_reset_clears_split(self):
        eng = _engine()
        eng.replace_card(0, 0, "8 of Hearts")
        eng.replace_card(0, 1, "8 of Spades")
        eng.set_split(0)
        eng.new_round()
        self.assertFalse(eng.get_snapshot()["seats"][0]["split"])


if __name__ == "__main__":
    unittest.main()
