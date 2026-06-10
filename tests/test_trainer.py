"""V2 Feature 7: trainer drill logic.

Run:  .venv\\Scripts\\python -m unittest tests.test_trainer -v
"""

import random
import tempfile
import unittest
from pathlib import Path

from lib.logic import trainer
from lib.logic.session_store import SessionStore
from lib.logic.strategy import StrategyAdvisor

ADV = StrategyAdvisor()


class Countdown(unittest.TestCase):
    def test_deck_and_count(self):
        deck = trainer.fresh_deck(random.Random(7))
        self.assertEqual(len(deck), 52)
        self.assertEqual(len(set(deck)), 52)
        # A full deck is balanced: running count ends at exactly 0.
        self.assertEqual(trainer.running_count(deck), 0)
        self.assertEqual(trainer.running_count(["2 of Clubs", "King of Hearts",
                                                "7 of Spades"]), 0)
        result = trainer.grade_countdown(deck, 0)
        self.assertTrue(result["right"])
        self.assertFalse(trainer.grade_countdown(deck, 3)["right"])


class Flashcards(unittest.TestCase):
    def test_items_follow_rules_profile(self):
        s17 = trainer.deviation_items(s17=True, surrender=False)
        self.assertEqual(len(s17), 17)            # the I18 minus insurance
        h17 = trainer.deviation_items(s17=False, surrender=False)
        keys = {(i["hand"], i["dealer"]) for i in h17}
        self.assertNotIn(("11", "A"), keys)       # basic strategy in H17
        self.assertIn(("16", "A"), keys)          # H17-only stand index
        with_fab = trainer.deviation_items(s17=True, surrender=True)
        self.assertEqual(len(with_fab), 17 + 4)

    def test_flashcard_correct_side_of_index(self):
        rng = random.Random(42)
        for _ in range(50):
            card = trainer.make_flashcard(rng)
            expected = card["above"] if card["tc"] >= card["index"] else card["below"]
            self.assertEqual(card["correct"], expected)
            self.assertIn(str(card["hand"]), card["question"])
        self.assertTrue(trainer.grade_flashcard(card, card["correct"]))
        wrong = "H" if card["correct"] != "H" else "S"
        self.assertFalse(trainer.grade_flashcard(card, wrong))

    def test_ev_cost_zero_for_best_positive_for_worse(self):
        card = {"hand": "16", "dealer": "10", "index": 0, "above": "S",
                "below": "H", "source": "I18", "tc": 3.0, "correct": "S"}
        best_cost = trainer.ev_cost(card, "S")
        wrong_cost = trainer.ev_cost(card, "H")
        self.assertAlmostEqual(best_cost, 0.0, places=9)  # stand IS best at +3
        self.assertGreater(wrong_cost, 0.0)
        self.assertIsNone(trainer.ev_cost(card, "P"))     # not a pair


class Replay(unittest.TestCase):
    def _store_with_round(self):
        store = SessionStore(Path(tempfile.mkdtemp()) / "s.db")
        store.record_round({
            "round": 1,
            "dealer": {"card": "9", "locked": True, "extras": []},
            "seats": [{"index": 0, "cards": ["10 of Hearts", "6 of Clubs"],
                       "total": "Hard 16", "advice": "Hit",
                       "optimal": "Optimal: Hit (-0.50)", "split": False},
                      {"index": 1, "cards": ["Ace of Hearts", "King of Clubs"],
                       "total": "Blackjack!", "advice": "", "optimal": "",
                       "split": False}],
            "count": {"running": 2, "true": 0.4, "decks_remaining": 5.0,
                      "cards_seen": 50},
            "insurance": None, "side_bets": [],
        })
        return store

    def test_items_and_grading(self):
        items = trainer.replay_items(self._store_with_round())
        self.assertEqual(len(items), 1)            # the blackjack hand is skipped
        item = items[0]
        self.assertEqual(item["cards"], ["10 of Hearts", "6 of Clubs"])
        result = trainer.grade_replay(item, "H", ADV)
        self.assertTrue(result["right"])           # 16v9: R/H -> falls back to Hit
        self.assertEqual(result["book"], "H")
        self.assertFalse(trainer.grade_replay(item, "S", ADV)["right"])

    def test_no_store_is_empty(self):
        self.assertEqual(trainer.replay_items(None), [])


if __name__ == "__main__":
    unittest.main()
