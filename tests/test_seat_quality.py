"""V2 Feature 8: seat play-quality scoring for Bet Behind.

Run:  .venv\\Scripts\\python -m unittest tests.test_seat_quality -v
"""

import unittest

from lib.logic.seat_quality import follows_book, score_settled_round
from lib.logic.strategy import StrategyAdvisor

ADV = StrategyAdvisor()


class FollowsBook(unittest.TestCase):
    def test_correct_stand(self):
        # 20 vs 9: book stands; two cards = followed.
        self.assertTrue(follows_book(["10 of Hearts", "King of Spades"], "9", ADV))

    def test_hit_when_book_stands(self):
        # Hard 17 vs 6 with a third card = hit where the book stands.
        self.assertFalse(follows_book(
            ["10 of Hearts", "7 of Spades", "3 of Clubs"], "6", ADV))

    def test_stood_when_book_hits(self):
        # 16 vs 10 left at two cards = stood where the book keeps hitting.
        self.assertFalse(follows_book(["10 of Hearts", "6 of Spades"], "King", ADV))

    def test_correct_hit_sequence(self):
        # 12 vs 10: hit -> 17 vs 10: stand. Exactly three cards = followed.
        self.assertTrue(follows_book(
            ["10 of Hearts", "2 of Spades", "5 of Clubs"], "King", ADV))

    def test_double_spot_wants_exactly_one_card(self):
        # 11 vs 6: double = one card then stop.
        self.assertTrue(follows_book(
            ["6 of Hearts", "5 of Spades", "9 of Clubs"], "6", ADV))
        self.assertFalse(follows_book(["6 of Hearts", "5 of Spades"], "6", ADV))
        self.assertFalse(follows_book(
            ["6 of Hearts", "5 of Spades", "2 of Clubs", "3 of Clubs"], "6", ADV))

    def test_busted_while_drawing_is_book(self):
        # 14 vs 10: book hits; drew and busted — that's book play.
        self.assertTrue(follows_book(
            ["10 of Hearts", "4 of Spades", "King of Clubs"], "King", ADV))

    def test_unsplit_pair_is_not_book(self):
        self.assertFalse(follows_book(["8 of Hearts", "8 of Spades"], "6", ADV))

    def test_post_split_pair_scores_by_total(self):
        # Post-split 8,8 = hard 16 vs 6 -> stand at two cards = followed.
        self.assertTrue(follows_book(["8 of Hearts", "8 of Spades"], "6", ADV,
                                     post_split=True))

    def test_blackjack_and_unknowns(self):
        self.assertTrue(follows_book(["Ace of Hearts", "King of Spades"], "9", ADV))
        self.assertIsNone(follows_book(["10 of Hearts"], "9", ADV))
        self.assertIsNone(follows_book(["10 of Hearts", "6 of Spades"], None, ADV))

    def test_surrender_row_falls_back(self):
        # 16 vs 10 is R/H in the CSV; online there's no surrender, so the
        # fallback (hit) is the book line: three cards reaching 17+ = followed.
        self.assertTrue(follows_book(
            ["10 of Hearts", "6 of Spades", "5 of Clubs"], "King", ADV))


class ScoreSettledRound(unittest.TestCase):
    def test_scores_per_seat(self):
        settle = {"seats": [
            {"index": 0, "hands": [{"cards": ["10 of Hearts", "King of Spades"]}]},
            {"index": 2, "hands": [{"cards": ["10 of Hearts", "6 of Spades"]}]},
        ]}
        seats_snap = [{"index": 0, "split": False}, {"index": 2, "split": False}]
        verdicts = score_settled_round(settle, seats_snap, "9", ADV)
        self.assertEqual(verdicts[0], [True])    # 20 v 9 stood
        self.assertEqual(verdicts[2], [False])   # 16 v 9 stood (book hits)


class EngineTally(unittest.TestCase):
    def test_settlement_updates_seat_play_and_snapshot(self):
        from lib.logic.engine import DetectionEngine
        eng = DetectionEngine(log=lambda *a, **k: None)
        eng.store = None
        eng.replace_card(2, 0, "10 of Hearts")
        eng.replace_card(2, 1, "King of Spades")   # 20 vs 9: book stand
        eng.replace_dealer("9")
        with eng._lock:                            # dealer 9 + 8 = 17
            eng.dealer_extras.append({"rank": "8", "cx": 1.0, "cy": 1.0})
            eng.counter.count_card("8")
        eng.publish_snapshot()
        eng.new_round()
        self.assertEqual(eng.seat_play[2], {"book": 1, "n": 1})
        snap = eng.get_snapshot()
        self.assertEqual(snap["seats"][2]["book_n"], 1)
        self.assertEqual(snap["seats"][2]["book_pct"], 1.0)
        self.assertIn("bet_behind", snap)
        self.assertIn("-EV", snap["bet_behind"])   # neutral count -> wait


if __name__ == "__main__":
    unittest.main()
