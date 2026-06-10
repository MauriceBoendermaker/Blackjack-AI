"""V2 Feature 1: round settlement and session P&L.

Run:  .venv\\Scripts\\python -m unittest tests.test_settlement -v
"""

import tempfile
import unittest
from pathlib import Path

from lib.common import constants
from lib.logic.settlement import dealer_final, settle_hand, settle_round


def seat(idx, cards, split=False, hand_of=None):
    return {"index": idx, "cards": cards, "split": split,
            "hand_of": hand_of or [0] * len(cards)}


class DealerFinal(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(dealer_final("King", ["7"]), (17, False, True))
        self.assertEqual(dealer_final("King", ["Ace"]), (21, True, True))
        self.assertEqual(dealer_final("Ace", ["5", "10"]), (16, False, False))
        self.assertEqual(dealer_final("10", ["6", "9"]), (25, False, True))  # bust
        self.assertEqual(dealer_final(None, []), (0, False, False))


class SettleHand(unittest.TestCase):
    def test_outcomes(self):
        # player 20 vs dealer 19 -> win
        self.assertEqual(settle_hand(["10 of Hearts", "King of Spades"], 19, False),
                         ("win", 1.0))
        # push
        self.assertEqual(settle_hand(["10 of Hearts", "9 of Spades"], 19, False),
                         ("push", 0.0))
        # lose
        self.assertEqual(settle_hand(["10 of Hearts", "6 of Spades"], 19, False),
                         ("lose", -1.0))
        # dealer bust -> win
        self.assertEqual(settle_hand(["10 of Hearts", "2 of Spades"], 25, False),
                         ("win", 1.0))
        # player bust loses even to dealer bust
        self.assertEqual(settle_hand(["10 of Hearts", "6 of Spades", "9 of Clubs"],
                                     25, False), ("lose", -1.0))
        # natural pays bj_pays
        self.assertEqual(settle_hand(["Ace of Hearts", "King of Spades"], 20, False),
                         ("blackjack", 1.5))
        # natural vs dealer BJ -> push; plain 20 vs dealer BJ -> lose
        self.assertEqual(settle_hand(["Ace of Hearts", "King of Spades"], 21, True),
                         ("push", 0.0))
        self.assertEqual(settle_hand(["10 of Hearts", "King of Spades"], 21, True),
                         ("lose", -1.0))
        # post-split two-card 21 is not a natural
        self.assertEqual(settle_hand(["Ace of Hearts", "King of Spades"], 20, False,
                                     natural_allowed=False), ("win", 1.0))


class SettleRound(unittest.TestCase):
    def test_full_round(self):
        seats = [
            seat(0, ["10 of Hearts", "King of Spades"]),      # 20 -> win
            seat(2, ["10 of Clubs", "6 of Spades", "9 of Clubs"]),  # bust
            seat(4, ["Ace of Hearts", "8 of Spades"]),         # 19 -> push
        ]
        result = settle_round(seats, "King", ["9"])  # dealer 19
        self.assertIsNotNone(result)
        self.assertEqual(result["dealer_total"], 19)
        by_idx = {s["index"]: s for s in result["seats"]}
        self.assertEqual(by_idx[0]["net_units"], 1.0)
        self.assertEqual(by_idx[2]["net_units"], -1.0)
        self.assertEqual(by_idx[4]["net_units"], 0.0)

    def test_split_seat_settles_per_hand(self):
        s = seat(0, ["8 of Hearts", "8 of Spades", "10 of Clubs", "5 of Clubs"],
                 split=True, hand_of=[0, 1, 0, 1])
        result = settle_round([s], "King", ["9"])  # dealer 19
        hands = result["seats"][0]["hands"]
        self.assertEqual(len(hands), 2)
        # hand0: 8+10=18 lose; hand1: 8+5=13 lose
        self.assertEqual(result["seats"][0]["net_units"], -2.0)

    def test_incomplete_dealer_refuses(self):
        seats = [seat(0, ["10 of Hearts", "King of Spades"])]
        self.assertIsNone(settle_round(seats, "King", []))   # dealer 10 only
        self.assertIsNone(settle_round(seats, None, []))
        # ...unless everyone busted (dealer doesn't draw then)
        busted = [seat(0, ["10 of Hearts", "6 of Spades", "9 of Clubs"])]
        result = settle_round(busted, "King", [])
        self.assertEqual(result["seats"][0]["net_units"], -1.0)

    def test_empty_round(self):
        self.assertIsNone(settle_round([seat(0, ["10 of Hearts"])], "King", ["9"]))


class EnginePnlIntegration(unittest.TestCase):
    def test_owned_seat_books_pnl_and_bankroll(self):
        from lib.logic.engine import DetectionEngine
        from lib.logic.session_store import SessionStore
        eng = DetectionEngine(log=lambda *a, **k: None)
        eng.store = SessionStore(Path(tempfile.mkdtemp()) / "session.db")
        bankroll_before = constants.BETTING["bankroll"]
        try:
            eng.set_my_seat(0, True)
            eng.set_bet_placed(50)
            eng.replace_card(0, 0, "10 of Hearts")
            eng.replace_card(0, 1, "King of Spades")   # 20
            eng.replace_card(1, 0, "10 of Clubs")
            eng.replace_card(1, 1, "5 of Clubs")       # 15 (not owned)
            eng.replace_dealer("King")
            with eng._lock:                            # dealer draws a 9 -> 19
                eng.dealer_extras.append({"rank": "9", "cx": 1.0, "cy": 1.0})
                eng.counter.count_card("9")
            eng.publish_snapshot()
            eng.new_round()
            eng.flush_advice(timeout=60)

            self.assertEqual(eng.session_pnl["units"], 1.0)   # win on owned seat
            self.assertEqual(eng.session_pnl["eur"], 50.0)
            self.assertAlmostEqual(constants.BETTING["bankroll"],
                                   bankroll_before + 50.0)
            snap = eng.get_snapshot()
            self.assertEqual(snap["session_pnl"]["rounds"], 1)

            stats = eng.store.stats(session_only=False)
            self.assertEqual(stats["settled_rounds"], 1)
            self.assertEqual(stats["net_eur"], 50.0)
            self.assertEqual(stats["hand_outcomes"]["win"], 1)
            self.assertEqual(stats["hand_outcomes"]["lose"], 1)  # seat 1 lost
        finally:
            constants.BETTING["bankroll"] = bankroll_before

    def test_unsettleable_round_books_nothing(self):
        from lib.logic.engine import DetectionEngine
        eng = DetectionEngine(log=lambda *a, **k: None)
        eng.store = None
        eng.set_my_seat(0, True)
        eng.replace_card(0, 0, "10 of Hearts")
        eng.replace_card(0, 1, "King of Spades")
        eng.replace_dealer("King")   # no playout -> dealer 10, incomplete
        eng.new_round()
        self.assertEqual(eng.session_pnl["rounds"], 0)


if __name__ == "__main__":
    unittest.main()
