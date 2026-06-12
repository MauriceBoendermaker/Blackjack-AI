"""V3 Feature 6: leak finder & coaching report.

Run:  .venv\\Scripts\\python -m unittest tests.test_leaks -v
"""

import os
import tempfile
import unittest
from pathlib import Path

from lib.logic import leaks
from lib.logic.session_store import SessionStore
from lib.logic.strategy import StrategyAdvisor

T, S = "10 of Hearts", "6 of Clubs"


def snap(round_number, tc, seats, settlement=None, side_bets=None,
         bet_placed=None, bet_suggested=None, bet_sit_out=False,
         edge_exact=None, dealer="King"):
    return {
        "round": round_number,
        "dealer": {"card": dealer, "locked": True, "extras": []},
        "seats": seats,
        "count": {"running": int(tc * 6), "true": tc,
                  "decks_remaining": 6.0, "cards_seen": 100},
        "insurance": None,
        "side_bets": side_bets or [],
        "settlement": settlement,
        "bet_placed": bet_placed,
        "bet_suggested": bet_suggested,
        "bet_sit_out": bet_sit_out,
        "edge_exact": edge_exact,
    }


def seat(cards, mine=True, split=False, index=0):
    return {"index": index, "cards": cards, "total": "", "advice": "",
            "optimal": "", "split": split, "mine": mine,
            "book_action": None, "optimal_action": None}


def settled(cards, units=-1.0, my_eur=-10.0, index=0, side_bets=None):
    out = {"dealer_total": 20, "dealer_bj": False,
           "seats": [{"index": index,
                      "hands": [{"cards": cards, "outcome": "lose",
                                 "units": units}],
                      "net_units": units}],
           "my_units": units, "my_eur": my_eur, "my_side_eur": 0.0}
    if side_bets:
        out["side_bets"] = side_bets
    return out


class ReplayDivergence(unittest.TestCase):
    def setUp(self):
        self.advisor = StrategyAdvisor()

    def div(self, cards, dealer="King", tc=0.0, post_split=False):
        return leaks.replay_divergence(cards, dealer, self.advisor, tc=tc,
                                       post_split=post_split)

    def test_stood_where_book_hits(self):
        d = self.div([T, S], tc=-1.0)  # 16 vs T below the index: hit
        self.assertEqual((d["expected"], d["played"]), ("H", "S"))
        self.assertEqual(d["hand_key"], "16")
        self.assertEqual(d["dealer"], "10")

    def test_index_stand_is_not_flagged(self):
        # 16 vs T at TC >= 0: the I18 index says stand — standing is right.
        self.assertIsNone(self.div([T, S], tc=1.0))

    def test_hit_where_book_stands(self):
        d = self.div([T, "4 of Clubs", "5 of Spades"], dealer="6 of Hearts")
        self.assertEqual((d["expected"], d["played"]), ("S", "H"))

    def test_double_spot_is_charitable_on_three_cards(self):
        self.assertIsNone(self.div(["6 of Hearts", "5 of Clubs",
                                    "9 of Spades"], dealer="6 of Hearts"))

    def test_stood_on_a_double_spot(self):
        d = self.div(["6 of Hearts", "5 of Clubs"], dealer="6 of Hearts")
        self.assertEqual((d["expected"], d["played"]), ("D", "S"))

    def test_unsplit_pair(self):
        d = self.div(["8 of Hearts", "8 of Clubs"])
        self.assertEqual((d["expected"], d["played"]), ("P", "no-split"))

    def test_impossible_sequence_abstains(self):
        self.assertIsNone(self.div([T, S, "9 of Spades", "2 of Hearts"],
                                   tc=-1.0))

    def test_non_hittable_split_aces_abstain(self):
        self.assertIsNone(self.div(["Ace of Hearts", "9 of Clubs"],
                                   post_split=True))


class FindLeaks(unittest.TestCase):
    def setUp(self):
        self.store = SessionStore(Path(tempfile.mkdtemp()) / "session.db")

    def test_play_bet_and_sidebet_leaks(self):
        # Round 1: clean book play; its end-of-round state says "sit out
        # next round" and prices 21+3 at -3%.
        self.store.record_round(snap(
            1, tc=-2.0,
            seats=[seat([T, "5 of Clubs", "6 of Spades"])],  # hit 15vT: book
            settlement=settled([T, "5 of Clubs", "6 of Spades"]),
            bet_placed=10.0, bet_suggested=10.0, bet_sit_out=True,
            edge_exact=-0.015,
            side_bets=[{"key": "21+3", "label": "21+3", "ev": -0.03,
                        "variance": 19.0, "stake": 5.0}]))
        # Round 2: played anyway (missed sit-out), stood on 16 vs T at the
        # negative count (play error), and a €5 21+3 stake settled lost.
        self.store.record_round(snap(
            2, tc=3.0,
            seats=[seat([T, S])],
            settlement=settled([T, S], side_bets={
                "0": {"21+3": {"result": "lose", "stake": 5.0, "eur": -5.0,
                               "label": "21+3", "key": "21+3"}}}),
            bet_placed=50.0, bet_suggested=100.0, bet_sit_out=False,
            edge_exact=0.01))
        # Round 3: underbet the raise spot round 2's state suggested.
        self.store.record_round(snap(
            3, tc=0.0,
            seats=[seat([T, "9 of Clubs"])],
            settlement=settled([T, "9 of Clubs"]),
            bet_placed=50.0))

        result = leaks.find_leaks(self.store)
        self.assertEqual(result["rounds"], 3)
        self.assertEqual(result["owned_rounds"], 3)

        # Play: exactly the 16vT stand (round 1's hit-15 line is book; the
        # pre-deal TC for round 2 is round 1's -2.0, so no index stand).
        self.assertEqual(len(result["play"]), 1)
        g = result["play"][0]
        self.assertEqual((g["hand_key"], g["expected"], g["played"]),
                         ("16", "H", "S"))
        self.assertEqual(g["count"], 1)

        bets = result["bets"]
        self.assertEqual(bets["missed_sit_outs"], 1)
        self.assertAlmostEqual(bets["sit_out_cost"], 50.0 * 0.015)
        self.assertEqual(bets["underbet_rounds"], 1)
        self.assertAlmostEqual(bets["underbet_ev"], 50.0 * 0.01)
        self.assertEqual(bets["reconstructed"], 0)

        self.assertIn("21+3", result["side_bets"])
        sb = result["side_bets"]["21+3"]
        self.assertEqual(sb["count"], 1)
        self.assertAlmostEqual(sb["cost_eur"], 5.0 * 0.03)

    def test_other_seats_are_not_my_leaks(self):
        self.store.record_round(snap(1, tc=0.0, seats=[seat([T, S])],
                                     settlement=settled([T, S])))
        self.store.record_round(snap(
            2, tc=0.0, seats=[seat([T, S], mine=False)],
            settlement={"dealer_total": 20, "dealer_bj": False,
                        "seats": [{"index": 0, "hands": [
                            {"cards": [T, S], "outcome": "lose",
                             "units": -1.0}], "net_units": -1.0}]}))
        result = leaks.find_leaks(self.store)
        # Round 2's stand-16 belongs to a stranger; round 1's own 16vT at
        # pre-deal TC 0 is an index stand — no play leaks at all.
        self.assertEqual(result["play"], [])

    def test_persisted_suggestion_columns_round_trip(self):
        self.store.record_round(snap(1, tc=1.0, seats=[seat([T, S])],
                                     bet_suggested=40.0, bet_sit_out=False,
                                     edge_exact=0.004))
        import sqlite3
        with sqlite3.connect(self.store.path) as con:
            row = con.execute("SELECT bet_suggested, bet_sit_out, edge_exact"
                              " FROM rounds").fetchone()
        self.assertEqual(row, (40.0, 0, 0.004))


class CostingAndDeck(unittest.TestCase):
    def setUp(self):
        os.environ["BJ_EV_INPROC"] = "1"  # keep the exact engine in-process

    def tearDown(self):
        os.environ.pop("BJ_EV_INPROC", None)

    def _result(self):
        return {"rounds": 10, "owned_rounds": 10, "bets": {}, "side_bets": {},
                "play": [{"hand_key": "16", "dealer": "10", "expected": "H",
                          "played": "S", "count": 3, "cost_units": None,
                          "pattern": "16 vs 10: stood — play says hit",
                          "examples": [{"cards": [T, S], "dealer": "King",
                                        "tc": -1.0, "split": False}]}]}

    def test_cost_play_leaks_prices_the_divergence(self):
        result = self._result()
        leaks.cost_play_leaks(result["play"])
        g = result["play"][0]
        self.assertIsNotNone(g["cost_units"])
        # Hitting 16vT beats standing by a few percent of a unit per error.
        self.assertGreater(g["cost_per_error"], 0.0)
        self.assertLess(g["cost_per_error"], 0.30)
        self.assertAlmostEqual(g["cost_units"], g["cost_per_error"] * 3)

    def test_drill_items_weighted_and_capped(self):
        result = self._result()
        items = leaks.drill_items(result, cap=10)
        self.assertEqual(len(items), 3)  # one example x count 3
        self.assertEqual(items[0]["cards"], [T, S])
        self.assertEqual(items[0]["dealer"], "King")
        items = leaks.drill_items(result, cap=2)
        self.assertEqual(len(items), 2)

    def test_report_html_mentions_everything(self):
        result = self._result()
        result["bets"] = {"overbet_risk_ce": 0.0, "underbet_ev": 1.25,
                          "overbet_neg_ev": 0.0, "missed_sit_outs": 2,
                          "sit_out_cost": 0.75, "compared": 9,
                          "reconstructed": 4, "overbet_rounds": 0,
                          "underbet_rounds": 3}
        result["side_bets"] = {"21+3": {"label": "21+3", "count": 2,
                                        "staked": 10.0, "cost_eur": 0.30}}
        html_text = leaks.report_html(result, session_label="test")
        for needle in ("16 vs 10", "sit-out", "underbet", "21+3",
                       "reconstructed", "Insurance"):
            self.assertIn(needle, html_text)


if __name__ == "__main__":
    unittest.main()
