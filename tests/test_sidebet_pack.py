"""Side-bet pack: variance/Kelly sizing, outcome evaluation, settlement
into the session P&L, and the dealer-playout tracking upgrades.

Run:  .venv\\Scripts\\python -m unittest tests.test_sidebet_pack -v
"""

import unittest

from lib.common import constants
from lib.logic import betting, sidebet_outcomes, sidebets
from lib.logic.ev_engine import full_shoe
from lib.logic.shoe import RANKS, SUITS

PT = {key: cfg["paytable"] for key, cfg in constants.SIDE_BETS.items()}


def full_comp52(decks=8):
    return {(r, s): 2 * decks for r in RANKS for s in SUITS}


class Moments(unittest.TestCase):
    def test_certain_outcome_has_zero_variance(self):
        # Only Queens of Hearts left: every two cards are a perfect pair.
        comp = {("Queen", "Hearts"): 8.0}
        ev, var = sidebets.moments_perfect_pairs(comp, PT["perfect_pairs"])
        self.assertAlmostEqual(ev, 25.0)
        self.assertAlmostEqual(var, 0.0)

    def test_moments_match_ev_wrappers(self):
        comp52, comp10 = full_comp52(8), full_shoe(8)
        for moments, ev in [
            (sidebets.moments_perfect_pairs(comp52, PT["perfect_pairs"]),
             sidebets.ev_perfect_pairs(comp52, PT["perfect_pairs"])),
            (sidebets.moments_three_card(comp52, "21+3", PT["21+3"]),
             sidebets.ev_three_card(comp52, "21+3", PT["21+3"])),
            (sidebets.moments_bust_it(comp10, PT["bust_it"], s17=True),
             sidebets.ev_bust_it(comp10, PT["bust_it"], s17=True)),
        ]:
            self.assertAlmostEqual(moments[0], ev)
            self.assertGreater(moments[1], 1.0)  # high-paytable variance

    def test_evaluate_all_carries_variance(self):
        results = sidebets.evaluate_all(full_comp52(8), full_shoe(8))
        for item in results:
            self.assertIn("variance", item)
            self.assertGreater(item["variance"], 0)


class KellyStake(unittest.TestCase):
    def test_formula_and_guards(self):
        cfg = {"bankroll": 1000.0, "kelly_fraction": 0.5}
        self.assertAlmostEqual(betting.side_bet_stake(0.04, 20.0, cfg), 1.0)
        self.assertEqual(betting.side_bet_stake(-0.02, 20.0, cfg), 0.0)
        self.assertEqual(betting.side_bet_stake(None, 20.0, cfg), 0.0)
        self.assertEqual(betting.side_bet_stake(0.04, None, cfg), 0.0)


class Outcomes(unittest.TestCase):
    def outcome(self, key, first_two, dealer=None, extras=()):
        return sidebet_outcomes.seat_outcomes(
            first_two, dealer, list(extras))[key]

    def test_perfect_pairs(self):
        win = self.outcome("perfect_pairs",
                           ["King of Hearts", "King of Hearts"])
        self.assertEqual((win["result"], win["tier"], win["pays"]),
                         ("win", "perfect", 25.0))
        colored = self.outcome("perfect_pairs",
                               ["King of Hearts", "King of Diamonds"])
        self.assertEqual(colored["tier"], "colored")
        mixed = self.outcome("perfect_pairs",
                             ["King of Hearts", "King of Spades"])
        self.assertEqual(mixed["tier"], "mixed")
        loss = self.outcome("perfect_pairs",
                            ["King of Hearts", "Queen of Hearts"])
        self.assertEqual(loss["result"], "lose")

    def test_21p3_with_and_without_dealer_suit(self):
        sf = self.outcome("21+3", ["5 of Hearts", "6 of Hearts"],
                          "7 of Hearts")
        self.assertEqual((sf["result"], sf["tier"]), ("win", "straight_flush"))
        # Rank-only dealer + suited player: straight vs straight-flush
        # depends on the unseen suit -> unknown.
        unk = self.outcome("21+3", ["5 of Hearts", "6 of Hearts"], "7")
        self.assertEqual(unk["result"], "unknown")
        # Off-suit player: every candidate dealer suit agrees -> decided.
        st = self.outcome("21+3", ["5 of Hearts", "6 of Clubs"], "7")
        self.assertEqual((st["result"], st["tier"]), ("win", "straight"))
        loss = self.outcome("21+3", ["2 of Hearts", "9 of Clubs"], "7")
        self.assertEqual(loss["result"], "lose")

    def test_bust_it_lengths_and_honesty(self):
        win = self.outcome("bust_it", ["2 of Hearts", "3 of Clubs"],
                           "10", ["6", "10"])
        self.assertEqual((win["result"], win["tier"], win["pays"]),
                         ("win", 3, 1.0))
        stood = self.outcome("bust_it", ["2 of Hearts", "3 of Clubs"],
                             "10", ["7"])
        self.assertEqual(stood["result"], "lose")
        incomplete = self.outcome("bust_it", ["2 of Hearts", "3 of Clubs"],
                                  "10", [])
        self.assertEqual(incomplete["result"], "unknown")

    def test_three_card_bets_wait_for_the_up_card(self):
        out = sidebet_outcomes.seat_outcomes(
            ["5 of Hearts", "6 of Hearts"], None, [])
        self.assertNotIn("21+3", out)        # not resolvable yet
        self.assertIn("perfect_pairs", out)  # player-cards-only bet resolves


class EngineSideBetSettlement(unittest.TestCase):
    def setUp(self):
        from lib.logic.engine import DetectionEngine
        self.eng = DetectionEngine(log=lambda *a, **k: None)
        self.eng.store = None
        self._betting = dict(constants.BETTING)
        self._stakes = {k: cfg.get("stake") for k, cfg in
                        constants.SIDE_BETS.items()}
        constants.BETTING["auto_bankroll"] = 0

    def tearDown(self):
        constants.BETTING.clear()
        constants.BETTING.update(self._betting)
        for key, stake in self._stakes.items():
            constants.SIDE_BETS[key]["stake"] = stake

    def test_owned_seat_side_bet_books_into_pnl(self):
        eng = self.eng
        eng.set_my_seat(0, True)
        eng.set_bet_placed(10.0)
        eng.set_side_bet_stake("perfect_pairs", 5.0)
        eng.replace_card(0, 0, "King of Hearts")
        eng.replace_card(0, 1, "King of Diamonds")  # colored pair, pays 12
        eng.replace_dealer("10 of Spades")
        eng.add_dealer_extra("7 of Spades")  # dealer 17: complete
        eng.new_round()
        pnl = eng.session_pnl
        self.assertAlmostEqual(pnl["side_eur"], 60.0)   # 5 x 12
        self.assertAlmostEqual(pnl["eur"], 70.0)        # KK 20 beats 17: +10
        self.assertEqual(pnl["rounds"], 1)

    def test_stake_frozen_at_first_card_not_at_settle(self):
        eng = self.eng
        eng.set_my_seat(0, True)
        eng.set_side_bet_stake("perfect_pairs", 5.0)
        eng.replace_card(0, 0, "King of Hearts")   # round stakes freeze here
        eng.replace_card(0, 1, "King of Diamonds")
        eng.set_side_bet_stake("perfect_pairs", 50.0)  # mid-round edit
        eng.replace_dealer("10 of Spades")
        eng.add_dealer_extra("7 of Spades")
        eng.new_round()
        self.assertAlmostEqual(eng.session_pnl["side_eur"], 60.0)  # 5 x 12

    def test_stale_round_guard_on_dealer_extras(self):
        eng = self.eng
        eng.replace_dealer("10 of Spades")
        stale = eng.round_number - 1
        eng.add_dealer_extra("7 of Clubs", expected_round=stale)
        self.assertEqual(eng.dealer_extras, [])
        eng.add_dealer_extra("7 of Clubs", expected_round=eng.round_number)
        self.assertEqual(len(eng.dealer_extras), 1)
        eng.replace_dealer_extra(0, None, expected_round=stale)
        self.assertEqual(len(eng.dealer_extras), 1)  # stale remove discarded

    def test_unknown_outcome_is_skipped_not_guessed(self):
        eng = self.eng
        eng.set_my_seat(0, True)
        eng.set_side_bet_stake("bust_it", 5.0)
        eng.replace_card(0, 0, "King of Hearts")
        eng.replace_card(0, 1, "Queen of Diamonds")
        eng.replace_dealer("10 of Spades")
        eng.add_dealer_extra("7 of Spades")  # 17 complete; round settles
        # bust_it on a 17 is a LOSS (decided); now make it undecidable:
        # remove the extra so the dealer hand is incomplete -> the main
        # settlement itself refuses, so nothing books at all.
        eng.replace_dealer_extra(0, None)
        eng.new_round()
        self.assertEqual(eng.session_pnl["rounds"], 0)


class DealerPlayoutTracking(unittest.TestCase):
    def setUp(self):
        from lib.logic.engine import DetectionEngine
        self.eng = DetectionEngine(log=lambda *a, **k: None)
        self.eng.store = None

    def test_pending_draw_survives_a_missed_cycle(self):
        eng = self.eng
        eng.replace_dealer("10 of Spades")
        pred = {"rank": "7 of Clubs", "cx": 500.0, "cy": 300.0}
        with eng._lock:
            eng._track_dealer_playout([pred])   # sighting 1 -> pending
            eng._track_dealer_playout([])       # occluded mid-draw
            eng._track_dealer_playout([pred])   # sighting 2 -> confirmed
        self.assertEqual([c["rank"] for c in eng.dealer_extras],
                         ["7 of Clubs"])

    def test_pending_draw_drops_after_tolerated_misses(self):
        eng = self.eng
        eng.replace_dealer("10 of Spades")
        pred = {"rank": "7 of Clubs", "cx": 500.0, "cy": 300.0}
        with eng._lock:
            eng._track_dealer_playout([pred])
            for _ in range(constants.DEALER_PENDING_MISS_TOLERANCE + 1):
                eng._track_dealer_playout([])
            self.assertEqual(eng._pending_dealer, {})

    def test_manual_extra_adopts_detected_position(self):
        eng = self.eng
        eng.replace_dealer("10 of Spades")
        eng.add_dealer_extra("7 of Clubs")
        pred = {"rank": "7 of Clubs", "cx": 480.0, "cy": 290.0}
        with eng._lock:
            eng._track_dealer_playout([pred])  # same card: adopts, no dupe
        self.assertEqual(len(eng.dealer_extras), 1)
        self.assertEqual(eng.dealer_extras[0]["cx"], 480.0)

    def test_manual_extra_does_not_swallow_other_suit(self):
        # Suit mode: a manual 7♣ must NOT absorb a genuinely different 7♥ —
        # the second seven goes through normal pending confirmation.
        eng = self.eng
        eng.replace_dealer("10 of Spades")
        eng.add_dealer_extra("7 of Clubs")
        pred = {"rank": "7 of Hearts", "cx": 480.0, "cy": 290.0}
        with eng._lock:
            eng._track_dealer_playout([pred])
            eng._track_dealer_playout([pred])  # confirmed on 2nd sighting
        self.assertEqual([c["rank"] for c in eng.dealer_extras],
                         ["7 of Clubs", "7 of Hearts"])
        self.assertIsNone(eng.dealer_extras[0]["cx"])  # 7♣ stays manual


if __name__ == "__main__":
    unittest.main()
