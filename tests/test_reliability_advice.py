"""Reliability pack (Core-2): advice-path hardening.

Covers the book-play fallback for delayed/failed EV jobs, per-key ERROR
logging with a per-round dedup set, the single split-aces decision path in
strategy.advice, and selective cache invalidation in refresh_settings.

Run:  .venv\\Scripts\\python -m unittest tests.test_reliability_advice -v
"""

import copy
import threading
import unittest
from unittest import mock

from lib.common import constants
from lib.logic import ev_engine
from lib.logic.strategy import StrategyAdvisor


def _engine(log=None):
    from lib.logic.engine import DetectionEngine
    eng = DetectionEngine(log=log or (lambda *a, **k: None))
    eng.store = None  # no DB writes from tests
    return eng


def _deal_hard_18(eng):
    """10+8 vs dealer 6 — the book row is an unambiguous Stand."""
    eng.replace_card(0, 0, "10 of Hearts")
    eng.replace_card(0, 1, "8 of Clubs")
    eng.replace_dealer("6")


class DelayedEvFallback(unittest.TestCase):
    def test_pending_past_timeout_falls_back_to_book(self):
        eng = _engine()
        release = threading.Event()
        real_advise = ev_engine.advise

        def slow_advise(*args, **kwargs):
            release.wait(30)
            return real_advise(*args, **kwargs)

        try:
            with mock.patch.object(ev_engine, "advise", side_effect=slow_advise):
                _deal_hard_18(eng)
                # The job is queued behind the blocked worker: placeholder.
                snap = eng.get_snapshot()
                self.assertEqual(snap["seats"][0]["optimal"], "Optimal: …")
                # Backdate the submit stamps instead of sleeping the timeout out.
                with eng._advice_lock:
                    for key in eng._advice_pending:
                        eng._advice_pending[key] -= constants.EV_ADVICE_TIMEOUT_S + 1.0
                eng.publish_snapshot()
                seat = eng.get_snapshot()["seats"][0]
                self.assertEqual(seat["optimal"], "Stand (book — EV delayed)")
                self.assertEqual(seat["optimal_color"], constants.ACTION_COLORS["S"])
        finally:
            release.set()
        # The late-landing result still upgrades to the normal Optimal line.
        self.assertTrue(eng.flush_advice(timeout=60))
        self.assertTrue(
            eng.get_snapshot()["seats"][0]["optimal"].startswith("Optimal: Stand"))


class FailedEvFallback(unittest.TestCase):
    def test_failed_job_shows_book_fallback_and_logs_each_new_key(self):
        logs = []
        eng = _engine(log=lambda msg, level=None: logs.append((level, msg)))

        def errors():
            return [msg for level, msg in logs if level == "ERROR"]

        with mock.patch.object(ev_engine, "advise",
                               side_effect=RuntimeError("boom")):
            _deal_hard_18(eng)
            self.assertTrue(eng.flush_advice(timeout=60))
            seat = eng.get_snapshot()["seats"][0]
            self.assertEqual(seat["optimal"], "Stand (book — EV failed)")
            self.assertEqual(len(errors()), 1)
            self.assertIn("RuntimeError('boom')", errors()[0])  # exception repr
            self.assertIn("10 of Hearts", errors()[0])          # hand context
            self.assertIn("vs 6", errors()[0])                  # dealer context

            # The errored key stays cached as None: re-publishing neither
            # resubmits the job nor logs the same key again.
            eng.publish_snapshot()
            self.assertTrue(eng.flush_advice(timeout=60))
            self.assertEqual(len(errors()), 1)

            # A new hand (new key) gets exactly one ERROR line of its own.
            eng.replace_card(1, 0, "9 of Hearts")
            eng.replace_card(1, 1, "7 of Clubs")
            self.assertTrue(eng.flush_advice(timeout=60))
            self.assertEqual(
                len([m for m in errors() if "9 of Hearts" in m]), 1)
            self.assertGreater(len(errors()), 1)

    def test_same_key_logs_once_until_round_reset(self):
        logs = []
        eng = _engine(log=lambda msg, level=None: logs.append((level, msg)))
        eng._log_ev_error(("k",), "first failure")
        eng._log_ev_error(("k",), "second failure (suppressed)")
        self.assertEqual([m for level, m in logs if level == "ERROR"],
                         ["first failure"])
        eng.new_round()  # round end clears the dedup set
        eng._log_ev_error(("k",), "fresh round failure")
        self.assertIn(("ERROR", "fresh round failure"), logs)


class SplitAcesSinglePath(unittest.TestCase):
    def setUp(self):
        self._hsa = constants.RULES["hit_split_aces"]

    def tearDown(self):
        constants.RULES["hit_split_aces"] = self._hsa

    def test_strategy_owns_the_decision(self):
        adv = StrategyAdvisor()
        drew_five = ["Ace of Hearts", "5 of Clubs"]
        drew_ace = ["Ace of Hearts", "Ace of Clubs"]
        constants.RULES["hit_split_aces"] = False
        for hand in (drew_five, drew_ace):
            action, text, color = adv.advice(hand, "6", post_split=True)
            self.assertEqual((action, text), ("S", "Stand (one card)"))
            self.assertEqual(color, constants.ACTION_COLORS["S"])
        # Rule flip: the same hand becomes a live CSV lookup (soft 16 vs 6).
        constants.RULES["hit_split_aces"] = True
        action, text, _ = adv.advice(drew_five, "6", post_split=True)
        self.assertIsNotNone(action)
        self.assertNotEqual(text, "Stand (one card)")

    def test_split_non_aces_drawing_an_ace_is_not_forced(self):
        constants.RULES["hit_split_aces"] = False
        adv = StrategyAdvisor()
        # Split 8s drew an ace: soft 19, a normal lookup — not split aces.
        _, text, _ = adv.advice(["8 of Hearts", "Ace of Clubs"], "6",
                                post_split=True)
        self.assertNotEqual(text, "Stand (one card)")

    def test_engine_snapshot_follows_the_rule_with_no_engine_special_case(self):
        constants.RULES["hit_split_aces"] = False
        eng = _engine()
        eng.replace_card(0, 0, "Ace of Hearts")
        eng.replace_card(0, 1, "Ace of Spades")
        eng.replace_dealer("6")
        eng.set_split(0)
        eng.replace_card(0, 99, "5 of Clubs")  # appended -> shorter hand (h0)
        eng.replace_card(0, 99, "9 of Clubs")  # -> other hand
        self.assertTrue(eng.flush_advice(timeout=60))
        seat = eng.get_snapshot()["seats"][0]
        self.assertIn("H1: Stand (one card)", seat["advice"])
        self.assertIn("H2: Stand (one card)", seat["advice"])
        # ev_engine.advise(post_split=True) does not model the one-card
        # rule, so the optimal line must stay suppressed for split aces.
        self.assertEqual(seat["optimal"], "")

        # Flipping the rule changes the snapshot through strategy.advice
        # alone — the engine has no split-aces special case left to desync.
        constants.RULES["hit_split_aces"] = True
        eng.publish_snapshot()
        self.assertTrue(eng.flush_advice(timeout=60))
        seat = eng.get_snapshot()["seats"][0]
        self.assertNotIn("Stand (one card)", seat["advice"])
        _, h1_text, _ = eng.strategy.advice(
            ["Ace of Hearts", "5 of Clubs"], "6", post_split=True)
        self.assertIn(f"H1: {h1_text}", seat["advice"])


class SelectiveCacheRefresh(unittest.TestCase):
    def setUp(self):
        self._rules = dict(constants.RULES)
        self._deck = constants.DECK_COUNT
        self._side = copy.deepcopy(constants.SIDE_BETS)
        self._betting = dict(constants.BETTING)

    def tearDown(self):
        constants.RULES.clear()
        constants.RULES.update(self._rules)
        constants.DECK_COUNT = self._deck
        constants.SIDE_BETS.clear()
        constants.SIDE_BETS.update(self._side)
        constants.BETTING.clear()
        constants.BETTING.update(self._betting)

    @staticmethod
    def _primed_engine():
        """Engine with sentinel cache contents (init's async side-bet job is
        flushed first so it can't overwrite the sentinels)."""
        eng = _engine()
        eng.flush_advice(timeout=60)
        with eng._advice_lock:
            eng._advice_cache[("sentinel",)] = {"best": "S", "evs": {"S": 0.0}}
            eng._predeal = {"edge": 0.0123, "sig": "sweep"}
            eng._sidebet_result = (["sb"], "sig")
        return eng

    def test_irrelevant_change_keeps_every_cache(self):
        eng = self._primed_engine()
        constants.BETTING["kelly_fraction"] = 0.25  # not part of the signature
        with mock.patch.object(eng, "publish_snapshot") as pub:
            eng.refresh_settings()
        pub.assert_called_once()
        with eng._advice_lock:
            self.assertIn(("sentinel",), eng._advice_cache)
            self.assertEqual(eng._predeal, {"edge": 0.0123, "sig": "sweep"})
            self.assertEqual(eng._sidebet_result, (["sb"], "sig"))

    def test_rules_change_clears_advice_and_predeal_only(self):
        eng = self._primed_engine()
        constants.RULES["s17"] = not constants.RULES["s17"]
        with mock.patch.object(eng, "publish_snapshot"):
            eng.refresh_settings()
        with eng._advice_lock:
            self.assertEqual(eng._advice_cache, {})
            self.assertEqual(eng._predeal, {"edge": None, "sig": None})
            self.assertEqual(eng._sidebet_result, (["sb"], "sig"))  # kept

    def test_deck_count_change_clears_advice_and_predeal(self):
        eng = self._primed_engine()
        constants.DECK_COUNT = 6
        with mock.patch.object(eng, "publish_snapshot"):
            eng.refresh_settings()
        self.assertEqual(eng.counter.deck_count, 6)
        with eng._advice_lock:
            self.assertEqual(eng._advice_cache, {})
            self.assertEqual(eng._predeal, {"edge": None, "sig": None})

    def test_sidebet_change_clears_sidebets_only(self):
        eng = self._primed_engine()
        constants.SIDE_BETS["hot3"]["paytable"]["21"] = 5  # paytable tweak
        with mock.patch.object(eng, "publish_snapshot"):
            eng.refresh_settings()
        with eng._advice_lock:
            self.assertEqual(eng._sidebet_result, ([], None))
            self.assertIn(("sentinel",), eng._advice_cache)
            self.assertEqual(eng._predeal, {"edge": 0.0123, "sig": "sweep"})

    def test_live_hands_not_recomputed_on_irrelevant_refresh(self):
        calls = []
        real_advise = ev_engine.advise

        def counting_advise(*args, **kwargs):
            calls.append(args)
            return real_advise(*args, **kwargs)

        with mock.patch.object(ev_engine, "advise", side_effect=counting_advise):
            eng = _engine()
            _deal_hard_18(eng)
            self.assertTrue(eng.flush_advice(timeout=60))
            n = len(calls)
            self.assertGreater(n, 0)
            constants.BETTING["kelly_fraction"] = 0.25
            eng.refresh_settings()           # nothing cache-relevant changed
            self.assertTrue(eng.flush_advice(timeout=60))
            self.assertEqual(len(calls), n)  # cache hit — no EV recompute
            constants.RULES["s17"] = not constants.RULES["s17"]
            eng.refresh_settings()           # rules change -> recompute
            self.assertTrue(eng.flush_advice(timeout=60))
            self.assertGreater(len(calls), n)


if __name__ == "__main__":
    unittest.main()
