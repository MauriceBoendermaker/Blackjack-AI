"""V3 E5: session guardrails — stop-loss / stop-win / max-rounds banner,
snapshot block, settings persistence, executor coupling (the disarm test
itself lives in tests/test_executor.py with the executor fixtures).

Run:  .venv\\Scripts\\python -m unittest tests.test_guardrails -v
"""

import unittest

from lib.common import constants, settings
from lib.logic import bankroll, betting


class EngineBlock(unittest.TestCase):
    def setUp(self):
        self._saved = dict(constants.GUARDRAILS)
        from lib.logic.engine import DetectionEngine
        self.eng = DetectionEngine(log=lambda *a, **k: None)
        self.eng.store = None

    def tearDown(self):
        constants.GUARDRAILS.clear()
        constants.GUARDRAILS.update(self._saved)

    def block(self, pnl=0.0, rounds=0, **cfg):
        constants.GUARDRAILS.update({"enabled": 1, "stop_loss_eur": 200.0,
                                     "stop_win_eur": 0.0, "max_rounds": 0,
                                     **cfg})
        self.eng.session_pnl["eur"] = pnl
        self.eng.session_pnl["rounds"] = rounds
        self.eng.publish_snapshot()
        return self.eng.get_snapshot()["guardrails"]

    def test_disabled_never_breaches(self):
        block = self.block(pnl=-5000.0, enabled=0)
        self.assertFalse(block["enabled"])
        self.assertFalse(block["breached"])

    def test_stop_loss(self):
        self.assertFalse(self.block(pnl=-199.99)["breached"])
        block = self.block(pnl=-200.0)
        self.assertTrue(block["breached"])
        self.assertEqual(block["kind"], "stop_loss")
        self.assertIn("walk away", block["text"])

    def test_stop_win(self):
        block = self.block(pnl=120.0, stop_win_eur=100.0)
        self.assertTrue(block["breached"])
        self.assertEqual(block["kind"], "stop_win")
        # stop_win 0 = off: a winning session alone never breaches.
        self.assertFalse(self.block(pnl=99_999.0)["breached"])

    def test_max_rounds(self):
        block = self.block(rounds=50, max_rounds=50)
        self.assertTrue(block["breached"])
        self.assertEqual(block["kind"], "max_rounds")
        self.assertFalse(self.block(rounds=49, max_rounds=50)["breached"])


class SettingsRoundTrip(unittest.TestCase):
    def test_persist_and_clamp(self):
        before = settings.snapshot()
        try:
            settings.apply({"guardrails": {"enabled": 1,
                                           "stop_loss_eur": 150,
                                           "stop_win_eur": "junk",
                                           "max_rounds": -5}})
            self.assertEqual(constants.GUARDRAILS["enabled"], 1)
            self.assertEqual(constants.GUARDRAILS["stop_loss_eur"], 150.0)
            self.assertEqual(constants.GUARDRAILS["max_rounds"], 0)
            self.assertEqual(
                settings.snapshot()["guardrails"]["stop_loss_eur"], 150.0)
        finally:
            settings.apply(before)


class MultiSeatModelStats(unittest.TestCase):
    """V3 E5 part 1: k-aware (mu, sigma) for the risk readouts."""

    CFG = {"bankroll": 100_000.0, "kelly_fraction": 0.5, "base_edge": -0.005,
           "edge_per_tc": 0.005, "variance": 1.33, "covariance": 0.479,
           "table_min": 10, "table_max": 5000}

    def test_two_seats_match_hand_computation(self):
        freqs = {3: 1.0}
        mu, sigma = bankroll.model_round_stats(self.CFG, freqs, seats=2)
        bet = betting.suggest(3, self.CFG, seats=2)["bet"]
        edge = betting.estimate_edge(3, self.CFG)
        self.assertAlmostEqual(mu, 2 * edge * bet, places=12)
        self.assertAlmostEqual(
            sigma ** 2, (2 * 1.33 + 2 * 0.479) * bet * bet, places=6)

    def test_one_seat_unchanged(self):
        freqs = {0: 0.5, 3: 0.5}
        self.assertEqual(bankroll.model_round_stats(self.CFG, freqs),
                         bankroll.model_round_stats(self.CFG, freqs, seats=1))


if __name__ == "__main__":
    unittest.main()
