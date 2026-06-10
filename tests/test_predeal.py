"""V2 Feature 3: exact pre-deal EV and the bet-ramp integration.

The full 8-deck sweep takes ~15 s, so these tests use small engineered
compositions with hand-checkable exact values, plus a one-deck sanity sweep.

Run:  .venv\\Scripts\\python -m unittest tests.test_predeal -v
"""

import unittest

from lib.common import constants
from lib.logic import betting
from lib.logic.ev_engine import ACE, TEN, Rules, full_shoe, predeal_ev


class PredealExactValues(unittest.TestCase):
    def test_all_tens_shoe_is_exactly_zero(self):
        # Only tens left: every hand is 20, the dealer always has 20 -> push.
        comp = tuple(0 if i != TEN else 64 for i in range(10))
        self.assertAlmostEqual(predeal_ev(comp, Rules()), 0.0, places=12)

    def test_naturals_rich_shoe_is_positive(self):
        # Half aces, half tens: naturals everywhere -> strongly +EV.
        comp = tuple(32 if i in (ACE, TEN) else 0 for i in range(10))
        ev = predeal_ev(comp, Rules())
        self.assertGreater(ev, 0.05)

    def test_one_deck_sane_and_composition_sensitive(self):
        base = predeal_ev(full_shoe(1), Rules())
        self.assertGreater(base, -0.03)
        self.assertLess(base, 0.01)
        # Strip low cards -> player edge must improve.
        rich = list(full_shoe(1))
        for i in range(1, 6):
            rich[i] -= 2
        self.assertGreater(predeal_ev(tuple(rich), Rules()), base)

    def test_too_depleted_returns_zero(self):
        comp = tuple(1 if i == TEN else 0 for i in range(10))
        self.assertEqual(predeal_ev(comp, Rules()), 0.0)


class BettingIntegration(unittest.TestCase):
    def test_exact_edge_overrides_estimate(self):
        cfg = {"bankroll": 100_000.0, "kelly_fraction": 0.5, "base_edge": -0.005,
               "edge_per_tc": 0.005, "variance": 1.33, "table_min": 10,
               "table_max": 5000, "auto_bankroll": 0, "use_exact_edge": 1}
        linear = betting.suggest(3.0, cfg)
        exact = betting.suggest(3.0, cfg, exact_edge=0.02)
        self.assertIn("TC est.", linear["text"])
        self.assertIn("exact", exact["text"])
        self.assertAlmostEqual(exact["edge"], 0.02)
        self.assertGreater(exact["bet"], linear["bet"])
        # Exact edge of zero/negative -> min bet regardless of the count.
        floor = betting.suggest(5.0, cfg, exact_edge=-0.01)
        self.assertEqual(floor["bet"], 10)

    def test_engine_gates_predeal_when_headless(self):
        from lib.logic.engine import DetectionEngine
        eng = DetectionEngine(log=lambda *a, **k: None)
        # No monitor selected -> no sweep is ever scheduled.
        snap = eng.get_snapshot()
        self.assertIsNone(snap["edge_exact"])
        self.assertFalse(eng._predeal_pending)
        self.assertIn("TC est.", snap["bet"])


if __name__ == "__main__":
    unittest.main()
