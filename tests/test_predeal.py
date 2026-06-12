"""V2 Feature 3: exact pre-deal EV and the bet-ramp integration.

The full 8-deck sweep takes ~15 s, so these tests use small engineered
compositions with hand-checkable exact values, plus a one-deck sanity sweep.

Run:  .venv\\Scripts\\python -m unittest tests.test_predeal -v
"""

import os
import unittest

from lib.common import constants
from lib.logic import betting, ev_offload
from lib.logic.ev_engine import (ACE, TEN, Rules, full_shoe, predeal_ev,
                                 predeal_ev_upcards)


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


class ParallelSweep(unittest.TestCase):
    """V3 E2: the sweep fans out per up-card; partials must sum to the
    monolithic value, and the offload fan-out must keep run()'s fallback
    contract."""

    def test_upcard_partials_sum_to_the_full_sweep(self):
        comp = full_shoe(1)
        full = predeal_ev(comp, Rules())
        parts = [predeal_ev_upcards(comp, Rules(), (u,)) for u in range(10)]
        self.assertAlmostEqual(sum(parts), full, places=10)
        # And the multi-up-card form gives the same total in one call.
        self.assertAlmostEqual(
            predeal_ev_upcards(comp, Rules(), tuple(range(10))), full,
            places=10)

    def test_partials_respect_the_depletion_guard(self):
        comp = tuple(1 if i == TEN else 0 for i in range(10))
        self.assertEqual(predeal_ev_upcards(comp, Rules(), (TEN,)), 0.0)

    def test_hand_slices_partition_an_upcard(self):
        comp = full_shoe(1)
        for u in (ACE, 1, TEN):
            whole = predeal_ev_upcards(comp, Rules(), (u,))
            for m in (2, 3):
                parts = [predeal_ev_upcards(comp, Rules(), (u,), (j, m))
                         for j in range(m)]
                self.assertAlmostEqual(sum(parts), whole, places=10,
                                       msg=f"upcard {u}, {m} slices")

    def test_predeal_jobs_cover_everything_exactly_once(self):
        from lib.logic.ev_engine import predeal_jobs
        comp = full_shoe(8)
        # Default grain keeps up-cards whole (measured: slicing burns more
        # than it buys); a small grain exercises the slicing partition.
        jobs = predeal_jobs(comp, 8, grain=3.0)
        # Every up-card present; each up-card's slices form one partition.
        seen = {}
        for upcards, slice_of in jobs:
            self.assertEqual(len(upcards), 1)
            seen.setdefault(upcards[0], []).append(slice_of)
        self.assertEqual(set(seen), set(range(10)))
        for u, slices in seen.items():
            if slices == [None]:
                continue
            m = slices[0][1]
            self.assertEqual(sorted(j for j, _ in slices), list(range(m)))
            self.assertTrue(all(mm == m for _, mm in slices))
        # workers <= 1: one monolithic job (the in-process fallback path).
        self.assertEqual(predeal_jobs(comp, 1),
                         [(tuple(range(10)), None)])
        # Default grain: whole up-cards, heaviest (low cards) first.
        whole = predeal_jobs(comp, 8)
        self.assertEqual(len(whole), 10)
        self.assertTrue(all(s is None for _, s in whole))
        self.assertEqual(whole[0][0], (1,))  # 2-up: the deepest dealer tree

    def test_run_many_in_process_fallback(self):
        os.environ["BJ_EV_INPROC"] = "1"
        try:
            self.assertEqual(ev_offload.run_many("predeal", abs,
                                                 [(-2,), (3,), (-4,)]),
                             [2, 3, 4])
            self.assertEqual(ev_offload.run_many("predeal", abs, []), [])
        finally:
            os.environ.pop("BJ_EV_INPROC", None)

    def test_run_many_through_real_workers(self):
        self.assertEqual(ev_offload.run_many("predeal", abs,
                                             [(-i,) for i in range(6)]),
                         list(range(6)))

    def test_predeal_pool_is_sized_advice_stays_single(self):
        self.assertEqual(ev_offload._workers("predeal"),
                         constants.PREDEAL_WORKERS)
        self.assertEqual(ev_offload._workers("advice"), 1)
        self.assertGreaterEqual(constants.PREDEAL_WORKERS, 1)


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
