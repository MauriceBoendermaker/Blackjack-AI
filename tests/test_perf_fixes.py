"""Perf & OCR-sync pack: EV subprocess offload, snapshot bankroll, deferred
settings persistence, and no-op bet commits.

Run:  .venv\\Scripts\\python -m unittest tests.test_perf_fixes -v
"""

import os
import unittest

from lib.common import constants
from lib.logic import ev_engine, ev_offload
from lib.logic.ev_engine import TEN, Rules


def _tens_only():
    return tuple(0 if i != TEN else 64 for i in range(10))


class OffloadRun(unittest.TestCase):
    def test_subprocess_result_matches_direct_call(self):
        # All-tens shoe: every hand pushes -> exactly 0. Cheap to sweep.
        direct = ev_engine.predeal_ev(_tens_only(), Rules())
        shipped = ev_offload.run("predeal", ev_engine.predeal_ev,
                                 _tens_only(), Rules())
        self.assertEqual(shipped, direct)

    def test_advise_through_worker(self):
        result = ev_offload.run(
            "advice", ev_engine.advise, ["10 of Hearts", "6 of Clubs"], "10",
            {"10": 3, "6": 1}, 8, Rules())
        self.assertIn(result["best"], ("S", "H", "D", "R"))
        self.assertIn("H", result["evs"])

    def test_inproc_env_bypasses_pools(self):
        os.environ["BJ_EV_INPROC"] = "1"
        try:
            value = ev_offload.run("predeal", ev_engine.predeal_ev,
                                   _tens_only(), Rules())
            self.assertEqual(value, 0.0)
        finally:
            del os.environ["BJ_EV_INPROC"]

    def test_prewarm_is_safe(self):
        ev_offload.prewarm("advice")  # must never raise


class EngineMoneySync(unittest.TestCase):
    def setUp(self):
        from lib.logic.engine import DetectionEngine
        self.eng = DetectionEngine(log=lambda *a, **k: None)
        self.eng.store = None
        self._betting = dict(constants.BETTING)
        self._ocr = dict(constants.OCR)
        constants.OCR.update(enabled=1, sync_bankroll=1, sync_bet=1)

    def tearDown(self):
        constants.BETTING.clear()
        constants.BETTING.update(self._betting)
        constants.OCR.clear()
        constants.OCR.update(self._ocr)

    def test_snapshot_carries_bankroll(self):
        constants.BETTING["bankroll"] = 1234.5
        self.eng.publish_snapshot()
        self.assertEqual(self.eng.get_snapshot()["bankroll"], 1234.5)

    def test_ocr_values_reach_snapshot_and_defer_settings_write(self):
        self.eng._apply_ocr_values({"balance": 987.65, "bet": 25.0,
                                    "result": None})
        snap = self.eng.get_snapshot()
        self.assertEqual(constants.BETTING["bankroll"], 987.65)
        self.assertEqual(snap["bankroll"], 987.65)
        self.assertEqual(snap["bet_placed"], 25.0)
        # Persistence is deferred to round end, not done per read.
        self.assertTrue(self.eng._settings_dirty)

    def test_noop_bet_commit_publishes_nothing(self):
        self.eng.set_bet_placed(40.0)
        seq = self.eng.get_snapshot()["seq"]
        self.eng.set_bet_placed(40.0)  # FocusOut traversal with same value
        self.assertEqual(self.eng.get_snapshot()["seq"], seq)


if __name__ == "__main__":
    unittest.main()
