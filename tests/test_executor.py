"""V4 Feature 2: ghost-mode executor + one-key assisted execute — plan
building, the §5 guard stack, never-reclick, verification, audit trail.

No clicks: the dispatcher is faked; io_submit runs inline.

Run:  .venv\\Scripts\\python -m unittest tests.test_executor -v
"""

import tempfile
import time
import types
import unittest
from pathlib import Path

from lib.common import constants
from lib.logic import executor as executor_mod
from lib.logic import phase
from lib.logic.executor import (ClickDispatcher, Executor, chip_sequence)
from lib.logic.session_store import SessionStore


class FakeDispatcher:
    def __init__(self):
        self.clicks = []

    def click(self, x, y):
        self.clicks.append((x, y))
        return "fake"

    def submit(self, fn, *args):
        fn(*args)  # synchronous: deterministic tests (RLock re-enters)


def make_snap(phase_name=phase.MY_TURN, action="H", seat=0, mine=True,
              frames=3, score=0.95, enabled=True, confident=True,
              n_cards=2, round_=5, split=False, action_list=None,
              bet_suggested=None, sit_out=False, ocr=None, pnl=0.0,
              bankroll=1000.0, bet_spot=None, timer=None):
    seats = [{"index": i, "cards": [], "mine": False, "split": False,
              "optimal_action": None} for i in range(7)]
    seats[seat] = {"index": seat,
                   "cards": ["10 of Hearts"] * n_cards,
                   "mine": mine, "split": split,
                   "optimal_action": action_list if action_list is not None
                   else action}
    control = {"H": "hit", "S": "stand", "D": "double", "P": "split"}.get(
        action or "H", "hit")
    buttons = {control: {"found": True, "enabled": enabled, "score": score,
                         "center": [340, 220]}}
    return {
        "seq": 1, "ts": time.time(), "round": round_, "seats": seats,
        "session_pnl": {"eur": pnl, "units": 0, "side_eur": 0, "rounds": 1},
        "bankroll": bankroll, "bet_placed": 10.0,
        "bet_suggested": bet_suggested, "bet_sit_out": sit_out,
        "ocr": ocr,
        "phase": {"phase": phase_name, "frames": frames, "timer_s": timer,
                  "my_seat": seat, "seat_confident": confident,
                  "buttons": buttons, "calibrated": True,
                  "bet_spot": bet_spot, "confidence": 0.9, "triage": None},
    }


class ChipSequence(unittest.TestCase):
    def test_decomposition(self):
        self.assertEqual(chip_sequence(40, [0.5, 1, 5, 25, 100]),
                         [25, 5, 5, 5])
        self.assertEqual(chip_sequence(7.5, [0.5, 1, 5]), [5, 1, 1, 0.5])
        self.assertIsNone(chip_sequence(37.3, [1, 5]))   # not representable
        self.assertIsNone(chip_sequence(0, [1]))
        self.assertIsNone(chip_sequence(None, [1]))


class Coordinates(unittest.TestCase):
    def test_to_absolute_corners(self):
        virtual = (0, 0, 2560, 1440)
        self.assertEqual(ClickDispatcher.to_absolute(virtual, 0, 0), (0, 0))
        self.assertEqual(ClickDispatcher.to_absolute(virtual, 2559, 1439),
                         (65535, 65535))
        # Negative-origin secondary monitor on the virtual desktop.
        virtual = (-2560, 0, 5120, 1440)
        nx, _ = ClickDispatcher.to_absolute(virtual, 0, 0)
        self.assertEqual(nx, round(2560 * 65535 / 5119))


class ExecutorBase(unittest.TestCase):
    def setUp(self):
        self._out = constants.OUTPUT_DIR
        self._exec = {k: (list(v) if isinstance(v, list) else v)
                      for k, v in constants.EXECUTOR.items()}
        constants.OUTPUT_DIR = Path(tempfile.mkdtemp())
        self.store = SessionStore()
        self.dispatcher = FakeDispatcher()
        self.snap_holder = {"snap": None}
        self.monitor = types.SimpleNamespace(x=0, y=0, width=2560, height=1440)
        self.ex = Executor(
            snapshot_fn=lambda: self.snap_holder["snap"],
            monitor_fn=lambda: self.monitor,
            store=self.store,
            io_submit=lambda fn, *a: fn(*a),
            log=lambda *a, **k: None,
            dispatcher=self.dispatcher,
            running_fn=lambda: True)

    def tearDown(self):
        constants.OUTPUT_DIR = self._out
        constants.EXECUTOR.clear()
        constants.EXECUTOR.update(self._exec)

    def _wait_dispatch(self, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.ex._lock:
                if not self.ex._firing:
                    return
            time.sleep(0.01)
        self.fail("dispatch thread did not finish")


class Planning(ExecutorBase):
    def test_ghost_plans_the_optimal_click(self):
        self.ex.set_mode("ghost")
        state = self.ex.step(make_snap(action="H"))
        self.assertIsNotNone(state["plan"])
        self.assertEqual(state["plan"]["target"], (340, 220))
        self.assertIn("Hit", state["plan"]["label"])
        self.assertIsNotNone(state["marker"])
        self.assertEqual(self.store.executor_stats()["plans"], 1)
        self.assertEqual(self.dispatcher.clicks, [])  # ghost NEVER clicks

    def test_abstains(self):
        self.ex.set_mode("ghost")
        cases = [
            make_snap(confident=False),
            make_snap(mine=False),
            make_snap(action=None),                  # EV pending
            make_snap(action="R"),                   # no surrender control
            make_snap(action="H", enabled=False),    # greyed button
            make_snap(action_list=["H", "S"]),       # split hands
        ]
        for snap in cases:
            state = self.ex.step(snap)
            self.assertIsNone(state["plan"], state["status"])
            self.assertIsNone(state["marker"])

    def test_off_mode_is_inert(self):
        state = self.ex.step(make_snap())
        self.assertIsNone(state["plan"])
        self.assertEqual(self.store.executor_stats()["plans"], 0)

    def test_bet_plan_is_ghost_only(self):
        self.ex.set_mode("ghost")
        snap = make_snap(phase_name=phase.BETTING_OPEN, bet_suggested=40.0,
                         bet_spot={"center": [800, 900], "changed": False})
        state = self.ex.step(snap)
        self.assertEqual(state["plan"]["kind"], "bet")
        self.assertEqual(state["plan"]["chips"], [25, 5, 5, 5])
        self.assertEqual(state["plan"]["target"], (800, 900))

    def test_bet_plan_skips_when_already_placed(self):
        self.ex.set_mode("ghost")
        on_spot = make_snap(phase_name=phase.BETTING_OPEN, bet_suggested=40.0,
                            bet_spot={"center": [800, 900], "changed": True})
        self.assertIsNone(self.ex.step(on_spot)["plan"])
        seen = make_snap(phase_name=phase.BETTING_OPEN, bet_suggested=40.0,
                         bet_spot={"center": [800, 900], "changed": False},
                         ocr={"bet": 40.0, "balance": None, "ts": time.time()})
        self.assertIsNone(self.ex.step(seen)["plan"])

    def test_bet_plan_respects_sit_out(self):
        self.ex.set_mode("ghost")
        snap = make_snap(phase_name=phase.BETTING_OPEN, bet_suggested=40.0,
                         sit_out=True,
                         bet_spot={"center": [800, 900], "changed": False})
        self.assertIsNone(self.ex.step(snap)["plan"])


class ConfirmAndGuards(ExecutorBase):
    def _armed(self):
        self.ex.set_mode("assist")
        self.assertTrue(self.ex.arm())

    def test_confirm_fires_once_and_verifies(self):
        self._armed()
        snap = make_snap(action="H", n_cards=2)
        self.snap_holder["snap"] = snap
        self.ex.step(snap)
        result = self.ex.confirm()
        self.assertIn("firing", result)
        self._wait_dispatch()
        self.assertEqual(self.dispatcher.clicks, [(340, 220)])

        # While the click is unverified, nothing else may fire.
        self.assertIn("still verifying", self.ex.confirm())

        # The card lands -> verification passes, audit row stamped.
        grown = make_snap(action="S", n_cards=3)
        self.ex.step(grown)
        self.assertTrue(self.ex.armed)
        self.assertEqual(self.ex.session["verified"], 1)
        stats = self.store.executor_stats()
        self.assertEqual(stats["fired"], 1)
        self.assertEqual(stats["matched"], 1)
        # plans counts planning rows only — a fired decision is not double-
        # counted by its confirmed row.
        self.assertEqual(stats["plans"], 2)  # the H plan + the new S plan

        # Never re-click the same decision (old snapshot still held).
        self.assertIn("never re-click", self.ex.confirm())

    def test_confirm_refuses_when_plan_changed_since_display(self):
        self._armed()
        shown = make_snap(action="H")
        self.ex.step(shown)  # the marker the human is looking at says Hit
        # Advice flips to Stand before the key lands (same decision key).
        self.snap_holder["snap"] = make_snap(action="S")
        result = self.ex.confirm()
        self.assertIn("plan changed", result)
        self.assertEqual(self.dispatcher.clicks, [])

    def test_confirm_refuses_stale_snapshot(self):
        self._armed()
        snap = make_snap(action="H")
        snap["ts"] = time.time() - 10  # frozen worker
        self.snap_holder["snap"] = snap
        self.ex.step(make_snap(action="H"))
        result = self.ex.confirm()
        self.assertIn("stalled", result)
        self.assertFalse(self.ex.armed)  # passive guard auto-disarmed
        self.assertEqual(self.dispatcher.clicks, [])

    def test_arm_requires_running_detection(self):
        self.ex.running_fn = lambda: False
        self.ex.set_mode("assist")
        self.assertFalse(self.ex.arm())
        self.assertFalse(self.ex.armed)
        self.assertIn("not running", self.ex.disarm_reason)

    def test_verification_timeout_disarms(self):
        constants.EXECUTOR["verify_timeout_s"] = 0.05
        self._armed()
        snap = make_snap(action="H", n_cards=2)
        self.snap_holder["snap"] = snap
        self.ex.step(snap)
        self.ex.confirm()
        self._wait_dispatch()
        time.sleep(0.08)
        self.ex.step(snap)  # nothing changed on screen
        self.assertFalse(self.ex.armed)
        self.assertIn("verification failed", self.ex.disarm_reason)
        self.assertEqual(self.ex.session["mismatches"], 1)

    def test_confirm_requires_arm_mode_and_frames(self):
        self.assertIn("not in assist", self.ex.confirm())
        self.ex.set_mode("assist")
        self.assertIn("not armed", self.ex.confirm())
        self.assertTrue(self.ex.arm())
        self.snap_holder["snap"] = make_snap(frames=1)
        self.assertIn("confirmed 1 frame", self.ex.confirm())
        self.snap_holder["snap"] = make_snap(score=0.5)
        self.assertIn("below", self.ex.confirm())
        self.assertEqual(self.dispatcher.clicks, [])

    def test_bets_never_fire(self):
        self._armed()
        self.snap_holder["snap"] = make_snap(
            phase_name=phase.BETTING_OPEN, bet_suggested=40.0,
            bet_spot={"center": [800, 900], "changed": False})
        self.assertIn("ghost-only", self.ex.confirm())
        self.assertEqual(self.dispatcher.clicks, [])

    def test_stop_loss_and_unknown_disarm(self):
        self._armed()
        self.ex.step(make_snap(pnl=-250.0))
        self.assertFalse(self.ex.armed)
        self.assertIn("stop-loss", self.ex.disarm_reason)

        self.assertTrue(self.ex.arm())
        self.ex.step(make_snap(phase_name=phase.UNKNOWN))
        self.assertFalse(self.ex.armed)
        self.assertIn("unidentified", self.ex.disarm_reason)

    def test_balance_mismatch_disarms(self):
        self._armed()
        snap = make_snap(bankroll=1000.0,
                         ocr={"balance": 900.0, "bet": None,
                              "ts": time.time()})
        self.ex.step(snap)
        self.assertFalse(self.ex.armed)
        self.assertIn("balance mismatch", self.ex.disarm_reason)

    def test_anchor_drift_disarms(self):
        # V3 E3: drifted calibration means every click target is suspect.
        self._armed()
        snap = make_snap()
        snap["anchors"] = {"status": "active", "drift": True}
        self.ex.step(snap)
        self.assertFalse(self.ex.armed)
        self.assertIn("anchor drift", self.ex.disarm_reason)

    def test_session_guardrail_disarms(self):
        # V3 E5: a breached session plan stops assisted clicks; the human
        # can keep playing manually but nothing fires.
        self._armed()
        snap = make_snap()
        snap["guardrails"] = {"enabled": True, "breached": True,
                              "kind": "stop_win",
                              "text": "STOP-WIN reached (€+120.00) — bank it"}
        self.ex.step(snap)
        self.assertFalse(self.ex.armed)
        self.assertIn("session guardrail", self.ex.disarm_reason)

    def test_double_beyond_max_bet_disarms(self):
        constants.EXECUTOR["max_bet_eur"] = 15.0
        self._armed()
        self.snap_holder["snap"] = make_snap(action="D")  # 10 -> 20 > 15
        self.assertIn("disarmed", self.ex.confirm())
        self.assertFalse(self.ex.armed)

    def test_kill_switch(self):
        self._armed()
        self.ex.kill("kill switch")
        self.assertFalse(self.ex.armed)
        self.assertEqual(self.ex.disarm_reason, "kill switch")

    def test_arm_requires_assist_mode(self):
        self.ex.set_mode("ghost")
        self.assertFalse(self.ex.arm())
        self.assertFalse(self.ex.armed)


if __name__ == "__main__":
    unittest.main()
