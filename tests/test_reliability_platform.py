"""Reliability pack (Platform): model warm-up refresh after long idle.

ModelProvider.refresh() rebuilds hosted sessions and re-validates local
weight files; DetectionEngine.health_check_and_refresh() retries it with
exponential backoff; the worker loop triggers it on a large cycle gap.
No test here touches the network — backends and _build are dummies.

Run:  .venv\\Scripts\\python -m unittest tests.test_reliability_platform -v
"""

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from lib.common import constants
from lib.logic.background import DetectionController
from lib.logic.engine import DetectionEngine
from lib.logic.models import ModelError, ModelProvider


class DummyHosted:
    name = "Dummy hosted"
    hosted = True

    def predict(self, *a, **k):
        return []


class DummyLocal:
    name = "Dummy local"
    hosted = False

    def __init__(self, path=None):
        self.path = path

    def predict(self, *a, **k):
        return []


class ProviderRefresh(unittest.TestCase):
    """ModelProvider.refresh(): hosted rebuild + local weights validation."""

    def setUp(self):
        self.provider = ModelProvider()
        self.entries = []  # (message, level)
        self.log = lambda msg, level=None: self.entries.append((msg, level))
        self.build_calls = []
        self._tmp = tempfile.TemporaryDirectory()
        self._old_models_dir = constants.MODELS_DIR
        constants.MODELS_DIR = Path(self._tmp.name)

    def tearDown(self):
        constants.MODELS_DIR = self._old_models_dir
        self._tmp.cleanup()

    def fake_build(self, result=None):
        def build(weights_name, project_id, model_version):
            self.build_calls.append(weights_name)
            if isinstance(result, Exception):
                raise result
            return result if result is not None else DummyHosted()
        self.provider._build = build

    def write_weights(self, name, size):
        path = Path(self._tmp.name) / name
        path.write_bytes(b"x" * size)
        return path

    def test_hosted_models_rebuilt_with_new_identity(self):
        old_players, old_dealer = DummyHosted(), DummyHosted()
        self.provider._players = old_players
        self.provider._dealer = old_dealer
        self.fake_build()
        self.provider.refresh(log=self.log)
        self.assertIsNot(self.provider._players, old_players)
        self.assertIsNot(self.provider._dealer, old_dealer)
        self.assertEqual(self.build_calls,
                         ["player_cards.pt", "dealer_cards.pt"])

    def test_uninitialized_slots_left_alone(self):
        self.provider._players = DummyHosted()
        self.fake_build()
        self.provider.refresh(log=self.log)
        self.assertIsNone(self.provider._dealer)
        self.assertEqual(self.build_calls, ["player_cards.pt"])

    def test_healthy_local_weights_reloaded_without_fallthrough(self):
        path = self.write_weights("player_cards.pt", 4096)
        old = DummyLocal(path)
        self.provider._players = old
        self.fake_build()
        self.provider.refresh(log=self.log)
        fresh = self.provider._players
        self.assertIsInstance(fresh, DummyLocal)
        self.assertIsNot(fresh, old)          # constructable == fresh swap
        self.assertEqual(fresh.path, path)
        self.assertEqual(self.build_calls, [])  # no fall-through needed

    def test_tiny_weights_file_rejected_with_fallthrough(self):
        self.write_weights("player_cards.pt", 10)
        self.provider._players = DummyLocal()
        replacement = DummyHosted()
        self.fake_build(replacement)
        self.provider.refresh(log=self.log)
        self.assertIs(self.provider._players, replacement)
        self.assertEqual(self.build_calls, ["player_cards.pt"])
        errors = [(m, lv) for m, lv in self.entries
                  if "player_cards.pt" in m and lv == "ERROR"]
        self.assertTrue(errors, f"no ERROR naming the file in {self.entries}")

    def test_missing_weights_file_rejected_with_fallthrough(self):
        self.provider._players = DummyLocal()
        replacement = DummyHosted()
        self.fake_build(replacement)
        self.provider.refresh(log=self.log)
        self.assertIs(self.provider._players, replacement)
        self.assertTrue([m for m, lv in self.entries if lv == "ERROR"])

    def test_hosted_rebuild_failure_propagates_model_error(self):
        # The engine's retry loop needs the ModelError to reach it.
        self.provider._players = DummyHosted()
        self.fake_build(ModelError("auth handshake failed"))
        with self.assertRaises(ModelError):
            self.provider.refresh(log=self.log)

    def test_has_hosted(self):
        self.assertFalse(self.provider.has_hosted())
        self.provider._players = DummyLocal()
        self.assertFalse(self.provider.has_hosted())
        self.provider._dealer = DummyHosted()
        self.assertTrue(self.provider.has_hosted())


class FlakyProvider:
    """Stands in for ModelProvider: fails `fail_times` refreshes, then works."""

    backend_name = "fake backend"

    def __init__(self, fail_times=0, hosted=True):
        self.fail_times = fail_times
        self.hosted = hosted
        self.calls = 0

    def has_hosted(self):
        return self.hosted

    def refresh(self, log=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ModelError("transient network error")


class HealthCheckRetry(unittest.TestCase):
    """DetectionEngine.health_check_and_refresh(): retries and backoff."""

    def make_engine(self, provider):
        self.entries = []
        eng = DetectionEngine(
            log=lambda msg, level=None: self.entries.append((msg, level)))
        eng.store = None
        eng.provider = provider
        return eng

    def expected_backoff(self, failures):
        return [mock.call(constants.HEALTH_CHECK_BACKOFF_BASE * (2 ** i))
                for i in range(failures)]

    def test_retry_backoff_sequence_on_transient_error(self):
        provider = FlakyProvider(fail_times=constants.HEALTH_CHECK_RETRY_COUNT - 1)
        eng = self.make_engine(provider)
        with mock.patch("lib.logic.engine.time.sleep") as sleep:
            self.assertTrue(eng.health_check_and_refresh())
        self.assertEqual(provider.calls, constants.HEALTH_CHECK_RETRY_COUNT)
        self.assertEqual(sleep.call_args_list,
                         self.expected_backoff(constants.HEALTH_CHECK_RETRY_COUNT - 1))
        self.assertIsNone(eng.last_error)
        self.assertIsNotNone(eng._model_refresh_ts)

    def test_gives_up_after_retry_count(self):
        provider = FlakyProvider(fail_times=constants.HEALTH_CHECK_RETRY_COUNT)
        eng = self.make_engine(provider)
        with mock.patch("lib.logic.engine.time.sleep") as sleep:
            self.assertFalse(eng.health_check_and_refresh())  # caller re-arms
        self.assertEqual(provider.calls, constants.HEALTH_CHECK_RETRY_COUNT)
        # The final failure is reported, not slept on.
        self.assertEqual(sleep.call_args_list,
                         self.expected_backoff(constants.HEALTH_CHECK_RETRY_COUNT - 1))
        self.assertIn("Model refresh failed", eng.last_error)
        self.assertTrue([m for m, lv in self.entries if lv == "ERROR"])
        self.assertIsNone(eng._model_refresh_ts)

    def test_local_only_setup_skips_refresh(self):
        provider = FlakyProvider(hosted=False)
        eng = self.make_engine(provider)
        self.assertTrue(eng.health_check_and_refresh())  # nothing to refresh
        self.assertEqual(provider.calls, 0)

    def test_snapshot_exposes_refresh_timestamp(self):
        eng = self.make_engine(FlakyProvider())
        self.assertIsNone(eng.get_snapshot()["model_refresh_ts"])
        eng.health_check_and_refresh()
        eng.publish_snapshot()
        self.assertIsNotNone(eng.get_snapshot()["model_refresh_ts"])


class IdleGapDetection(unittest.TestCase):
    """DetectionController._maybe_idle_refresh(): fake-clock gap check.
    Returns True when the gap is handled (none, or refresh succeeded) and the
    caller may advance its baseline; False keeps the gap open for a retry."""

    def setUp(self):
        self.entries = []
        self.controller = DetectionController(
            log=lambda msg, level=None: self.entries.append((msg, level)))
        self.refreshes = []
        self.refresh_result = True
        self.controller.engine = types.SimpleNamespace(
            health_check_and_refresh=self._fake_refresh)

    def _fake_refresh(self):
        self.refreshes.append(True)
        return self.refresh_result

    def announcements(self):
        return [m for m, _ in self.entries if "refreshing model backends" in m]

    def test_gap_over_threshold_triggers_refresh(self):
        handled = self.controller._maybe_idle_refresh(
            1000.0, 1000.0 + constants.IDLE_REFRESH_GAP_S + 0.5)
        self.assertTrue(handled)
        self.assertEqual(len(self.refreshes), 1)
        self.assertTrue(self.announcements())

    def test_normal_cadence_does_not_trigger(self):
        for gap in (constants.CYCLE_SLEEP_WAITING,
                    constants.IDLE_REFRESH_GAP_S):  # boundary is exclusive
            handled = self.controller._maybe_idle_refresh(1000.0, 1000.0 + gap)
            self.assertTrue(handled)  # nothing to do -> baseline advances
        self.assertEqual(self.refreshes, [])
        self.assertEqual(self.announcements(), [])

    def test_failed_refresh_rearms_until_success(self):
        # Wake-from-sleep with Wi-Fi still down: the failed burst must NOT
        # disarm recovery — the frozen baseline re-triggers it every cycle.
        self.refresh_result = False
        wake = 1000.0 + constants.IDLE_REFRESH_GAP_S + 5.0
        self.assertFalse(self.controller._maybe_idle_refresh(1000.0, wake))
        self.assertFalse(self.controller._maybe_idle_refresh(1000.0, wake + 2.0))
        self.assertEqual(len(self.refreshes), 2)
        self.assertEqual(len(self.announcements()), 1)  # retries don't re-spam
        self.refresh_result = True  # the network comes back
        self.assertTrue(self.controller._maybe_idle_refresh(1000.0, wake + 4.0))
        self.assertEqual(len(self.refreshes), 3)
        # A later, fresh gap announces itself again.
        self.assertTrue(self.controller._maybe_idle_refresh(
            wake + 4.0, wake + 4.0 + constants.IDLE_REFRESH_GAP_S + 1.0))
        self.assertEqual(len(self.announcements()), 2)


if __name__ == "__main__":
    unittest.main()
