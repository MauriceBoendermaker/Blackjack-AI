"""Feature 9: persisted table profiles and dynamic rules pickup.

Run:  .venv\\Scripts\\python -m unittest tests.test_settings -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from lib.common import constants, settings
from lib.logic import ev_engine


class SettingsRoundTrip(unittest.TestCase):
    def setUp(self):
        self._before = settings.snapshot()
        self._path = settings.SETTINGS_PATH
        settings.SETTINGS_PATH = Path(tempfile.mkdtemp()) / "settings.json"

    def tearDown(self):
        settings.apply(self._before)
        settings.SETTINGS_PATH = self._path

    def test_apply_and_snapshot(self):
        settings.apply({"rules": {"s17": False, "peek": True},
                        "deck_count": 6, "base_bet": 25,
                        "side_bets": {"lucky_lucky": {"enabled": True},
                                      "hot3": {"enabled": False}}})
        self.assertFalse(constants.RULES["s17"])
        self.assertTrue(constants.RULES["peek"])
        self.assertEqual(constants.DECK_COUNT, 6)
        self.assertEqual(constants.BASE_BET, 25)
        self.assertTrue(constants.SIDE_BETS["lucky_lucky"]["enabled"])
        self.assertFalse(constants.SIDE_BETS["hot3"]["enabled"])
        snap = settings.snapshot()
        self.assertEqual(snap["deck_count"], 6)
        self.assertFalse(snap["rules"]["s17"])

    def test_save_load_round_trip(self):
        settings.apply({"rules": {"surrender": True}, "deck_count": 4})
        settings.save()
        # Trash the live values, then restore from disk.
        settings.apply({"rules": {"surrender": False}, "deck_count": 8})
        self.assertTrue(settings.load_and_apply())
        self.assertTrue(constants.RULES["surrender"])
        self.assertEqual(constants.DECK_COUNT, 4)

    def test_unknown_keys_ignored_and_clamped(self):
        settings.apply({"rules": {"bogus": 1}, "deck_count": 99, "junk": True})
        self.assertNotIn("bogus", constants.RULES)
        self.assertEqual(constants.DECK_COUNT, 8)  # clamped to max

    def test_corrupt_file_is_survived(self):
        settings.SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings.SETTINGS_PATH.write_text("{not json", encoding="utf-8")
        self.assertFalse(settings.load_and_apply())

    def test_rules_changes_reach_the_ev_engine(self):
        per_rank = {k: 0 for k in ("Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10")}
        per_rank.update({"10": 2, "6": 1})
        hand = ["10 of Hearts", "6 of Clubs"]
        settings.apply({"rules": {"peek": False}})
        enhc = ev_engine.advise(hand, "King", per_rank)["evs"]["S"]
        settings.apply({"rules": {"peek": True}})
        peek = ev_engine.advise(hand, "King", per_rank)["evs"]["S"]
        # ENHC stand EV mixes in the dealer-BJ loss -> strictly worse.
        self.assertLess(enhc, peek)


class AppSettingsRoundTrip(unittest.TestCase):
    """ui/app/ocr sections from the settings dialog's App tab."""

    _ATTRS = ("EMPTY_FRAMES_FOR_RESET", "CUTTING_CARD_CONFIRM_FRAMES",
              "DEALER_USE_PLAYER_MODEL",
              "EV_ADVICE_TIMEOUT_S", "IDLE_REFRESH_GAP_S", "SNAPSHOT_POLL_MS")

    def setUp(self):
        # ConstantsGuard pattern: module attrs need explicit save/restore on
        # top of the snapshot, since tests below clamp/coerce them directly.
        self._before = settings.snapshot()
        self._attrs = {name: getattr(constants, name) for name in self._ATTRS}
        self._ui = dict(constants.UI)
        self._ocr = dict(constants.OCR)
        self._path = settings.SETTINGS_PATH
        settings.SETTINGS_PATH = Path(tempfile.mkdtemp()) / "settings.json"

    def tearDown(self):
        settings.apply(self._before)
        for name, value in self._attrs.items():
            setattr(constants, name, value)
        constants.UI.clear()
        constants.UI.update(self._ui)
        constants.OCR.clear()
        constants.OCR.update(self._ocr)
        settings.SETTINGS_PATH = self._path

    def test_snapshot_has_ui_app_and_ocr_sections(self):
        snap = settings.snapshot()
        self.assertEqual(snap["ui"], dict(constants.UI))
        self.assertEqual(set(snap["app"]), set(self._ATTRS))
        self.assertEqual(set(snap["ocr"]),
                         {"enabled", "sync_bankroll", "sync_bet"})

    def test_apply_sets_module_attrs_with_types(self):
        settings.apply({"ui": {"scale": 150},
                        "app": {"EMPTY_FRAMES_FOR_RESET": "7",
                                "CUTTING_CARD_CONFIRM_FRAMES": 2,
                                "EV_ADVICE_TIMEOUT_S": 5,
                                "IDLE_REFRESH_GAP_S": 60,
                                "SNAPSHOT_POLL_MS": 200.0},
                        "ocr": {"enabled": False, "sync_bet": True}})
        self.assertEqual(constants.UI["scale"], 150)
        self.assertEqual(constants.EMPTY_FRAMES_FOR_RESET, 7)  # str coerced
        self.assertEqual(constants.CUTTING_CARD_CONFIRM_FRAMES, 2)
        self.assertEqual(constants.EV_ADVICE_TIMEOUT_S, 5.0)
        self.assertIsInstance(constants.EV_ADVICE_TIMEOUT_S, float)
        self.assertEqual(constants.IDLE_REFRESH_GAP_S, 60.0)
        self.assertIsInstance(constants.IDLE_REFRESH_GAP_S, float)
        self.assertEqual(constants.SNAPSHOT_POLL_MS, 200)
        self.assertIsInstance(constants.SNAPSHOT_POLL_MS, int)
        self.assertEqual(constants.OCR["enabled"], 0)  # bools stored as 1/0
        self.assertEqual(constants.OCR["sync_bet"], 1)

    def test_save_load_round_trip(self):
        settings.apply({"ui": {"scale": 125},
                        "app": {"EMPTY_FRAMES_FOR_RESET": 8,
                                "SNAPSHOT_POLL_MS": 90},
                        "ocr": {"sync_bankroll": 0}})
        settings.save()
        # Trash the live values, then restore from disk.
        settings.apply({"ui": {"scale": 0},
                        "app": {"EMPTY_FRAMES_FOR_RESET": 3,
                                "SNAPSHOT_POLL_MS": 400},
                        "ocr": {"sync_bankroll": 1}})
        self.assertTrue(settings.load_and_apply())
        self.assertEqual(constants.UI["scale"], 125)
        self.assertEqual(constants.EMPTY_FRAMES_FOR_RESET, 8)
        self.assertEqual(constants.SNAPSHOT_POLL_MS, 90)
        self.assertEqual(constants.OCR["sync_bankroll"], 0)

    def test_app_values_clamped_to_dialog_bounds(self):
        settings.apply({"app": {"EMPTY_FRAMES_FOR_RESET": 99,
                                "CUTTING_CARD_CONFIRM_FRAMES": 0,
                                "EV_ADVICE_TIMEOUT_S": 0.1,
                                "IDLE_REFRESH_GAP_S": 10_000,
                                "SNAPSHOT_POLL_MS": 5}})
        self.assertEqual(constants.EMPTY_FRAMES_FOR_RESET, 10)
        self.assertEqual(constants.CUTTING_CARD_CONFIRM_FRAMES, 1)
        self.assertEqual(constants.EV_ADVICE_TIMEOUT_S, 1.0)
        self.assertEqual(constants.IDLE_REFRESH_GAP_S, 600.0)
        self.assertEqual(constants.SNAPSHOT_POLL_MS, 60)

    def test_ui_scale_clamped_zero_means_auto(self):
        settings.apply({"ui": {"scale": -25}})
        self.assertEqual(constants.UI["scale"], 0)  # non-positive -> auto
        settings.apply({"ui": {"scale": 37}})
        self.assertEqual(constants.UI["scale"], 75)  # scaling module minimum
        settings.apply({"ui": {"scale": 999}})
        self.assertEqual(constants.UI["scale"], 300)  # scaling module maximum

    def test_garbage_values_ignored_unknown_attrs_not_set(self):
        before = constants.EMPTY_FRAMES_FOR_RESET
        settings.apply({"ui": {"scale": "huge"},
                        "app": {"EMPTY_FRAMES_FOR_RESET": "lots",
                                "DEALER_CONFIRM_FRAMES": 99}})
        self.assertEqual(constants.UI["scale"], self._ui["scale"])
        self.assertEqual(constants.EMPTY_FRAMES_FOR_RESET, before)
        # Only whitelisted attrs are setattr'd — no arbitrary constant pokes.
        self.assertEqual(constants.DEALER_CONFIRM_FRAMES, 2)


if __name__ == "__main__":
    unittest.main()
