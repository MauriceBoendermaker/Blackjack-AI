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


if __name__ == "__main__":
    unittest.main()
