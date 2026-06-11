"""Named table profiles for region calibration (V3.2).

Run:  .venv\\Scripts\\python -m unittest tests.test_region_profiles -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from lib.common import constants
from lib.logic import monitor_utils, ocr, region_profiles


class ProfileStore(unittest.TestCase):
    RES = (1920, 1080)

    def setUp(self):
        self._out = constants.OUTPUT_DIR
        constants.OUTPUT_DIR = Path(tempfile.mkdtemp())

    def tearDown(self):
        constants.OUTPUT_DIR = self._out

    def test_fresh_install_has_default_and_creates_no_files(self):
        self.assertEqual(region_profiles.active_name(), "Default")
        self.assertEqual([p["name"] for p in region_profiles.list_profiles()],
                         ["Default"])
        self.assertFalse(region_profiles.store_path().exists())

    def test_legacy_files_become_the_default_profile(self):
        out = constants.OUTPUT_DIR
        regions = {"players": [[[1, 2], [3, 4], [5, 6]]], "dealer": [1, 2, 3, 4]}
        ocr_rects = {"balance": [10, 20, 200, 60]}
        (out / "regions_1920x1080.json").write_text(json.dumps(regions),
                                                    encoding="utf-8")
        (out / "ocr_regions_1920x1080.json").write_text(json.dumps(ocr_rects),
                                                        encoding="utf-8")

        self.assertEqual(region_profiles.get_regions(self.RES), regions)
        self.assertEqual(region_profiles.get_ocr(self.RES), ocr_rects)
        profile = region_profiles.list_profiles()[0]
        self.assertEqual(profile["name"], "Default")
        self.assertIsNotNone(profile["saved"])  # stamped from file mtime
        # Legacy files renamed so they are never re-imported.
        self.assertFalse((out / "regions_1920x1080.json").exists())
        self.assertTrue((out / "regions_1920x1080.json.migrated").exists())
        self.assertTrue(region_profiles.store_path().exists())

    def test_named_save_creates_activates_and_stamps(self):
        region_profiles.set_regions(self.RES, {"players": [], "dealer": []},
                                    profile="Table A")
        self.assertEqual(region_profiles.active_name(), "Table A")
        names = [p["name"] for p in region_profiles.list_profiles()]
        self.assertEqual(names, ["Default", "Table A"])  # Default sorts first
        saved = next(p["saved"] for p in region_profiles.list_profiles()
                     if p["name"] == "Table A")
        self.assertRegex(saved, r"^\d{4}-\d{2}-\d{2}T")

    def test_switching_profiles_switches_what_loads(self):
        players = monitor_utils.default_player_regions(self.RES)
        monitor_utils.save_custom_regions(self.RES, players, (1, 2, 3, 4),
                                          profile="Table A")
        ocr.save_regions(self.RES, {"balance": [10, 20, 200, 60]})  # active=A

        self.assertEqual(monitor_utils.dealer_area_rect(self.RES), (1, 2, 3, 4))
        self.assertEqual(ocr.load_regions(self.RES),
                         {"balance": [10, 20, 200, 60]})

        # Default has no calibration for this resolution -> shipped defaults.
        self.assertTrue(region_profiles.set_active("Default"))
        self.assertIsNone(monitor_utils.load_custom_regions(self.RES))
        self.assertIsNone(ocr.load_regions(self.RES))

        self.assertTrue(region_profiles.set_active("Table A"))
        self.assertEqual(monitor_utils.dealer_area_rect(self.RES), (1, 2, 3, 4))
        self.assertFalse(region_profiles.set_active("No Such Table"))

    def test_blank_name_falls_back_to_active(self):
        region_profiles.set_regions(self.RES, {"players": [], "dealer": []},
                                    profile="   ")
        self.assertEqual(region_profiles.active_name(), "Default")
        self.assertEqual([p["name"] for p in region_profiles.list_profiles()],
                         ["Default"])

    def test_corrupt_store_is_quarantined_not_overwritten(self):
        path = region_profiles.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{broken", encoding="utf-8")
        self.assertEqual(region_profiles.active_name(), "Default")
        corrupt = path.with_name(path.name + ".corrupt")
        self.assertTrue(corrupt.exists())   # original bytes preserved
        self.assertFalse(path.exists())     # next save starts a fresh store
        region_profiles.set_regions(self.RES, {"players": [], "dealer": []},
                                    profile="Table A")
        self.assertTrue(path.exists())
        self.assertEqual(corrupt.read_text(encoding="utf-8"), "{broken")

    def test_non_dict_kind_values_are_healed(self):
        path = region_profiles.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"active": "Default", "profiles": {
            "Default": {"saved": None, "regions": None, "ocr": []}}}),
            encoding="utf-8")
        # Loads fall back to defaults instead of raising AttributeError...
        self.assertIsNone(monitor_utils.load_custom_regions(self.RES))
        self.assertIsNone(ocr.load_regions(self.RES))
        # ...and a save self-heals the store.
        ocr.save_regions(self.RES, {"balance": [10, 20, 200, 60]})
        self.assertEqual(ocr.load_regions(self.RES),
                         {"balance": [10, 20, 200, 60]})

    def test_delete_only_touches_active_profile_resolution(self):
        monitor_utils.save_custom_regions(
            self.RES, monitor_utils.default_player_regions(self.RES),
            (1, 2, 3, 4), profile="Table A")
        ocr.save_regions(self.RES, {"balance": [10, 20, 200, 60]})
        monitor_utils.delete_custom_regions(self.RES)
        self.assertIsNone(monitor_utils.load_custom_regions(self.RES))
        # The OCR rects of the same profile survive a regions reset.
        self.assertEqual(ocr.load_regions(self.RES),
                         {"balance": [10, 20, 200, 60]})


if __name__ == "__main__":
    unittest.main()
