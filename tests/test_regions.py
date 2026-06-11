"""Feature 10: per-resolution calibrated region profiles.

Run:  .venv\\Scripts\\python -m unittest tests.test_regions -v
"""

import tempfile
import unittest
from pathlib import Path

from lib.common import constants
from lib.logic import monitor_utils, region_profiles


class RegionProfiles(unittest.TestCase):
    RES = (1920, 1080)

    def setUp(self):
        self._out = constants.OUTPUT_DIR
        constants.OUTPUT_DIR = Path(tempfile.mkdtemp())

    def tearDown(self):
        constants.OUTPUT_DIR = self._out

    def test_defaults_when_no_profile(self):
        self.assertIsNone(monitor_utils.load_custom_regions(self.RES))
        regions = monitor_utils.scaled_player_regions(self.RES)
        self.assertEqual(len(regions), constants.NUM_SEATS)
        # 1920/2560 scaling applied to the first base vertex.
        sx = 1920 / constants.BASE_RESOLUTION[0]
        self.assertAlmostEqual(regions[0].vertices[0][0],
                               constants.BASE_PLAYER_REGIONS[0][0][0] * sx)
        rect = monitor_utils.dealer_area_rect(self.RES)
        self.assertEqual(rect[0], int(constants.DEALER_AREA_LEFT * sx))

    def test_save_load_round_trip_and_priority(self):
        players = monitor_utils.default_player_regions(self.RES)
        players[0][0] = [123.0, 456.0]  # calibrated tweak
        dealer = (10, 20, 800, 600)
        monitor_utils.save_custom_regions(self.RES, players, dealer)

        loaded = monitor_utils.load_custom_regions(self.RES)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["dealer"], [10, 20, 800, 600])

        regions = monitor_utils.scaled_player_regions(self.RES)
        self.assertAlmostEqual(regions[0].vertices[0][0], 123.0)
        self.assertEqual(monitor_utils.dealer_area_rect(self.RES), (10, 20, 800, 600))

        # Profiles are per resolution: another resolution stays default.
        self.assertIsNone(monitor_utils.load_custom_regions((1280, 720)))

        monitor_utils.delete_custom_regions(self.RES)
        self.assertIsNone(monitor_utils.load_custom_regions(self.RES))

    def test_corrupt_or_wrong_shape_falls_back(self):
        path = region_profiles.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{broken", encoding="utf-8")
        self.assertIsNone(monitor_utils.load_custom_regions(self.RES))
        # Wrong seat count inside an otherwise valid store entry.
        region_profiles.set_regions(
            self.RES, {"players": [[[1, 2], [3, 4], [5, 6]]],
                       "dealer": [1, 2, 3, 4]})
        self.assertIsNone(monitor_utils.load_custom_regions(self.RES))
        self.assertEqual(len(monitor_utils.scaled_player_regions(self.RES)),
                         constants.NUM_SEATS)

    def test_polygon_containment_with_custom_regions(self):
        players = [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]
        players += monitor_utils.default_player_regions(self.RES)[1:]
        monitor_utils.save_custom_regions(self.RES, players, (0, 0, 50, 50))
        regions = monitor_utils.scaled_player_regions(self.RES)
        self.assertTrue(regions[0].contains(50, 50))
        self.assertFalse(regions[0].contains(150, 50))


if __name__ == "__main__":
    unittest.main()
