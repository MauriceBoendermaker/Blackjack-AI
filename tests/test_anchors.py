"""V3 E3: anchor-based, resolution-independent calibration.

Synthetic screens: a "calibrated" frame with distinctive noise patches as
anchors, and "live" frames built by scaling + offsetting it — the solver
must recover the exact transform, the engine must remap its cached
geometry from it, and a moved anchor must latch the drift warning.

Run:  .venv\\Scripts\\python -m unittest tests.test_anchors -v
"""

import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

from lib.common import constants
from lib.logic import anchors, ocr, region_profiles
from lib.logic.monitor_utils import save_custom_regions

CALIB_RES = (800, 600)
ANCHOR_RECTS = {"anchor_a": (60, 50, 140, 110),
                "anchor_b": (640, 70, 730, 140),
                "anchor_c": (340, 480, 430, 540)}


def calib_frame():
    """Structured patches (like real logos/buttons) — pure noise would be
    the template-matching worst case under a scale mismatch and is not
    what anchors are captured from."""
    frame = np.full((600, 800, 3), 30, dtype=np.uint8)
    for i, (left, top, right, bottom) in enumerate(ANCHOR_RECTS.values()):
        color = [(40, 160, 220), (220, 160, 40), (60, 200, 60)][i]
        cv2.rectangle(frame, (left, top), (right - 1, bottom - 1),
                      color, -1)
        cv2.circle(frame, ((left + right) // 2, (top + bottom) // 2),
                   (bottom - top) // 3, (250, 250, 250), -1)
        cv2.line(frame, (left, top), (right, bottom), (10, 10, 10), 5)
        cv2.rectangle(frame, (left + 8, top + 8),
                      (right - 9, bottom - 9), (10, 10, 120), 3)
    return frame


def live_frame(scale=1.3, dx=24, dy=18, size=(880, 1120)):
    base = calib_frame()
    resized = cv2.resize(base, (int(800 * scale), int(600 * scale)),
                         interpolation=cv2.INTER_LINEAR)
    live = np.full((size[0], size[1], 3), 30, dtype=np.uint8)
    h = min(resized.shape[0], size[0] - dy)
    w = min(resized.shape[1], size[1] - dx)
    live[dy:dy + h, dx:dx + w] = resized[:h, :w]
    return live


class TempStore(unittest.TestCase):
    def setUp(self):
        self._out = constants.OUTPUT_DIR
        self._controls = constants.CONTROLS_DIR
        tmp = Path(tempfile.mkdtemp())
        constants.OUTPUT_DIR = tmp / "output"
        constants.CONTROLS_DIR = tmp / "controls"

    def tearDown(self):
        constants.OUTPUT_DIR = self._out
        constants.CONTROLS_DIR = self._controls

    def save_test_anchors(self, keys=("anchor_a", "anchor_b", "anchor_c")):
        frame = calib_frame()
        captures = {}
        for key in keys:
            left, top, right, bottom = ANCHOR_RECTS[key]
            captures[key] = {"rect": list(ANCHOR_RECTS[key]),
                             "crop": frame[top:bottom, left:right].copy()}
        anchors.save_anchors(CALIB_RES, captures)


class Storage(TempStore):
    def test_round_trip_and_resolutions(self):
        self.save_test_anchors()
        loaded = anchors.load_anchors(CALIB_RES)
        self.assertEqual(set(loaded), set(ANCHOR_RECTS))
        self.assertEqual(loaded["anchor_a"]["rect"],
                         ANCHOR_RECTS["anchor_a"])
        self.assertEqual(loaded["anchor_a"]["template"].shape, (60, 80, 3))
        self.assertEqual(region_profiles.anchor_resolutions(), ["800x600"])
        self.assertEqual(anchors.calibrated_resolution((1920, 1080)),
                         CALIB_RES)
        self.assertEqual(anchors.calibrated_resolution(CALIB_RES), CALIB_RES)
        region_profiles.delete_anchors(CALIB_RES)
        self.assertEqual(anchors.load_anchors(CALIB_RES), {})
        self.assertIsNone(anchors.calibrated_resolution((1920, 1080)))


class Solve(TempStore):
    def test_recovers_scale_and_offset(self):
        self.save_test_anchors()
        anchor_set = anchors.load_anchors(CALIB_RES)
        fit = anchors.solve(live_frame(1.3, 24, 18), anchor_set, CALIB_RES)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["scale"], 1.3, delta=0.02)
        self.assertAlmostEqual(fit["dx"], 24, delta=5)
        self.assertAlmostEqual(fit["dy"], 18, delta=5)
        self.assertEqual(fit["matched"], 3)

    def test_identity_when_nothing_changed(self):
        self.save_test_anchors()
        anchor_set = anchors.load_anchors(CALIB_RES)
        fit = anchors.solve(calib_frame(), anchor_set, CALIB_RES)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["scale"], 1.0, delta=0.01)
        self.assertAlmostEqual(fit["dx"], 0, delta=2)
        self.assertAlmostEqual(fit["dy"], 0, delta=2)

    def test_fail_disabled_on_blank_screen(self):
        self.save_test_anchors()
        anchor_set = anchors.load_anchors(CALIB_RES)
        blank = np.full((600, 800, 3), 30, dtype=np.uint8)
        self.assertIsNone(anchors.solve(blank, anchor_set, CALIB_RES))

    def test_fail_disabled_below_min_anchors(self):
        self.save_test_anchors(keys=("anchor_a",))
        anchor_set = anchors.load_anchors(CALIB_RES)
        self.assertIsNone(anchors.solve(live_frame(), anchor_set, CALIB_RES))


class Transforms(unittest.TestCase):
    FIT = {"scale": 2.0, "dx": 10.0, "dy": -5.0}

    def test_rect_and_regions(self):
        self.assertEqual(anchors.transform_rect((5, 10, 20, 30), self.FIT),
                         [20, 15, 50, 55])
        payload = {"players": [[[0, 0], [10, 0], [10, 10], [0, 0]]] * 7,
                   "dealer": [5, 10, 20, 30]}
        out = anchors.transform_regions(payload, self.FIT)
        self.assertEqual(out["dealer"], [20, 15, 50, 55])
        self.assertEqual(out["players"][0][1], [30.0, -5.0])
        self.assertIsNone(anchors.transform_regions(None, self.FIT))

    def test_ocr_and_controls(self):
        self.assertEqual(anchors.transform_ocr({"balance": [1, 2, 3, 4]},
                                               self.FIT),
                         {"balance": [12, -1, 16, 3]})
        template = np.zeros((10, 20, 3), dtype=np.uint8)
        out = anchors.transform_controls(
            {"hit": {"rect": (5, 10, 20, 30), "template": template}},
            self.FIT)
        self.assertEqual(out["hit"]["rect"], (20, 15, 50, 55))
        self.assertEqual(out["hit"]["template"].shape, (20, 40, 3))


class Drift(TempStore):
    def test_consistent_screen_is_ok_then_drift_latches(self):
        self.save_test_anchors()
        anchor_set = anchors.load_anchors(CALIB_RES)
        live = live_frame(1.3, 24, 18)
        fit = anchors.solve(live, anchor_set, CALIB_RES)
        check = anchors.check_drift(live, anchor_set, fit)
        self.assertTrue(check["ok"])
        # The casino window moved 60 px after the solve.
        moved = live_frame(1.3, 84, 18)
        check = anchors.check_drift(moved, anchor_set, fit)
        self.assertFalse(check["ok"])


class EngineIntegration(TempStore):
    def test_resolve_remaps_engine_geometry(self):
        from lib.logic.engine import DetectionEngine
        self.save_test_anchors()
        # Calibrate regions + OCR at the calibrated resolution only.
        players = [[[10 + i, 10], [60 + i, 10], [60 + i, 60], [10 + i, 10]]
                   for i in range(7)]
        save_custom_regions(CALIB_RES, players, [100, 20, 200, 80])
        ocr.save_regions(CALIB_RES, {"balance": [50, 50, 150, 80]})

        eng = DetectionEngine(log=lambda *a, **k: None)
        eng.store = None
        eng.set_monitor(types.SimpleNamespace(x=0, y=0, width=1120,
                                              height=880, is_primary=False))
        self.assertTrue(eng._anchor_pending)
        self.assertIsNone(eng._ocr_regions)  # 1120x880 has no calibration

        eng._maybe_anchors(live_frame(1.3, 24, 18, size=(880, 1120)))
        snap = eng.get_snapshot()
        self.assertEqual(snap["anchors"]["status"], "active")
        self.assertAlmostEqual(snap["anchors"]["scale"], 1.3, delta=0.02)
        self.assertEqual(snap["anchors"]["calib_res"], "800x600")
        # OCR rect mapped: 50*1.3+24 = 89 (±rounding of the solved fit).
        balance = eng._ocr_regions["balance"]
        self.assertAlmostEqual(balance[0], 89, delta=4)
        self.assertAlmostEqual(balance[3], 80 * 1.3 + 18, delta=4)
        # Seat polygons mapped too: bounds of seat 0 start near 10*1.3+24.
        left, top, _, _ = eng.regions[0].bounds
        self.assertAlmostEqual(left, 10 * 1.3 + 24, delta=4)

    def test_identity_fit_keeps_exact_saved_geometry(self):
        # Same resolution, nothing moved: the EXACT saved calibration must
        # win over a fitted approximation (rounding jitter shaves crops).
        from lib.logic.engine import DetectionEngine
        self.save_test_anchors()
        players = [[[10, 10], [60, 10], [60, 60], [10, 10]]] * 7
        save_custom_regions(CALIB_RES, players, [100, 20, 200, 80])
        ocr.save_regions(CALIB_RES, {"balance": [50, 50, 150, 80]})
        eng = DetectionEngine(log=lambda *a, **k: None)
        eng.store = None
        eng.set_monitor(types.SimpleNamespace(x=0, y=0, width=800,
                                              height=600, is_primary=False))
        regions_before = eng.regions
        ocr_before = eng._ocr_regions
        eng._maybe_anchors(calib_frame())
        snap = eng.get_snapshot()
        self.assertEqual(snap["anchors"]["status"], "active")
        self.assertIs(eng.regions, regions_before)
        self.assertIs(eng._ocr_regions, ocr_before)

    def test_failed_solve_keeps_geometry_and_says_so(self):
        from lib.logic.engine import DetectionEngine
        self.save_test_anchors()
        eng = DetectionEngine(log=lambda *a, **k: None)
        eng.store = None
        eng.set_monitor(types.SimpleNamespace(x=0, y=0, width=1120,
                                              height=880, is_primary=False))
        before = eng.regions
        blank = np.full((880, 1120, 3), 30, dtype=np.uint8)
        eng._maybe_anchors(blank)
        snap = eng.get_snapshot()
        self.assertEqual(snap["anchors"]["status"], "failed")
        self.assertIs(eng.regions, before)  # fail-disabled: nothing moved


if __name__ == "__main__":
    unittest.main()
