"""V4: Claude vision assist — sizing math, gating, and coordinate rescale.

No network: the API call is mocked; available() is exercised through the
key/enabled gates only.

Run:  .venv\\Scripts\\python -m unittest tests.test_vision_assist -v
"""

import json
import os
import unittest
from unittest import mock

import numpy as np

from lib.common import constants
from lib.logic import vision_assist


class ResizedSize(unittest.TestCase):
    def test_docs_golden(self):
        # The A4-page example from the vision docs (default limits).
        self.assertEqual(vision_assist.resized_size(1075, 1520), (924, 1307))

    def test_fits_unchanged(self):
        self.assertEqual(vision_assist.resized_size(800, 600), (800, 600))
        # 2560x1440 costs exactly 4784 visual tokens — fits the hi-res tier
        # natively, so calibration coords map 1:1 on the game monitor.
        self.assertEqual(
            vision_assist.resized_size(2560, 1440, 2576, 4784), (2560, 1440))

    def test_token_count(self):
        self.assertEqual(vision_assist.count_image_tokens(200, 200), 64)
        self.assertEqual(vision_assist.count_image_tokens(2560, 1440), 4784)


class Gating(unittest.TestCase):
    def setUp(self):
        self._vision = dict(constants.VISION)
        self._env = os.environ.pop("ANTHROPIC_API_KEY", None)

    def tearDown(self):
        constants.VISION.clear()
        constants.VISION.update(self._vision)
        if self._env is not None:
            os.environ["ANTHROPIC_API_KEY"] = self._env

    def test_disabled_without_key_or_toggle(self):
        constants.VISION["enabled"] = 1
        constants.VISION["api_key"] = ""
        self.assertFalse(vision_assist.available())
        constants.VISION["api_key"] = "sk-test"
        self.assertTrue(vision_assist.available())
        constants.VISION["enabled"] = 0
        self.assertFalse(vision_assist.available())

    def test_env_key_overrides(self):
        constants.VISION["enabled"] = 1
        constants.VISION["api_key"] = ""
        os.environ["ANTHROPIC_API_KEY"] = "sk-env"
        try:
            self.assertTrue(vision_assist.available())
        finally:
            del os.environ["ANTHROPIC_API_KEY"]


class Bootstrap(unittest.TestCase):
    def setUp(self):
        self._vision = dict(constants.VISION)

    def tearDown(self):
        constants.VISION.clear()
        constants.VISION.update(self._vision)

    def test_rescales_to_native_and_routes_keys(self):
        # Force the low-res tier so a real rescale happens.
        constants.VISION["model"] = "claude-haiku-4-5"
        w, h = 2560, 1440
        rw, rh = vision_assist.resized_size(w, h, 1568, 1568)
        self.assertLess(rw, w)
        reply = json.dumps({
            "balance": [10, 20, 110, 40],
            "hit": [200, 300, 280, 340],
            "bogus": [1, 2, 3, 4],          # unknown key: dropped
            "split": None,                   # not visible: dropped
            "timer": [5, 5, 6, 6],           # degenerate sliver: dropped
        })
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        with mock.patch.object(vision_assist, "_post", return_value=reply):
            out = vision_assist.bootstrap(frame)
        sx, sy = w / rw, h / rh
        self.assertEqual(out["ocr"]["balance"],
                         [int(round(10 * sx)), int(round(20 * sy)),
                          int(round(110 * sx)), int(round(40 * sy))])
        self.assertEqual(out["controls"]["hit"][0], int(round(200 * sx)))
        self.assertNotIn("bogus", out["controls"])
        self.assertNotIn("split", out["controls"])
        self.assertNotIn("timer", out["ocr"])

    def test_json_block_tolerates_fences(self):
        text = "Here you go:\n```json\n{\"label\": \"lobby\", \"detail\": \"x\"}\n```"
        self.assertEqual(vision_assist._json_block(text)["label"], "lobby")

    def test_triage_formats_label(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        reply = json.dumps({"label": "modal_dialog",
                            "detail": "an inactivity dialog covers the table"})
        with mock.patch.object(vision_assist, "_post", return_value=reply):
            label = vision_assist.triage(frame)
        self.assertEqual(
            label, "modal_dialog — an inactivity dialog covers the table")


if __name__ == "__main__":
    unittest.main()
