"""V2 Feature 6: training-data capture, ONNX helpers, dealer-suit flow.

Run:  .venv\\Scripts\\python -m unittest tests.test_training -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lib.common import constants
from lib.logic.models import letterbox, nms
from lib.logic.training_data import CLASS_NAMES, TrainingDataCollector


def frame(w=1920, h=1080):
    return np.full((h, w, 3), 60, dtype=np.uint8)


class Collector(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.col = TrainingDataCollector(self.dir)
        self._training = dict(constants.TRAINING)
        constants.TRAINING["enabled"] = 1
        constants.TRAINING["max_files"] = 5000

    def tearDown(self):
        constants.TRAINING.update(self._training)

    def test_save_sample_writes_image_label_meta(self):
        ok = self.col.save_sample(frame(), 500.0, 400.0, "King of Hearts",
                                  "correction")
        self.assertTrue(ok)
        jpgs = list(self.dir.glob("correction_*King-of-Hearts.jpg"))
        self.assertEqual(len(jpgs), 1)
        label = jpgs[0].with_suffix(".txt").read_text().split()
        self.assertEqual(len(label), 5)
        self.assertEqual(int(label[0]), CLASS_NAMES.index("King of Hearts"))
        for v in label[1:]:
            self.assertTrue(0.0 <= float(v) <= 1.0)
        meta = json.loads(jpgs[0].with_suffix(".json").read_text())
        self.assertEqual(meta["kind"], "correction")
        self.assertTrue((self.dir / "classes.txt").exists())

    def test_rank_only_label_skips_yolo_line(self):
        self.col.save_sample(frame(), 500.0, 400.0, "King", "lowconf")
        jpgs = list(self.dir.glob("lowconf_*.jpg"))
        self.assertEqual(len(jpgs), 1)
        self.assertFalse(jpgs[0].with_suffix(".txt").exists())  # no suit, no class
        self.assertTrue(jpgs[0].with_suffix(".json").exists())

    def test_edge_of_frame_and_disabled(self):
        # Near-edge centers still save (the crop clamps to the frame)...
        self.assertTrue(self.col.save_sample(frame(), 2.0, 2.0, "2 of Clubs",
                                             "confirmed"))
        # ...but a frame smaller than a usable crop is refused.
        self.assertFalse(self.col.save_sample(frame(30, 30), 15.0, 15.0,
                                              "2 of Clubs", "confirmed"))
        constants.TRAINING["enabled"] = 0
        self.assertFalse(self.col.save_sample(frame(), 500, 400, "2 of Clubs",
                                              "confirmed"))

    def test_max_files_cap(self):
        constants.TRAINING["max_files"] = 2
        self.assertTrue(self.col.save_sample(frame(), 500, 400, "2 of Clubs", "confirmed"))
        self.assertTrue(self.col.save_sample(frame(), 600, 400, "3 of Clubs", "confirmed"))
        self.assertFalse(self.col.save_sample(frame(), 700, 400, "4 of Clubs", "confirmed"))
        self.assertEqual(self.col.stats()["total"], 2)

    def test_stats_by_kind(self):
        self.col.save_sample(frame(), 500, 400, "2 of Clubs", "confirmed")
        self.col.save_sample(frame(), 600, 400, "3 of Clubs", "correction")
        stats = self.col.stats()
        self.assertEqual(stats["confirmed"], 1)
        self.assertEqual(stats["correction"], 1)


class OnnxHelpers(unittest.TestCase):
    def test_letterbox_shapes_and_mapping(self):
        img = frame(1280, 720)
        padded, ratio, (px, py) = letterbox(img, 640)
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertAlmostEqual(ratio, 0.5)
        self.assertEqual(px, 0)
        self.assertEqual(py, 140)  # (640 - 360) / 2
        # A point at frame center maps to letterbox center.
        self.assertAlmostEqual(1280 / 2 * ratio + px, 320)
        self.assertAlmostEqual(720 / 2 * ratio + py, 320)

    def test_nms_suppresses_overlaps(self):
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 300, 300]],
                         dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, iou_threshold=0.5)
        self.assertEqual(keep, [0, 2])  # near-duplicate dropped
        self.assertEqual(nms(np.empty((0, 4)), np.empty(0), 0.5), [])


class DealerSuitFlow(unittest.TestCase):
    def test_player_model_classes_give_dealer_suits(self):
        from lib.logic.engine import DetectionEngine
        eng = DetectionEngine(log=lambda *a, **k: None)
        eng.store = None
        pred = {"class": "c13", "confidence": 0.9, "cx": 100.0, "cy": 100.0}
        with eng._lock:
            eng._process_dealer([pred])   # consensus frame 1
            eng._process_dealer([pred])   # frame 2 -> lock
        self.assertEqual(eng.dealer_card, "King of Spades")
        self.assertEqual(eng.counter.suit_seen.get("King of Spades"), 1)
        # Rank-only detections of the same card still match the up-card.
        with eng._lock:
            self.assertTrue(eng._matches_dealer_card(
                {"rank": "King", "cx": 100.0, "cy": 100.0}, 80.0))

    def test_manual_dealer_correction_keeps_suit(self):
        from lib.logic.engine import DetectionEngine
        eng = DetectionEngine(log=lambda *a, **k: None)
        eng.store = None
        eng.replace_dealer("Queen of Hearts")
        self.assertEqual(eng.dealer_card, "Queen of Hearts")
        self.assertEqual(eng.counter.suit_seen.get("Queen of Hearts"), 1)
        eng.replace_dealer("King")  # rank-only still accepted
        self.assertEqual(eng.dealer_card, "King")


if __name__ == "__main__":
    unittest.main()
