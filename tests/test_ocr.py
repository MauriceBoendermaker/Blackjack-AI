"""V2 Feature 5: casino-UI OCR — parsing, regions, and a live engine check.

Run:  .venv\\Scripts\\python -m unittest tests.test_ocr -v
"""

import tempfile
import unittest
from pathlib import Path

from lib.common import constants
from lib.logic import ocr
from lib.logic.ocr import parse_amount, snap_result


class ParseAmount(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_amount("? 1,234.56"), 1234.56)   # € OCRs as '?'
        self.assertEqual(parse_amount("€1.234,56"), 1234.56)    # EU style
        self.assertEqual(parse_amount("Balance: 50"), 50.0)
        self.assertEqual(parse_amount("12,50"), 12.50)          # comma decimal
        self.assertEqual(parse_amount("1.000"), 1000.0)         # thousands dot
        self.assertEqual(parse_amount("987.5"), 987.5)
        self.assertEqual(parse_amount("12,000"), 12000.0)       # thousands comma
        self.assertIsNone(parse_amount(""))
        self.assertIsNone(parse_amount(None))
        self.assertIsNone(parse_amount("no numbers here"))


class SnapResult(unittest.TestCase):
    def test_vocabulary(self):
        self.assertEqual(snap_result("WIN"), "win")
        self.assertEqual(snap_result("You won €25"), "win")
        self.assertEqual(snap_result("LOSS"), "lose")
        self.assertEqual(snap_result("Bust!"), "lose")
        self.assertEqual(snap_result(" push "), "push")
        self.assertEqual(snap_result("BLACKJACK!"), "blackjack")
        self.assertIsNone(snap_result("Place your bets"))
        self.assertIsNone(snap_result(""))


class RegionProfiles(unittest.TestCase):
    RES = (1920, 1080)

    def setUp(self):
        self._out = constants.OUTPUT_DIR
        constants.OUTPUT_DIR = Path(tempfile.mkdtemp())

    def tearDown(self):
        constants.OUTPUT_DIR = self._out

    def test_round_trip_and_validation(self):
        self.assertIsNone(ocr.load_regions(self.RES))
        ocr.save_regions(self.RES, {"balance": [10, 20, 200, 60],
                                    "result": [50, 50, 40, 60]})  # invalid rect
        loaded = ocr.load_regions(self.RES)
        self.assertEqual(loaded, {"balance": [10, 20, 200, 60]})  # bad one dropped
        ocr.delete_regions(self.RES)
        self.assertIsNone(ocr.load_regions(self.RES))


@unittest.skipUnless(ocr.OCR_AVAILABLE, "winocr not installed")
class LiveOcr(unittest.TestCase):
    def test_reads_rendered_ui_text(self):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (560, 160), "#0b2818")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("segoeui.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
        draw.text((20, 15), "€ 1,234.56", fill="white", font=font)
        draw.text((20, 60), "25", fill="white", font=font)
        draw.text((20, 105), "WIN", fill="#ffd700", font=font)
        frame = np.asarray(img)[:, :, ::-1].copy()  # RGB -> BGR

        regions = {"balance": [0, 0, 560, 52], "bet": [0, 50, 560, 100],
                   "result": [0, 98, 560, 160]}
        values = ocr.interpret(ocr.read_regions(frame, regions))
        self.assertEqual(values["balance"], 1234.56)
        self.assertEqual(values["bet"], 25.0)
        self.assertEqual(values["result"], "win")


if __name__ == "__main__":
    unittest.main()
