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


class ApplyOcrValues(unittest.TestCase):
    """The engine's sync of interpreted OCR readings into bankroll /
    bet_placed. A genuine 0 must apply (it's the screen's truth); a
    failed read (None) must not."""

    def setUp(self):
        from lib.logic.engine import DetectionEngine
        self._betting = dict(constants.BETTING)
        self._ocr = dict(constants.OCR)
        constants.OCR.update(enabled=1, sync_bankroll=1, sync_bet=1)
        constants.BETTING["bankroll"] = 975.0
        self.eng = DetectionEngine(log=lambda *a, **k: None)
        self.eng.store = None
        self.eng.bet_placed = 5.0

    def tearDown(self):
        constants.BETTING.clear()
        constants.BETTING.update(self._betting)
        constants.OCR.clear()
        constants.OCR.update(self._ocr)

    def _apply(self, **vals):
        base = {"balance": None, "bet": None, "result": None,
                "status": None, "timer": None, "stray_text": False}
        self.eng._apply_ocr_values({**base, **vals})

    def test_zero_balance_and_bet_sync(self):
        # The reported bug: screen shows 0/0, app keeps stale 975/5.
        self._apply(balance=0.0, bet=0.0)
        self.assertEqual(constants.BETTING["bankroll"], 0.0)
        self.assertEqual(self.eng.bet_placed, 0.0)

    def test_failed_read_does_not_zero_state(self):
        self._apply(balance=None, bet=None)
        self.assertEqual(constants.BETTING["bankroll"], 975.0)
        self.assertEqual(self.eng.bet_placed, 5.0)

    def test_nonzero_still_syncs(self):
        self._apply(balance=1200.0, bet=40.0)
        self.assertEqual(constants.BETTING["bankroll"], 1200.0)
        self.assertEqual(self.eng.bet_placed, 40.0)

    def test_sync_toggles_respected(self):
        constants.OCR["sync_bankroll"] = 0
        constants.OCR["sync_bet"] = 0
        self._apply(balance=0.0, bet=0.0)
        self.assertEqual(constants.BETTING["bankroll"], 975.0)
        self.assertEqual(self.eng.bet_placed, 5.0)


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
