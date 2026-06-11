"""V4 Feature 1: game-phase & turn detection — OCR parsing, the control
template library, template matching, and the state machine.

Run:  .venv\\Scripts\\python -m unittest tests.test_phase -v
"""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from lib.common import constants
from lib.logic import ocr, phase, region_profiles


class ParseTimer(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(ocr.parse_timer("9"), 9)
        self.assertEqual(ocr.parse_timer("0:12"), 12)
        self.assertEqual(ocr.parse_timer("8s"), 8)
        self.assertEqual(ocr.parse_timer(" 14 "), 14)
        self.assertIsNone(ocr.parse_timer(""))
        self.assertIsNone(ocr.parse_timer(None))
        self.assertIsNone(ocr.parse_timer("no digits"))
        self.assertIsNone(ocr.parse_timer("850"))  # > 120 s = misread


class SnapStatus(unittest.TestCase):
    def test_vocabulary(self):
        self.assertEqual(ocr.snap_status("PLACE YOUR BETS"), "betting")
        self.assertEqual(ocr.snap_status("Place  your\nbets"), "betting")
        self.assertEqual(ocr.snap_status("BETS OPEN"), "betting")
        self.assertEqual(ocr.snap_status("NO MORE BETS"), "closed")
        self.assertEqual(ocr.snap_status("Bets closed"), "closed")
        self.assertIsNone(ocr.snap_status("WIN"))
        self.assertIsNone(ocr.snap_status(""))
        self.assertIsNone(ocr.snap_status(None))

    def test_localized_banners(self):
        self.assertEqual(ocr.snap_status("PLAATS UW INZET"), "betting")
        self.assertEqual(ocr.snap_status("Platzieren Sie Ihre Einsätze"),
                         "betting")
        self.assertEqual(ocr.snap_status("RIEN NE VA PLUS"), "closed")
        self.assertEqual(ocr.snap_status("no más apuestas"), "closed")

    def test_interpret_carries_status_and_timer(self):
        values = ocr.interpret({"result": "PLACE YOUR BETS", "timer": "12"})
        self.assertEqual(values["status"], "betting")
        self.assertEqual(values["timer"], 12)
        self.assertIsNone(values["result"])
        self.assertFalse(values["stray_text"])
        # Result banner still wins when it shows a result.
        values = ocr.interpret({"result": "BLACKJACK!"})
        self.assertEqual(values["result"], "blackjack")
        self.assertIsNone(values["status"])

    def test_stray_text_flags_unrecognized_dialogs(self):
        values = ocr.interpret({"result": "Are you still there? Click OK"})
        self.assertTrue(values["stray_text"])
        # Known vocabulary, short text, or no text never count as stray.
        self.assertFalse(ocr.interpret({"result": "WIN"})["stray_text"])
        self.assertFalse(ocr.interpret({"result": "21"})["stray_text"])
        self.assertFalse(ocr.interpret({})["stray_text"])


def _frame(width=640, height=360, color=(20, 80, 40)):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _draw_button(frame, rect, color):
    """A 'button': filled rect + contrasting glyph bar (matchable shape)."""
    left, top, right, bottom = rect
    frame[top:bottom, left:right] = color
    frame[top + 8:bottom - 8, left + 10:left + 20] = (255, 255, 255)


class TemplateLibrary(unittest.TestCase):
    RES = (640, 360)

    def setUp(self):
        self._out = constants.OUTPUT_DIR
        self._controls = constants.CONTROLS_DIR
        tmp = Path(tempfile.mkdtemp())
        constants.OUTPUT_DIR = tmp / "output"
        constants.CONTROLS_DIR = tmp / "controls"

    def tearDown(self):
        constants.OUTPUT_DIR = self._out
        constants.CONTROLS_DIR = self._controls

    def test_save_load_delete_round_trip(self):
        frame = _frame()
        _draw_button(frame, (100, 200, 180, 240), (60, 160, 60))
        crop = frame[200:240, 100:180].copy()
        phase.save_controls(self.RES, {"hit": {"rect": [100, 200, 180, 240],
                                               "crop": crop}})
        loaded = phase.load_controls(self.RES)
        self.assertIn("hit", loaded)
        self.assertEqual(loaded["hit"]["rect"], (100, 200, 180, 240))
        self.assertEqual(loaded["hit"]["template"].shape, crop.shape)
        # Template PNG landed under the profile slug + resolution folder.
        folder = phase.template_dir(region_profiles.active_name(), self.RES)
        self.assertTrue((folder / "hit.png").exists())
        phase.delete_controls(self.RES)
        self.assertEqual(phase.load_controls(self.RES), {})

    def test_profile_slug_is_collision_proof(self):
        self.assertTrue(phase.profile_slug("Table A").startswith("table-a-"))
        # Distinct names that sanitize identically still get distinct
        # folders — otherwise one profile's templates overwrite another's.
        self.assertNotEqual(phase.profile_slug("Table A"),
                            phase.profile_slug("table a"))
        self.assertNotEqual(phase.profile_slug("Table A"),
                            phase.profile_slug("Table-A"))
        self.assertTrue(phase.profile_slug("!!!").startswith("profile-"))

    def test_save_controls_merges_with_earlier_captures(self):
        frame = _frame()
        _draw_button(frame, (100, 200, 180, 240), (60, 160, 60))
        _draw_button(frame, (200, 200, 280, 240), (160, 120, 40))
        phase.save_controls(self.RES, {
            "hit": {"rect": [100, 200, 180, 240],
                    "crop": frame[200:240, 100:180].copy()}})
        # A later session captures only Stand — Hit must survive untouched.
        phase.save_controls(self.RES, {
            "stand": {"rect": [200, 200, 280, 240],
                      "crop": frame[200:240, 200:280].copy()}})
        loaded = phase.load_controls(self.RES)
        self.assertEqual(set(loaded), {"hit", "stand"})
        self.assertIsNotNone(loaded["hit"]["template"])

    def test_save_controls_whitespace_profile_uses_active(self):
        frame = _frame()
        _draw_button(frame, (100, 200, 180, 240), (60, 160, 60))
        phase.save_controls(self.RES, {
            "hit": {"rect": [100, 200, 180, 240],
                    "crop": frame[200:240, 100:180].copy()}}, profile="   ")
        # Rects and PNGs both landed under the ACTIVE profile, not a
        # phantom whitespace one.
        self.assertEqual(region_profiles.active_name(), "Default")
        self.assertIn("hit", phase.load_controls(self.RES))


class Matching(unittest.TestCase):
    def test_locate_enabled_and_greyed(self):
        frame = _frame()
        rect = (300, 200, 380, 240)
        _draw_button(frame, rect, (60, 160, 60))
        template = frame[200:240, 300:380].copy()

        hit = phase.locate_control(frame, rect, template)
        self.assertGreaterEqual(hit["score"], 0.95)
        self.assertLess(hit["color_dist"], 5.0)

        # Greyed: same shape, washed-out color — found but NOT enabled.
        grey = _frame()
        _draw_button(grey, rect, (110, 110, 110))
        hit_grey = phase.locate_control(grey, rect, template)
        self.assertGreaterEqual(hit_grey["score"],
                                constants.PHASE["match_threshold"])
        self.assertGreater(hit_grey["color_dist"],
                           constants.PHASE["enabled_color_dist"])

        # Absent: plain felt — the shape does not match.
        felt = _frame()
        miss = phase.locate_control(felt, rect, template)
        self.assertLess(miss["score"], constants.PHASE["match_threshold"])

    def test_locate_tolerates_small_shift(self):
        frame = _frame()
        _draw_button(frame, (310, 205, 390, 245), (60, 160, 60))
        template_frame = _frame()
        _draw_button(template_frame, (300, 200, 380, 240), (60, 160, 60))
        template = template_frame[200:240, 300:380].copy()
        hit = phase.locate_control(frame, (300, 200, 380, 240), template)
        self.assertGreaterEqual(hit["score"], 0.95)
        cx, cy = hit["center"]
        self.assertAlmostEqual(cx, 350, delta=4)   # shifted +10
        self.assertAlmostEqual(cy, 225, delta=4)   # shifted +5

    def test_patch_changed_detects_chip(self):
        frame = _frame()
        rect = (200, 300, 260, 340)
        template = frame[300:340, 200:260].copy()  # empty spot
        self.assertLess(phase.patch_changed(frame, rect, template), 1.0)
        chip = frame.copy()
        chip[305:335, 210:250] = (40, 40, 200)  # a red chip landed
        self.assertGreater(phase.patch_changed(chip, rect, template),
                           constants.PHASE["enabled_color_dist"] / 2)


class StateMachine(unittest.TestCase):
    RES = (640, 360)

    def setUp(self):
        self._out = constants.OUTPUT_DIR
        self._controls = constants.CONTROLS_DIR
        tmp = Path(tempfile.mkdtemp())
        constants.OUTPUT_DIR = tmp / "output"
        constants.CONTROLS_DIR = tmp / "controls"

        self.hit_rect = (300, 200, 380, 240)
        self.stand_rect = (400, 200, 480, 240)
        enabled = _frame()
        _draw_button(enabled, self.hit_rect, (60, 160, 60))
        _draw_button(enabled, self.stand_rect, (160, 120, 40))
        phase.save_controls(self.RES, {
            "hit": {"rect": list(self.hit_rect),
                    "crop": enabled[200:240, 300:380].copy()},
            "stand": {"rect": list(self.stand_rect),
                      "crop": enabled[200:240, 400:480].copy()},
        })
        self.enabled_frame = enabled
        self.detector = phase.PhaseDetector(log=lambda *a, **k: None)
        self.detector.configure(self.RES)

    def tearDown(self):
        constants.OUTPUT_DIR = self._out
        constants.CONTROLS_DIR = self._controls

    @staticmethod
    def _signals(activity="waiting", cards=0, undecided=(), mine=(),
                 **ocr_vals):
        ocr_dict = {"ts": time.time(), **ocr_vals} if ocr_vals else {"ts": 0.0}
        return {"activity": activity, "cards_on_table": cards,
                "undecided_mine": list(undecided), "my_seats": list(mine),
                "ocr": ocr_dict}

    def test_my_turn_needs_confirm_frames(self):
        felt = _frame()
        state = self.detector.update(felt, self._signals())
        self.assertEqual(state["phase"], phase.IDLE)
        self.assertTrue(state["calibrated"])

        signals = self._signals(activity="complete", cards=5,
                                undecided=[2], mine=[2])
        state = self.detector.update(self.enabled_frame, signals)
        self.assertNotEqual(state["phase"], phase.MY_TURN)  # 1 frame: not yet
        state = self.detector.update(self.enabled_frame, signals)
        self.assertEqual(state["phase"], phase.MY_TURN)
        self.assertEqual(state["my_seat"], 2)
        self.assertTrue(state["seat_confident"])
        self.assertTrue(state["buttons"]["hit"]["enabled"])

    def test_greyed_buttons_are_not_my_turn(self):
        grey = _frame()
        _draw_button(grey, self.hit_rect, (110, 110, 110))
        _draw_button(grey, self.stand_rect, (110, 110, 110))
        signals = self._signals(activity="complete", cards=5)
        for _ in range(3):
            state = self.detector.update(grey, signals)
        self.assertNotEqual(state["phase"], phase.MY_TURN)

    def test_my_turn_has_exit_hysteresis(self):
        signals = self._signals(activity="complete", cards=5,
                                undecided=[2], mine=[2])
        for _ in range(2):
            state = self.detector.update(self.enabled_frame, signals)
        self.assertEqual(state["phase"], phase.MY_TURN)

        # One frame with a hover-brightened Hit button (shape matches,
        # color drifts) must NOT flap the turn away...
        hover = self.enabled_frame.copy()
        _draw_button(hover, self.hit_rect, (140, 230, 140))
        state = self.detector.update(hover, signals)
        self.assertEqual(state["phase"], phase.MY_TURN)
        # ...but a sustained exit (confirm_frames bad frames) does leave.
        state = self.detector.update(hover, signals)
        self.assertNotEqual(state["phase"], phase.MY_TURN)

    def test_stray_center_text_means_unknown(self):
        felt = _frame()
        state = self.detector.update(
            felt, self._signals(activity="complete", cards=5,
                                stray_text=True))
        self.assertEqual(state["phase"], phase.UNKNOWN)

    def test_betting_open_from_banner_and_settle(self):
        felt = _frame()
        signals = self._signals(status="betting", timer=12)
        self.detector.update(felt, signals)
        state = self.detector.update(felt, signals)
        self.assertEqual(state["phase"], phase.BETTING_OPEN)
        self.assertEqual(state["timer_s"], 12)

        state = self.detector.update(
            felt, self._signals(activity="complete", cards=6, result="win"))
        self.assertEqual(state["phase"], phase.SETTLE)

    def test_stale_ocr_is_ignored(self):
        felt = _frame()
        signals = {"activity": "waiting", "cards_on_table": 0,
                   "undecided_mine": [], "my_seats": [],
                   "ocr": {"status": "betting", "ts": time.time() - 60}}
        for _ in range(3):
            state = self.detector.update(felt, signals)
        self.assertEqual(state["phase"], phase.IDLE)

    def test_half_occluded_buttons_mean_unknown(self):
        # Hit visible, Stand covered by a "modal" — the calibrated pair
        # disagreeing is the modal/overlay signal.
        covered = self.enabled_frame.copy()
        covered[195:250, 395:490] = (240, 240, 240)
        signals = self._signals(activity="complete", cards=5)
        state = self.detector.update(covered, signals)
        self.assertEqual(state["phase"], phase.UNKNOWN)

    def test_dealing_and_waiting(self):
        felt = _frame()
        state = self.detector.update(felt, self._signals(activity="dealing",
                                                         cards=3))
        self.assertEqual(state["phase"], phase.DEALING)
        state = self.detector.update(felt, self._signals(activity="complete",
                                                         cards=6))
        self.assertEqual(state["phase"], phase.WAITING)

    def test_seat_attribution_ambiguity(self):
        felt = _frame()
        state = self.detector.update(
            felt, self._signals(cards=4, undecided=[1, 4], mine=[1, 4]))
        self.assertEqual(state["my_seat"], 1)
        self.assertFalse(state["seat_confident"])


class EngineIntegration(unittest.TestCase):
    """Headless engine: snapshot carries the phase block; the discipline
    judge scores observed actions against the advice."""

    def setUp(self):
        self._out = constants.OUTPUT_DIR
        constants.OUTPUT_DIR = Path(tempfile.mkdtemp())
        from lib.logic.engine import DetectionEngine
        self.eng = DetectionEngine(log=lambda *a, **k: None)

    def tearDown(self):
        constants.OUTPUT_DIR = self._out

    def test_snapshot_has_phase_block(self):
        snap = self.eng.get_snapshot()
        self.assertIn("phase", snap)
        self.assertEqual(snap["phase"]["phase"], phase.IDLE)
        self.assertIn("discipline", snap["phase"])

    def test_discipline_judges_stand_vs_hit(self):
        eng = self.eng
        with eng._lock:
            eng.my_seats.add(0)
            eng.seats[0].cards = [
                {"name": "10 of Hearts", "confidence": 1.0, "cx": 1, "cy": 1,
                 "manual": True, "counted": True, "hand": 0},
                {"name": "6 of Clubs", "confidence": 1.0, "cx": 2, "cy": 2,
                 "manual": True, "counted": True, "hand": 0}]
            # Buttons went live with advice = Hit; then the turn ended with
            # no new card -> the user stood against a Hit call.
            eng._my_turn_baseline = {0: {"cards": 2, "split": False,
                                         "optimal": "H"}}
            eng._on_phase_transition(phase.MY_TURN, phase.WAITING, {})
            eng._pending_discipline[0]["deadline"] = 0  # due immediately
            eng._check_discipline()
        self.assertEqual(eng.discipline, {"checked": 1, "matched": 0})

    def test_discipline_matches_hit(self):
        eng = self.eng
        with eng._lock:
            eng.my_seats.add(0)
            eng.seats[0].cards = [
                {"name": "10 of Hearts", "confidence": 1.0, "cx": 1, "cy": 1,
                 "manual": True, "counted": True, "hand": 0}] * 3
            eng._pending_discipline = [{"seat": 0, "cards": 2, "split": False,
                                        "optimal": "H", "round": 1,
                                        "deadline": 0}]
            eng._check_discipline()
        self.assertEqual(eng.discipline, {"checked": 1, "matched": 1})

    def test_discipline_drops_stale_round_entries(self):
        eng = self.eng
        with eng._lock:
            eng._pending_discipline = [{"seat": 0, "cards": 2, "split": False,
                                        "optimal": "H", "round": 99,
                                        "deadline": 0}]
            eng._check_discipline()
        self.assertEqual(eng.discipline, {"checked": 0, "matched": 0})

    def test_my_turn_reentry_cancels_pending_judgments(self):
        eng = self.eng
        with eng._lock:
            eng._pending_discipline = [{"seat": 0, "cards": 2, "split": False,
                                        "optimal": "H", "round": 1,
                                        "deadline": time.monotonic() + 99}]
            eng._on_phase_transition(phase.WAITING, phase.MY_TURN,
                                     {"my_seat": None})
            self.assertEqual(eng._pending_discipline, [])


if __name__ == "__main__":
    unittest.main()
