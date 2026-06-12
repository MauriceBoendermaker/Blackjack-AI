"""Dealer pipeline reliability: the up-card and the playout draws.

Covers the field-reported failure pair: the region preview recognized the
dealer card (player model, full frame) while the live app rarely updated
it (a different model on a small crop + a confirmation streak that any
single occluded frame reset), and the dealer's flipped second card /
playout draws never registered (swallowed by a too-wide 'flicker' radius
around the up-card, missed entirely when the whole-frame diff skipped the
small dealer-area change).

Run:  .venv\\Scripts\\python -m unittest tests.test_dealer_detection -v
"""

import types
import unittest

import numpy as np

from lib.common import constants
from lib.common.card_mappings import CUTTING_CARD_CLASS
from lib.logic.engine import DetectionEngine


def player_pred(cls="a13", cx=300.0, cy=150.0, conf=0.9):
    """Full-frame player-model prediction ('a13' = King of Hearts)."""
    return {"class": cls, "confidence": conf, "cx": cx, "cy": cy}


class _FakeModel:
    def __init__(self, preds=()):
        self.preds = list(preds)
        self.calls = 0

    def predict(self, *a, **k):
        self.calls += 1
        return list(self.preds)


def engine_with_models(player_preds=(), rank_preds=()):
    eng = DetectionEngine(log=lambda *a, **k: None)
    eng.store = None
    eng.regions = []  # no seats: nothing in the dealer area double-routes
    eng._dealer_rect = (0, 0, 600, 300)
    players = _FakeModel(player_preds)
    rank = _FakeModel(rank_preds)
    eng.provider = types.SimpleNamespace(players_model=lambda: players,
                                         dealer_model=lambda: rank)
    return eng, players, rank


class UnifiedDealerSource(unittest.TestCase):
    """Dealer cards come from the SAME full-frame player-model pass the
    preview draws — in both legacy modes of the old toggle."""

    def _run(self, use_player_model):
        old = constants.DEALER_USE_PLAYER_MODEL
        constants.DEALER_USE_PLAYER_MODEL = use_player_model
        try:
            eng, players, rank = engine_with_models(
                player_preds=[player_pred(cx=300, cy=150)])
            frame = np.zeros((400, 700, 3), dtype=np.uint8)
            for _ in range(constants.DEALER_CONFIRM_FRAMES):
                eng._detect(frame)
            self.assertEqual(eng.dealer_card, "King of Hearts")
            self.assertTrue(eng.dealer_locked)
            # The rank model is for the cutting card only — polled every
            # Nth cycle, not run per frame.
            self.assertLessEqual(rank.calls,
                                 1 + constants.DEALER_CONFIRM_FRAMES
                                 // constants.CUTTING_CARD_CHECK_EVERY)
        finally:
            constants.DEALER_USE_PLAYER_MODEL = old

    def test_suit_mode(self):
        self._run(True)

    def test_legacy_rank_mode_uses_the_same_path(self):
        self._run(False)

    def test_outside_the_dealer_rect_is_not_the_dealer(self):
        eng, players, _ = engine_with_models(
            player_preds=[player_pred(cx=300, cy=350)])  # below the rect
        frame = np.zeros((400, 700, 3), dtype=np.uint8)
        for _ in range(constants.DEALER_CONFIRM_FRAMES + 1):
            eng._detect(frame)
        self.assertIsNone(eng.dealer_card)

    def test_draw_fanning_outside_a_tight_rect_still_tracked(self):
        # The reported failure: a right-edge-pinned dealer rect (like the
        # user's [1955,0,2560,554]). The up-card locks inside it; a playout
        # card fans LEFT, outside [left,right] but in the dealer's vertical
        # band — it must still be counted (the preview shows it; the old
        # tight-rect filter dropped it).
        eng, players, _ = engine_with_models()
        eng._dealer_rect = (1955, 0, 2560, 554)
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        players.preds = [player_pred(cx=2100, cy=200)]  # King of Hearts up
        for _ in range(constants.DEALER_CONFIRM_FRAMES):
            eng._detect(frame)
        self.assertEqual(eng.dealer_card, "King of Hearts")
        # Draw fans left to x=1700, well outside [1955, 2560].
        players.preds = [player_pred(cx=2100, cy=200),
                         {"class": "d7", "confidence": 0.9,
                          "cx": 1700.0, "cy": 210.0}]
        for _ in range(constants.EXTRA_CARD_CONFIRM_CYCLES):
            eng._detect(frame)
        self.assertEqual([c["rank"] for c in eng.dealer_extras],
                         ["7 of Clubs"])

    def test_seat_card_in_the_dealer_band_is_not_a_dealer_card(self):
        # A card inside a seat polygon is never routed to the dealer, even
        # if it falls within the (widened) dealer horizontal/vertical band.
        from lib.logic.monitor_utils import Polygon
        eng, players, _ = engine_with_models()
        eng._dealer_rect = (1955, 0, 2560, 554)
        eng.regions = [Polygon([[1600, 100], [1800, 100],
                                [1800, 300], [1600, 300]])]  # seat in the band
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        players.preds = [player_pred(cx=2100, cy=200)]  # up-card
        for _ in range(constants.DEALER_CONFIRM_FRAMES):
            eng._detect(frame)
        # A card landing inside that seat polygon must not become a draw.
        players.preds = [player_pred(cx=2100, cy=200),
                         {"class": "d7", "confidence": 0.9,
                          "cx": 1700.0, "cy": 200.0}]  # inside the seat poly
        for _ in range(constants.EXTRA_CARD_CONFIRM_CYCLES + 1):
            eng._detect(frame)
        self.assertEqual(eng.dealer_extras, [])

    def test_cutting_card_coords_shift_to_frame_space(self):
        eng, players, rank = engine_with_models(
            rank_preds=[{"class": CUTTING_CARD_CLASS, "confidence": 0.9,
                         "cx": 50.0, "cy": 40.0}])
        eng._dealer_rect = (100, 80, 600, 300)
        frame = np.zeros((400, 700, 3), dtype=np.uint8)
        for _ in range(constants.CUTTING_CARD_CHECK_EVERY
                       * constants.CUTTING_CARD_CONFIRM_FRAMES):
            eng._detect(frame)
        self.assertTrue(eng.cutting_card_seen)


class UpCardOcclusionTolerance(unittest.TestCase):
    """The dealer's hands crossing the cards must not reset the
    confirmation streak — only a sustained empty run does."""

    def setUp(self):
        self.eng = DetectionEngine(log=lambda *a, **k: None)
        self.eng.store = None

    def sight(self, rank="K"):
        with self.eng._lock:
            self.eng._process_dealer([{"class": rank, "confidence": 0.8,
                                       "cx": 500.0, "cy": 200.0}])

    def miss(self, n=1):
        for _ in range(n):
            with self.eng._lock:
                self.eng._process_dealer([])

    def test_single_miss_does_not_reset_the_streak(self):
        self.sight()
        self.miss()  # occluded frame
        self.sight()
        self.assertEqual(self.eng.dealer_card, "King")

    def test_sustained_gap_still_resets(self):
        self.sight()
        self.miss(constants.DEALER_PENDING_MISS_TOLERANCE + 1)
        self.sight()
        self.assertIsNone(self.eng.dealer_card)  # streak was cleared
        self.sight()
        self.assertEqual(self.eng.dealer_card, "King")

    def test_disagreeing_sightings_still_blocked(self):
        self.sight("K")
        self.sight("Q")
        self.assertIsNone(self.eng.dealer_card)


class PlayoutFan(unittest.TestCase):
    """The dealer's 2nd/3rd/4th cards land fanned right next to the
    up-card and must register as draws, not be swallowed as flicker."""

    def setUp(self):
        self.eng = DetectionEngine(log=lambda *a, **k: None)
        self.eng.store = None
        for _ in range(constants.DEALER_CONFIRM_FRAMES):
            with self.eng._lock:
                self.eng._process_dealer([{"class": "K", "confidence": 0.8,
                                           "cx": 500.0, "cy": 200.0}])
        self.assertTrue(self.eng.dealer_locked)

    def playout_frame(self, preds):
        with self.eng._lock:
            self.eng._process_dealer(
                [{"class": "K", "confidence": 0.8, "cx": 500.0, "cy": 200.0}]
                + preds)

    def test_adjacent_fanned_draw_registers(self):
        draw = {"class": "7", "confidence": 0.8, "cx": 530.0, "cy": 205.0}
        for _ in range(constants.EXTRA_CARD_CONFIRM_CYCLES):
            self.playout_frame([dict(draw)])
        self.assertEqual([c["rank"] for c in self.eng.dealer_extras], ["7"])

    def test_misread_on_the_up_card_is_still_flicker(self):
        # Different class within a few px of the up-card = misread.
        ghost = {"class": "Q", "confidence": 0.8, "cx": 505.0, "cy": 202.0}
        for _ in range(constants.EXTRA_CARD_CONFIRM_CYCLES + 2):
            self.playout_frame([dict(ghost)])
        self.assertEqual(self.eng.dealer_extras, [])

    def test_box_wobble_across_a_cell_boundary_still_confirms(self):
        # 50px-quantization boundary sits at 25/75: cx 222<->228 flips the
        # cell every frame — the hits must still accumulate as ONE card.
        for cx in (222.0, 228.0):
            self.playout_frame([{"class": "5", "confidence": 0.8,
                                 "cx": cx, "cy": 240.0}])
        self.assertEqual([c["rank"] for c in self.eng.dealer_extras], ["5"])

    def test_draws_keep_arriving(self):
        for rank, cx in (("7", 530.0), ("9", 560.0), ("4", 590.0)):
            draw = {"class": rank, "confidence": 0.8, "cx": cx, "cy": 205.0}
            for _ in range(constants.EXTRA_CARD_CONFIRM_CYCLES):
                self.playout_frame([dict(draw)])
        self.assertEqual([c["rank"] for c in self.eng.dealer_extras],
                         ["7", "9", "4"])


class DealerDrawingState(unittest.TestCase):
    """_dealer_drawing gates both the fast pacing and the frame-skip bypass."""

    def setUp(self):
        self.eng = DetectionEngine(log=lambda *a, **k: None)
        self.eng.store = None
        self.eng.replace_card(0, 0, "10 of Hearts")
        self.eng.replace_card(0, 1, "9 of Spades")
        self.eng.replace_card(1, 0, "9 of Clubs")
        self.eng.replace_card(1, 1, "7 of Diamonds")

    def test_true_while_hand_not_final(self):
        self.eng.replace_dealer("6")  # total 6, must draw
        self.assertTrue(self.eng._dealer_drawing())

    def test_false_when_no_dealer_locked(self):
        self.assertFalse(self.eng._dealer_drawing())  # no up-card yet

    def test_false_once_hand_is_final(self):
        self.eng.replace_dealer("10")
        self.eng.add_dealer_extra("7 of Clubs")  # 17 — dealer stands
        self.assertFalse(self.eng._dealer_drawing())

    def test_false_on_all_bust(self):
        eng = DetectionEngine(log=lambda *a, **k: None)
        eng.store = None
        eng.replace_card(0, 0, "10 of Hearts")
        eng.replace_card(0, 1, "6 of Spades")
        eng.replace_card(0, 2, "King of Clubs")  # 26, bust
        eng.replace_dealer("6")
        self.assertFalse(eng._dealer_drawing())  # dealer never plays out


class PlayoutBypassesFrameSkip(unittest.TestCase):
    """The core fix: during the dealer's playout, a card that lands and then
    sits static must still confirm — the frame-skip is bypassed so the
    confirmation sightings accumulate."""

    def setUp(self):
        self._phase = constants.PHASE["enabled"]
        constants.PHASE["enabled"] = 0
        self.eng = DetectionEngine(log=lambda *a, **k: None)
        self.eng.store = None
        self.eng.regions = []  # seats come from replace_card, not the model
        self.eng._dealer_rect = (400, 100, 600, 300)
        for seat, idx, name in ((0, 0, "10 of Hearts"), (0, 1, "9 of Spades"),
                                (1, 0, "9 of Clubs"), (1, 1, "7 of Diamonds")):
            self.eng.replace_card(seat, idx, name)
        self.eng.replace_dealer("6")  # total 6 — the dealer must draw
        self.frame = np.full((400, 700, 3), 40, dtype=np.uint8)
        self.eng.capture = types.SimpleNamespace(
            grab_bgr=lambda: self.frame, resolution=(700, 400), monitor=None,
            close_local=lambda: None)

    def tearDown(self):
        constants.PHASE["enabled"] = self._phase

    def _set_model(self, preds):
        players = _FakeModel(preds)
        rank = _FakeModel([])
        self.eng.provider = types.SimpleNamespace(
            players_model=lambda: players, dealer_model=lambda: rank,
            backend_name="fake")
        return players

    def test_static_playout_card_still_confirms(self):
        # The dealer's hole/draw card is detected in the dealer area on a
        # frame that never changes after it lands.
        self._set_model([{"class": "d7", "confidence": 0.9,
                          "cx": 500.0, "cy": 200.0}])
        for _ in range(constants.EXTRA_CARD_CONFIRM_CYCLES + 1):
            self.eng.run_cycle()
        self.assertEqual([c["rank"] for c in self.eng.dealer_extras],
                         ["7 of Clubs"])
        # Run-cycle reports the fast 'dealing' cadence during playout.
        self.assertEqual(self.eng.run_cycle(), "dealing")

    def test_idle_static_frame_still_skips(self):
        # Control: with nothing happening, the skip optimization holds.
        eng = DetectionEngine(log=lambda *a, **k: None)
        eng.store = None
        eng.regions = []
        eng._dealer_rect = (0, 0, 200, 100)
        old = constants.PHASE["enabled"]
        constants.PHASE["enabled"] = 0
        players = _FakeModel([])
        eng.provider = types.SimpleNamespace(
            players_model=lambda: players, dealer_model=lambda: _FakeModel([]),
            backend_name="fake")
        frame = np.full((400, 700, 3), 40, dtype=np.uint8)
        eng.capture = types.SimpleNamespace(
            grab_bgr=lambda: frame, resolution=(700, 400),
            close_local=lambda: None, monitor=None)
        try:
            eng.run_cycle()  # first: prev is None -> detects
            eng.run_cycle()  # static, not drawing -> skipped
            eng.run_cycle()
            self.assertEqual(players.calls, 1)
        finally:
            constants.PHASE["enabled"] = old


class DealerAreaFrameDiff(unittest.TestCase):
    """A card flip changes only the dealer area; the whole-frame diff is
    blind to it and must not skip the detect cycle."""

    def setUp(self):
        self.eng = DetectionEngine(log=lambda *a, **k: None)
        self.eng.store = None
        self.eng._dealer_rect = (0, 0, 200, 100)

    def frame(self, dealer_card=False):
        frame = np.full((400, 700, 3), 40, dtype=np.uint8)
        if dealer_card:
            frame[30:80, 60:100] = 255  # a card flips in the dealer area
        return frame

    def test_identical_frames_still_skip(self):
        self.assertFalse(self.eng._frame_unchanged(self.frame()))  # first
        self.assertTrue(self.eng._frame_unchanged(self.frame()))

    def test_dealer_area_change_defeats_the_skip(self):
        self.eng._frame_unchanged(self.frame())
        self.assertFalse(self.eng._frame_unchanged(self.frame(dealer_card=True)))
        # And it settles again once the dealer area is static.
        self.assertTrue(self.eng._frame_unchanged(self.frame(dealer_card=True)))


if __name__ == "__main__":
    unittest.main()
