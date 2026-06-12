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
