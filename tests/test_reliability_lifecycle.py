"""Reliability pack (Core-1): round-end gate, cutting-card confirmation,
and the dismissible reshuffle badge.

Run:  .venv\\Scripts\\python -m unittest tests.test_reliability_lifecycle -v
"""

import types
import unittest

import numpy as np

from lib.common import constants
from lib.common.card_mappings import CUTTING_CARD_CLASS
from lib.logic.engine import DetectionEngine


def cc_pred(cx=400.0, cy=300.0):
    return {"class": CUTTING_CARD_CLASS, "confidence": 0.9, "cx": cx, "cy": cy}


def dealer_pred(rank="K", cx=500.0, cy=200.0):
    return {"class": rank, "confidence": 0.8, "cx": cx, "cy": cy}


class EngineCase(unittest.TestCase):
    """Headless engine with a recording logger; no DB writes."""

    def setUp(self):
        self.entries = []  # (message, level) tuples
        self.engine = DetectionEngine(
            log=lambda msg, level=None: self.entries.append((msg, level)))
        self.engine.store = None

    def logged(self, fragment, level=None):
        return [(m, lv) for m, lv in self.entries
                if fragment in m and (level is None or lv == level)]

    def complete_round(self):
        """Two seats with two cards each plus a locked dealer up-card."""
        eng = self.engine
        eng.replace_card(0, 0, "10 of Hearts")
        eng.replace_card(0, 1, "King of Spades")
        eng.replace_card(1, 0, "9 of Clubs")
        eng.replace_card(1, 1, "7 of Diamonds")
        eng.replace_dealer("King")
        self.assertEqual(eng._activity(), "complete")

    def quiet_frame(self):
        """One detection cycle where both models return nothing."""
        eng = self.engine
        with eng._lock:
            eng._process_dealer([])
            eng._process_players([])
            eng._maybe_auto_new_round([])

    def latch_cutting_card(self):
        for _ in range(constants.CUTTING_CARD_CONFIRM_FRAMES):
            with self.engine._lock:
                self.engine._process_dealer([cc_pred()])
        self.assertTrue(self.engine.cutting_card_seen)


class RoundEndGate(EngineCase):
    def test_needs_five_consecutive_empty_frames(self):
        self.complete_round()
        start = self.engine.round_number
        for _ in range(constants.EMPTY_FRAMES_FOR_RESET - 1):
            self.quiet_frame()
        self.assertEqual(self.engine.round_number, start)
        self.assertTrue(self.engine.seats[0].cards)
        self.quiet_frame()
        self.assertEqual(self.engine.round_number, start + 1)
        self.assertFalse(self.engine.seats[0].cards)
        self.assertEqual(self.logged("delayed"), [])

    def test_reset_log_is_warning_with_round_context(self):
        self.complete_round()
        start = self.engine.round_number
        for _ in range(constants.EMPTY_FRAMES_FOR_RESET):
            self.quiet_frame()
        warns = self.logged("Table cleared", level="WARNING")
        self.assertEqual(len(warns), 1)
        msg = warns[0][0]
        self.assertIn(f"round {start + 1}", msg)   # round numbers
        self.assertIn("P1", msg)                   # seats that had cards
        self.assertIn("King", msg)                 # dealer state

    def test_hiccup_shorter_than_window_does_not_reset(self):
        self.complete_round()
        start = self.engine.round_number
        for _ in range(constants.EMPTY_FRAMES_FOR_RESET - 1):
            self.quiet_frame()
        with self.engine._lock:  # the stream recovers: a card is seen again
            self.engine._process_dealer([])
            self.engine._maybe_auto_new_round(
                [{"class": "a1", "confidence": 0.9, "cx": 1.0, "cy": 1.0}])
        for _ in range(constants.EMPTY_FRAMES_FOR_RESET - 1):
            self.quiet_frame()
        self.assertEqual(self.engine.round_number, start)  # run restarted
        self.quiet_frame()
        self.assertEqual(self.engine.round_number, start + 1)

    def test_occupied_dealer_area_blocks_and_logs_once(self):
        self.complete_round()
        start = self.engine.round_number
        eng = self.engine
        # Player model goes quiet but the dealer rank model still sees the
        # up-card: the gate must hold the round and say so once.
        for _ in range(constants.EMPTY_FRAMES_FOR_RESET + 2):
            with eng._lock:
                eng._process_dealer([dealer_pred("K")])
                eng._process_players([])
                eng._maybe_auto_new_round([])
        self.assertEqual(eng.round_number, start)
        self.assertTrue(eng.seats[0].cards)
        delays = self.logged("delayed")
        self.assertEqual(len(delays), 1)
        self.assertIn(delays[0][1], (None, "INFO"))
        # Once the dealer area clears for a full window, the reset fires.
        for _ in range(constants.EMPTY_FRAMES_FOR_RESET):
            self.quiet_frame()
        self.assertEqual(eng.round_number, start + 1)
        self.assertEqual(len(self.logged("delayed")), 1)  # still only once

    def test_manual_new_round_stays_info(self):
        self.complete_round()
        self.engine.new_round()
        resets = self.logged("Round reset")
        self.assertEqual(len(resets), 1)
        self.assertIn(resets[0][1], (None, "INFO"))


class CuttingCardConfirmation(EngineCase):
    def test_single_sighting_does_not_latch(self):
        with self.engine._lock:
            self.engine._process_dealer([cc_pred()])
        self.assertFalse(self.engine.cutting_card_seen)

    def test_three_sightings_latch_and_warn(self):
        eng = self.engine
        for _ in range(constants.CUTTING_CARD_CONFIRM_FRAMES - 1):
            with eng._lock:
                eng._process_dealer([cc_pred()])
            self.assertFalse(eng.cutting_card_seen)
        with eng._lock:
            eng._process_dealer([cc_pred()])
        self.assertTrue(eng.cutting_card_seen)
        self.assertEqual(
            len(self.logged("Cutting card confirmed", level="WARNING")), 1)

    def test_checked_empty_frame_decays_pending(self):
        eng = self.engine
        for _ in range(constants.CUTTING_CARD_CONFIRM_FRAMES - 1):
            with eng._lock:
                eng._process_dealer([cc_pred()])
        with eng._lock:
            eng._process_dealer([])  # checked frame without a sighting
        for _ in range(constants.CUTTING_CARD_CONFIRM_FRAMES - 1):
            with eng._lock:
                eng._process_dealer([cc_pred()])
        self.assertFalse(eng.cutting_card_seen)  # the run restarted
        with eng._lock:
            eng._process_dealer([cc_pred()])
        self.assertTrue(eng.cutting_card_seen)

    def test_unchecked_frame_preserves_pending(self):
        eng = self.engine
        for _ in range(constants.CUTTING_CARD_CONFIRM_FRAMES - 1):
            with eng._lock:
                eng._process_dealer([cc_pred()])
        with eng._lock:
            eng._process_dealer([], cc_checked=False)  # cc model not polled
        with eng._lock:
            eng._process_dealer([cc_pred()])
        self.assertTrue(eng.cutting_card_seen)

    def test_sightings_at_different_positions_do_not_accumulate(self):
        eng = self.engine
        for i in range(constants.CUTTING_CARD_CONFIRM_FRAMES):
            with eng._lock:
                eng._process_dealer([cc_pred(cx=400.0 + i * 200.0)])
        self.assertFalse(eng.cutting_card_seen)

    def test_duplicate_boxes_in_one_frame_count_as_one_sighting(self):
        # An elongated cutting card can NMS into two overlapping boxes that
        # quantize to the same 50px cell — still one sighting per frame.
        eng = self.engine
        frame = [cc_pred(400.0, 300.0), cc_pred(415.0, 308.0)]
        for _ in range(constants.CUTTING_CARD_CONFIRM_FRAMES - 1):
            with eng._lock:
                eng._process_dealer(list(frame))
            self.assertFalse(eng.cutting_card_seen)
        with eng._lock:
            eng._process_dealer(list(frame))
        self.assertTrue(eng.cutting_card_seen)


class CuttingCardPollCadence(unittest.TestCase):
    """Suit mode polls the rank model every CUTTING_CARD_CHECK_EVERY cycles —
    but once a sighting is pending it must poll EVERY cycle."""

    class _FakeModel:
        def __init__(self, preds):
            self.preds = preds
            self.calls = 0

        def predict(self, *a, **k):
            self.calls += 1
            return list(self.preds)

    def test_pending_sighting_polls_every_cycle(self):
        eng = DetectionEngine(log=lambda *a, **k: None)
        eng.store = None
        eng._dealer_rect = (0, 0, 100, 100)
        players = self._FakeModel([])
        rank = self._FakeModel([cc_pred()])
        eng.provider = types.SimpleNamespace(
            players_model=lambda: players, dealer_model=lambda: rank)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        old = constants.DEALER_USE_PLAYER_MODEL
        constants.DEALER_USE_PLAYER_MODEL = True
        try:
            for _ in range(constants.CUTTING_CARD_CHECK_EVERY - 1):
                eng._detect(frame)
            self.assertEqual(rank.calls, 0)  # off-cycle, nothing pending
            eng._detect(frame)               # scheduled poll -> first sighting
            self.assertEqual(rank.calls, 1)
            self.assertFalse(eng.cutting_card_seen)
            for _ in range(constants.CUTTING_CARD_CONFIRM_FRAMES - 1):
                eng._detect(frame)           # pending -> polled every cycle
            self.assertTrue(eng.cutting_card_seen)
            self.assertEqual(rank.calls, constants.CUTTING_CARD_CONFIRM_FRAMES)
            eng._detect(frame)               # confirmed -> polling stops
            self.assertEqual(rank.calls, constants.CUTTING_CARD_CONFIRM_FRAMES)
        finally:
            constants.DEALER_USE_PLAYER_MODEL = old


class ReshuffleBadge(EngineCase):
    def test_badge_shows_on_confirmation_and_hides_on_dismiss(self):
        eng = self.engine
        self.assertFalse(eng.get_snapshot()["reshuffle_badge"])
        self.latch_cutting_card()
        eng.publish_snapshot()
        snap = eng.get_snapshot()
        self.assertTrue(snap["reshuffle_badge"])
        self.assertTrue(snap["cutting_card_seen"])
        eng.dismiss_reshuffle_badge()
        snap = eng.get_snapshot()
        self.assertFalse(snap["reshuffle_badge"])
        self.assertTrue(snap["cutting_card_seen"])  # flag survives dismissal

    def test_reset_shoe_clears_flag_badge_and_dismissal(self):
        eng = self.engine
        self.latch_cutting_card()
        eng.dismiss_reshuffle_badge()
        eng.reset_shoe()
        snap = eng.get_snapshot()
        self.assertFalse(snap["cutting_card_seen"])
        self.assertFalse(snap["reshuffle_badge"])
        # A fresh confirmation in the new shoe shows the badge again.
        self.latch_cutting_card()
        eng.publish_snapshot()
        self.assertTrue(eng.get_snapshot()["reshuffle_badge"])


if __name__ == "__main__":
    unittest.main()
