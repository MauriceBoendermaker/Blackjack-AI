"""Reliability pack (Test-Split): split-detection robustness.

Synthetic misaligned/overlapping hand geometries are fed through the REAL
grouping code — _process_players -> _lock_card -> _assign_hand, plus
set_split() retro-routing and replace_card(hand_index=...) — never via
manual hand tags. Hit cards (3rd+ per seat) respect the real confirmation
window: EXTRA_CARD_CONFIRM_CYCLES consecutive sightings of the same
quantized prediction.

Run:  .venv\\Scripts\\python -m unittest tests.test_split_robustness -v
"""

import unittest

from lib.common import constants
from lib.common.card_mappings import PLAYER_CLASS_MAP
from lib.logic.engine import DetectionEngine
from lib.logic.monitor_utils import Polygon

# Synthetic table geometry: seat i owns x in [i*1000, i*1000+999], any y in
# [0, 2000]. Deterministic, independent of any calibrated regions profile on
# this machine (output/regions_*.json is a user file).
SEAT_W = 1000.0


def seat_rect(i):
    x0, x1 = i * SEAT_W, i * SEAT_W + SEAT_W - 1.0
    return Polygon([[x0, 0.0], [x1, 0.0], [x1, 2000.0], [x0, 2000.0],
                    [x0, 0.0]])


def pred(cls, cx, cy, conf=0.9, **extra):
    """A synthetic player-model prediction (class codes: a=Hearts, b=Diamonds,
    c=Spades, d=Clubs; 1=Ace .. 13=King)."""
    return {"class": cls, "confidence": conf,
            "cx": float(cx), "cy": float(cy), **extra}


class SplitCase(unittest.TestCase):
    def setUp(self):
        self.eng = DetectionEngine(log=lambda *a, **k: None)
        self.eng.store = None  # no DB writes from tests
        with self.eng._lock:
            self.eng.regions = [seat_rect(i) for i in range(constants.NUM_SEATS)]
            self.eng._dist_scale = 1.0

    def feed(self, preds):
        """One detection cycle's worth of player predictions."""
        with self.eng._lock:
            self.eng._process_players(list(preds))

    def tags(self, seat_idx=0):
        return [c["hand"] for c in self.eng.seats[seat_idx].cards]

    def names(self, seat_idx=0):
        return [c["name"] for c in self.eng.seats[seat_idx].cards]

    def feed_hit_confirmed(self, seat_idx, p, jitter=0.0):
        """Drive a 3rd+ card through the real confirmation window, asserting
        it locks exactly on the EXTRA_CARD_CONFIRM_CYCLES-th sighting.
        `jitter` shifts cx/cy a little per frame (stays inside the 50 px
        pending-key quantization bucket, like a live stream)."""
        eng = self.eng
        before = len(eng.seats[seat_idx].cards)
        for i in range(constants.EXTRA_CARD_CONFIRM_CYCLES - 1):
            self.feed([{**p, "cx": p["cx"] + i * jitter,
                        "cy": p["cy"] + i * jitter}])
            self.assertEqual(len(eng.seats[seat_idx].cards), before,
                             "hit card locked before the confirmation window")
        last = constants.EXTRA_CARD_CONFIRM_CYCLES - 1
        self.feed([{**p, "cx": p["cx"] + last * jitter,
                    "cy": p["cy"] + last * jitter}])
        self.assertEqual(len(eng.seats[seat_idx].cards), before + 1,
                         "hit card did not lock after the confirmation window")
        return eng.seats[seat_idx].cards[-1]

    def deal_split_pair(self, seat_idx, cls_a, cls_b, xa, xb, cy=1000.0):
        """Initial deal through the real ingest path, then the user splits.
        One card locks per cycle (best confidence first), so two frames."""
        a = pred(cls_a, xa, cy, conf=0.95)
        b = pred(cls_b, xb, cy, conf=0.90)
        self.feed([a, b])
        self.feed([a, b])
        seat = self.eng.seats[seat_idx]
        self.assertEqual(self.names(seat_idx),
                         [PLAYER_CLASS_MAP[cls_a], PLAYER_CLASS_MAP[cls_b]])
        self.eng.set_split(seat_idx)
        self.assertTrue(seat.split)
        self.assertEqual(self.tags(seat_idx), [0, 1])
        return seat


class DetectedRouting(SplitCase):
    """Detections flow through _process_players; routing by _assign_hand."""

    def test_overlapping_boxes_route_by_mean_x(self):
        # Hands at x=200 / x=400. Card boxes are ~120 px wide, so a hit at
        # x=260 overlaps hand 0's box and one at x=380 overlaps hand 1's
        # pair card — routing must use center-x distance, not box geometry.
        self.deal_split_pair(0, "a8", "c8", 200, 400)
        c = self.feed_hit_confirmed(
            0, pred("d5", 260, 1100, width=120, height=170))
        self.assertEqual(c["hand"], 0)   # |260-200| = 60 < |260-400| = 140
        c = self.feed_hit_confirmed(
            0, pred("c13", 380, 1100, width=120, height=170))
        # mean(h0) = (200+260)/2 = 230, mean(h1) = 400 -> 380 goes right.
        self.assertEqual(c["hand"], 1)
        self.assertEqual(self.tags(0), [0, 1, 0, 1])

    def test_exact_midpoint_tie_breaks_to_hand_zero(self):
        # _assign_hand uses `<=` (engine.py): equidistant -> hand 0.
        seat = self.deal_split_pair(0, "a8", "c8", 200, 400)
        self.assertEqual(self.eng._assign_hand(seat, 300.0), 0)  # the rule
        c = self.feed_hit_confirmed(0, pred("d5", 300, 1000))
        self.assertEqual(c["hand"], 0)
        # One pixel past the midpoint flips deterministically to hand 1.
        self.deal_split_pair(1, "a9", "c9", 1200, 1400)
        c = self.feed_hit_confirmed(1, pred("d5", 1301, 1000))
        self.assertEqual(c["hand"], 1)

    def test_vertical_jitter_and_offset_do_not_flip_routing(self):
        # Only x matters: a hit 700 px BELOW both hands (with per-frame
        # cx/cy jitter during confirmation) still joins the x-nearer hand,
        # and a hit far ABOVE joins the other.
        self.deal_split_pair(0, "a8", "c8", 200, 400)
        c = self.feed_hit_confirmed(0, pred("d5", 220, 1700), jitter=2.0)
        self.assertEqual(c["hand"], 0)
        c = self.feed_hit_confirmed(0, pred("a2", 430, 300))
        self.assertEqual(c["hand"], 1)
        self.assertEqual(self.tags(0), [0, 1, 0, 1])

    def test_one_hand_temporarily_empty(self):
        # Hand 1's card is removed (misread correction): a detection to the
        # RIGHT of hand 0's mean must seed the empty hand 1.
        self.deal_split_pair(0, "a8", "c8", 200, 400)
        self.eng.replace_card(0, 1, None)
        self.assertEqual(self.tags(0), [0])
        self.feed([pred("d5", 500, 1000)])  # len<2 -> locks without window
        self.assertEqual(self.tags(0), [0, 1])
        # Mirror case: hand 0 empty, detection LEFT of hand 1's mean -> hand 0.
        self.deal_split_pair(1, "a9", "c9", 1200, 1400)
        self.eng.replace_card(1, 0, None)
        self.assertEqual(self.tags(1), [1])
        self.feed([pred("d5", 1100, 1000)])
        self.assertEqual(self.tags(1), [1, 0])

    def test_no_positioned_cards_defaults_to_hand_zero(self):
        # A fully manual split (both cards cx=None) gives _assign_hand no
        # positions at all: a detected hit must still land deterministically.
        self.eng.replace_card(2, 0, "8 of Hearts")
        self.eng.replace_card(2, 1, "8 of Diamonds")
        self.eng.set_split(2)
        self.assertTrue(self.eng.seats[2].split)
        c = self.feed_hit_confirmed(2, pred("d5", 2500, 1000))
        self.assertEqual(c["hand"], 0)

    def test_set_split_retro_routes_pre_existing_extras(self):
        # A confirmed 3rd card exists BEFORE the user clicks Split: set_split
        # must re-route it by x (card 1 seeds hand 0, card 2 seeds hand 1).
        a, b = pred("a8", 200, 1000, conf=0.95), pred("c8", 400, 1000, conf=0.90)
        self.feed([a, b])
        self.feed([a, b])
        self.feed_hit_confirmed(0, pred("d5", 390, 1100))
        self.assertEqual(self.tags(0), [0, 0, 0])  # no split yet
        self.eng.set_split(0)
        self.assertEqual(self.tags(0), [0, 1, 1])  # 390 sits next to 400


class ManualRouting(SplitCase):
    """replace_card adds/replacements interact with the same hand routing."""

    def test_manual_add_falls_back_to_count_balancing(self):
        # cx=None adds cannot route by position: counts balance, ties -> hand 0.
        self.deal_split_pair(0, "a8", "c8", 200, 400)
        self.eng.replace_card(0, 99, "5 of Clubs")    # 1-1 tie -> hand 0
        self.eng.replace_card(0, 99, "2 of Hearts")   # 2-1     -> hand 1
        self.eng.replace_card(0, 99, "3 of Clubs")    # 2-2 tie -> hand 0
        self.assertEqual(self.tags(0), [0, 1, 0, 1, 0])

    def test_explicit_hand_index_overrides_routing(self):
        self.deal_split_pair(0, "a8", "c8", 200, 400)
        self.eng.replace_card(0, 99, "5 of Clubs", hand_index=1)
        # Count-balancing would now pick hand 0 (1 vs 2) — explicit wins:
        self.eng.replace_card(0, 99, "2 of Hearts", hand_index=1)
        self.eng.replace_card(0, 99, "3 of Clubs", hand_index=0)
        self.assertEqual(self.tags(0), [0, 1, 1, 1, 0])

    def test_out_of_range_hand_index_falls_back_to_automatic(self):
        self.deal_split_pair(0, "a8", "c8", 200, 400)
        self.eng.replace_card(0, 99, "5 of Clubs", hand_index=2)    # -> auto: 0
        self.eng.replace_card(0, 99, "2 of Hearts", hand_index=-1)  # -> auto: 1
        self.assertEqual(self.tags(0), [0, 1, 0, 1])

    def test_hand_index_ignored_on_non_split_seat(self):
        self.eng.replace_card(3, 99, "5 of Clubs", hand_index=1)
        self.assertEqual(self.tags(3), [0])

    def test_replace_keeps_hand_tag_and_position(self):
        # Replacing a misread must keep the card's hand tag (even against a
        # contradicting hand_index) and its screen position, so subsequent
        # detections still route off the corrected card's x.
        self.deal_split_pair(0, "a8", "c8", 200, 400)
        self.feed_hit_confirmed(0, pred("d5", 380, 1100))
        self.assertEqual(self.tags(0), [0, 1, 1])
        self.eng.replace_card(0, 2, "Queen of Spades", hand_index=0)
        card = self.eng.seats[0].cards[2]
        self.assertEqual(card["name"], "Queen of Spades")
        self.assertEqual(card["hand"], 1)          # tag survives the replace
        self.assertEqual(card["cx"], 380.0)        # position survives too
        self.assertTrue(card["manual"])
        # mean(h1) = (400+380)/2 = 390: a hit at 350 is nearer hand 1 than
        # hand 0's 200 — proves the replaced card still anchors routing.
        c = self.feed_hit_confirmed(0, pred("d7", 350, 1200))
        self.assertEqual(c["hand"], 1)

    def test_manual_card_adopts_detected_position_without_duplicate(self):
        # A manual add (cx=None) later seen by the model is the SAME card:
        # it adopts the detected position, no duplicate, hand tag intact.
        self.deal_split_pair(0, "a8", "c8", 200, 400)
        self.eng.replace_card(0, 99, "5 of Clubs", hand_index=1)
        for _ in range(constants.EXTRA_CARD_CONFIRM_CYCLES + 1):
            self.feed([pred("d5", 450, 1100)])
        self.assertEqual(self.names(0),
                         ["8 of Hearts", "8 of Spades", "5 of Clubs"])
        card = self.eng.seats[0].cards[2]
        self.assertEqual(card["hand"], 1)
        self.assertEqual(card["cx"], 450.0)


class FrameStability(SplitCase):
    """Confirmation cadence and dedup across consecutive re-detections."""

    def test_hit_confirmation_window_and_decay(self):
        self.deal_split_pair(0, "a8", "c8", 200, 400)
        p = pred("d5", 260, 1100)
        for _ in range(constants.EXTRA_CARD_CONFIRM_CYCLES - 1):
            self.feed([p])
        self.assertEqual(len(self.eng.seats[0].cards), 2)  # still pending
        self.feed([])  # one missed frame drops the pending run
        self.assertEqual(self.eng._pending_extra, {})
        for _ in range(constants.EXTRA_CARD_CONFIRM_CYCLES - 1):
            self.feed([p])
        self.assertEqual(len(self.eng.seats[0].cards), 2)  # run restarted
        self.feed([p])
        self.assertEqual(len(self.eng.seats[0].cards), 3)
        self.assertEqual(self.tags(0), [0, 1, 0])

    def test_redetection_of_same_cards_neither_duplicates_nor_reroutes(self):
        self.deal_split_pair(0, "a8", "c8", 200, 400)
        self.feed_hit_confirmed(0, pred("d5", 260, 1100))
        self.feed_hit_confirmed(0, pred("c13", 380, 1100))
        names, tags = self.names(0), self.tags(0)
        self.assertEqual(tags, [0, 1, 0, 1])
        per_rank = dict(self.eng.counter.per_rank)
        round_no = self.eng.round_number
        # The stream keeps re-detecting the same four physical cards with
        # positional wobble (well inside SAME_CARD_DISTANCE_PX), plus a
        # near-position different-class flicker that must be ignored.
        for i in range(6):
            dx = (-1) ** i * 15.0
            frame = [pred("a8", 200 + dx, 1000 - dx),
                     pred("c8", 400 + dx, 1000 + dx),
                     pred("d5", 260 + dx, 1100 - dx),
                     pred("c13", 380 + dx, 1100 + dx),
                     pred("d2", 205, 1005)]  # <32 px from the 8H = misread
            self.feed(frame)
        self.assertEqual(self.names(0), names)          # no duplicates
        self.assertEqual(self.tags(0), tags)            # no re-routing
        self.assertEqual(self.eng._pending_extra, {})   # no phantom pendings
        self.assertEqual(dict(self.eng.counter.per_rank), per_rank)
        self.assertEqual(self.eng.round_number, round_no)


class SnapshotIntegration(SplitCase):
    """The routed hands surface coherently in the published snapshot."""

    def test_snapshot_reflects_detected_split_routing(self):
        self.eng.replace_dealer("6")
        self.deal_split_pair(0, "a8", "c8", 200, 400)
        self.feed_hit_confirmed(0, pred("d5", 260, 1100))
        self.feed_hit_confirmed(0, pred("c13", 380, 1100))
        self.eng.publish_snapshot()
        self.assertTrue(self.eng.flush_advice(timeout=60))
        seat = self.eng.get_snapshot()["seats"][0]
        self.assertTrue(seat["split"])
        self.assertEqual(seat["hand_of"], [0, 1, 0, 1])
        self.assertEqual(seat["cards"], ["8 of Hearts", "8 of Spades",
                                         "5 of Clubs", "King of Spades"])
        self.assertIn("H1:", seat["total"])   # 8+5 and 8+K, hand by hand
        self.assertIn("H2:", seat["total"])
        self.assertIn("H1:", seat["advice"])
        self.assertIn("H2:", seat["advice"])


if __name__ == "__main__":
    unittest.main()
