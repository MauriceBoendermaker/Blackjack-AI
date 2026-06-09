"""Detection engine: capture -> inference -> game state -> display snapshot.

Runs entirely on the worker thread; owns NO Tkinter objects. The GUI reads
immutable snapshots via `get_snapshot()` and mutates state only through the
thread-safe public methods (replace_card, new_round, ...).

Performance design (vs. the old implementation):
  * frames stay in memory end-to-end (no JPEG round-trips through disk)
  * dealer + player inference run in parallel; the dealer call stops entirely
    once the up-card is confirmed ("locked") for the round
  * statically unchanged frames are detected with a cheap thumbnail diff and
    skipped before any inference happens
  * hosted-API uploads are downscaled (see models.RoboflowModel)
"""

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2

from ..common import constants
from ..common.card_mappings import PLAYER_CLASS_MAP, DEALER_CLASS_MAP, CUTTING_CARD_CLASS
from . import cards
from .counting import CardCounter
from .models import ModelProvider, ModelError
from .monitor_utils import ScreenCapture, scaled_player_regions, dealer_area_rect, scaling_factors
from .strategy import StrategyAdvisor


class Seat:
    __slots__ = ("index", "cards")

    def __init__(self, index):
        self.index = index
        self.cards = []  # list of dicts: name, confidence, cx, cy, manual, counted


class DetectionEngine:
    def __init__(self, log=print):
        self.log = log
        self.capture = ScreenCapture()
        self.provider = ModelProvider.get()
        self.strategy = StrategyAdvisor()
        self.counter = CardCounter()

        self._lock = threading.RLock()
        self.seats = [Seat(i) for i in range(constants.NUM_SEATS)]
        self.dealer_card = None          # rank string, e.g. "King" or "10"
        self.dealer_locked = False
        self._dealer_counted = False
        self._dealer_history = deque(maxlen=4)
        self.round_number = 1
        self.cutting_card_seen = False

        self.regions = None
        self._dealer_rect = None
        self._dist_scale = 1.0

        self._pool = ThreadPoolExecutor(max_workers=2)
        self._pending_extra = {}   # (seat_idx, name, qx, qy) -> consecutive sightings
        self._prev_thumb = None
        self._skipped_cycles = 0
        self._empty_frames = 0     # consecutive inference frames with zero detections
        self._last_activity = "waiting"
        self.last_error = None

        self._snapshot_lock = threading.Lock()
        self._snapshot = None
        self._seq = 0
        self._metrics = {"cycle_ms": 0.0, "inference_ms": 0.0, "skipped": False}

        self.publish_snapshot()

    # ------------------------------------------------------------------ setup

    def set_monitor(self, monitor):
        with self._lock:
            self.capture.set_monitor(monitor)
            res = self.capture.resolution
            self.regions = scaled_player_regions(res)
            self._dealer_rect = dealer_area_rect(res)
            sx, _ = scaling_factors(res)
            self._dist_scale = sx

    def warm_up(self):
        """Initialize models (network handshake for the hosted API). Call from
        the worker thread before the first cycle so the GUI never blocks."""
        self.provider.players_model()
        self.provider.dealer_model()

    # ------------------------------------------------------------ main cycle

    def run_cycle(self) -> str:
        """One capture+detect pass. Returns activity: dealing|complete|waiting."""
        t0 = time.perf_counter()
        skipped = False
        try:
            frame = self.capture.grab_bgr()
            if self._frame_unchanged(frame):
                skipped = True
                if self._empty_frames > 0:
                    # Unchanged since an empty frame == still empty.
                    with self._lock:
                        self._maybe_auto_new_round([])
            else:
                self._detect(frame)
        except ModelError as e:
            self.last_error = str(e)
            self.log(f"Error: {e}")
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self.log(f"Error in detection cycle: {self.last_error}")

        with self._lock:
            activity = self._activity()
        self._last_activity = activity
        self._metrics["cycle_ms"] = (time.perf_counter() - t0) * 1000.0
        self._metrics["skipped"] = skipped
        self.publish_snapshot()
        return activity

    def _frame_unchanged(self, frame) -> bool:
        thumb = cv2.cvtColor(cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY)
        prev, self._prev_thumb = self._prev_thumb, thumb
        if prev is None:
            return False
        diff = float(cv2.absdiff(thumb, prev).mean())
        if diff < constants.FRAME_DIFF_THRESHOLD and self._skipped_cycles < constants.MAX_SKIPPED_CYCLES:
            self._skipped_cycles += 1
            return True
        self._skipped_cycles = 0
        return False

    def _detect(self, frame):
        t0 = time.perf_counter()
        players_future = self._pool.submit(
            self.provider.players_model().predict, frame,
            constants.PREDICTION_CONFIDENCE_PLAYERS, constants.PREDICTION_OVERLAP_PLAYERS)

        dealer_preds = []
        run_dealer = not self.dealer_locked
        if run_dealer:
            left, top, right, bottom = self._dealer_rect
            crop = frame[top:bottom, left:right]
            dealer_future = self._pool.submit(
                self.provider.dealer_model().predict, crop,
                constants.PREDICTION_CONFIDENCE_DEALER, constants.PREDICTION_OVERLAP_DEALER)

        player_preds = players_future.result()
        if run_dealer:
            dealer_preds = dealer_future.result()
        self._metrics["inference_ms"] = (time.perf_counter() - t0) * 1000.0
        self.last_error = None

        with self._lock:
            if run_dealer:
                self._process_dealer(dealer_preds)
            self._process_players(player_preds)
            self._maybe_auto_new_round(player_preds)

    # --------------------------------------------------------------- dealer

    def _process_dealer(self, predictions):
        best = None
        for p in predictions:
            if p["class"] == CUTTING_CARD_CLASS:
                if not self.cutting_card_seen:
                    self.cutting_card_seen = True
                    self.log("Cutting card seen — the shoe will be reshuffled soon.")
                continue
            rank = DEALER_CLASS_MAP.get(p["class"])
            if rank is None:
                continue
            if best is None or p["confidence"] > best[1]:
                best = (rank, p["confidence"])
        if best is None:
            # A frame with no dealer card breaks the consecutive-agreement run;
            # without this, an old transient misread could pair with a later one.
            self._dealer_history.clear()
            return
        self._dealer_history.append(best[0])
        recent = list(self._dealer_history)[-constants.DEALER_CONFIRM_FRAMES:]
        if len(recent) == constants.DEALER_CONFIRM_FRAMES and len(set(recent)) == 1:
            self._set_dealer(recent[0], manual=False)

    def _set_dealer(self, rank, manual):
        if self._dealer_counted and self.dealer_card is not None:
            self.counter.uncount_card(self.dealer_card)
            self._dealer_counted = False
        self.dealer_card = rank
        self.dealer_locked = rank is not None
        if rank is not None:
            self.counter.count_card(rank)
            self._dealer_counted = True
            self.log(f"Dealer up-card: {rank}" + (" (manual)" if manual else " (locked)"))

    # --------------------------------------------------------------- players

    def _process_players(self, predictions):
        candidates = {}  # seat_idx -> list of candidate dicts
        for p in predictions:
            name = PLAYER_CLASS_MAP.get(p["class"])
            if name is None:
                continue
            seat_idx = self._seat_for_point(p["cx"], p["cy"])
            if seat_idx is None:
                continue
            candidates.setdefault(seat_idx, []).append(p | {"name": name})

        seen_pending = set()
        for seat_idx, plist in candidates.items():
            seat = self.seats[seat_idx]
            if len(seat.cards) >= constants.MAX_CARDS_PER_SEAT:
                continue
            fresh = [p for p in plist if not self._matches_existing(seat, p)]
            if not fresh:
                continue
            best = max(fresh, key=lambda p: p["confidence"])
            if len(seat.cards) < 2:
                self._lock_card(seat, best)
            else:
                # Hits need confirmation across consecutive cycles to avoid phantoms.
                key = (seat_idx, best["name"],
                       round(best["cx"] / 50.0), round(best["cy"] / 50.0))
                seen_pending.add(key)
                count = self._pending_extra.get(key, 0) + 1
                if count >= constants.EXTRA_CARD_CONFIRM_CYCLES:
                    self._lock_card(seat, best)
                    self._pending_extra.pop(key, None)
                else:
                    self._pending_extra[key] = count
        # Drop pending hits that were not seen this cycle.
        for key in [k for k in self._pending_extra if k not in seen_pending]:
            del self._pending_extra[key]

    def _seat_for_point(self, x, y):
        for i, region in enumerate(self.regions):
            if region.contains(x, y):
                return i
        return None

    def _matches_existing(self, seat, pred) -> bool:
        """Is this detection the same physical card as one already locked?"""
        limit_same = constants.SAME_CARD_DISTANCE_PX * self._dist_scale
        limit_flicker = limit_same * 0.4
        for c in seat.cards:
            if c["cx"] is None:
                # Manually added card without a screen position: if the model
                # now sees the same rank+suit in this seat, that IS this card —
                # adopt the detected position so normal dedup applies from here.
                if c["name"] == pred["name"]:
                    c["cx"], c["cy"] = pred["cx"], pred["cy"]
                    return True
                continue
            dist = ((c["cx"] - pred["cx"]) ** 2 + (c["cy"] - pred["cy"]) ** 2) ** 0.5
            if c["name"] == pred["name"] and dist < limit_same:
                return True
            if dist < limit_flicker:  # same spot, different class = misread flicker
                return True
            if c["manual"] and dist < limit_same:
                return True  # never fight a manual correction at this position
        return False

    def _lock_card(self, seat, pred):
        seat.cards.append({
            "name": pred["name"], "confidence": pred["confidence"],
            "cx": pred["cx"], "cy": pred["cy"], "manual": False, "counted": True,
        })
        self.counter.count_card(pred["name"])
        self.log(f"P{seat.index + 1} card {len(seat.cards)}: "
                 f"{pred['name']} ({pred['confidence'] * 100:.0f}%)")

    # ----------------------------------------------------------- round flow

    def _activity(self):
        active = [s for s in self.seats if s.cards]
        if not active:
            return "waiting"
        if any(len(s.cards) < 2 for s in active):
            return "dealing"
        if self.dealer_locked or self.dealer_card:
            return "complete"
        return "dealing"

    def _maybe_auto_new_round(self, player_preds):
        """When a completed round's table is cleared, start the next round.

        `player_preds` is the UNFILTERED full-frame prediction list, so a
        lingering dealer card (which the player model also detects) keeps the
        frame "non-empty" and blocks a premature reset. No extra inference —
        and therefore no network call — happens here; this runs under _lock.
        """
        if player_preds:
            self._empty_frames = 0
            return
        if self._activity() != "complete":
            return
        self._empty_frames += 1
        if self._empty_frames < 2:
            return
        self._empty_frames = 0
        self.log(f"Table cleared — starting round {self.round_number + 1}.")
        self._reset_round_state()

    def _reset_round_state(self):
        for seat in self.seats:
            seat.cards.clear()
        self._pending_extra.clear()
        self._dealer_history.clear()
        self.dealer_card = None
        self.dealer_locked = False
        self._dealer_counted = False
        self.round_number += 1
        if self.cutting_card_seen:
            self.log("Reminder: cutting card was seen — reset the shoe count after the shuffle.")

    # ------------------------------------------------- GUI-facing mutations

    def new_round(self):
        with self._lock:
            self._reset_round_state()
        self.log(f"Round reset — now at round {self.round_number}.")
        self.publish_snapshot()

    def reset_shoe(self):
        with self._lock:
            self.counter.reset_shoe()
            self.cutting_card_seen = False
            # Cards already on the table belong to the pre-reset history;
            # detach them so later corrections can't drive the fresh count negative.
            for seat in self.seats:
                for card in seat.cards:
                    card["counted"] = False
            self._dealer_counted = False
        self.log("Shoe counts reset.")
        self.publish_snapshot()

    def replace_card(self, seat_idx, slot, card_name, expected_round=None):
        """Manual correction from the UI. card_name=None removes the card.

        `expected_round` is the round number the user was looking at when the
        picker opened; if the round advanced meanwhile (auto reset), the
        correction targets a hand that no longer exists and is discarded.
        """
        with self._lock:
            if expected_round is not None and expected_round != self.round_number:
                self.log("Correction discarded — the round changed while the picker was open.")
                return
            seat = self.seats[seat_idx]
            if slot < len(seat.cards):
                old = seat.cards[slot]
                if old["counted"]:
                    self.counter.uncount_card(old["name"])
                if card_name is None:
                    seat.cards.pop(slot)
                    self.log(f"P{seat_idx + 1} card {slot + 1} removed.")
                else:
                    # Correcting a misread keeps the card's counted status: a
                    # card detached by reset_shoe stays out of the fresh count.
                    seat.cards[slot] = {"name": card_name, "confidence": 1.0,
                                        "cx": old["cx"], "cy": old["cy"],
                                        "manual": True, "counted": old["counted"]}
                    if old["counted"]:
                        self.counter.count_card(card_name)
                    self.log(f"P{seat_idx + 1} card {slot + 1} set to {card_name}.")
            elif card_name is not None and len(seat.cards) < constants.MAX_CARDS_PER_SEAT:
                seat.cards.append({"name": card_name, "confidence": 1.0,
                                   "cx": None, "cy": None, "manual": True, "counted": True})
                self.counter.count_card(card_name)
                self.log(f"P{seat_idx + 1} card {len(seat.cards)} added: {card_name}.")
        self.publish_snapshot()

    def replace_dealer(self, card_name):
        """Manual dealer correction; card_name is a full name, rank, or None."""
        with self._lock:
            rank = cards.rank_of(card_name) if card_name else None
            self._set_dealer(rank, manual=True)
            if rank is None:
                self._dealer_history.clear()
                self.log("Dealer card cleared.")
        self.publish_snapshot()

    def adjust_counter(self, rank_key, delta):
        self.counter.adjust_manual(rank_key, delta)
        self.publish_snapshot()

    # ------------------------------------------------------------- snapshot

    def publish_snapshot(self):
        with self._lock:
            dealer_rank = self.dealer_card
            seats = []
            for seat in self.seats:
                names = [c["name"] for c in seat.cards]
                action, text, color = self.strategy.advice(names, dealer_rank)
                seats.append({
                    "index": seat.index,
                    "cards": names,
                    "manual": [c["manual"] for c in seat.cards],
                    "total": cards.describe_hand(names),
                    "advice": text,
                    "advice_color": color,
                })
            count = self.counter.snapshot()
            snapshot = {
                "seq": 0,
                "seats": seats,
                "dealer": {"card": dealer_rank, "locked": self.dealer_locked},
                "count": count,
                "bet": StrategyAdvisor.bet_suggestion(count["true"]),
                "round": self.round_number,
                "cutting_card_seen": self.cutting_card_seen,
                "activity": self._last_activity,
                "metrics": dict(self._metrics),
                "backend": self.provider.backend_name,
                "error": self.last_error,
            }
            # Publish while still holding the state lock so a snapshot built
            # from older state can never be stored with a newer seq.
            with self._snapshot_lock:
                self._seq += 1
                snapshot["seq"] = self._seq
                self._snapshot = snapshot

    def get_snapshot(self):
        with self._snapshot_lock:
            return self._snapshot
