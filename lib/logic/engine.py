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
from . import betting
from . import cards
from . import deviations
from . import ev_engine
from . import ocr
from . import settlement
from . import shoe
from . import sidebets
from .counting import CardCounter, counter_key
from .models import ModelProvider, ModelError
from .monitor_utils import ScreenCapture, scaled_player_regions, dealer_area_rect, scaling_factors
from .session_store import SessionStore
from .strategy import StrategyAdvisor
from .training_data import TrainingDataCollector

_ACTION_NAMES = {"S": "Stand", "H": "Hit", "D": "Double", "P": "Split", "R": "Surrender"}
_ACTION_COLOR_KEYS = {"S": "S", "H": "H", "D": "D/H", "P": "P", "R": "R/H"}


class Seat:
    __slots__ = ("index", "cards", "split")

    def __init__(self, index):
        self.index = index
        self.cards = []  # dicts: name, confidence, cx, cy, manual, counted, hand
        self.split = False  # seat plays two hands; cards carry a hand tag (0/1)


class DetectionEngine:
    def __init__(self, log=print):
        self.log = log
        self.capture = ScreenCapture()
        self.provider = ModelProvider.get()
        self.strategy = StrategyAdvisor()
        self.counter = CardCounter()
        try:
            self.store = SessionStore()
        except Exception as e:  # a broken DB must never block detection
            self.store = None
            self.log(f"Session store unavailable: {e}")

        self._lock = threading.RLock()
        self.seats = [Seat(i) for i in range(constants.NUM_SEATS)]
        self.dealer_card = None          # rank string, e.g. "King" or "10"
        self.dealer_locked = False
        self._dealer_counted = False
        self._dealer_history = deque(maxlen=4)
        self._dealer_pos = None          # up-card (cx, cy) in dealer-crop coords
        self.dealer_extras = []          # playout cards: dicts rank, cx, cy
        self._pending_dealer = {}        # (rank, qx, qy) -> consecutive sightings
        self.round_number = 1
        self.cutting_card_seen = False
        self.my_seats = set()            # seat indices whose P&L the user tracks
        self.bet_placed = float(constants.BASE_BET)  # EUR per owned seat this round
        self.session_pnl = {"units": 0.0, "eur": 0.0, "rounds": 0}

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
        self._ev_error_logged = False

        # Advice (exact EV + side bets) is computed on a dedicated single
        # worker thread, never under self._lock and never on the Tk thread:
        # publish_snapshot only does cache lookups and submits jobs; finished
        # jobs publish a fresh snapshot themselves.
        self._advice_pool = ThreadPoolExecutor(max_workers=1,
                                               thread_name_prefix="advice")
        self._advice_lock = threading.Lock()
        self._advice_cache = {}     # advice key -> ev_engine result (or None)
        self._advice_pending = set()
        self._sidebet_result = ([], None)   # (evs, composition signature)
        self._sidebet_pending = False
        # The exact pre-deal EV sweep (~15 s pure Python) gets its own thread
        # so it can never delay per-seat advice; at most one round stale.
        self._predeal_pool = ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix="predeal")
        self._predeal = {"edge": None, "sig": None}
        self._predeal_pending = False
        self._ocr_regions = None
        self._ocr_last = {"balance": None, "bet": None, "result": None, "ts": 0.0}
        self._ocr_pending = False
        self._ocr_next = 0.0
        self.training = TrainingDataCollector()
        self._last_frame = None
        self._confirm_count = 0
        self._cc_cycle = 0

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
            self._ocr_regions = ocr.load_regions(res)

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
                self._maybe_ocr(frame)
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

        # The dealer area is watched for the whole round: before the lock to
        # find the up-card, after it to count the dealer's playout cards —
        # otherwise the shoe composition silently drifts every round.
        self._last_frame = frame  # retained for training-data crops
        left, top, right, bottom = self._dealer_rect
        crop = frame[top:bottom, left:right]
        if constants.DEALER_USE_PLAYER_MODEL:
            # Suit-aware dealer detection via the 52-class player model.
            dealer_future = self._pool.submit(
                self.provider.players_model().predict, crop,
                constants.PREDICTION_CONFIDENCE_PLAYERS,
                constants.PREDICTION_OVERLAP_PLAYERS)
        else:
            dealer_future = self._pool.submit(
                self.provider.dealer_model().predict, crop,
                constants.PREDICTION_CONFIDENCE_DEALER,
                constants.PREDICTION_OVERLAP_DEALER)

        player_preds = players_future.result()
        dealer_preds = dealer_future.result()
        if constants.DEALER_USE_PLAYER_MODEL:
            # The cutting card only exists in the rank model — poll it cheaply.
            self._cc_cycle += 1
            if (not self.cutting_card_seen
                    and self._cc_cycle % constants.CUTTING_CARD_CHECK_EVERY == 0):
                try:
                    cc_preds = self.provider.dealer_model().predict(
                        crop, constants.PREDICTION_CONFIDENCE_DEALER,
                        constants.PREDICTION_OVERLAP_DEALER)
                    dealer_preds = dealer_preds + [
                        p for p in cc_preds if p["class"] == CUTTING_CARD_CLASS]
                except ModelError:
                    pass
        self._metrics["inference_ms"] = (time.perf_counter() - t0) * 1000.0
        self.last_error = None

        with self._lock:
            self._process_dealer(dealer_preds)
            self._process_players(player_preds)
            self._maybe_auto_new_round(player_preds)

    # --------------------------------------------------------------- dealer

    def _process_dealer(self, predictions):
        cards_seen = []
        for p in predictions:
            if p["class"] == CUTTING_CARD_CLASS:
                if not self.cutting_card_seen:
                    self.cutting_card_seen = True
                    self.log("Cutting card seen — the shoe will be reshuffled soon.")
                continue
            rank = DEALER_CLASS_MAP.get(p["class"]) or PLAYER_CLASS_MAP.get(p["class"])
            if rank is not None:  # full "King of Hearts" names in suit mode
                cards_seen.append(p | {"rank": rank})

        if self.dealer_locked:
            self._track_dealer_playout(cards_seen)
            return

        best = max(cards_seen, key=lambda p: p["confidence"], default=None)
        if best is None:
            # A frame with no dealer card breaks the consecutive-agreement run;
            # without this, an old transient misread could pair with a later one.
            self._dealer_history.clear()
            return
        self._dealer_history.append(best["rank"])
        recent = list(self._dealer_history)[-constants.DEALER_CONFIRM_FRAMES:]
        if len(recent) == constants.DEALER_CONFIRM_FRAMES and len(set(recent)) == 1:
            self._dealer_pos = (best["cx"], best["cy"])
            self._set_dealer(recent[0], manual=False)

    def _track_dealer_playout(self, cards_seen):
        """Count the dealer's hole/hit cards after the up-card locks. Same
        machinery as player hits: position dedupe + multi-cycle confirmation."""
        limit_same = constants.SAME_CARD_DISTANCE_PX * self._dist_scale
        seen_pending = set()
        for p in cards_seen:
            if self._matches_dealer_card(p, limit_same):
                continue
            key = (p["rank"], round(p["cx"] / 50.0), round(p["cy"] / 50.0))
            seen_pending.add(key)
            count = self._pending_dealer.get(key, 0) + 1
            if count >= constants.EXTRA_CARD_CONFIRM_CYCLES:
                self.dealer_extras.append({"rank": p["rank"], "cx": p["cx"], "cy": p["cy"]})
                self.counter.count_card(p["rank"])
                self._pending_dealer.pop(key, None)
                self.log(f"Dealer draws: {p['rank']}")
            else:
                self._pending_dealer[key] = count
        for key in [k for k in self._pending_dealer if k not in seen_pending]:
            del self._pending_dealer[key]

    def _matches_dealer_card(self, pred, limit_same):
        """Is this detection the up-card or an already-counted playout card?
        Up-card comparison is by rank — a manual full-name correction must
        still match rank-only detections (and vice versa in suit mode)."""
        limit_flicker = limit_same * 0.4
        if (self.dealer_card
                and cards.rank_of(pred["rank"]) == cards.rank_of(self.dealer_card)):
            if self._dealer_pos is None:
                self._dealer_pos = (pred["cx"], pred["cy"])  # adopt (manual lock)
                return True
            dx = pred["cx"] - self._dealer_pos[0]
            dy = pred["cy"] - self._dealer_pos[1]
            if dx * dx + dy * dy < limit_same * limit_same:
                return True
        if self._dealer_pos is not None:
            dx = pred["cx"] - self._dealer_pos[0]
            dy = pred["cy"] - self._dealer_pos[1]
            if dx * dx + dy * dy < limit_flicker * limit_flicker:
                return True  # same spot, different class = misread flicker
        for c in self.dealer_extras:
            dx, dy = c["cx"] - pred["cx"], c["cy"] - pred["cy"]
            dist_sq = dx * dx + dy * dy
            if pred["rank"] == c["rank"] and dist_sq < limit_same * limit_same:
                return True
            if dist_sq < limit_flicker * limit_flicker:
                return True
        return False

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
            # Warm the shared EV memos for this up-card before player hits
            # arrive — a deep dummy hand explores the subtrees the real seat
            # evaluations will need for the same composition.
            per_rank = dict(self.counter.per_rank)
            self._advice_pool.submit(self._warm_advice_job, rank, per_rank)

    def _warm_advice_job(self, dealer_rank, per_rank):
        try:
            ev_engine.advise(["2 of Hearts", "3 of Clubs"], dealer_rank, per_rank,
                             self.counter.deck_count)
        except Exception:
            pass

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
        # Drop pending hits that were not seen this cycle — and save them as
        # low-confidence training samples: they're the model's blind spots.
        for key in [k for k in self._pending_extra if k not in seen_pending]:
            _, name, qx, qy = key
            if self._last_frame is not None:
                self._advice_pool.submit(self.training.save_sample, self._last_frame,
                                         qx * 50.0, qy * 50.0, name, "lowconf")
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
        hand = self._assign_hand(seat, pred["cx"]) if seat.split else 0
        seat.cards.append({
            "name": pred["name"], "confidence": pred["confidence"],
            "cx": pred["cx"], "cy": pred["cy"], "manual": False, "counted": True,
            "hand": hand,
        })
        self.counter.count_card(pred["name"])
        tag = f" (hand {hand + 1})" if seat.split else ""
        self.log(f"P{seat.index + 1} card {len(seat.cards)}{tag}: "
                 f"{pred['name']} ({pred['confidence'] * 100:.0f}%)")
        # Active learning: sample every Nth confirmed lock as training data.
        self._confirm_count += 1
        every = constants.TRAINING.get("confirmed_every", 25)
        if self._last_frame is not None and self._confirm_count % every == 0:
            self._advice_pool.submit(self.training.save_sample, self._last_frame,
                                     pred["cx"], pred["cy"], pred["name"], "confirmed")

    @staticmethod
    def _assign_hand(seat, cx):
        """After a split, route a new detection to the nearer hand by the mean
        x of each hand's positioned cards (split hands sit side by side)."""
        if cx is None:
            counts = [sum(1 for c in seat.cards if c["hand"] == h) for h in (0, 1)]
            return 0 if counts[0] <= counts[1] else 1
        means = []
        for h in (0, 1):
            xs = [c["cx"] for c in seat.cards if c["hand"] == h and c["cx"] is not None]
            means.append(sum(xs) / len(xs) if xs else None)
        if means[0] is None and means[1] is None:
            return 0
        if means[0] is None:
            return 0 if cx < means[1] else 1
        if means[1] is None:
            return 1 if cx > means[0] else 0
        return 0 if abs(cx - means[0]) <= abs(cx - means[1]) else 1

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
        # Settle and persist the round that just ended (and the shoe state, so
        # a restart mid-shoe doesn't lose the count) before wiping the table.
        snap = self.get_snapshot()
        if snap and (snap["dealer"]["card"]
                     or any(s["cards"] for s in snap["seats"])):
            settle = self._settle_round(snap)
            snap = {**snap, "settlement": settle, "bet_placed": self.bet_placed}
            if self.store is not None:
                self._advice_pool.submit(
                    self._persist_round_job, snap, self.counter.get_state(),
                    self.round_number, self.cutting_card_seen)
        for seat in self.seats:
            seat.cards.clear()
            seat.split = False
        self._pending_extra.clear()
        self._dealer_history.clear()
        self._pending_dealer.clear()
        self.dealer_extras = []   # counted cards stay counted; list just resets
        self._dealer_pos = None
        self.dealer_card = None
        self.dealer_locked = False
        self._dealer_counted = False
        self._ev_error_logged = False
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
        if self.store is not None:
            self._advice_pool.submit(self._save_state_job)
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
                                        "manual": True, "counted": old["counted"],
                                        "hand": old.get("hand", 0)}
                    if old["counted"]:
                        self.counter.count_card(card_name)
                    self.log(f"P{seat_idx + 1} card {slot + 1} set to {card_name}.")
                    # A corrected misread is GOLD training data.
                    if (not old["manual"] and old["cx"] is not None
                            and card_name != old["name"]
                            and self._last_frame is not None):
                        self._advice_pool.submit(
                            self.training.save_sample, self._last_frame,
                            old["cx"], old["cy"], card_name, "correction")
            elif card_name is not None and len(seat.cards) < constants.MAX_CARDS_PER_SEAT:
                hand = self._assign_hand(seat, None) if seat.split else 0
                seat.cards.append({"name": card_name, "confidence": 1.0,
                                   "cx": None, "cy": None, "manual": True,
                                   "counted": True, "hand": hand})
                self.counter.count_card(card_name)
                self.log(f"P{seat_idx + 1} card {len(seat.cards)} added: {card_name}.")
        self.publish_snapshot()

    def set_split(self, seat_idx, on=True):
        """Split a seat's pair into two hands (or undo). Card 1 seeds hand 1,
        card 2 seeds hand 2; later cards route to the nearer hand by position."""
        with self._lock:
            seat = self.seats[seat_idx]
            if on:
                if seat.split or len(seat.cards) < 2 or not cards.is_pair(
                        [c["name"] for c in seat.cards[:2]]):
                    return
                seat.split = True
                seat.cards[0]["hand"] = 0
                seat.cards[1]["hand"] = 1
                for c in seat.cards[2:]:
                    c["hand"] = self._assign_hand(seat, c["cx"])
                self.log(f"P{seat_idx + 1} split into two hands.")
            else:
                if not seat.split:
                    return
                seat.split = False
                for c in seat.cards:
                    c["hand"] = 0
                self.log(f"P{seat_idx + 1} split undone.")
        self.publish_snapshot()

    def replace_dealer(self, card_name):
        """Manual dealer correction; card_name is a full name, rank, or None.
        Full names are kept — the suit refines the 52-cell composition."""
        with self._lock:
            if card_name and " of " in str(card_name):
                rank = card_name
            else:
                rank = cards.rank_of(card_name) if card_name else None
            self._set_dealer(rank, manual=True)
            if rank is None:
                self._dealer_history.clear()
                self.log("Dealer card cleared.")
        self.publish_snapshot()

    def adjust_counter(self, rank_key, delta):
        self.counter.adjust_manual(rank_key, delta)
        self.publish_snapshot()

    def _settle_round(self, snap):
        """Settle the ending round and roll owned-seat results into the
        session P&L / bankroll. Returns the settlement dict (or None when the
        dealer hand was incomplete — then nothing is booked)."""
        try:
            settle = settlement.settle_round(
                snap["seats"], snap["dealer"]["card"],
                snap["dealer"].get("extras", []))
        except Exception as e:
            self.log(f"Settlement error: {type(e).__name__}: {e}")
            return None
        if settle is None:
            if snap["dealer"]["card"]:
                self.log(f"Round {self.round_number} not settled — "
                         "dealer hand incomplete.")
            return None

        parts = [f"P{s['index'] + 1} {s['net_units']:+g}u" for s in settle["seats"]]
        bj = " BJ" if settle["dealer_bj"] else ""
        self.log(f"Round {self.round_number} settled (dealer {settle['dealer_total']}{bj}): "
                 + ", ".join(parts))

        mine = [s for s in settle["seats"] if s["index"] in self.my_seats]
        if mine:
            units = sum(s["net_units"] for s in mine)
            eur = units * self.bet_placed
            # Cross-check against a recent OCR'd result banner (your result).
            banner = self._ocr_last.get("result")
            if banner and time.time() - self._ocr_last.get("ts", 0) < 30:
                expected = 1 if units > 0 else (-1 if units < 0 else 0)
                seen = {"win": 1, "blackjack": 1, "lose": -1, "push": 0}[banner]
                if expected != seen:
                    self.log(f"⚠ Result banner says '{banner}' but settlement "
                             f"computed {units:+g}u — check for a misread card.")
            self.session_pnl["units"] += units
            self.session_pnl["eur"] += eur
            self.session_pnl["rounds"] += 1
            settle["my_units"] = units
            settle["my_eur"] = eur
            if constants.BETTING.get("auto_bankroll") and eur:
                constants.BETTING["bankroll"] = max(0.0, constants.BETTING["bankroll"] + eur)
                self.log(f"Bankroll {eur:+.2f} EUR -> {constants.BETTING['bankroll']:g} "
                         "(auto-settled)")
                self._advice_pool.submit(self._save_settings_job)
        return settle

    @staticmethod
    def _save_settings_job():
        try:
            from ..common import settings
            settings.save()
        except Exception:
            pass

    def _persist_round_job(self, snap, counter_state, round_number, cutting):
        try:
            self.store.record_round(snap)
            self.store.save_shoe_state(counter_state, round_number, cutting)
        except Exception as e:
            if not self._ev_error_logged:
                self._ev_error_logged = True
                self.log(f"Session store error: {type(e).__name__}: {e}")

    def _save_state_job(self):
        try:
            self.store.save_shoe_state(self.counter.get_state(),
                                       self.round_number, self.cutting_card_seen)
        except Exception:
            pass

    def apply_shoe_state(self, info):
        """Restore a persisted shoe (counter state + round + cutting card)."""
        with self._lock:
            self.counter.apply_state(info["state"])
            self.round_number = max(1, int(info.get("round_number") or 1))
            self.cutting_card_seen = bool(info.get("cutting_card_seen"))
        self.log(f"Shoe restored — round {self.round_number}, "
                 f"{self.counter.cards_seen} cards seen, "
                 f"running count {self.counter.running_count:+d}.")
        self.publish_snapshot()

    def set_my_seat(self, seat_idx, mine=None):
        """Toggle (or set) whether a seat's results count toward the session
        P&L and the bankroll auto-update."""
        with self._lock:
            if mine is None:
                mine = seat_idx not in self.my_seats
            if mine:
                self.my_seats.add(seat_idx)
            else:
                self.my_seats.discard(seat_idx)
        self.log(f"P{seat_idx + 1} {'is now tracked as yours' if mine else 'untracked'}.")
        self.publish_snapshot()

    def set_bet_placed(self, eur):
        with self._lock:
            self.bet_placed = max(0.0, float(eur))
        self.publish_snapshot()

    def refresh_settings(self):
        """Re-read runtime settings (rules, deck count, side bets) and drop
        advice caches keyed on the old config. Called after the settings
        dialog saves."""
        with self._lock:
            self.counter.deck_count = constants.DECK_COUNT
        with self._advice_lock:
            self._advice_cache.clear()
            self._sidebet_result = ([], None)
            self._predeal = {"edge": None, "sig": None}
        self.log("Settings applied — table rules and paytables refreshed.")
        self.publish_snapshot()

    # ------------------------------------------------------------- snapshot

    def _optimal_advice(self, names, dealer_rank, per_rank, csv_action,
                        post_split=False):
        """Exact composition-dependent advice for a seat ('' when no decision).

        Non-blocking: returns the cached EV result when this exact
        (hand, dealer, composition, rules) was already computed; otherwise
        submits a job to the advice thread and returns a placeholder — the
        finished job publishes a fresh snapshot with the real line. Heavy EV
        recursion therefore never runs under self._lock or on the Tk thread."""
        hand = [c for c in names if c and c != "-"]
        if len(hand) < 2 or not dealer_rank or cards.hand_value(hand) >= 21:
            return "", constants.ACTION_COLORS["-"]
        key = (tuple(sorted(hand)), dealer_rank, post_split,
               tuple(sorted(per_rank.items())), self.counter.deck_count,
               tuple(sorted(constants.RULES.items())))
        with self._advice_lock:
            if key not in self._advice_cache:
                if key not in self._advice_pending:
                    self._advice_pending.add(key)
                    self._advice_pool.submit(self._advice_job, key, list(hand),
                                             dealer_rank, dict(per_rank), post_split)
                return "Optimal: …", constants.ACTION_COLORS["-"]
            result = self._advice_cache[key]
        if result is None:
            return "", constants.ACTION_COLORS["-"]
        best = result["best"]
        text = f"Optimal: {_ACTION_NAMES[best]} ({result['evs'][best]:+.3f})"
        if self._csv_primary(csv_action, result["evs"]) not in (None, best):
            text += " ≠ book"
        return text, constants.ACTION_COLORS[_ACTION_COLOR_KEYS[best]]

    def _advice_job(self, key, hand, dealer_rank, per_rank, post_split=False):
        try:
            result = ev_engine.advise(hand, dealer_rank, per_rank,
                                      self.counter.deck_count,
                                      post_split=post_split)
        except Exception as e:
            result = None
            if not self._ev_error_logged:
                self._ev_error_logged = True
                self.log(f"EV engine error: {type(e).__name__}: {e}")
        with self._advice_lock:
            if len(self._advice_cache) > 1024:
                self._advice_cache.clear()
            self._advice_cache[key] = result
            self._advice_pending.discard(key)
        self.publish_snapshot()

    def flush_advice(self, timeout=15.0) -> bool:
        """Block until all queued advice/side-bet jobs have landed and the
        snapshot reflects them. For tests and debugging only."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            barrier = self._advice_pool.submit(lambda: None)
            barrier.result(max(0.1, deadline - time.monotonic()))
            with self._advice_lock:
                if not self._advice_pending and not self._sidebet_pending:
                    return True
        return False

    @staticmethod
    def _csv_primary(action, available_evs):
        """First letter of a CSV code ('D/H' -> 'D') that is actually available
        under the table rules; None when the CSV has no usable advice."""
        if not action:
            return None
        for part in action.split("/"):
            if part in available_evs:
                return part
        return None

    @staticmethod
    def _index_advice(names, dealer_rank, true_count):
        """Illustrious 18 / Fab 4 annotation for a seat ('' when none applies)."""
        hand = [c for c in names if c and c != "-"]
        dealer = cards.dealer_strategy_rank(dealer_rank)
        if len(hand) < 2 or dealer is None or cards.hand_value(hand) >= 21:
            return "", constants.ACTION_COLORS["-"]
        dev = deviations.index_advice(cards.hand_key(hand), dealer, true_count,
                                      two_cards=len(hand) == 2)
        if dev is None:
            return "", constants.ACTION_COLORS["-"]
        rel = ">=" if dev["triggered"] else "<"
        text = (f"Index: {_ACTION_NAMES[dev['action']]} "
                f"(TC {true_count:+.1f} {rel} {dev['index']:+d})")
        color = (constants.ACTION_COLORS[_ACTION_COLOR_KEYS[dev["action"]]]
                 if dev["triggered"] else constants.ACTION_COLORS["-"])
        return text, color

    def _insurance_advice(self, dealer_rank, per_rank):
        """Table-level insurance call when the dealer shows an ace, else None."""
        if not dealer_rank or cards.dealer_strategy_rank(dealer_rank) != "A":
            return None
        try:
            info = ev_engine.insurance_advice(per_rank)
        except Exception as e:
            if not self._ev_error_logged:
                self._ev_error_logged = True
                self.log(f"EV engine error: {type(e).__name__}: {e}")
            return None
        verb = "TAKE" if info["take"] else "Decline"
        info["text"] = f"Insurance: {verb} ({info['ev']:+.3f}/unit)"
        info["color"] = constants.ACTION_COLORS["H" if info["take"] else "R/H"]
        return info

    def _side_bet_evs(self, count):
        """Pre-deal side-bet EVs. Non-blocking: returns the latest computed
        values (possibly one composition behind) and lets the advice thread
        catch up; the finished job publishes a fresh snapshot."""
        sig = (count["cards_seen"],
               tuple(sorted(count["per_rank"].items())),
               tuple(sorted(count["suit_seen"].items())),
               tuple(sorted(count["rank_seen_nosuit"].items())),
               self.counter.deck_count, constants.RULES["s17"],
               tuple(k for k, v in constants.SIDE_BETS.items() if v.get("enabled")))
        with self._advice_lock:
            result, cached_sig = self._sidebet_result
            if cached_sig != sig and not self._sidebet_pending:
                self._sidebet_pending = True
                self._advice_pool.submit(self._sidebet_job, sig, {
                    "per_rank": dict(count["per_rank"]),
                    "suit_seen": dict(count["suit_seen"]),
                    "rank_seen_nosuit": dict(count["rank_seen_nosuit"]),
                })
            return result

    def _predeal_edge(self, count):
        """Latest exact pre-deal EV (V2 Feature 3), refreshed on its own
        thread whenever the composition changes. None until the first sweep
        lands, when disabled, or when no monitor is selected (headless)."""
        if not constants.BETTING.get("use_exact_edge") or self.capture.monitor is None:
            return None
        sig = (count["cards_seen"], tuple(sorted(count["per_rank"].items())),
               self.counter.deck_count, tuple(sorted(constants.RULES.items())))
        with self._advice_lock:
            if self._predeal["sig"] != sig and not self._predeal_pending:
                self._predeal_pending = True
                self._predeal_pool.submit(self._predeal_job, sig,
                                          dict(count["per_rank"]))
            return self._predeal["edge"]

    def _predeal_job(self, sig, per_rank):
        try:
            comp = ev_engine.comp_from_per_rank(per_rank, self.counter.deck_count)
            edge = (ev_engine.predeal_ev(comp, ev_engine.current_rules())
                    if sum(comp) >= 52 else None)
        except Exception as e:
            edge = None
            if not self._ev_error_logged:
                self._ev_error_logged = True
                self.log(f"Pre-deal EV error: {type(e).__name__}: {e}")
        with self._advice_lock:
            self._predeal = {"edge": edge, "sig": sig}
            self._predeal_pending = False
        if edge is not None:
            self.log(f"Exact pre-deal edge: {edge:+.3%}")
        self.publish_snapshot()

    def _maybe_ocr(self, frame):
        """Throttled screen-OCR of balance/bet/result crops (advice thread)."""
        if (not ocr.OCR_AVAILABLE or not self._ocr_regions
                or not constants.OCR.get("enabled")):
            return
        now = time.monotonic()
        if now < self._ocr_next or self._ocr_pending:
            return
        self._ocr_next = now + float(constants.OCR.get("interval_s", 1.0))
        h, w = frame.shape[:2]
        crops = {}
        for key, (left, top, right, bottom) in self._ocr_regions.items():
            left, top = max(0, left), max(0, top)
            right, bottom = min(w, right), min(h, bottom)
            if right - left >= 4 and bottom - top >= 4:
                crops[key] = frame[top:bottom, left:right].copy()
        if not crops:
            return
        self._ocr_pending = True
        self._advice_pool.submit(self._ocr_job, crops)

    def _ocr_job(self, crops):
        try:
            texts = {}
            for key, img in crops.items():
                texts.update(ocr.read_regions(
                    img, {key: [0, 0, img.shape[1], img.shape[0]]}))
            values = ocr.interpret(texts)
        except Exception as e:
            values = {}
            if not self._ev_error_logged:
                self._ev_error_logged = True
                self.log(f"OCR error: {type(e).__name__}: {e}")
        finally:
            self._ocr_pending = False
        if not values:
            return

        balance = values.get("balance")
        if (balance and constants.OCR.get("sync_bankroll")
                and abs(balance - constants.BETTING["bankroll"]) >= 0.01):
            constants.BETTING["bankroll"] = balance
            self.log(f"Bankroll synced from screen: €{balance:g}")
            self._save_settings_job()
        bet = values.get("bet")
        if bet and constants.OCR.get("sync_bet") and bet != self.bet_placed:
            with self._lock:
                self.bet_placed = bet
            self.log(f"Bet placed synced from screen: €{bet:g}")
        self._ocr_last = {**values, "ts": time.time()}
        self.publish_snapshot()

    def _sidebet_job(self, sig, count):
        try:
            comp52 = shoe.from_counter_snapshot(count, self.counter.deck_count)
            comp10 = ev_engine.comp_from_per_rank(count["per_rank"],
                                                  self.counter.deck_count)
            result = sidebets.evaluate_all(comp52, comp10)
        except Exception as e:
            result = []
            if not self._ev_error_logged:
                self._ev_error_logged = True
                self.log(f"Side-bet engine error: {type(e).__name__}: {e}")
        with self._advice_lock:
            self._sidebet_result = (result, sig)
            self._sidebet_pending = False
        self.publish_snapshot()

    def _split_seat_snapshot(self, seat, names, dealer_rank, ev_count, true_count):
        """Per-hand advice for a split seat. The combined lines show both
        hands; the flat card list keeps engine order so the picker still
        addresses cards by slot."""
        split_aces = (cards.rank_of(names[0]) == "Ace"
                      and not constants.RULES["hit_split_aces"])
        hand_lines = {"total": [], "advice": [], "optimal": []}
        advice_color = optimal_color = constants.ACTION_COLORS["-"]
        for h in (0, 1):
            hand_names = [c["name"] for c in seat.cards if c.get("hand", 0) == h]
            if not hand_names:
                continue
            if split_aces and len(hand_names) >= 2:
                action, text, color = "S", "Stand (one card)", constants.ACTION_COLORS["S"]
                optimal, opt_color = "", constants.ACTION_COLORS["-"]
            else:
                action, text, color = self.strategy.advice(
                    hand_names, dealer_rank, post_split=True)
                optimal, opt_color = self._optimal_advice(
                    hand_names, dealer_rank, ev_count["per_rank"], action,
                    post_split=True)
            label = cards.describe_hand(hand_names) if hand_names else "—"
            hand_lines["total"].append(f"H{h + 1}: {label}")
            if text:
                hand_lines["advice"].append(f"H{h + 1}: {text}")
                advice_color = color
            if optimal:
                hand_lines["optimal"].append(
                    f"H{h + 1} {optimal.replace('Optimal: ', '')}")
                optimal_color = opt_color
        return {
            "index": seat.index,
            "cards": names,
            "manual": [c["manual"] for c in seat.cards],
            "hand_of": [c.get("hand", 0) for c in seat.cards],
            "split": True,
            "mine": seat.index in self.my_seats,
            "can_split": False,
            "total": "  ·  ".join(hand_lines["total"]),
            "advice": "\n".join(hand_lines["advice"]),
            "advice_color": advice_color,
            "optimal": "\n".join(hand_lines["optimal"]),
            "optimal_color": optimal_color,
            "index_advice": "",
            "index_color": constants.ACTION_COLORS["-"],
        }

    def _ev_count(self, count):
        """The composition the EV engines must see. After a mid-round shoe
        reset, cards still on the table are detached from the counter
        (counted=False) but they ARE seen — fold them back in for EV inputs
        (the displayed counters stay as the user set them)."""
        extra = {}
        for seat in self.seats:
            for c in seat.cards:
                if not c["counted"]:
                    key = counter_key(c["name"])
                    extra[key] = extra.get(key, 0) + 1
        if not extra:
            return count
        per_rank = dict(count["per_rank"])
        for key, n in extra.items():
            per_rank[key] = per_rank.get(key, 0) + n
        return {**count, "per_rank": per_rank,
                "cards_seen": count["cards_seen"] + sum(extra.values())}

    def publish_snapshot(self):
        with self._lock:
            dealer_rank = self.dealer_card
            count = self.counter.snapshot()
            ev_count = self._ev_count(count)
            insurance = self._insurance_advice(dealer_rank, ev_count["per_rank"])
            seats = []
            for seat in self.seats:
                names = [c["name"] for c in seat.cards]
                if seat.split:
                    seats.append(self._split_seat_snapshot(
                        seat, names, dealer_rank, ev_count, count["true"]))
                    continue
                action, text, color = self.strategy.advice(names, dealer_rank)
                optimal, optimal_color = self._optimal_advice(
                    names, dealer_rank, ev_count["per_rank"], action)
                if (insurance is not None and not optimal and len(names) == 2
                        and cards.hand_value(names) == 21):
                    # Natural blackjack vs an ace: the even-money decision.
                    take = insurance["even_money_edge"] > 0
                    optimal = (f"Even money: {'TAKE' if take else 'Decline'} "
                               f"({insurance['even_money_edge']:+.3f})")
                    optimal_color = constants.ACTION_COLORS["H" if take else "R/H"]
                index_text, index_color = self._index_advice(
                    names, dealer_rank, count["true"])
                seats.append({
                    "index": seat.index,
                    "cards": names,
                    "manual": [c["manual"] for c in seat.cards],
                    "hand_of": [c.get("hand", 0) for c in seat.cards],
                    "split": False,
                    "mine": seat.index in self.my_seats,
                    "can_split": (len(names) == 2 and cards.is_pair(names)
                                  and cards.hand_value(names) < 21),
                    "total": cards.describe_hand(names),
                    "advice": text,
                    "advice_color": color,
                    "optimal": optimal,
                    "optimal_color": optimal_color,
                    "index_advice": index_text,
                    "index_color": index_color,
                })
            snapshot = {
                "seq": 0,
                "seats": seats,
                "dealer": {"card": dealer_rank, "locked": self.dealer_locked,
                           "extras": [c["rank"] for c in self.dealer_extras]},
                "insurance": insurance,
                "side_bets": self._side_bet_evs(ev_count),
                "count": count,
                "bet": betting.suggest(count["true"],
                                       exact_edge=self._predeal_edge(ev_count))["text"],
                "edge_exact": self._predeal["edge"],
                "bet_placed": self.bet_placed,
                "session_pnl": dict(self.session_pnl),
                "ocr": dict(self._ocr_last) if self._ocr_regions else None,
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
