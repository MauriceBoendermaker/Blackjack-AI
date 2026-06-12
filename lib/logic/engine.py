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

import inspect
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
from . import ev_offload
from . import ocr
from . import phase
from . import seat_quality
from . import settlement
from . import shoe
from . import sidebet_outcomes
from . import sidebets
from . import anchors as anchors_mod
from .counting import CardCounter, counter_key
from .models import ModelProvider, ModelError
from .monitor_utils import (Polygon, ScreenCapture, dealer_area_rect,
                            load_custom_regions, scaled_player_regions,
                            scaling_factors)
from .session_store import SessionStore
from .strategy import StrategyAdvisor
from .training_data import TrainingDataCollector

_ACTION_NAMES = {"S": "Stand", "H": "Hit", "D": "Double", "P": "Split", "R": "Surrender"}
_ACTION_COLOR_KEYS = {"S": "S", "H": "H", "D": "D/H", "P": "P", "R": "R/H"}


def _level_aware(log_fn):
    """Engine logs pass level="WARNING" for entries that must stand out.
    LogManager.add_log accepts that kwarg, but the GUI injects plain print
    (its stdout is redirected into the LogManager), which does not — probe
    once and fold the level into the message for such loggers."""
    try:
        inspect.signature(log_fn).bind("probe", level="INFO")
        return log_fn
    except (TypeError, ValueError):
        def adapted(message, level=None):
            log_fn(f"{level}: {message}" if level and level != "INFO" else message)
        return adapted


class Seat:
    __slots__ = ("index", "cards", "split")

    def __init__(self, index):
        self.index = index
        self.cards = []  # dicts: name, confidence, cx, cy, manual, counted, hand
        self.split = False  # seat plays two hands; cards carry a hand tag (0/1)


class DetectionEngine:
    def __init__(self, log=print):
        self.log = _level_aware(log)
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
        self._pending_cutting_card = {}  # (qx, qy) -> consecutive sightings
        self._reshuffle_dismissed = False  # badge hidden; the flag itself stays
        self.my_seats = set()            # seat indices whose P&L the user tracks
        self.bet_placed = float(constants.BASE_BET)  # EUR per owned seat this round
        # "eur" is the total incl. side bets; "side_eur" breaks them out.
        self.session_pnl = {"units": 0.0, "eur": 0.0, "side_eur": 0.0,
                            "rounds": 0}
        # Side-bet stakes frozen at the round's first card (bets are
        # immutable once dealing starts — a stake edited mid-round must not
        # change what THIS round settles for). None until the first card.
        self._round_stakes = None
        # Rounds whose bet suggestion hit the table max (session-scoped). The
        # latch collapses the many snapshots a round publishes into one tick,
        # consumed at round end in _reset_round_state.
        self.bet_capped_rounds = 0
        self._bet_was_capped = False
        # Payout fingerprint for this shoe — a mid-shoe paytable change taints
        # the recorded EVs, so it stays flagged until the next reset_shoe().
        self._shoe_paytable_hash = settlement.paytable_hash()
        self.paytable_changed_midshoe = False
        # Per-seat book-play tally for Bet Behind (session-scoped — players
        # come and go, so history from another session would be misleading).
        self.seat_play = {i: {"book": 0, "n": 0} for i in range(constants.NUM_SEATS)}

        self.regions = None
        self._dealer_rect = None
        self._dist_scale = 1.0

        self._pool = ThreadPoolExecutor(max_workers=2)
        self._pending_extra = {}   # (seat_idx, name, qx, qy) -> consecutive sightings
        self._prev_thumb = None
        self._skipped_cycles = 0
        self._empty_frames = 0     # consecutive inference frames with zero detections
        self._dealer_empty_frames = 0  # consecutive dealer-area frames with no cards
        self._reset_delay_logged = False  # one INFO per dealer-blocked reset window
        self._last_activity = "waiting"
        self.last_error = None
        self._model_refresh_ts = None  # time.time() of the last idle refresh
        self._ev_error_keys = set()  # advice-path errors logged once per key per round

        # Advice (exact EV + side bets) is computed on a dedicated single
        # worker thread, never under self._lock and never on the Tk thread:
        # publish_snapshot only does cache lookups and submits jobs; finished
        # jobs publish a fresh snapshot themselves.
        self._advice_pool = ThreadPoolExecutor(max_workers=1,
                                               thread_name_prefix="advice")
        # Disk and OCR work gets its own worker so a balance read or a
        # SQLite write never queues behind EV recursion (and vice versa).
        # All SQLite writers stay on this ONE thread so shoe-state writes
        # keep their order.
        self._io_pool = ThreadPoolExecutor(max_workers=1,
                                           thread_name_prefix="io")
        self._advice_lock = threading.Lock()
        self._advice_cache = {}     # advice key -> ev_engine result (or None)
        self._advice_pending = {}   # advice key -> submit time (time.monotonic)
        self._sidebet_result = ([], None)   # (evs, composition signature)
        self._sidebet_pending = False
        # The exact pre-deal EV sweep (~15 s pure Python) gets its own thread
        # so it can never delay per-seat advice; at most one round stale.
        self._predeal_pool = ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix="predeal")
        self._predeal = {"edge": None, "sig": None}
        self._predeal_pending = False
        # Settings signature backing the selective cache invalidation in
        # refresh_settings — a display-only save must not drop EV caches.
        self._rules_sig = self._settings_sig()
        self._ocr_regions = None
        self._ocr_last = {"balance": None, "bet": None, "result": None, "ts": 0.0}
        self._ocr_pending = False
        self._ocr_next = 0.0
        self._settings_dirty = False  # OCR bankroll syncs persist at round end
        self.training = TrainingDataCollector()
        self._last_frame = None
        self._confirm_count = 0
        self._cc_cycle = 0

        # Game-phase / turn detection (V4 Feature 1). The detector itself is
        # worker-thread-only; its result dict is committed under self._lock
        # and published as snapshot["phase"].
        self.phase_detector = phase.PhaseDetector(log=self.log)
        self._phase_state = phase.empty_state()
        self._phase_prev = phase.IDLE
        self._my_turn_baseline = {}   # seat -> {"cards", "split", "optimal"}
        self._pending_discipline = []  # judged a grace period after MY_TURN
        self.discipline = {"checked": 0, "matched": 0}
        self._unknown_since = None    # monotonic ts the UNKNOWN streak began
        self._triage_pending = False
        self._triage_last_ts = 0.0

        # Anchor calibration (V3 E3): solved on the worker (a full-frame
        # template search must never block the Tk thread). State mutated
        # under self._lock; published as snapshot["anchors"].
        self._anchor_state = {"status": "none", "fit": None, "drift": False,
                              "calib_res": None}
        self._anchor_set = {}
        self._anchor_pending = False
        self._anchor_drift_at = 0.0

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
            # Anchor remap (V3 E3): solved on the next worker cycle — a
            # full-frame template search here would hitch the Tk thread.
            self._anchor_state = {"status": "none", "fit": None,
                                  "drift": False, "calib_res": None}
            self._anchor_set = {}
            self._anchor_pending = bool(constants.ANCHORS.get("enabled"))
        self.phase_detector.configure(res)
        if (self._ocr_regions is None and ocr.OCR_AVAILABLE
                and constants.OCR.get("enabled")):
            # Region files are keyed by exact capture resolution; without
            # this line a monitor change silently kills balance/bet OCR.
            self.log(f"OCR regions not calibrated for {res[0]}x{res[1]} — "
                     "balance/bet OCR paused (use OCR Regions to calibrate).",
                     level="WARNING")

    @property
    def ocr_calibrated(self) -> bool:
        """OCR regions exist for the current capture resolution."""
        return self._ocr_regions is not None

    def warm_up(self):
        """Initialize models (network handshake for the hosted API). Call from
        the worker thread before the first cycle so the GUI never blocks."""
        self.provider.players_model()
        self.provider.dealer_model()
        # Spawn the EV worker processes now so the first advice/pre-deal
        # job doesn't pay the process start-up mid-round.
        ev_offload.prewarm("advice", "predeal")

    def health_check_and_refresh(self) -> bool:
        """Refresh model backends after a long idle gap (system sleep/restore).

        Hosted Roboflow sessions do not survive a sleep/wake cycle, so they
        are rebuilt with HEALTH_CHECK_RETRY_COUNT retries and exponential
        backoff; local-only setups are skipped (loaded weights never go
        stale). Runs on the worker thread — the sleeps and the blocking
        network rebuild happen OUTSIDE self._lock, and ModelProvider swaps
        each rebuilt model atomically under its own lock.

        Returns False only when every attempt failed (the burst lasts ~2 s,
        often shorter than Wi-Fi reassociation after a wake) so the caller
        can keep retrying at its own cadence; True when refreshed or when
        there was nothing hosted to refresh.
        """
        if not self.provider.has_hosted():
            return True
        for attempt in range(constants.HEALTH_CHECK_RETRY_COUNT):
            try:
                self.provider.refresh(log=self.log)
            except ModelError as e:
                if attempt + 1 >= constants.HEALTH_CHECK_RETRY_COUNT:
                    with self._lock:
                        self.last_error = f"Model refresh failed: {e}"
                    self.log(f"Model refresh failed after "
                             f"{constants.HEALTH_CHECK_RETRY_COUNT} attempts: {e}",
                             level="ERROR")
                    return False
                delay = constants.HEALTH_CHECK_BACKOFF_BASE * (2 ** attempt)
                self.log(f"Model refresh failed ({e}); retrying in {delay:.1f}s",
                         level="WARNING")
                time.sleep(delay)
            else:
                with self._lock:
                    self._model_refresh_ts = time.time()
                    self.last_error = None
                self.log(f"Model backends refreshed "
                         f"({self.provider.backend_name}).")
                return True
        return False

    # ------------------------------------------------------------ main cycle

    def run_cycle(self) -> str:
        """One capture+detect pass. Returns activity: dealing|complete|waiting."""
        t0 = time.perf_counter()
        skipped = False
        try:
            frame = self.capture.grab_bgr()
            # OCR runs regardless of the frame diff: a balance/bet text
            # change in a small corner never trips the whole-frame
            # threshold, so a static table would defer reads ~10 s.
            # _maybe_ocr self-throttles to OCR["interval_s"].
            self._maybe_ocr(frame)
            # Phase detection too — a button enabling or a countdown digit
            # is a small-region change below FRAME_DIFF_THRESHOLD.
            self._update_phase(frame)
            # Anchor solve/drift check (V3 E3): self-throttled like OCR.
            self._maybe_anchors(frame)
            if self._frame_unchanged(frame):
                skipped = True
                if self._empty_frames > 0:
                    # Unchanged since an empty frame == still empty. The dealer
                    # area keeps its last observed state too: extend its empty
                    # run only if one was already going (a card the rank model
                    # saw on that frame is still there on an unchanged frame).
                    with self._lock:
                        if self._dealer_empty_frames > 0:
                            self._dealer_empty_frames += 1
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
            if (activity == "complete" and self.dealer_locked
                    and self._empty_frames == 0):
                extras = [c["rank"] for c in self.dealer_extras]
                all_bust = all(
                    cards.hand_value([c["name"] for c in s.cards]) > 21
                    for s in self.seats if s.cards)
                if (not settlement.dealer_final(self.dealer_card, extras)[2]
                        and not all_bust):
                    # The dealer is still drawing: sample at the fast
                    # "dealing" cadence — playout draws land seconds apart
                    # and the slow "complete" pace missed cards. All-bust
                    # rounds skip this (the dealer doesn't draw then), and
                    # a clearing table (_empty_frames) drops back to the
                    # slow pace even if a missed draw left the hand
                    # incomplete. (Only the pacing/status string changes;
                    # _maybe_auto_new_round keeps using _activity().)
                    activity = "dealing"
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
        cc_checked = True  # the rank model reports the cutting card every frame
        if constants.DEALER_USE_PLAYER_MODEL:
            # The cutting card only exists in the rank model — poll it cheaply
            # every Nth cycle, but EVERY cycle once a sighting is pending so
            # the confirmation run is not stretched across poll gaps.
            self._cc_cycle += 1
            cc_checked = False
            if (not self.cutting_card_seen
                    and (self._pending_cutting_card
                         or self._cc_cycle % constants.CUTTING_CARD_CHECK_EVERY == 0)):
                try:
                    cc_preds = self.provider.dealer_model().predict(
                        crop, constants.PREDICTION_CONFIDENCE_DEALER,
                        constants.PREDICTION_OVERLAP_DEALER)
                    dealer_preds = dealer_preds + [
                        p for p in cc_preds if p["class"] == CUTTING_CARD_CLASS]
                    cc_checked = True
                except ModelError:
                    pass
        self._metrics["inference_ms"] = (time.perf_counter() - t0) * 1000.0
        self.last_error = None

        with self._lock:
            self._process_dealer(dealer_preds, cc_checked=cc_checked)
            self._process_players(player_preds)
            self._maybe_auto_new_round(player_preds)

    # --------------------------------------------------------------- dealer

    def _process_dealer(self, predictions, cc_checked=True):
        """cc_checked: whether this frame's predictions could have contained
        the cutting card (False on suit-mode cycles that skipped the poll)."""
        cards_seen = []
        cc_keys = set()
        for p in predictions:
            if p["class"] == CUTTING_CARD_CLASS:
                self._note_cutting_card(p, cc_keys)
                continue
            rank = DEALER_CLASS_MAP.get(p["class"]) or PLAYER_CLASS_MAP.get(p["class"])
            if rank is not None:  # full "King of Hearts" names in suit mode
                cards_seen.append(p | {"rank": rank})
        if cc_checked:
            # A checked frame without the sighting breaks the confirmation run
            # (same decay rule as pending dealer hits).
            for key in [k for k in self._pending_cutting_card if k not in cc_keys]:
                del self._pending_cutting_card[key]
        # The auto-reset gate wants the dealer area provably clear, not just
        # the player model gone quiet (see _maybe_auto_new_round).
        if cards_seen:
            self._dealer_empty_frames = 0
        else:
            self._dealer_empty_frames += 1

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

    def _note_cutting_card(self, pred, keys_seen):
        """One cutting-card sighting proves nothing — require
        CUTTING_CARD_CONFIRM_FRAMES consecutive checked-frame sightings at the
        same quantized position before latching the reshuffle flag."""
        if self.cutting_card_seen:
            return
        key = (round(pred["cx"] / 50.0), round(pred["cy"] / 50.0))
        if key in keys_seen:
            return  # overlapping boxes on one frame are still one sighting
        keys_seen.add(key)
        count = self._pending_cutting_card.get(key, 0) + 1
        if count >= constants.CUTTING_CARD_CONFIRM_FRAMES:
            self.cutting_card_seen = True
            self._pending_cutting_card.clear()
            self.log("Cutting card confirmed — the shoe will be reshuffled soon.",
                     level="WARNING")
        else:
            self._pending_cutting_card[key] = count

    def _track_dealer_playout(self, cards_seen):
        """Count the dealer's hole/hit cards after the up-card locks. Same
        machinery as player hits — position dedupe + multi-cycle
        confirmation — but with MISS TOLERANCE: the dealer's hand briefly
        occludes cards mid-draw, and a hard consecutive-sighting rule made
        fast playouts lose cards. A pending draw survives
        DEALER_PENDING_MISS_TOLERANCE missed cycles before it is dropped."""
        limit_same = constants.SAME_CARD_DISTANCE_PX * self._dist_scale
        seen_pending = set()
        for p in cards_seen:
            if self._matches_dealer_card(p, limit_same):
                continue
            key = (p["rank"], round(p["cx"] / 50.0), round(p["cy"] / 50.0))
            seen_pending.add(key)
            hits, _ = self._pending_dealer.get(key, (0, 0))
            hits += 1
            if hits >= constants.EXTRA_CARD_CONFIRM_CYCLES:
                self.dealer_extras.append({"rank": p["rank"], "cx": p["cx"], "cy": p["cy"]})
                self.counter.count_card(p["rank"])
                self._pending_dealer.pop(key, None)
                self.log(f"Dealer draws: {p['rank']}")
            else:
                self._pending_dealer[key] = (hits, 0)  # a sighting resets misses
        for key in [k for k in self._pending_dealer if k not in seen_pending]:
            hits, misses = self._pending_dealer[key]
            if misses + 1 > constants.DEALER_PENDING_MISS_TOLERANCE:
                del self._pending_dealer[key]
            else:
                self._pending_dealer[key] = (hits, misses + 1)

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
            if c["cx"] is None:
                # Manually added draw without a screen position: a detection
                # of the same card IS this card — adopt the position so
                # normal dedup applies from here (mirrors seat cards). With
                # suits on both sides the names must match EXACTLY: a
                # rank-only comparison would swallow a genuinely different
                # same-rank draw (5♥ added manually eats a detected 5♠).
                pred_name, c_name = str(pred["rank"]), str(c["rank"])
                if " of " in pred_name and " of " in c_name:
                    same = pred_name == c_name
                else:
                    same = cards.rank_of(pred_name) == cards.rank_of(c_name)
                if same:
                    c["cx"], c["cy"] = pred["cx"], pred["cy"]
                    return True
                continue
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
            ev_offload.run("advice", ev_engine.advise,
                           ["2 of Hearts", "3 of Clubs"], dealer_rank, per_rank,
                           self.counter.deck_count, ev_engine.current_rules())
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
                self._io_pool.submit(self.training.save_sample, self._last_frame,
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

    def _capture_round_stakes(self):
        """Freeze the staked side bets the moment the round's first card
        lands. Called under self._lock from every card-adding path."""
        if self._round_stakes is None:
            self._round_stakes = {
                key: float(cfg.get("stake") or 0.0)
                for key, cfg in constants.SIDE_BETS.items()
                if cfg.get("enabled") and float(cfg.get("stake") or 0.0) > 0}

    def _lock_card(self, seat, pred):
        self._capture_round_stakes()
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
            self._io_pool.submit(self.training.save_sample, self._last_frame,
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
        frame "non-empty" and blocks a premature reset. The dealer area must
        additionally have been clear (per the dealer model, tracked in
        _process_dealer) for the same EMPTY_FRAMES_FOR_RESET stretch. No extra
        inference — and therefore no network call — happens here; this runs
        under _lock.
        """
        if player_preds:
            self._empty_frames = 0
            self._reset_delay_logged = False
            return
        if self._activity() != "complete":
            return
        self._empty_frames += 1
        if self._empty_frames < constants.EMPTY_FRAMES_FOR_RESET:
            return
        if self._dealer_empty_frames < constants.EMPTY_FRAMES_FOR_RESET:
            # The dealer model still sees cards the player model missed —
            # likely a stream hiccup, not a cleared table. Hold the round.
            if not self._reset_delay_logged:
                self._reset_delay_logged = True
                self.log("Auto round reset delayed — dealer area still occupied "
                         f"after {self._empty_frames} empty player frames.")
            return
        self._empty_frames = 0
        self._reset_delay_logged = False
        occupied = ", ".join(f"P{s.index + 1}×{len(s.cards)}"
                             for s in self.seats if s.cards)
        dealer_state = (f"{self.dealer_card} +{len(self.dealer_extras)} draws"
                        if self.dealer_card else "none")
        self.log(f"Table cleared — auto-starting round {self.round_number + 1} "
                 f"(round {self.round_number} had seats [{occupied}], "
                 f"dealer {dealer_state}).", level="WARNING")
        self._reset_round_state()

    def _reset_round_state(self):
        # Settle and persist the round that just ended (and the shoe state, so
        # a restart mid-shoe doesn't lose the count) before wiping the table.
        snap = self.get_snapshot()
        had_activity = bool(snap and (snap["dealer"]["card"]
                                      or any(s["cards"] for s in snap["seats"])))
        if had_activity:
            settle = self._settle_round(snap)
            snap = {**snap, "settlement": settle, "bet_placed": self.bet_placed}
            if self.store is not None:
                self._io_pool.submit(
                    self._persist_round_job, snap, self.counter.get_state(),
                    self.round_number, self.cutting_card_seen,
                    self._shoe_paytable_hash)
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
        with self._advice_lock:
            self._ev_error_keys.clear()
        # Pending discipline judgments belong to the round that just ended;
        # judging them against the next round's cards would be noise.
        self._pending_discipline = []
        self._my_turn_baseline = {}
        if self._bet_was_capped:
            self._bet_was_capped = False
            # Only count rounds where cards were actually dealt — otherwise
            # idle new-round ticks with a capped config inflate the telemetry.
            if had_activity:
                self.bet_capped_rounds += 1
                self.log(f"Bet capped at table max "
                         f"({self.bet_capped_rounds}× this session)", level="WARNING")
        if self._settings_dirty:
            # OCR bankroll syncs persist once per round, not per read.
            self._settings_dirty = False
            self._io_pool.submit(self._save_settings_job)
        self._round_stakes = None  # next round freezes at its first card
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
            self._pending_cutting_card.clear()
            self._reshuffle_dismissed = False  # next confirmation shows a fresh badge
            # A fresh shoe starts clean under whatever paytables now apply.
            self._shoe_paytable_hash = settlement.paytable_hash()
            self.paytable_changed_midshoe = False
            # Cards already on the table belong to the pre-reset history;
            # detach them so later corrections can't drive the fresh count negative.
            for seat in self.seats:
                for card in seat.cards:
                    card["counted"] = False
            self._dealer_counted = False
        if self.store is not None:
            self._io_pool.submit(self._save_state_job)
        self.log("Shoe counts reset.")
        self.publish_snapshot()

    def dismiss_reshuffle_badge(self):
        """Hide the reshuffle badge for the rest of this shoe. The underlying
        cutting_card_seen flag stays set (persistence + end-of-round reminder)."""
        with self._lock:
            self._reshuffle_dismissed = True
        self.publish_snapshot()

    def replace_card(self, seat_idx, slot, card_name, expected_round=None,
                     hand_index=None):
        """Manual correction from the UI. card_name=None removes the card.

        `expected_round` is the round number the user was looking at when the
        picker opened; if the round advanced meanwhile (auto reset), the
        correction targets a hand that no longer exists and is discarded.

        `hand_index` (0 or 1) routes an ADDED card to that hand of a split
        seat; None (or any out-of-range value) keeps the automatic
        nearest-hand routing. Replacing keeps the card's existing hand tag.
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
                        self._io_pool.submit(
                            self.training.save_sample, self._last_frame,
                            old["cx"], old["cy"], card_name, "correction")
            elif card_name is not None and len(seat.cards) < constants.MAX_CARDS_PER_SEAT:
                self._capture_round_stakes()
                if seat.split and hand_index in (0, 1):
                    hand = hand_index
                else:
                    hand = self._assign_hand(seat, None) if seat.split else 0
                seat.cards.append({"name": card_name, "confidence": 1.0,
                                   "cx": None, "cy": None, "manual": True,
                                   "counted": True, "hand": hand})
                self.counter.count_card(card_name)
                tag = f" (hand {hand + 1})" if seat.split else ""
                self.log(f"P{seat_idx + 1} card {len(seat.cards)}{tag} "
                         f"added: {card_name}.")
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

    def add_dealer_extra(self, card_name, expected_round=None):
        """Manually record a dealer draw the detector missed. Position-less
        like manual seat cards; a later detection of the same card adopts
        the screen position (see _matches_dealer_card). `expected_round`
        guards a picker left open across an auto round reset — a stale pick
        must not plant a phantom draw in the NEXT round's dealer hand."""
        with self._lock:
            if expected_round is not None and expected_round != self.round_number:
                self.log("Correction discarded — the round changed while the picker was open.")
                return
            if not self.dealer_locked or not card_name:
                return
            self.dealer_extras.append({"rank": card_name, "cx": None, "cy": None})
            self.counter.count_card(card_name)
        self.log(f"Dealer draw added manually: {card_name}.")
        self.publish_snapshot()

    def replace_dealer_extra(self, idx, card_name, expected_round=None):
        """Correct a tracked dealer draw (card_name=None removes it)."""
        with self._lock:
            if expected_round is not None and expected_round != self.round_number:
                self.log("Correction discarded — the round changed while the picker was open.")
                return
            if not 0 <= idx < len(self.dealer_extras):
                return
            old = self.dealer_extras[idx]
            self.counter.uncount_card(old["rank"])
            if card_name is None:
                self.dealer_extras.pop(idx)
                self.log(f"Dealer draw {old['rank']} removed.")
            else:
                self.dealer_extras[idx] = {**old, "rank": card_name}
                self.counter.count_card(card_name)
                self.log(f"Dealer draw corrected to {card_name}.")
        self.publish_snapshot()

    def set_side_bet_stake(self, key, eur):
        """EUR the user actually places on this side bet per owned seat
        (0 = not playing it). Settlement books stake x paytable."""
        cfg = constants.SIDE_BETS.get(key)
        if cfg is None:
            return
        eur = max(0.0, float(eur))
        with self._lock:
            if eur == float(cfg.get("stake") or 0.0):
                return
            cfg["stake"] = eur
        self.log(f"{cfg.get('label', key)} stake set to €{eur:g}.")
        self.persist_settings_async()
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
            if self._round_stakes and self.my_seats:
                # Decided side bets exist even in refused rounds (a Perfect
                # Pair needs no dealer hand) — booking only on settled
                # rounds keeps the P&L semantics whole, but the user must
                # know money went unbooked.
                self.log("Staked side bets not booked — the round itself "
                         "did not settle; adjust the bankroll manually if "
                         "they paid.", level="WARNING")
            return None

        parts = [f"P{s['index'] + 1} {s['net_units']:+g}u" for s in settle["seats"]]
        bj = " BJ" if settle["dealer_bj"] else ""
        self.log(f"Round {self.round_number} settled (dealer {settle['dealer_total']}{bj}): "
                 + ", ".join(parts))

        # Book-play tally per seat (Bet Behind quality signal).
        try:
            verdicts = seat_quality.score_settled_round(
                settle, snap["seats"], snap["dealer"]["card"], self.strategy)
            for idx, hand_verdicts in verdicts.items():
                tally = self.seat_play[idx]
                tally["n"] += len(hand_verdicts)
                tally["book"] += sum(hand_verdicts)
        except Exception as e:
            self.log(f"Seat-quality error: {type(e).__name__}: {e}")

        mine = [s for s in settle["seats"] if s["index"] in self.my_seats]
        if mine:
            units = sum(s["net_units"] for s in mine)
            eur = units * self.bet_placed
            side_eur, side_detail = self._settle_side_bets(snap)
            settle["side_bets"] = side_detail
            # Cross-check against a recent OCR'd result banner (your result).
            banner = self._ocr_last.get("result")
            if banner and time.time() - self._ocr_last.get("ts", 0) < 30:
                expected = 1 if units > 0 else (-1 if units < 0 else 0)
                seen = {"win": 1, "blackjack": 1, "lose": -1, "push": 0}[banner]
                if expected != seen:
                    self.log(f"⚠ Result banner says '{banner}' but settlement "
                             f"computed {units:+g}u — check for a misread card.")
            self.session_pnl["units"] += units
            self.session_pnl["eur"] += eur + side_eur
            self.session_pnl["side_eur"] += side_eur
            self.session_pnl["rounds"] += 1
            settle["my_units"] = units
            settle["my_eur"] = eur
            settle["my_side_eur"] = side_eur
            total_eur = eur + side_eur
            if constants.BETTING.get("auto_bankroll") and total_eur:
                constants.BETTING["bankroll"] = max(0.0, constants.BETTING["bankroll"] + total_eur)
                self.log(f"Bankroll {total_eur:+.2f} EUR -> {constants.BETTING['bankroll']:g} "
                         "(auto-settled)")
                self._io_pool.submit(self._save_settings_job)
        return settle

    def _settle_side_bets(self, snap):
        """(eur_delta, per-seat detail) for the user's staked side bets on
        owned seats — at the stakes FROZEN when the round's first card
        landed, not the live entry values (bets close at the deal). Same
        honesty rule as the main settlement: an outcome the detections
        can't decide is skipped (and logged), never guessed."""
        staked = self._round_stakes if self._round_stakes is not None else {}
        if not staked:
            return 0.0, {}
        dealer_up = snap["dealer"]["card"]
        extras = snap["dealer"].get("extras", [])
        total = 0.0
        detail = {}
        for seat in snap["seats"]:
            if seat["index"] not in self.my_seats or len(seat["cards"]) < 2:
                continue
            outcomes = sidebet_outcomes.seat_outcomes(
                seat["cards"][:2], dealer_up, extras)
            entries = {}
            for key, stake in staked.items():
                outcome = outcomes.get(key)
                if outcome is None:
                    # 3-card bets with no detected up-card never resolve —
                    # say so instead of silently dropping staked money.
                    self.log(f"P{seat['index'] + 1} "
                             f"{constants.SIDE_BETS[key].get('label', key)} "
                             "not settled — no dealer up-card detected.",
                             level="WARNING")
                    continue
                if outcome["result"] == "unknown":
                    self.log(f"P{seat['index'] + 1} "
                             f"{constants.SIDE_BETS[key].get('label', key)} "
                             f"not settled — {outcome.get('reason', 'undecidable')}.",
                             level="WARNING")
                    continue
                eur = (stake * outcome["pays"] if outcome["result"] == "win"
                       else -stake)
                total += eur
                entries[key] = {**outcome, "stake": stake, "eur": eur}
                if outcome["result"] == "win":
                    self.log(f"P{seat['index'] + 1} "
                             f"{constants.SIDE_BETS[key].get('label', key)}: "
                             f"{sidebet_outcomes.tier_label(outcome['tier'])} "
                             f"pays {outcome['pays']:g}:1 -> +€{eur:.2f}")
            if entries:
                detail[seat["index"]] = entries
        if total:
            self.log(f"Side bets settled: {total:+.2f} EUR")
        return total, detail

    @staticmethod
    def _save_settings_job():
        try:
            from ..common import settings
            settings.save()
        except Exception:
            pass

    def persist_settings_async(self):
        """Write settings.json off the calling thread (GUI commits use this
        so a slow disk never hitches the Tk loop)."""
        self._io_pool.submit(self._save_settings_job)

    def submit_io(self, fn, *args):
        """Run a small disk/DB job on the single io worker — external
        SQLite writers (the executor's audit trail) must share this one
        thread to keep the single-writer ordering contract."""
        return self._io_pool.submit(fn, *args)

    def _persist_round_job(self, snap, counter_state, round_number, cutting,
                           paytable_hash):
        # The hash is captured under _lock at round end like every other
        # argument — this job can queue behind slow EV work, and a paytable
        # saved in the settings dialog meanwhile must not restamp a round
        # that was settled under the old payouts.
        try:
            self.store.record_round(snap, paytable_hash=paytable_hash)
            self.store.save_shoe_state(counter_state, round_number, cutting)
        except Exception as e:
            self._log_ev_error(("persist", round_number),
                               f"Session store error persisting round "
                               f"{round_number}: {e!r}")

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
        eur = max(0.0, float(eur))
        with self._lock:
            if eur == self.bet_placed:
                return  # no-op commits (FocusOut traversals) skip the rebuild
            self.bet_placed = eur
        self.publish_snapshot()

    @staticmethod
    def _settings_sig():
        """Signature of the settings the advice caches are keyed on. The
        (RULES, DECK_COUNT) part feeds the EV/pre-deal caches, the side-bet
        part (enabled flags + paytables) only the side-bet EVs. Paytable keys
        stringify because bust_it uses ints where the others use strs."""
        side = tuple(sorted(
            (key, bool(cfg.get("enabled")),
             tuple(sorted((str(k), v) for k, v in (cfg.get("paytable") or {}).items())))
            for key, cfg in constants.SIDE_BETS.items()))
        return (tuple(sorted(constants.RULES.items())), constants.DECK_COUNT, side)

    def refresh_settings(self):
        """Re-read runtime settings (rules, deck count, side bets) and drop
        only the advice caches keyed on what actually changed — a display
        preference must not discard a finished ~15 s pre-deal sweep. Called
        after the settings dialog saves; the closing publish_snapshot
        resubmits advice jobs for live hands (the proactive recompute)."""
        old_sig, new_sig = self._rules_sig, self._settings_sig()
        new_hash = settlement.paytable_hash()
        with self._lock:
            self.counter.deck_count = constants.DECK_COUNT
            self._rules_sig = new_sig
            if new_hash != self._shoe_paytable_hash:
                # The shoe was dealt under other payouts: every EV recorded
                # earlier this shoe is suspect. Sticky until reset_shoe().
                self._shoe_paytable_hash = new_hash
                self.paytable_changed_midshoe = True
                self.log(f"Paytables changed mid-shoe (round {self.round_number})"
                         " — EVs recorded earlier this shoe used the old"
                         " payouts.", level="WARNING")
        with self._advice_lock:
            if new_sig[:2] != old_sig[:2]:   # RULES or DECK_COUNT changed
                self._advice_cache.clear()
                self._predeal = {"edge": None, "sig": None}
            if new_sig[2] != old_sig[2]:     # side-bet set or paytables changed
                self._sidebet_result = ([], None)
        self.log("Settings applied — table rules and paytables refreshed.")
        self.publish_snapshot()

    # ------------------------------------------------------------- snapshot

    def _log_ev_error(self, dedup_key, message):
        """ERROR-log an advice-path failure, deduped per key per round (the
        set clears in _reset_round_state) so a flapping job can't flood the
        log while every distinct failure still leaves a trace."""
        with self._advice_lock:
            if dedup_key in self._ev_error_keys:
                return
            self._ev_error_keys.add(dedup_key)
        self.log(message, level="ERROR")

    @staticmethod
    def _book_fallback(book, reason):
        """The book play dressed as the seat's optimal line while the EV job
        is delayed or failed; '' when the seat has no book advice either."""
        book_text, book_color = book
        if not book_text or book_text == "-":
            return "", constants.ACTION_COLORS["-"]
        return (f"{book_text} (book — {reason})",
                book_color or constants.ACTION_COLORS["-"])

    def _optimal_advice(self, names, dealer_rank, per_rank, csv_action,
                        book=("", None), post_split=False):
        """Exact composition-dependent advice for a seat: (text, color, code)
        — '' / None code when no decision applies. The code is the raw
        best-EV action letter ('S', 'H', ...) once the exact result is in,
        None for placeholders and book fallbacks; it rides into the recorded
        seats JSON next to the CSV book code.

        Non-blocking: returns the cached EV result when this exact
        (hand, dealer, composition, rules) was already computed; otherwise
        submits a job to the advice thread and returns a placeholder — the
        finished job publishes a fresh snapshot with the real line. Heavy EV
        recursion therefore never runs under self._lock or on the Tk thread.
        `book` is the seat's already-computed basic-strategy (text, color):
        a job pending past EV_ADVICE_TIMEOUT_S falls back to it instead of a
        stuck placeholder, an errored job (None in the cache) likewise — and
        a late-landing result still upgrades to the normal Optimal line."""
        hand = [c for c in names if c and c != "-"]
        if len(hand) < 2 or not dealer_rank or cards.hand_value(hand) >= 21:
            return "", constants.ACTION_COLORS["-"], None
        key = (tuple(sorted(hand)), dealer_rank, post_split,
               tuple(sorted(per_rank.items())), self.counter.deck_count,
               tuple(sorted(constants.RULES.items())))
        with self._advice_lock:
            if key not in self._advice_cache:
                submitted = self._advice_pending.get(key)
                if submitted is None:
                    self._advice_pending[key] = time.monotonic()
                    self._advice_pool.submit(self._advice_job, key, list(hand),
                                             dealer_rank, dict(per_rank), post_split)
                elif time.monotonic() - submitted > constants.EV_ADVICE_TIMEOUT_S:
                    return (*self._book_fallback(book, "EV delayed"), None)
                return "Optimal: …", constants.ACTION_COLORS["-"], None
            result = self._advice_cache[key]
        if result is None:
            return (*self._book_fallback(book, "EV failed"), None)
        best = result["best"]
        text = f"Optimal: {_ACTION_NAMES[best]} ({result['evs'][best]:+.3f})"
        if self._csv_primary(csv_action, result["evs"]) not in (None, best):
            text += " ≠ book"
        return text, constants.ACTION_COLORS[_ACTION_COLOR_KEYS[best]], best

    def _advice_job(self, key, hand, dealer_rank, per_rank, post_split=False):
        try:
            # Computed in a persistent worker process: the EV recursion is
            # pure Python and would otherwise starve the Tk thread of the
            # GIL. Rules pass explicitly — the child has default constants.
            result = ev_offload.run("advice", ev_engine.advise, hand,
                                    dealer_rank, per_rank,
                                    self.counter.deck_count,
                                    ev_engine.current_rules(), post_split)
        except Exception as e:
            result = None
            self._log_ev_error(key, f"EV engine error for {hand} vs "
                                    f"{dealer_rank}: {e!r}")
        with self._advice_lock:
            if len(self._advice_cache) > 1024:
                self._advice_cache.clear()
            self._advice_cache[key] = result
            self._advice_pending.pop(key, None)
        self.publish_snapshot()

    def flush_advice(self, timeout=15.0) -> bool:
        """Block until all queued advice/side-bet/persistence jobs have
        landed and the snapshot reflects them. For tests and debugging only."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            barrier = self._advice_pool.submit(lambda: None)
            barrier.result(max(0.1, deadline - time.monotonic()))
            io_barrier = self._io_pool.submit(lambda: None)
            io_barrier.result(max(0.1, deadline - time.monotonic()))
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
            self._log_ev_error(("insurance", dealer_rank),
                               f"EV engine error for insurance vs dealer "
                               f"{dealer_rank}: {e!r}")
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
            # Sweeps run between rounds only ("waiting"): the edge prices
            # the NEXT round's bet, so re-sweeping after every dealt card
            # is pure CPU waste — the pre-round edge stays valid mid-round.
            # Live _activity(), not _last_activity: the cached value only
            # updates while the detection worker runs, and a stop mid-round
            # would close the gate forever for manual play. (Caller holds
            # self._lock — publish_snapshot — which _activity requires.)
            if (self._predeal["sig"] != sig and not self._predeal_pending
                    and self._activity() == "waiting"):
                self._predeal_pending = True
                self._predeal_pool.submit(self._predeal_job, sig,
                                          dict(count["per_rank"]))
            return self._predeal["edge"]

    def _predeal_job(self, sig, per_rank):
        try:
            comp = ev_engine.comp_from_per_rank(per_rank, self.counter.deck_count)
            # The sweep is seconds of pure Python — fan weight-balanced
            # up-card/hand slices across the predeal worker processes
            # (V3 E2) so the exact bet call lands inside the betting
            # window instead of one round late, and nothing contends with
            # the Tk thread's GIL. Slicing trades away some memo sharing;
            # the worker count must beat that, hence the sized pool.
            if sum(comp) >= 52:
                rules = ev_engine.current_rules()
                jobs = ev_engine.predeal_jobs(
                    comp, max(1, int(constants.PREDEAL_WORKERS)))
                parts = ev_offload.run_many(
                    "predeal", ev_engine.predeal_ev_upcards,
                    [(comp, rules, upcards, slice_of)
                     for upcards, slice_of in jobs])
                edge = sum(parts)
            else:
                edge = None
        except Exception as e:
            edge = None
            self._log_ev_error(("predeal", sig),
                               f"Pre-deal EV error ({sum(per_rank.values())} "
                               f"cards seen): {e!r}")
        with self._advice_lock:
            self._predeal = {"edge": edge, "sig": sig}
            self._predeal_pending = False
        if edge is not None:
            self.log(f"Exact pre-deal edge: {edge:+.3%}")
        self.publish_snapshot()

    # --------------------------------------------------- anchor calibration

    def request_anchor_resolve(self):
        """Re-solve the anchor transform on the next worker cycle (the
        GUI's one-click re-anchor; set_monitor queues the same thing)."""
        with self._lock:
            self._anchor_pending = True
            self._anchor_state["drift"] = False
        self.publish_snapshot()

    def _maybe_anchors(self, frame):
        """Worker-side anchor work: a queued solve, else the throttled
        drift re-check of an active fit."""
        if not constants.ANCHORS.get("enabled"):
            return
        if self._anchor_pending:
            self._anchor_pending = False
            self._resolve_anchors(frame)
            return
        with self._lock:
            fit = self._anchor_state["fit"]
            drifted = self._anchor_state["drift"]
            anchor_set = self._anchor_set
        if fit is None or drifted or not anchor_set:
            return
        now = time.monotonic()
        if now < self._anchor_drift_at:
            return
        self._anchor_drift_at = now + float(constants.ANCHORS["drift_check_s"])
        drift = anchors_mod.check_drift(frame, anchor_set, fit)
        if not drift["ok"]:
            with self._lock:
                self._anchor_state["drift"] = True
            shift = drift["max_shift"]
            self.log("Anchor drift detected (worst score "
                     f"{drift['worst_score']:.2f}"
                     + (f", shift {shift:.0f}px" if shift is not None else "")
                     + ") — calibration may be misaligned; use Re-anchor.",
                     level="WARNING")
            self.publish_snapshot()

    def _resolve_anchors(self, frame):
        """Solve the calibrated->live transform and remap every calibrated
        geometry kind. Fail-disabled: no fit keeps the current geometry."""
        res = self.capture.resolution
        if res is None:
            return
        calib_res = anchors_mod.calibrated_resolution(res)
        anchor_set = anchors_mod.load_anchors(calib_res) if calib_res else {}
        if len(anchor_set) < int(constants.ANCHORS["min_anchors"]):
            with self._lock:
                self._anchor_state = {"status": "none", "fit": None,
                                      "drift": False, "calib_res": None}
                self._anchor_set = {}
            return
        fit = anchors_mod.solve(frame, anchor_set, calib_res)
        if fit is None:
            with self._lock:
                self._anchor_state = {
                    "status": "failed", "fit": None, "drift": False,
                    "calib_res": f"{calib_res[0]}x{calib_res[1]}"}
                self._anchor_set = {}
            self.log("Anchor solve failed (anchors not found on this "
                     "screen) — keeping the current calibration; geometry "
                     "may be wrong here.", level="WARNING")
            self.publish_snapshot()
            return
        regions_payload = load_custom_regions(calib_res)
        ocr_payload = ocr.load_regions(calib_res)
        controls = phase.load_controls(calib_res)
        t_regions = anchors_mod.transform_regions(regions_payload, fit)
        t_ocr = anchors_mod.transform_ocr(ocr_payload, fit)
        with self._lock:
            if t_regions:
                self.regions = [Polygon(p) for p in t_regions["players"]]
                self._dealer_rect = tuple(t_regions["dealer"])
                self._dist_scale = (calib_res[0]
                                    / constants.BASE_RESOLUTION[0]
                                    ) * fit["scale"]
            if t_ocr:
                self._ocr_regions = t_ocr
            self._anchor_set = anchor_set
            self._anchor_state = {
                "status": "active", "fit": fit, "drift": False,
                "calib_res": f"{calib_res[0]}x{calib_res[1]}"}
        self._anchor_drift_at = (time.monotonic()
                                 + float(constants.ANCHORS["drift_check_s"]))
        if controls:
            self.phase_detector.set_controls(
                anchors_mod.transform_controls(controls, fit))
        mapped = [name for name, ok in (("regions", bool(t_regions)),
                                        ("ocr", bool(t_ocr)),
                                        ("controls", bool(controls))) if ok]
        self.log(f"Anchors solved: scale {fit['scale']:.3f}, offset "
                 f"({fit['dx']:+.0f}, {fit['dy']:+.0f}), {fit['matched']} "
                 f"anchor(s), score {fit['score']:.2f} — mapped "
                 f"{', '.join(mapped) or 'nothing'} from "
                 f"{calib_res[0]}x{calib_res[1]}.")
        self.publish_snapshot()

    # ----------------------------------------------------- phase detection

    @property
    def current_phase(self) -> str:
        """The latest detected game phase (for the controller's pacing)."""
        return self._phase_state.get("phase", phase.IDLE)

    def reload_phase_controls(self):
        """Re-read the active profile's control templates after a
        Capture Controls save or a profile switch."""
        if self.capture.monitor is not None:
            self.phase_detector.configure(self.capture.resolution)

    def _update_phase(self, frame):
        """Per-cycle game-phase step (V4 Feature 1). The template matching
        is cheap C work and runs OUTSIDE self._lock; only the signal
        gathering and the state commit take it. Runs before the frame-diff
        skip for the same reason OCR does."""
        if not constants.PHASE.get("enabled"):
            # Disabled mid-run: clear the latched state, or the HUD keeps a
            # stale YOUR TURN banner and the controller stays pinned at the
            # fast cadence forever.
            if self._phase_state.get("phase") != phase.IDLE:
                with self._lock:
                    self._phase_state = phase.empty_state()
                    self._phase_prev = phase.IDLE
                    self._pending_discipline = []
                    self._my_turn_baseline = {}
                self.phase_detector.reset()
            return
        with self._lock:
            cards_on_table = sum(len(s.cards) for s in self.seats)
            if self.dealer_card:
                cards_on_table += 1 + len(self.dealer_extras)
            undecided = [s.index for s in self.seats
                         if s.index in self.my_seats and s.cards
                         and self._seat_undecided(s)]
            signals = {
                "activity": self._activity(),
                "cards_on_table": cards_on_table,
                "undecided_mine": undecided,
                "my_seats": sorted(self.my_seats),
                "ocr": dict(self._ocr_last),
            }
        state = self.phase_detector.update(frame, signals)
        with self._lock:
            prev, new = self._phase_prev, state["phase"]
            # A triage label sticks while the screen stays unidentified.
            state["triage"] = (self._phase_state.get("triage")
                               if new == phase.UNKNOWN else None)
            self._phase_state = state
            if new != prev:
                self._phase_prev = new
                self._on_phase_transition(prev, new, state)
            self._check_discipline()
            if (new == phase.BETTING_OPEN and self._activity() == "complete"
                    and self._empty_frames >= 1
                    and self._dealer_empty_frames >= 1):
                # Fast-path settle: the next round's betting banner PLUS at
                # least one provably clear inference pass (player AND dealer
                # area empty). The banner alone is not enough — it routinely
                # appears while the previous round's cards are still being
                # swept, and resetting then would re-lock and RE-COUNT those
                # cards into the shoe (a money-correctness bug). With the
                # one-frame evidence this settles ~4 frames sooner than the
                # EMPTY_FRAMES_FOR_RESET stretch, never sooner than safe.
                self.log("Betting window open over a cleared table — "
                         "settling the finished round now (phase fast-path).")
                self._empty_frames = 0
                self._dealer_empty_frames = 0
                self._reset_round_state()
        self._maybe_triage(frame, state)

    @staticmethod
    def _seat_undecided(seat) -> bool:
        """Does this seat still have a hand a decision could apply to?
        Split seats are judged PER HAND — the combined card total of both
        hands is meaningless and wrongly excluded split seats from turn
        attribution."""
        if seat.split:
            return any(
                hand and cards.hand_value(hand) < 21
                for hand in ([c["name"] for c in seat.cards
                              if c.get("hand", 0) == h] for h in (0, 1)))
        return cards.hand_value([c["name"] for c in seat.cards]) < 21

    def _on_phase_transition(self, prev, new, state):
        """React to a confirmed phase change. Caller holds self._lock."""
        if new == phase.MY_TURN:
            # Re-entering the turn supersedes any judgment queued by a brief
            # exit flap — judging "you stood" while the buttons are
            # demonstrably live again would be a false accusation.
            self._pending_discipline = []
            # Baseline every owned seat so the observed action (cards grew /
            # stood pat / split) can be judged against the advice that was
            # showing when the buttons went live.
            self._my_turn_baseline = {}
            snap = self.get_snapshot() or {}
            seats_snap = {s["index"]: s for s in snap.get("seats", [])}
            for idx in self.my_seats:
                seat = self.seats[idx]
                if not seat.cards:
                    continue
                entry = seats_snap.get(idx, {})
                code = entry.get("optimal_action")
                if isinstance(code, list):
                    code = next((c for c in code if c), None)
                if not code:
                    book = entry.get("book_action")
                    if isinstance(book, list):
                        book = next((b for b in book if b), None)
                    code = str(book).split("/")[0] if book else None
                self._my_turn_baseline[idx] = {"cards": len(seat.cards),
                                               "split": seat.split,
                                               "optimal": code}
            seat_txt = (f" (P{state['my_seat'] + 1})"
                        if state.get("my_seat") is not None else "")
            self.log(f"YOUR TURN{seat_txt} — action buttons are live.",
                     level="WARNING")
        elif prev == phase.MY_TURN:
            grace = float(constants.PHASE.get("discipline_grace_s", 3.0))
            for idx, base in self._my_turn_baseline.items():
                if base["optimal"] in (None, "", "R"):
                    continue  # nothing judgeable (surrender is unobservable)
                self._pending_discipline.append(
                    {"seat": idx, **base, "round": self.round_number,
                     "deadline": time.monotonic() + grace})
            self._my_turn_baseline = {}
        if new == phase.BETTING_OPEN:
            timer = state.get("timer_s")
            self.log("Betting window open"
                     + (f" — {timer}s left" if timer is not None else "")
                     + ".")

    def _check_discipline(self):
        """Judge observed actions against the advice once their grace
        period lapses (live discipline feedback). Caller holds self._lock."""
        if not self._pending_discipline:
            return
        now = time.monotonic()
        due = [p for p in self._pending_discipline if p["deadline"] <= now]
        if not due:
            return
        self._pending_discipline = [p for p in self._pending_discipline
                                    if p["deadline"] > now]
        for p in due:
            if p.get("round") != self.round_number:
                continue  # the round rolled mid-grace: stale, never judge
            seat = self.seats[p["seat"]]
            if len(seat.cards) < p["cards"]:
                continue  # the round reset mid-grace: nothing to judge
            grew = len(seat.cards) > p["cards"]
            split_now = seat.split and not p["split"]
            optimal = p["optimal"]
            if split_now:
                observed, matched = "split", optimal == "P"
            elif grew:
                # A split is detected via the split flag above; a grown hand
                # against optimal "P" is a deviation, not a match.
                observed, matched = "hit/doubled", optimal in ("H", "D")
            else:
                observed, matched = "stood", optimal == "S"
            self.discipline["checked"] += 1
            if matched:
                self.discipline["matched"] += 1
            else:
                name = _ACTION_NAMES.get(optimal, optimal)
                self.log(f"Discipline: P{p['seat'] + 1} — you {observed}; "
                         f"optimal was {name}.", level="WARNING")

    def _maybe_triage(self, frame, state):
        """Label a persistently UNKNOWN screen (modal / disconnect / lobby)
        via the optional Claude vision assist. Network work goes to the io
        thread; the hot loop never blocks on it."""
        if state["phase"] != phase.UNKNOWN:
            self._unknown_since = None
            return
        from . import vision_assist
        if not vision_assist.available() or not constants.VISION.get("triage"):
            return
        now = time.monotonic()
        if self._unknown_since is None:
            self._unknown_since = now
            return
        if (now - self._unknown_since
                < float(constants.VISION.get("triage_after_s", 15.0))
                or self._triage_pending
                or now - self._triage_last_ts
                < float(constants.VISION.get("triage_min_gap_s", 60.0))):
            return
        self._triage_pending = True
        # Dedicated thread, NOT the io pool: a slow API round-trip must
        # never queue the 1 Hz OCR reads (the phase machine's own input)
        # or the SQLite writes behind it.
        threading.Thread(target=self._triage_job, args=(frame.copy(),),
                         daemon=True, name="vision-triage").start()

    def _triage_job(self, frame):
        from . import vision_assist
        try:
            label = vision_assist.triage(frame)
        except Exception as e:
            label = None
            self._log_ev_error(("triage",), f"Vision triage failed: {e!r}")
        finally:
            # Timestamp BEFORE clearing the pending flag — the other order
            # opens a window where the worker sees pending=False with the
            # old timestamp and double-submits.
            self._triage_last_ts = time.monotonic()
            self._triage_pending = False
        if label:
            with self._lock:
                self._phase_state = {**self._phase_state, "triage": label}
            self.log(f"Screen-state triage: {label}", level="WARNING")
        self.publish_snapshot()

    # ----------------------------------------------------------------- ocr

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
        self._io_pool.submit(self._ocr_job, crops)

    def _ocr_job(self, crops):
        try:
            texts = {}
            for key, img in crops.items():
                texts.update(ocr.read_regions(
                    img, {key: [0, 0, img.shape[1], img.shape[0]]}))
            values = ocr.interpret(texts)
        except Exception as e:
            values = {}
            self._log_ev_error(("ocr", tuple(sorted(crops))),
                               f"OCR error reading {sorted(crops)}: {e!r}")
        finally:
            self._ocr_pending = False
        if not values:
            return
        self._apply_ocr_values(values)

    def _apply_ocr_values(self, values):
        """Apply interpreted OCR readings to the live state (split out of
        _ocr_job so tests can exercise it without an OCR backend)."""
        balance = values.get("balance")
        if balance and constants.OCR.get("sync_bankroll"):
            if balance > 10_000_000:
                # Same bound the GUI enforces on manual entry — an OCR misread
                # must not push the bankroll where its display formatting (and
                # the bet ramp) stops making sense.
                self.log(f"OCR balance €{balance:g} above €10,000,000 — "
                         f"clamped (likely misread).", level="WARNING")
                balance = 10_000_000.0
            if abs(balance - constants.BETTING["bankroll"]) >= 0.01:
                constants.BETTING["bankroll"] = balance
                self.log(f"Bankroll synced from screen: €{balance:g}")
                # Persist once at round end, not per read — payout
                # animations would otherwise rewrite settings.json at 1 Hz.
                self._settings_dirty = True
        bet = values.get("bet")
        if bet and constants.OCR.get("sync_bet"):
            table_max = float(constants.BETTING.get("table_max") or 0)
            clamped = (min(max(bet, 0.0), table_max) if table_max > 0
                       else max(bet, 0.0))
            if clamped != bet:
                # A legit bet can never exceed the table max — this is a
                # misread (8.50 seen as 850), so clamp it, loudly.
                self.log(f"OCR bet €{bet:g} outside [0, €{table_max:g}] — "
                         f"clamped to €{clamped:g}.", level="WARNING")
                bet = clamped
            if bet != self.bet_placed:
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
            self._log_ev_error(("sidebet", sig),
                               f"Side-bet engine error ({sig[0]} cards "
                               f"seen): {e!r}")
        with self._advice_lock:
            self._sidebet_result = (result, sig)
            self._sidebet_pending = False
        self.publish_snapshot()

    def _split_seat_snapshot(self, seat, names, dealer_rank, ev_count, true_count):
        """Per-hand advice for a split seat. The combined lines show both
        hands; the flat card list keeps engine order so the picker still
        addresses cards by slot."""
        hand_lines = {"total": [], "advice": [], "optimal": []}
        book_actions, optimal_actions = [], []  # raw codes, H1/H2 order
        advice_color = optimal_color = constants.ACTION_COLORS["-"]
        for h in (0, 1):
            hand_names = [c["name"] for c in seat.cards if c.get("hand", 0) == h]
            if not hand_names:
                continue
            action, text, color = self.strategy.advice(
                hand_names, dealer_rank, post_split=True)
            if (cards.rank_of(hand_names[0]) == "Ace"
                    and not constants.RULES["hit_split_aces"]):
                # ev_engine.advise(post_split=True) still prices Hit/Double
                # on a split-ace hand — it does not model the one-card-only
                # rule — so an EV line here would assume illegal actions.
                optimal, opt_color, opt_action = ("", constants.ACTION_COLORS["-"],
                                                  None)
            else:
                optimal, opt_color, opt_action = self._optimal_advice(
                    hand_names, dealer_rank, ev_count["per_rank"], action,
                    book=(text, color), post_split=True)
            book_actions.append(action)
            optimal_actions.append(opt_action)
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
            "book_n": self.seat_play[seat.index]["n"],
            "book_pct": (self.seat_play[seat.index]["book"]
                         / self.seat_play[seat.index]["n"]
                         if self.seat_play[seat.index]["n"] else None),
            "can_split": False,
            "total": "  ·  ".join(hand_lines["total"]),
            "advice": "\n".join(hand_lines["advice"]),
            "advice_color": advice_color,
            "book_action": book_actions,
            "optimal": "\n".join(hand_lines["optimal"]),
            "optimal_color": optimal_color,
            "optimal_action": optimal_actions,
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

    def _bet_behind_text(self, edge):
        """Who to bet behind, given the current edge and the seat tallies."""
        if edge <= 0:
            return f"-EV now ({edge:+.2%}) — wait for a positive count"
        candidates = [(i, t["book"] / t["n"], t["n"])
                      for i, t in self.seat_play.items()
                      if t["n"] >= 10 and i not in self.my_seats]
        if not candidates:
            return f"+EV ({edge:+.2%}) — no seat with 10+ scored hands yet"
        idx, pct, n = max(candidates, key=lambda c: c[1])
        if pct < 0.8:
            return f"+EV ({edge:+.2%}) but best seat plays only {pct:.0%} book"
        return f"P{idx + 1} ✓ ({edge:+.2%} edge, {pct:.0%} book over {n} hands)"

    def publish_snapshot(self):
        with self._lock:
            dealer_rank = self.dealer_card
            dealer_extra_ranks = [c["rank"] for c in self.dealer_extras]
            count = self.counter.snapshot()
            ev_count = self._ev_count(count)
            insurance = self._insurance_advice(dealer_rank, ev_count["per_rank"])
            seats = []
            for seat in self.seats:
                names = [c["name"] for c in seat.cards]
                outcomes = sidebet_outcomes.seat_outcomes(
                    names[:2], dealer_rank, dealer_extra_ranks)
                if seat.split:
                    snap_seat = self._split_seat_snapshot(
                        seat, names, dealer_rank, ev_count, count["true"])
                    snap_seat["side_outcomes"] = outcomes
                    seats.append(snap_seat)
                    continue
                action, text, color = self.strategy.advice(names, dealer_rank)
                optimal, optimal_color, optimal_action = self._optimal_advice(
                    names, dealer_rank, ev_count["per_rank"], action,
                    book=(text, color))
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
                    "book_n": self.seat_play[seat.index]["n"],
                    "book_pct": (self.seat_play[seat.index]["book"]
                                 / self.seat_play[seat.index]["n"]
                                 if self.seat_play[seat.index]["n"] else None),
                    "can_split": (len(names) == 2 and cards.is_pair(names)
                                  and cards.hand_value(names) < 21),
                    "total": cards.describe_hand(names),
                    "advice": text,
                    "advice_color": color,
                    "book_action": action,
                    "optimal": optimal,
                    "optimal_color": optimal_color,
                    "optimal_action": optimal_action,
                    "index_advice": index_text,
                    "index_color": index_color,
                    "side_outcomes": outcomes,
                })
            side_bet_items = []
            for item in self._side_bet_evs(ev_count):
                cfg = constants.SIDE_BETS.get(item["key"], {})
                side_bet_items.append({
                    **item,
                    "stake": float(cfg.get("stake") or 0.0),
                    "stake_suggested": betting.side_bet_stake(
                        item.get("ev"), item.get("variance")),
                })
            suggestion = betting.suggest(count["true"],
                                         exact_edge=self._predeal_edge(ev_count))
            if suggestion["capped"]:
                # Latch only — _reset_round_state turns the many snapshots a
                # round publishes into a single bet_capped_rounds tick.
                self._bet_was_capped = True
            snapshot = {
                "seq": 0,
                # Publish time: the executor refuses to fire from a stale
                # snapshot (a hung worker must not leave a frozen MY_TURN
                # that passes every guard forever).
                "ts": time.time(),
                "seats": seats,
                "dealer": {"card": dealer_rank, "locked": self.dealer_locked,
                           "extras": dealer_extra_ranks},
                "insurance": insurance,
                "side_bets": side_bet_items,
                "count": count,
                "bet": suggestion["text"],
                "bet_suggested": suggestion["bet"],
                "bet_sit_out": suggestion["sit_out"],
                "bet_behind": self._bet_behind_text(suggestion["edge"]),
                "edge_exact": self._predeal["edge"],
                "bet_placed": self.bet_placed,
                "bankroll": float(constants.BETTING["bankroll"]),
                "bet_capped_rounds": self.bet_capped_rounds,
                "session_pnl": dict(self.session_pnl),
                "ocr": dict(self._ocr_last) if self._ocr_regions else None,
                "phase": {**self._phase_state,
                          "discipline": dict(self.discipline)},
                "anchors": {
                    "status": self._anchor_state["status"],
                    "drift": self._anchor_state["drift"],
                    "calib_res": self._anchor_state["calib_res"],
                    "scale": (self._anchor_state["fit"] or {}).get("scale"),
                },
                "round": self.round_number,
                "cutting_card_seen": self.cutting_card_seen,
                "reshuffle_badge": (self.cutting_card_seen
                                    and not self._reshuffle_dismissed),
                "paytable_changed_midshoe": self.paytable_changed_midshoe,
                "activity": self._last_activity,
                "metrics": dict(self._metrics),
                "backend": self.provider.backend_name,
                "error": self.last_error,
                "model_refresh_ts": self._model_refresh_ts,
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
