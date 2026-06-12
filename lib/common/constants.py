"""Central configuration: paths, detection settings, game rules, and UI theme."""

import os
from pathlib import Path

VERSION = "3.1"
TITLE = f"Blackjack AI - v{VERSION}"

# ---------------------------------------------------------------------------
# Paths (anchored to the project root so the app works from any CWD)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
CARDS_DIR = ASSETS_DIR / "cards"
CONTROLS_DIR = ASSETS_DIR / "controls"  # captured control templates, per profile
STRATEGY_CSV_PATH = ASSETS_DIR / "strategy.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

CARD_BACK_IMAGE_PATH = CARDS_DIR / "card_back.png"


def card_image_path(card_name: str) -> Path:
    """'Ace of Spades' -> assets/cards/ace_of_spades.png (files are lowercase)."""
    return CARDS_DIR / f"{card_name.replace(' ', '_').lower()}.png"


# ---------------------------------------------------------------------------
# Detection models
# ---------------------------------------------------------------------------
# The API key can be overridden without touching code: set ROBOFLOW_API_KEY.
# ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "WBy7jG6AiiqjzifOfiNH")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "Yp7H8pUXd3U4fW3bJxDs")

PROJECT_ID_PLAYERS = "dey022"
MODEL_VERSION_PLAYERS = 1
PROJECT_ID_DEALER = "carddetection-v1hqz"
MODEL_VERSION_DEALER = 17

PREDICTION_CONFIDENCE_PLAYERS = 70
PREDICTION_OVERLAP_PLAYERS = 100
PREDICTION_CONFIDENCE_DEALER = 55
PREDICTION_OVERLAP_DEALER = 45

# Frames sent to the hosted API are downscaled to this width before upload
# (predictions are scaled back). Big upload-time win, negligible accuracy loss.
API_UPLOAD_MAX_WIDTH = 1280

# LEGACY (kept so persisted settings files keep loading): dealer cards now
# always come from the full-frame player-model pass the seats use — the
# same detections the region preview draws. The old toggle chose a
# dedicated dealer-crop inference (rank model when off) that routinely
# missed cards the preview clearly boxed. The rank model is still polled
# for the cutting card, which only exists there.
DEALER_USE_PLAYER_MODEL = True
CUTTING_CARD_CHECK_EVERY = 5
# Cutting-card sightings on consecutive checked frames (position-quantized,
# like dealer hits) before the reshuffle flag latches — one misread must not
# flag the whole shoe.
CUTTING_CARD_CONFIRM_FRAMES = 3

# Active-learning capture (V2 Feature 6): save labeled crops of corrections,
# sampled confirmed locks, and confirmation-flapping detections to
# output/training_data/ for review and fine-tuning.
TRAINING = {
    "enabled": 1,
    "confirmed_every": 25,   # save every Nth confirmed lock
    "max_files": 5000,
}

# ---------------------------------------------------------------------------
# Table layout (screen regions, defined at the base capture resolution)
# ---------------------------------------------------------------------------
BASE_RESOLUTION = (2560, 1440)
BASE_PLAYER_REGIONS = [
    [[476, 1096], [604, 1228], [1072, 948], [1076, 832], [476, 1096]],
    [[604, 1228], [820, 1332], [1216, 944], [1072, 948], [604, 1228]],
    [[820, 1332], [1112, 1392], [1308, 944], [1216, 944], [820, 1332]],
    [[1112, 1392], [1424, 1392], [1372, 944], [1308, 944], [1112, 1392]],
    [[1424, 1392], [1716, 1340], [1456, 940], [1372, 944], [1424, 1392]],
    [[1716, 1336], [1940, 1236], [1572, 940], [1456, 940], [1716, 1336]],
    [[1940, 1236], [2072, 1096], [1572, 840], [1572, 940], [1940, 1236]],
]
NUM_SEATS = len(BASE_PLAYER_REGIONS)

DEALER_AREA_LEFT = 1548
DEALER_AREA_UPPER = 12
DEALER_AREA_WIDTH, DEALER_AREA_HEIGHT = 1000, 800

# ---------------------------------------------------------------------------
# Detection engine tuning
# ---------------------------------------------------------------------------
# Producer loop pacing (seconds) by activity state.
CYCLE_SLEEP_DEALING = 0.4
CYCLE_SLEEP_COMPLETE = 1.0
CYCLE_SLEEP_WAITING = 1.2
# Time-critical phases (bets open / your turn) sample faster — the betting
# window is ~12-15 s and the decision timer ~10-13 s, so a 1.2 s cadence
# would burn a third of the window just noticing it opened.
CYCLE_SLEEP_ACTION = 0.3

# A gap between worker cycles longer than this means the machine slept or the
# process was suspended — hosted model sessions are refreshed before reuse,
# retried with exponential backoff (HEALTH_CHECK_BACKOFF_BASE * 2**attempt s).
IDLE_REFRESH_GAP_S = 30.0
HEALTH_CHECK_RETRY_COUNT = 3
HEALTH_CHECK_BACKOFF_BASE = 0.5

# Mean absolute pixel difference (0-255 scale, on a small grayscale thumbnail)
# below which the frame is considered unchanged and inference is skipped.
FRAME_DIFF_THRESHOLD = 2.0
# The dealer area gets its OWN thumbnail diff: one card flipping there moves
# the whole-frame mean by ~0.07 — invisible to the global threshold — and a
# quiet table would skip straight past the dealer's reveal/playout. Higher
# than the global value because the crop is small: a real flip moves its
# mean by 10+, stream compression shimmer stays in low single digits.
DEALER_FRAME_DIFF_THRESHOLD = 3.0
# Never skip more than this many consecutive cycles, even if the frame looks static.
MAX_SKIPPED_CYCLES = 8

# A detection within this many pixels (at base resolution) of an already
# locked card with the same rank+suit is treated as the same physical card.
SAME_CARD_DISTANCE_PX = 80
# Cards beyond the first two per seat (hits) must be seen in this many
# consecutive cycles before they are accepted.
EXTRA_CARD_CONFIRM_CYCLES = 2
# A pending DEALER draw survives this many missed cycles before it is
# dropped (the dealer's hand briefly occludes cards mid-draw; a hard
# consecutive requirement made fast playouts lose cards).
DEALER_PENDING_MISS_TOLERANCE = 2
# Dealer card must agree across this many consecutive frames to lock.
DEALER_CONFIRM_FRAMES = 2
MAX_CARDS_PER_SEAT = 6
# A completed round auto-advances only after this many consecutive empty
# player frames, with the dealer area clear for the same stretch — a brief
# stream hiccup must never wipe a live round.
EMPTY_FRAMES_FOR_RESET = 5

# How often the GUI polls the engine for a fresh snapshot (ms).
SNAPSHOT_POLL_MS = 120

# An EV advice job still pending after this many seconds falls back to the
# book play ("{book} (book — EV delayed)") until the exact result lands.
EV_ADVICE_TIMEOUT_S = 3.0

# Workers in the "predeal" EV pool (V3 E2): the exact pre-deal sweep fans
# out weight-balanced whole-up-card jobs, cutting the ~14 s sweep to
# ~9-10 s so the exact bet call lands inside the 12-15 s betting window
# instead of one round late. Six is enough: the LPT makespan floors at
# the heaviest single up-card, and each worker holds its own memo caches
# (low hundreds of MB at the fresh-shoe peak). Measured (16-thread box):
# warm caches do NOT help the next round (states key on the exact
# composition), so parallelism is the only honest lever; the documented
# next tier is mypyc (V2 F9).
PREDEAL_WORKERS = max(1, min(6, (os.cpu_count() or 8) // 2))

# ---------------------------------------------------------------------------
# Game rules
# ---------------------------------------------------------------------------
BASE_BET = 10
DECK_COUNT = 8

# Table rules for the exact-EV engine. Defaults match standard Evolution live
# blackjack (8 decks, S17, double any two, DAS, split once, no surrender).
# VERIFY peek/dealer_bj_takes in the specific table's game help — Evolution
# titles vary on no-hole-card handling, and it changes the EVs.
RULES = {
    "s17": True,              # dealer stands on all 17s
    "peek": False,            # True = US hole-card peek; False = no hole card (ENHC)
    "dealer_bj_takes": "all", # ENHC only: dealer BJ takes "all" bets or "obo"
    "das": True,              # double after split allowed
    "double_on": "any",       # "any" | "9-11" | "10-11"
    "hit_split_aces": False,
    "surrender": False,       # late surrender offered
    "bj_pays": 1.5,           # 3:2; 6:5 tables use 1.2
}

# Bet sizing (Feature 7): edge ~ base_edge + edge_per_tc * true_count; wager
# = bankroll * kelly_fraction * edge / variance, clamped to the table limits.
# All persisted with the table profile (settings.json); bankroll is editable
# straight from the left panel.
BETTING = {
    "bankroll": 1000.0,
    "kelly_fraction": 0.5,   # half-Kelly: near-optimal growth, far less ruin
    "base_edge": -0.005,     # house edge off the top for the configured rules
    "edge_per_tc": 0.005,    # standard Hi-Lo shoe-game slope
    "variance": 1.33,        # per-hand variance in squared units
    # Covariance between simultaneous hands at the same table (they share
    # the dealer; Wizard of Odds 0.479). Drives the per-seat Kelly shrink
    # when multiple seats are starred (V3 E5): two hands at ~73.5% each.
    "covariance": 0.479,
    "table_min": 10,
    "table_max": 5000,       # 0 = no max
    "auto_bankroll": 1,      # settle owned seats into the bankroll (1/0)
    "use_exact_edge": 1,     # exact pre-deal EV instead of the linear TC model
    # Per-TC bet table installed by the ramp designer (V3 Feature 5):
    # {str(floored_tc): bet_eur}, 0 = sit out. Empty = formula ramp.
    "bet_table": {},
}

# Casino-UI OCR (balance / bet / result banner). Needs winocr (Windows) and
# per-resolution regions drawn with the OCR-region editor.
OCR = {
    "enabled": 1,
    "interval_s": 1.0,       # read cadence; every engine besides EasyOCR is fine at 1 Hz
    "sync_bankroll": 1,      # screen balance is ground truth for the bankroll
    "sync_bet": 1,           # screen bet fills the "bet placed" field
}

# Game-phase / turn detection (V4 Feature 1, lib/logic/phase.py). The
# detector reads countdown digits + the "place your bets" banner via the
# OCR regions and template-matches the captured action buttons every cycle.
PHASE = {
    "enabled": 1,
    # Template match score (cv2.TM_CCOEFF_NORMED) below which a control is
    # treated as absent. Evolution buttons are crisp renders; 0.70 tolerates
    # stream compression while rejecting felt/chips at the same spot.
    "match_threshold": 0.70,
    # Mean per-channel color distance (0-255) between the matched patch and
    # the captured template above which the button counts as DISABLED
    # (greyed). Templates must be captured while the buttons are enabled.
    # Kept tight: rendered UI under stream compression drifts <10, a
    # grey-out shifts ~50 — and the safe failure mode is "disabled".
    "enabled_color_dist": 35.0,
    # Search padding around a control's calibrated rect (fraction of the
    # rect size) — tolerates small client re-layouts without a blind match
    # across the whole frame.
    "search_pad": 0.35,
    # Consecutive confirming cycles before MY_TURN / BETTING_OPEN latch.
    # One frame of a half-rendered button must never flash "YOUR TURN".
    "confirm_frames": 2,
    # An OCR'd status/result/timer older than this is stale for phase logic.
    "ocr_fresh_s": 3.0,
    # BETTING_OPEN confirmed for this many cycles while the finished round
    # is still on the table fast-paths the auto round reset (the betting
    # banner is a stronger "table cleared" signal than N empty frames).
    "reset_confirm_frames": 2,
    # MY_TURN with the countdown at/below this many seconds and no action
    # observed yet raises the missed-decision alarm on the HUD.
    "alarm_timer_s": 4.0,
    # Seconds after MY_TURN ends before the observed action is judged
    # against the advice (a clicked Hit takes a moment to land a card).
    "discipline_grace_s": 3.0,
}

# Session guardrails (V3 E5): the math panel knows the risk; this is the
# piece that enforces the plan. Visual-only (banner + HUD line) plus
# executor auto-disarm on breach — no audio, no enforcement of the human.
# FAIL-DISABLED by default (V4 standing decision); 0 disables a limit.
GUARDRAILS = {
    "enabled": 0,
    "stop_loss_eur": 200.0,   # breach when session P&L <= -stop_loss
    "stop_win_eur": 0.0,      # breach when session P&L >= stop_win (0 = off)
    "max_rounds": 0,          # breach after this many settled owned rounds
}

# Anchor-based resolution-independent calibration (V3 E3,
# lib/logic/anchors.py). 2-3 template crops of stable UI elements per
# table profile; on startup/demand the engine matches them on the live
# frame, solves scale+offset, and maps the one calibrated region set to
# the actual screen. FAIL-DISABLED: an unsolved fit keeps the current
# geometry and warns — it never guesses a transform that would move
# real OCR crops or executor clicks.
ANCHORS = {
    "enabled": 1,
    # TM_CCOEFF_NORMED below this = anchor not found on the live frame.
    "match_threshold": 0.60,
    # Anchors required for a trusted fit (1 can't separate scale errors
    # from offset errors; capture at least 2, ideally 3 spread out).
    "min_anchors": 2,
    # Worst allowed |matched - fitted| anchor distance: above this the
    # geometry is not a uniform scale+offset (rotated/cropped stream) and
    # the fit is refused.
    "max_residual_px": 12.0,
    # Drift re-check cadence on the worker (cheap local searches).
    "drift_check_s": 20.0,
    # An anchor moving farther than this from its fitted position latches
    # the drift warning (and disarms the executor).
    "drift_shift_px": 14.0,
}

# Ghost-mode executor / one-key assisted execute (V4 Feature 2 —
# AUTONOMY_PLAN.md stages 1-2; stage 3 full unattended autonomy is
# deliberately NOT implemented). Mode and limits persist in settings.json;
# the ARM state NEVER persists — every session starts disarmed.
EXECUTOR = {
    "mode": "off",             # off | ghost | assist
    # Stricter than the display threshold: acting needs the phase confirmed
    # across this many consecutive cycles AND the button matched this well.
    "confirm_frames": 3,
    "min_button_score": 0.80,
    # Money limits — any breach auto-disarms (never silently clamps).
    "max_bet_eur": 50.0,
    "stop_loss_eur": 200.0,    # session net loss beyond this disarms
    "stop_win_eur": 0.0,       # 0 = no stop-win
    "max_actions_per_round": 6,
    "balance_tolerance_eur": 25.0,  # OCR balance vs bankroll divergence
    "verify_timeout_s": 4.0,   # post-click confirmation deadline
    # A confirm against a snapshot older than this is refused — a stalled
    # or stopped worker must never leave a fireable frozen MY_TURN.
    "max_snapshot_age_s": 2.5,
    # Chip denominations available at the table (bet-plan decomposition).
    "chips": [0.5, 1, 5, 25, 100, 500],
    # Click delivery: OS SendInput (stdlib ctypes) is the default; the CDP
    # path needs `pip install playwright` and Chrome started with
    # --remote-debugging-port. Falls back to OS input when CDP fails.
    "use_cdp": 0,
    "cdp_port": 9222,
    "cdp_url_match": "evolution",
    # Global hotkeys (virtual-key codes): F8 confirm, F9 kill switch.
    "confirm_vk": 0x77,
    "kill_vk": 0x78,
}

# Claude vision assist (optional, hybrid per design decision: the per-frame
# hot loop stays local; the API is only used for one-shot calibration
# bootstrap and unknown-state triage). Uses the plain REST API via
# `requests` — no SDK dependency. Key can also come from ANTHROPIC_API_KEY.
VISION = {
    "enabled": 0,
    "api_key": "",
    "model": "claude-opus-4-8",
    "triage": 1,             # label unknown screens (modals, disconnects)
    "triage_after_s": 15.0,  # UNKNOWN phase persisting this long triggers it
    "triage_min_gap_s": 60.0,
}

# Side bets offered by the table and their paytables (X means pays X:1).
# Defaults = Evolution live blackjack. House edges (8 decks, full shoe):
# Perfect Pairs 4.10%, 21+3 3.70%, Hot 3 5.40%, Bust It 6.18% — all verified
# against wizardofodds.com. Lucky Lucky / Lucky Ladies are off-Evolution bets,
# disabled by default; enable per table. "stake" is the EUR amount the user
# actually places per owned seat (0 = not playing it); settlement books
# stake x paytable into the session P&L.
SIDE_BETS = {
    "perfect_pairs": {
        "label": "Perfect Pairs", "enabled": True, "stake": 0.0,
        "paytable": {"perfect": 25, "colored": 12, "mixed": 6},
    },
    "21+3": {
        "label": "21+3", "enabled": True, "stake": 0.0,
        "paytable": {"suited_trips": 100, "straight_flush": 40, "trips": 30,
                     "straight": 10, "flush": 5},
    },
    "hot3": {
        "label": "Hot 3", "enabled": True, "stake": 0.0,
        "paytable": {"777": 100, "suited_21": 20, "21": 4, "20": 2, "19": 1},
    },
    "bust_it": {
        "label": "Bust It", "enabled": True, "stake": 0.0,
        "paytable": {3: 1, 4: 2, 5: 9, 6: 50, 7: 100, 8: 250},  # 8 = 8+ cards
    },
    "lucky_lucky": {
        "label": "Lucky Lucky", "enabled": False, "stake": 0.0,
        "paytable": {"suited_777": 200, "suited_678": 100, "777": 50, "678": 30,
                     "suited_21": 15, "21": 3, "20": 2, "19": 2},
    },
    "lucky_ladies": {
        "label": "Lucky Ladies", "enabled": False, "stake": 0.0,
        "paytable": {"qh_pair_dealer_bj": 1000, "qh_pair": 125, "matched_20": 19,
                     "suited_20": 9, "any_20": 4},
    },
}

CARD_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
CARD_SUITS = ["Spades", "Hearts", "Diamonds", "Clubs"]

VALUE_MAPPING = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "Jack": 10, "Queen": 10, "King": 10, "Ace": 11,
}

ACTION_MAPPING = {
    "H": "Hit",
    "S": "Stand",
    "D/H": "Double / Hit",
    "D/S": "Double / Stand",
    "P": "Split",
    "P/H": "Split / Hit",
    "R/H": "Surrender / Hit",
    "-": "-",
}

# ---------------------------------------------------------------------------
# UI theme (single source of truth for every window)
# ---------------------------------------------------------------------------
# UI scale override percent; 0 = auto (per-monitor DPI).
UI = {"scale": 0}

FONT_FAMILY = "Segoe UI"

# 96-dpi baseline tuples; lib/interfaces/scaling.py replaces these attributes
# with live named Font objects (DPI-scaled) once a Tk root exists.
FONT_TITLE = (FONT_FAMILY, 17, "bold")
FONT_SECTION = (FONT_FAMILY, 11, "bold")
FONT_BODY = (FONT_FAMILY, 10)
FONT_BODY_BOLD = (FONT_FAMILY, 10, "bold")
FONT_SMALL = (FONT_FAMILY, 9)
FONT_BIG_VALUE = (FONT_FAMILY, 13, "bold")
TOOLTIP_FONT = (FONT_FAMILY, 9)

COLORS = {
    "bg_primary": "#f8f9fa",
    "bg_secondary": "#ffffff",
    "bg_canvas": "#0b5d3b",       # casino felt green
    "bg_canvas_soft": "#0e6e46",
    "accent": "#0d6efd",
    "accent_hover": "#0a58ca",
    "text_primary": "#212529",
    "text_secondary": "#6c757d",
    "text_on_felt": "#e9f5ee",
    "success": "#198754",
    "success_hover": "#146c43",
    "warning": "#ffc107",
    "danger": "#dc3545",
    "danger_hover": "#b02a37",
    "border": "#dee2e6",
    "card_bg": "#ffffff",
    "badge_bg": "#13301f",
}

ACTION_COLORS = {
    "H": "#2fbf71",      # hit - green
    "S": "#4dabf7",      # stand - blue
    "D/H": "#ffa94d",    # double - orange
    "D/S": "#ffa94d",
    "P": "#c084fc",      # split - purple
    "P/H": "#c084fc",
    "R/H": "#ff6b6b",    # surrender - red
    "-": "#e9f5ee",
}

CARD_RENDER_SIZE = (60, 88)        # on-table card size in px
DEALER_CARD_RENDER_SIZE = (84, 123)
PICKER_CARD_SIZE = (56, 82)

DEBUG_MODE = False
