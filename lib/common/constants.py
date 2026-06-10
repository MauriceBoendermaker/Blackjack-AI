"""Central configuration: paths, detection settings, game rules, and UI theme."""

import os
from pathlib import Path

VERSION = "3.0"
TITLE = f"Blackjack AI - v{VERSION}"

# ---------------------------------------------------------------------------
# Paths (anchored to the project root so the app works from any CWD)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
CARDS_DIR = ASSETS_DIR / "cards"
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
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "WBy7jG6AiiqjzifOfiNH")

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

# Run the 52-class (suit-aware) player model on the dealer crop instead of the
# rank-only dealer model. Gives dealer suits to the composition; validate the
# player model's accuracy on dealer-area imagery before enabling. The cutting
# card only exists in the rank model, so it is still checked every Nth cycle.
DEALER_USE_PLAYER_MODEL = False
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

# A gap between worker cycles longer than this means the machine slept or the
# process was suspended — hosted model sessions are refreshed before reuse,
# retried with exponential backoff (HEALTH_CHECK_BACKOFF_BASE * 2**attempt s).
IDLE_REFRESH_GAP_S = 30.0
HEALTH_CHECK_RETRY_COUNT = 3
HEALTH_CHECK_BACKOFF_BASE = 0.5

# Mean absolute pixel difference (0-255 scale, on a small grayscale thumbnail)
# below which the frame is considered unchanged and inference is skipped.
FRAME_DIFF_THRESHOLD = 2.0
# Never skip more than this many consecutive cycles, even if the frame looks static.
MAX_SKIPPED_CYCLES = 8

# A detection within this many pixels (at base resolution) of an already
# locked card with the same rank+suit is treated as the same physical card.
SAME_CARD_DISTANCE_PX = 80
# Cards beyond the first two per seat (hits) must be seen in this many
# consecutive cycles before they are accepted.
EXTRA_CARD_CONFIRM_CYCLES = 2
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
    "table_min": 10,
    "table_max": 5000,       # 0 = no max
    "auto_bankroll": 1,      # settle owned seats into the bankroll (1/0)
    "use_exact_edge": 1,     # exact pre-deal EV instead of the linear TC model
}

# Casino-UI OCR (balance / bet / result banner). Needs winocr (Windows) and
# per-resolution regions drawn with the OCR-region editor.
OCR = {
    "enabled": 1,
    "interval_s": 1.0,       # read cadence; every engine besides EasyOCR is fine at 1 Hz
    "sync_bankroll": 1,      # screen balance is ground truth for the bankroll
    "sync_bet": 1,           # screen bet fills the "bet placed" field
}

# Side bets offered by the table and their paytables (X means pays X:1).
# Defaults = Evolution live blackjack. House edges (8 decks, full shoe):
# Perfect Pairs 4.10%, 21+3 3.70%, Hot 3 5.40%, Bust It 6.18% — all verified
# against wizardofodds.com. Lucky Lucky / Lucky Ladies are off-Evolution bets,
# disabled by default; enable per table.
SIDE_BETS = {
    "perfect_pairs": {
        "label": "Perfect Pairs", "enabled": True,
        "paytable": {"perfect": 25, "colored": 12, "mixed": 6},
    },
    "21+3": {
        "label": "21+3", "enabled": True,
        "paytable": {"suited_trips": 100, "straight_flush": 40, "trips": 30,
                     "straight": 10, "flush": 5},
    },
    "hot3": {
        "label": "Hot 3", "enabled": True,
        "paytable": {"777": 100, "suited_21": 20, "21": 4, "20": 2, "19": 1},
    },
    "bust_it": {
        "label": "Bust It", "enabled": True,
        "paytable": {3: 1, 4: 2, 5: 9, 6: 50, 7: 100, 8: 250},  # 8 = 8+ cards
    },
    "lucky_lucky": {
        "label": "Lucky Lucky", "enabled": False,
        "paytable": {"suited_777": 200, "suited_678": 100, "777": 50, "678": 30,
                     "suited_21": 15, "21": 3, "20": 2, "19": 2},
    },
    "lucky_ladies": {
        "label": "Lucky Ladies", "enabled": False,
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
FONT_FAMILY = "Segoe UI"

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
