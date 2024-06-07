VERSION = "1.6"
TITLE = f"Blackjack AI - v{VERSION}"
SIZE = "1200x800"
RESOLUTION = "2560x1440"

BASE_RESOLUTION = (2560, 1440)
BASE_PLAYER_REGIONS = [
    [[476, 1096], [604, 1228], [1072, 948], [1076, 832], [476, 1096]],
    [[604, 1228], [820, 1332], [1216, 944], [1072, 948], [604, 1228]],
    [[820, 1332], [1112, 1392], [1308, 944], [1216, 944], [820, 1332]],
    [[1112, 1392], [1424, 1392], [1372, 944], [1308, 944], [1112, 1392]],
    [[1424, 1392], [1716, 1340], [1456, 940], [1372, 944], [1424, 1392]],
    [[1716, 1336], [1940, 1236], [1572, 940], [1456, 940], [1716, 1336]],
    [[1940, 1236], [2072, 1096], [1572, 840], [1572, 940], [1940, 1236]]
]

# Roboflow API
ROBOFLOW_API_KEY = "WBy7jG6AiiqjzifOfiNH"

# Initializes Roboflow API for Player Card detection
PROJECT_ID_PLAYERS = "dey022"  # API Project ID for player cards
MODEL_VERSION_PLAYERS = 1
PREDICTION_CONFIDENCE_PLAYERS = 70
PREDICTION_OVERLAP_PLAYERS = 100

# Initializes Roboflow API for Dealer Card detection
PROJECT_ID_DEALER = "carddetection-v1hqz"  # API Project ID for player cards
MODEL_VERSION_DEALER = 17
PREDICTION_CONFIDENCE_DEALER = 55
PREDICTION_OVERLAP_DEALER = 45

# File paths
CARD_FOLDER_PATH = "../api/cards"
DEFAULT_CARD_IMAGE_PATH = "../api/cards/red card.jpg"
INPUT_SCREENSHOT_PATH = "INPUT_current_screen.jpg"
OUTPUT_PREDICTION_PATH = "OUTPUT_prediction.jpg"
OUTPUT_FINAL_PREDICTION_PATH = "OUTPUT_final_prediction_with_players.jpg"
OUTPUT_DEBUG_IMAGE_PATH = "OUTPUT_dealer_area_current_view.jpg"
CSV_FILE_PATH = "Blackjack cheat sheet table.csv"

# Mapping
CARD_VALUES = ["Ace", "10", "9", "8", "7", "6", "5", "4", "3", "2"]

VALUE_MAPPING = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "Jack": 10, "Queen": 10, "King": 10, "Ace": 1  # Count Ace as 1 for simplicity
}

ACTION_MAPPING = {  # No need to map D/H, D/S, P/H and R/H
    "H": "Hit",
    "S": "Stand",
    "P": "Split",
}

ACTION_COLORS = {
    "H": "blue",
    "S": "red",
    "D/H": "green",
    "D/S": "green",
    "P": "skyblue",
    "R/H": "purple",
    "P/H": "orange"
}

# Misc
BASE_BET = 10
DECK_COUNT = 8

# Card dimensions
CARD_WIDTH = 73
CARD_HEIGHT = 98
CARD_SPACING = 25

# Dealer's card detection area
DEALER_AREA_LEFT = 1548
DEALER_AREA_UPPER = 12
DEALER_AREA_WIDTH, DEALER_AREA_HEIGHT = 1000, 800
DEALER_AREA_RIGHT = DEALER_AREA_LEFT + DEALER_AREA_WIDTH  # Calculate the right boundary
DEALER_AREA_LOWER = DEALER_AREA_UPPER + DEALER_AREA_HEIGHT  # Calculate the lower boundary
