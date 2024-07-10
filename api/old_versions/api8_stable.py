import os
import csv
import time
import tempfile
import numpy as np

from mss import mss
from PIL import Image
from roboflow import Roboflow
from matplotlib.path import Path
from screeninfo import get_monitors
from collections import defaultdict

# Dictionary mapping class IDs to card types
class_mapping = {
    # Hearts
    "a1": "Ace of Hearts",
    "a2": "2 of Hearts",
    "a3": "3 of Hearts",
    "a4": "4 of Hearts",
    "a5": "5 of Hearts",
    "a6": "6 of Hearts",
    "a7": "7 of Hearts",
    "a8": "8 of Hearts",
    "a9": "9 of Hearts",
    "a10": "10 of Hearts",
    "a11": "Jack of Hearts",
    "a12": "Queen of Hearts",
    "a13": "King of Hearts",
    # Diamonds
    "b1": "Ace of Diamonds",
    "b2": "2 of Diamonds",
    "b3": "3 of Diamonds",
    "b4": "4 of Diamonds",
    "b5": "5 of Diamonds",
    "b6": "6 of Diamonds",
    "b7": "7 of Diamonds",
    "b8": "8 of Diamonds",
    "b9": "9 of Diamonds",
    "b10": "10 of Diamonds",
    "b11": "Jack of Diamonds",
    "b12": "Queen of Diamonds",
    "b13": "King of Diamonds",
    # Spades
    "c1": "Ace of Spades",
    "c2": "2 of Spades",
    "c3": "3 of Spades",
    "c4": "4 of Spades",
    "c5": "5 of Spades",
    "c6": "6 of Spades",
    "c7": "7 of Spades",
    "c8": "8 of Spades",
    "c9": "9 of Spades",
    "c10": "10 of Spades",
    "c11": "Jack of Spades",
    "c12": "Queen of Spades",
    "c13": "King of Spades",
    # Clubs
    "d1": "Ace of Clubs",
    "d2": "2 of Clubs",
    "d3": "3 of Clubs",
    "d4": "4 of Clubs",
    "d5": "5 of Clubs",
    "d6": "6 of Clubs",
    "d7": "7 of Clubs",
    "d8": "8 of Clubs",
    "d9": "9 of Clubs",
    "d10": "10 of Clubs",
    "d11": "Jack of Clubs",
    "d12": "Queen of Clubs",
    "d13": "King of Clubs",
    # Misc classes
    "10": "10",
    "9": "9",
    "g": "G",
    "gg": "GG",
    "b87": "8 of Diamonds",
    "continue": "Continue",
    "endred": "End Red",
    "Unknown": "Unknown"
}

# Dictionary mapping class IDs to card types
dealer_class_mapping = {
    # Hearts
    "a1": "Ace of Hearts",
    "a2": "2 of Hearts",
    "a3": "3 of Hearts",
    "a4": "4 of Hearts",
    "a5": "5 of Hearts",
    "a6": "6 of Hearts",
    "a7": "7 of Hearts",
    "a8": "8 of Hearts",
    "a9": "9 of Hearts",
    "a10": "10 of Hearts",
    "a11": "Jack of Hearts",
    "a12": "Queen of Hearts",
    "a13": "King of Hearts",
    "ea1": "Ace of Hearts",
    "ea2": "2 of Hearts",
    "ea3": "3 of Hearts",
    "ea4": "4 of Hearts",
    "ea5": "5 of Hearts",
    "ea6": "6 of Hearts",
    "ea7": "7 of Hearts",
    "ea8": "8 of Hearts",
    "ea9": "9 of Hearts",
    "ea10": "10 of Hearts",
    "ea11": "Jack of Hearts",
    "ea12": "Queen of Hearts",
    "ea13": "King of Hearts",
    # Diamonds
    "b1": "Ace of Diamonds",
    "b2": "2 of Diamonds",
    "b3": "3 of Diamonds",
    "b4": "4 of Diamonds",
    "b5": "5 of Diamonds",
    "b6": "6 of Diamonds",
    "b7": "7 of Diamonds",
    "b8": "8 of Diamonds",
    "b9": "9 of Diamonds",
    "b10": "10 of Diamonds",
    "b11": "Jack of Diamonds",
    "b12": "Queen of Diamonds",
    "b13": "King of Diamonds",
    "eb1": "Ace of Diamonds",
    "eb2": "2 of Diamonds",
    "eb3": "3 of Diamonds",
    "eb4": "4 of Diamonds",
    "eb5": "5 of Diamonds",
    "eb6": "6 of Diamonds",
    "eb7": "7 of Diamonds",
    "eb8": "8 of Diamonds",
    "eb9": "9 of Diamonds",
    "eb10": "10 of Diamonds",
    "eb11": "Jack of Diamonds",
    "eb12": "Queen of Diamonds",
    "eb13": "King of Diamonds",
    # Spades
    "c1": "Ace of Spades",
    "c2": "2 of Spades",
    "c3": "3 of Spades",
    "c4": "4 of Spades",
    "c5": "5 of Spades",
    "c6": "6 of Spades",
    "c7": "7 of Spades",
    "c8": "8 of Spades",
    "c9": "9 of Spades",
    "c10": "10 of Spades",
    "c11": "Jack of Spades",
    "c12": "Queen of Spades",
    "c13": "King of Spades",
    "ec1": "Ace of Spades",
    "ec2": "2 of Spades",
    "ec3": "3 of Spades",
    "ec4": "4 of Spades",
    "ec5": "5 of Spades",
    "ec6": "6 of Spades",
    "ec7": "7 of Spades",
    "ec8": "8 of Spades",
    "ec9": "9 of Spades",
    "ec10": "10 of Spades",
    "ec11": "Jack of Spades",
    "ec12": "Queen of Spades",
    "ec13": "King of Spades",
    # Clubs
    "d1": "Ace of Clubs",
    "d2": "2 of Clubs",
    "d3": "3 of Clubs",
    "d4": "4 of Clubs",
    "d5": "5 of Clubs",
    "d6": "6 of Clubs",
    "d7": "7 of Clubs",
    "d8": "8 of Clubs",
    "d9": "9 of Clubs",
    "d10": "10 of Clubs",
    "d11": "Jack of Clubs",
    "d12": "Queen of Clubs",
    "d13": "King of Clubs",
    "ed1": "Ace of Clubs",
    "ed2": "2 of Clubs",
    "ed3": "3 of Clubs",
    "ed4": "4 of Clubs",
    "ed5": "5 of Clubs",
    "ed6": "6 of Clubs",
    "ed7": "7 of Clubs",
    "ed8": "8 of Clubs",
    "ed9": "9 of Clubs",
    "ed10": "10 of Clubs",
    "ed11": "Jack of Clubs",
    "ed12": "Queen of Clubs",
    "ed13": "King of Clubs",
    # Misc - Assuming these might be incorrect class labels or not directly mapping to specific cards
    "10": "misc",
    "9": "misc",
    "C6": "misc",
    "b87": "misc",
    "c7-": "misc",
    "cb1": "misc",
    "cb5": "misc",
    "continue": "misc",
    "endred": "misc",
    "gfg": "misc",
    "gg": "misc",
    "gg-": "misc",
}

# TODO: Convert player_regions to 1920x1080 resolution (now 2560x1440).
# Define player regions with numpy arrays or convert from your provided points
player_regions = [
    Path(np.array([[476, 1096], [604, 1228], [1072, 948], [1076, 832], [476, 1096]])),
    Path(np.array([[604, 1228], [820, 1332], [1216, 944], [1072, 948], [604, 1228]])),
    Path(np.array([[820, 1332], [1112, 1392], [1308, 944], [1216, 944], [820, 1332]])),
    Path(np.array([[1112, 1392], [1424, 1392], [1372, 944], [1308, 944], [1112, 1392]])),
    Path(np.array([[1424, 1392], [1716, 1340], [1456, 940], [1372, 944], [1424, 1392]])),
    Path(np.array([[1716, 1336], [1940, 1236], [1572, 940], [1456, 940], [1716, 1336]])),
    Path(np.array([[1940, 1236], [2072, 1096], [1572, 840], [1572, 940], [1940, 1236]]))
]

action_mapping = {
    "H": "Hit",
    "S": "Stand",
    "P": "Split",
    # No need to map D/H, D/S, P/H and R/H
}

# Path to your uploaded CSV file
csv_file_path = '../Blackjack cheat sheet table.csv'

# Initialize a dictionary to hold the strategy
blackjack_strategy = {}

for key in list(blackjack_strategy.keys())[:10]:  # Print first 10 keys for inspection
    print(key)

# Read the CSV file and populate the strategy dictionary
with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        dealer_card, player_hand, action = row
        blackjack_strategy[(dealer_card, player_hand)] = action

api_key = "WBy7jG6AiiqjzifOfiNH"

# Initialize Roboflow for player cards
project_id_players = "dey022"  # The project ID for player cards
rf_players = Roboflow(api_key=api_key)
project_players = rf_players.workspace().project(project_id_players)
model_players = project_players.version(1).model

# Initialize Roboflow for dealer cards
project_id_dealer = "card-nfetv"  # The project ID for dealer cards
rf_dealer = Roboflow(api_key=api_key)
project_dealer = rf_dealer.workspace().project(project_id_dealer)
model_dealer = project_dealer.version(1).model

# Initialize a dictionary to store the cards for each player
player_cards = defaultdict(lambda: {"cards": ["-", "-"], "confidences": [0.0, 0.0]})


# Function to get card name from class label
def get_card_name(class_label):
    return class_mapping.get(class_label, "Unknown")


# Mapping function for card names to Blackjack values
def get_card_value(card_name):
    value_mapping = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "Jack": 10, "Queen": 10, "King": 10, "Ace": 1  # Count Ace as 1 for simplicity
    }
    # Extract the card's face value (ignore suit for now, e.g., " of Spades")
    card_value = card_name.split(' ')[0]
    return value_mapping.get(card_value, 0)  # Return 0 if card name is not recognized


# Initialize a list to store the values of detected cards
detected_card_values = []

# Initialize outside to maintain counts across rounds
card_value_counts = defaultdict(int)

# Global variables for tracking card detections
first_cards_detected = set()
second_cards_detected = set()
players_received_first_card = set()

# Initialize outside to maintain unique counts across rounds
counted_cards_this_round = set()


# Continuously capture screen and perform object detection
def capture_screen_and_track_cards(monitor_number=0):
    global player_cards, card_value_counts, counted_cards_this_round

    detection_start_time = time.time()
    minimum_detection_duration = 2  # Minimum duration in seconds before checking if all cards have been detected

    while True:
        # Capture screen
        monitor_info = get_monitors()[monitor_number]
        monitor = {
            "left": monitor_info.x,
            "top": monitor_info.y,
            "width": monitor_info.width,
            "height": monitor_info.height
        }
        sct = mss()
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file_path = temp_file.name
            img.save(temp_file_path)

        # Dealer's card detection
        left = 1548
        upper = 12
        width = 1000
        height = 800
        right = left + width  # Calculate the right boundary
        lower = upper + height  # Calculate the lower boundary

        dealer_area = img.crop((left, upper, right, lower))  # Crop to dealer's area
        dealer_cards = capture_dealer_cards(dealer_area, model_dealer)  # Pass cropped image to dealer's model
        print(f"Dealer's cards: {dealer_cards}")

        # Predict using object detection model
        predictions = model_players.predict(temp_file_path, confidence=62, overlap=100).json()['predictions']
        print(f"{len(predictions)} predictions")

        # Process each prediction
        for prediction in predictions:
            x, y = prediction['x'], prediction['y']
            class_label = prediction['class']
            confidence = prediction['confidence']
            card_name = get_card_name(class_label)

            for i, region in enumerate(player_regions):
                if region.contains_point([x, y]):
                    card_id = f"{i}_{card_name}"  # Unique identifier for a card detected for a player

                    if card_id not in counted_cards_this_round:
                        # Update player cards and counts
                        if i not in first_cards_detected:
                            player_cards[i]['cards'][0] = card_name
                            player_cards[i]['confidences'][0] = confidence
                            first_cards_detected.add(i)
                            players_received_first_card.add(i)
                            card_value = get_card_value(card_name)
                            card_value_counts[card_value] += 1
                            counted_cards_this_round.add(card_id)
                        elif i in first_cards_detected and card_name != player_cards[i]['cards'][0] and confidence > \
                                player_cards[i]['confidences'][1]:
                            player_cards[i]['cards'][1] = card_name
                            player_cards[i]['confidences'][1] = confidence
                            second_cards_detected.add(i)
                            card_value = get_card_value(card_name)
                            card_value_counts[card_value] += 1
                            counted_cards_this_round.add(card_id)

        # Sort player_cards by player index before processing
        sorted_player_cards = sorted(player_cards.items(), key=lambda x: x[0])

        # After all players have received their second card
        dealer_up_card = dealer_cards[0] if dealer_cards else "Unknown"
        for player_index, player_data in sorted_player_cards:
            cards = player_data['cards']
            hand_value = calculate_hand_value(cards)

            # Blackjack decision logic
            # decision = blackjack_decision(hand_value, cards, dealer_up_card)
            decision = blackjack_decision(cards, dealer_up_card)

            # Adjusted logic to determine if additional cards need to be detected
            if hand_value > 21:
                message = f"Player {player_index + 1} card value is {hand_value} and busts."
            elif hand_value == 21:
                message = f"Player {player_index + 1} card value is {hand_value} and has Blackjack."
            else:
                message = f"Player {player_index + 1} card value is {hand_value}. Recommended action: {', '.join(decision)}."

            print(message)

        # Delete temporary file
        os.unlink(temp_file_path)

        # Check if the minimum detection duration has elapsed before considering to break the loop
        current_time = time.time()
        if (current_time - detection_start_time) > minimum_detection_duration:
            if second_cards_detected and players_received_first_card.issubset(second_cards_detected):
                print_all_cards()
                break


def capture_dealer_cards(image, model):
    # Save the cropped image for debugging
    debug_image_path = "dealer_area_current_view.jpg"
    image.save(debug_image_path)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file_path = temp_file.name
        image.save(temp_file_path)

    # Proceed with the prediction
    predictions = model.predict(temp_file_path, confidence=50, overlap=40).json()['predictions']
    os.unlink(temp_file_path)  # Delete the temp file after prediction

    dealer_cards = []
    for prediction in predictions:
        class_label = prediction['class']
        card_name = dealer_class_mapping.get(class_label, "Unknown")
        dealer_cards.append(card_name)

    return dealer_cards


def is_soft_hand(cards):
    """Check if the hand is a soft hand (contains an Ace counted as 11)."""
    values = [get_card_value(card.split(' ')[0]) for card in cards]
    return 1 in values and sum(values) + 10 <= 21


def is_pair_hand(cards):
    """Check if the hand is a pair."""
    if len(cards) != 2:
        return False
    return get_card_value(cards[0].split(' ')[0]) == get_card_value(cards[1].split(' ')[0])


def get_hand_representation(cards):
    # Check for pairs of tens or face cards
    if len(cards) == 2 and all(card.split(' ')[0] in ['10', 'Jack', 'Queen', 'King'] for card in cards):
        return "10,10"  # Represent as a pair of tens for strategy lookup

    # Existing logic for soft hands
    if is_soft_hand(cards):
        non_ace_total = sum([get_card_value(card.split(' ')[0]) for card in cards if card.split(' ')[0] != 'Ace'])
        return f"A,{non_ace_total}"

    # Check if the hand is a pair
    elif is_pair_hand(cards):
        value = get_card_value(cards[0].split(' ')[0])
        return f"{value},{value}"

    # For other hands, return the total value
    else:
        total = calculate_hand_value(cards)
        return str(total)


def blackjack_decision(player_cards, dealer_up_card):
    dealer_value = str(get_card_value(dealer_up_card.split(' ')[0])) if dealer_up_card != "Unknown" else "0"
    hand_representation = get_hand_representation(player_cards)

    # Special handling for pairs of 10-value cards (including face cards)
    if hand_representation == "10,10" or (len(player_cards) == 2 and calculate_hand_value(player_cards) == 20):
        action = "Stand"  # Directly recommend "Stand" for a total value of 20 from two cards
    else:
        # First, check if it's a pair for special handling
        if is_pair_hand(player_cards):
            pair_value = get_card_value(player_cards[0].split(' ')[0])
            pair_key = (dealer_value, f"{pair_value},{pair_value}")
            action = blackjack_strategy.get(pair_key,
                                            "?")  # Default to "?" if not found, but this case should be caught above

        # If not handled by pair logic, lookup using the general hand representation
        else:
            action_key = (dealer_value, hand_representation)
            action = blackjack_strategy.get(action_key, "?")  # Default to "?" if not found

    # Convert shorthand action to full word, if applicable, using the action_mapping
    return [action_mapping.get(action, action)]  # Use the mapping to convert "H" to "Hit", etc.


def calculate_hand_value(cards):
    total = 0
    aces = 0
    # Calculate the total value and count the aces
    for card in cards:
        card_value = get_card_value(card.split(' ')[0])  # Extract the value part (e.g., "Ace" from "Ace of Diamonds")
        if card_value == 1:  # Ace is counted as 1 initially
            aces += 1
        else:
            total += card_value
    # Adjust for Aces after calculating the total
    for _ in range(aces):
        # If adding 11 keeps the total 21 or under, use 11 for Ace; otherwise, use 1
        total += 11 if total + 11 <= 21 else 1
    return total


def print_all_cards():
    global card_value_counts

    for player_index in sorted(player_cards):
        cards = player_cards[player_index]['cards']
        confidences = player_cards[player_index]['confidences']
        cards_info = [f"[{i + 1}] {card} (C: {conf * 100:.2f}%)" for i, (card, conf) in
                      enumerate(zip(cards, confidences))]
        print(f"P{player_index + 1}: {' // '.join(cards_info)}")

    formatted_card_counts = [f"{value} => {count}x" for value, count in
                             sorted(card_value_counts.items(), key=lambda item: item[0])]
    total_cards = sum(card_value_counts.values())  # Sum up the counts of all cards for the total
    print(f"Card counts ({total_cards}):\n", "\n".join(formatted_card_counts))


def wait_for_next_round():
    print("Waiting for next round, press Enter to continue...")
    input()


def reset_for_new_round():
    global player_cards, first_cards_detected, second_cards_detected, players_received_first_card, card_value_counts, counted_cards_this_round
    player_cards.clear()
    first_cards_detected.clear()
    second_cards_detected.clear()
    players_received_first_card.clear()
    counted_cards_this_round.clear()  # Reset the set of counted cards for the new round
    print("Reset for new round.")


if __name__ == "__main__":
    while True:
        reset_for_new_round()  # Prepare for a new round
        capture_screen_and_track_cards(monitor_number=0)
        wait_for_next_round()  # Wait for user input to proceed to the next round
