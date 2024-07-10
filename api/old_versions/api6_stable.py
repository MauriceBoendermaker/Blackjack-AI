import os
import time
import numpy as np
from PIL import Image
from roboflow import Roboflow
from mss import mss
from screeninfo import get_monitors
from matplotlib.path import Path
from collections import defaultdict
import tempfile

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

# TODO: Convert player_regions to 1920x1080 resolution (now 2560x1440).
# Define player regions with numpy arrays or convert from your provided points
player_regions = [
    Path(np.array([[604, 1036], [572, 1132], [716, 1156], [1164, 984], [1000, 916], [604, 1036]])),
    Path(np.array([[708, 1148], [732, 1252], [900, 1252], [1268, 1020], [1136, 972], [708, 1148]])),
    Path(np.array([[884, 1244], [972, 1332], [1140, 1304], [1344, 1052], [1224, 1004], [884, 1244]])),
    Path(np.array([[1128, 1296], [1284, 1364], [1412, 1296], [1456, 1048], [1316, 1040], [1128, 1296]])),
    Path(np.array([[1416, 1296], [1588, 1340], [1660, 1248], [1544, 988], [1384, 1040], [1416, 1296]])),
    Path(np.array([[1652, 1244], [1844, 1264], [1848, 1156], [1572, 928], [1516, 1004], [1652, 1244]])),
    Path(np.array([[1832, 1156], [2000, 1136], [1960, 1032], [1600, 900], [1552, 952], [1832, 1156]]))
]

# Initialize Roboflow
api_key = "WBy7jG6AiiqjzifOfiNH"  # Replace with your actual API key
project_id = "dey022"  # Replace with your actual project ID
rf = Roboflow(api_key=api_key)
project = rf.workspace().project(project_id)
model = project.version(1).model

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

        # Predict using object detection model
        predictions = model.predict(temp_file_path, confidence=62, overlap=100).json()['predictions']
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

        # # Check if any player has received their second card
        # if second_cards_detected:
        #     # Check if all players who received the first card also received their second card
        #     if players_received_first_card.issubset(second_cards_detected):
        #         print_all_cards()
        #         break

        # Delete temporary file
        os.unlink(temp_file_path)

        # Check if the minimum detection duration has elapsed before considering to break the loop
        current_time = time.time()
        if (current_time - detection_start_time) > minimum_detection_duration:
            if second_cards_detected and players_received_first_card.issubset(second_cards_detected):
                print_all_cards()
                break


def print_all_cards():
    global card_value_counts

    for player_index in sorted(player_cards):
        cards = player_cards[player_index]['cards']
        confidences = player_cards[player_index]['confidences']
        cards_info = [f"[{i + 1}] {card} (C: {conf * 100:.2f}%)" for i, (card, conf) in
                      enumerate(zip(cards, confidences))]
        print(f"P{player_index + 1}: {' // '.join(cards_info)}")

    formatted_card_counts = [f"{value} => {count}x" for value, count in
                             sorted(card_value_counts.items(), key=lambda item: str(item[0]))]
    print("Card counts:\n", "\n".join(formatted_card_counts))


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
