import os
import numpy as np
from PIL import Image
from roboflow import Roboflow
from mss import mss
from screeninfo import get_monitors
from matplotlib.path import Path
from collections import defaultdict
import tempfile
import sys

from class_mapping import class_mapping

# TODO: Convert player_regions to 1920x1080 resolution (now 2560x1440).
# Define player regions with numpy arrays or convert from your provided points
player_regions = [
    Path(np.array([[652, 1026], [760, 1126], [1112, 994], [996, 902], [652, 1026]])),
    Path(np.array([[776, 1130], [924, 1218], [1184, 1026], [1072, 962], [776, 1130]])),
    Path(np.array([[920, 1218], [1168, 1254], [1292, 1050], [1176, 1018], [920, 1218]])),
    Path(np.array([[1168, 1254], [1420, 1250], [1412, 1042], [1276, 1046], [1168, 1254]])),
    Path(np.array([[1412, 1246], [1616, 1202], [1524, 1010], [1392, 1042], [1412, 1246]])),
    Path(np.array([[1616, 1202], [1796, 1130], [1616, 958], [1512, 1014], [1616, 1202]])),
    Path(np.array([[1796, 1130], [1924, 1018], [1696, 894], [1608, 962], [1796, 1130]]))
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


# Continuously capture screen and perform object detection
def capture_screen_and_track_cards(card_value_counts, monitor_number=0):
    global player_cards, detected_card_values
    first_cards_detected = set()
    second_cards_detected = set()
    players_received_first_card = set()

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
        print(f"Got {len(predictions)} predictions")

        for prediction in predictions:
            x, y = prediction['x'], prediction['y']
            class_label = prediction['class']
            confidence = prediction['confidence']
            card_name = get_card_name(class_label)

            for i, region in enumerate(player_regions):
                if region.contains_point([x, y]):
                    # Ensure we don't re-assign the first card if already detected
                    if i not in first_cards_detected:
                        player_cards[i]['cards'][0] = card_name
                        player_cards[i]['confidences'][0] = confidence
                        first_cards_detected.add(i)
                        players_received_first_card.add(i)
                    elif i in first_cards_detected and card_name != player_cards[i]['cards'][0]:
                        # Only update the second card if it's different from the first card
                        if confidence > player_cards[i]['confidences'][1]:
                            player_cards[i]['cards'][1] = card_name
                            player_cards[i]['confidences'][1] = confidence
                            second_cards_detected.add(i)

        # Check if any player has received their second card
        if second_cards_detected:
            # Check if all players who received the first card also received their second card
            if players_received_first_card.issubset(second_cards_detected):
                print_all_cards()
                break

        # Delete temporary file
        os.unlink(temp_file_path)


def print_all_cards():
    for player_index in sorted(player_cards):
        cards = player_cards[player_index]['cards']
        confidences = player_cards[player_index]['confidences']
        cards_info = []
        for i, (card, conf) in enumerate(zip(cards, confidences)):
            # Use a special case for Aces to maintain clarity in the output
            if "Ace" in card:
                card_value_counts["Ace"] += 1  # Increment the count for Aces specifically
            else:
                # Calculate the card value for each finalized card
                card_value = get_card_value(card)
                card_value_counts[card_value] += 1  # Increment the count for this card value
            cards_info.append(f"[{i + 1}] {card} (C: {conf * 100:.2f}%)")

        print(f"P{player_index + 1}: {' // '.join(cards_info)}")

    # Format and print the counts of each card value, converting keys to strings for sorting
    formatted_card_counts = [f"{value} => {count}x" for value, count in
                             sorted(card_value_counts.items(), key=lambda item: str(item[0]))]
    print("Card counts:\n", "\n".join(formatted_card_counts))


def wait_for_next_round():
    print("Waiting for next round, press Enter to continue...")
    input()


def reset_for_new_round():
    global player_cards, first_cards_detected, second_cards_detected, players_received_first_card
    player_cards = defaultdict(lambda: {"cards": ["-", "-"], "confidences": [0.0, 0.0]})
    first_cards_detected = set()
    second_cards_detected = set()
    players_received_first_card = set()


if __name__ == "__main__":
    while True:
        reset_for_new_round()  # Prepare for a new round
        capture_screen_and_track_cards(card_value_counts=card_value_counts, monitor_number=0)
        print_all_cards()
        wait_for_next_round()
