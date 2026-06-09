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

from class_mapping import class_mapping

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

        # Sort player_cards by player index before processing
        sorted_player_cards = sorted(player_cards.items(), key=lambda x: x[0])

        # After processing predictions, calculate hand values and check for stand/bust conditions
        for player_index, player_data in sorted_player_cards:
            cards = player_data['cards']
            hand_value = calculate_hand_value(cards)

            # Adjusted logic to determine if additional cards need to be detected
            if hand_value > 21:
                message = f"Player {player_index + 1} card value is {hand_value} and busts."
            elif hand_value < 21:
                message = f"Player {player_index + 1} card value is {hand_value} and can receive more cards."
            elif hand_value == 21:
                message = f"Player {player_index + 1} card value is {hand_value} and has Blackjack."
            else:
                message = f"??? Player {player_index + 1} has {hand_value}"

            print(message)

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
