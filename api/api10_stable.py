# TODO1: Implement counter after 2 initial cards
# TODO: Implement system to detect splitted hands
# TODO: Fix 2x Ace being counted as "12", instead of 2 (and thus split)
# TODO: Implement OCR for Saldo and Current bet
# TODO: Make player_regions Scalable (e.g. 1920x1080, now 2560x1440)
# TODO: Finetune accuracy for both models (adjust parameters)
# TODO: Refine player regions
# TODO: Clean up code

import os
import csv
import time
# import keyboard
import tempfile
import threading
import numpy as np

from mss import mss
from PIL import Image
from roboflow import Roboflow
from matplotlib.path import Path
from screeninfo import get_monitors
from collections import defaultdict

from class_mapping import class_mapping
from dealer_class_mapping2 import dealer_class_mapping

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

base_bet = 10

# Initialize the card counter variables
running_count = 0
true_count = 0
deck_count = 8  # 8-deck shoe at the start

# Path to Blackjack cheat sheet file
csv_file_path = 'Blackjack cheat sheet table.csv'

# Initialize a dictionary to hold the strategy
blackjack_strategy = {}

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
project_id_dealer = "carddetection-v1hqz"  # The project ID for dealer cards
rf_dealer = Roboflow(api_key=api_key)
project_dealer = rf_dealer.workspace().project(project_id_dealer)
model_dealer = project_dealer.version(17).model

# Initialize a dictionary to store the cards for each player
player_cards = defaultdict(lambda: {"cards": ["-", "-"], "confidences": [0.0, 0.0]})

# Flag to signal the end of the round
end_round_flag = False


def listen_for_arrow_up():
    print("Press Arrow Up to manually end the round...")
    # keyboard.wait('up')
    global end_round_flag
    end_round_flag = True
    print("Arrow Up pressed. Ending the round.")


def update_count(card_name):
    global running_count
    card_value = get_card_value(card_name)
    if card_value >= 2 and card_value <= 6:
        running_count += 1
    elif card_value == 10 or card_name == "Ace":
        running_count -= 1


def calculate_true_count():
    global true_count, running_count, deck_count
    decks_remaining = (deck_count * 52 - len(counted_cards_this_round)) / 52
    true_count = running_count / decks_remaining if decks_remaining > 0 else running_count


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

# Initialize a counter for the number of rounds observed
rounds_observed = 0


# Continuously capture screen and perform object detection
def capture_screen_and_track_cards(monitor_number=0):
    global player_cards, card_value_counts, counted_cards_this_round, true_count, rounds_observed, end_round_flag
    end_round_flag = False  # Reset the flag at the start of each round

    # Start the Escape key listener in a separate thread
    arrow_up_listener_thread = threading.Thread(target=listen_for_arrow_up)
    arrow_up_listener_thread.daemon = True  # Make sure the thread is marked as a daemon so it doesn't prevent the program from exiting
    arrow_up_listener_thread.start()

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

        # Dealer's card detection area
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

            # Update the count based on the detected card
            update_count(card_name)

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

        # Calculate true_count before making decisions
        calculate_true_count()  # Make sure this function properly updates true_count based on the cards seen and decks remaining

        dealer_up_card = dealer_cards[0] if dealer_cards else "Unknown"

        if end_round_flag:
            print("Manually ending the round.")
            break

        if not sorted_player_cards:
            print("")
        else:
            for player_index, player_data in sorted_player_cards:
                cards = player_data['cards']
                hand_value = calculate_hand_value(cards)

                # Passing true_count as an argument for decision making
                decision_recommendations = blackjack_decision(cards, dealer_up_card, true_count, base_bet)

                # if hand_value > 21:
                #     message = f"Player {player_index + 1} card value is {hand_value} and busts."
                # elif hand_value == 21:
                #     message = f"Player {player_index + 1} card value is {hand_value} and has Blackjack."
                # else:
                #     message = f"Player {player_index + 1} card value is {hand_value}. Recommended action: {', '.join(decision_recommendations)}."

                message = f"Player {player_index + 1} card value is {hand_value}. Recommended action: {', '.join(decision_recommendations)}."
                print(message)

        if rounds_observed > 3:
            betting_strategy = bet_strategy(true_count, base_bet)
            print(f"\nBetting strategy for next round: {betting_strategy}")
        else:
            print(
                f"\nAccumulating card count data, betting strategy recommendations will start in {3 - rounds_observed} rounds")

        # Delete temporary file
        os.unlink(temp_file_path)

        # Check if the minimum detection duration has elapsed before considering to break the loop
        current_time = time.time()
        if (current_time - detection_start_time) > minimum_detection_duration:
            if second_cards_detected and players_received_first_card.issubset(second_cards_detected):
                print_all_cards()
                break

    rounds_observed += 1


def capture_dealer_cards(image, model):
    # Save the cropped image for debugging
    debug_image_path = "dealer_area_current_view.jpg"
    image.save(debug_image_path)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file_path = temp_file.name
        image.save(temp_file_path)

    # Proceed with the prediction
    predictions = model.predict(temp_file_path, confidence=55, overlap=45).json()['predictions']
    os.unlink(temp_file_path)  # Delete the temp file after prediction

    dealer_cards = []
    for prediction in predictions:
        class_label = prediction['class']
        card_name = dealer_class_mapping.get(class_label, "Unknown")
        dealer_cards.append(card_name)

    return dealer_cards


def is_soft_hand(cards):
    # Check if the hand is a soft hand (contains an Ace counted as 11)
    values = [get_card_value(card.split(' ')[0]) for card in cards]
    return 1 in values and sum(values) + 10 <= 21


def is_pair_hand(cards):
    # Check if the hand is a pair
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


def blackjack_decision(player_cards, dealer_up_card, true_count, base_bet):
    dealer_value = get_dealer_card_value(dealer_up_card) if dealer_up_card != "Unknown" else "0"
    hand_value = calculate_hand_value(player_cards)
    recommendations = []

    # Directly handle values where the strategy is universally to stand
    if hand_value >= 17:
        action = "S"  # Stand on 17, 18, 19, 20, 21
    else:
        hand_representation = get_hand_representation(player_cards)

        if is_pair_hand(player_cards):
            pair_value = str(get_card_value(player_cards[0].split(' ')[0]))
            pair_value = "10" if pair_value in ["J", "Q", "K"] else pair_value
            action_key = (dealer_value, f"Pair {pair_value}")
        else:
            action_key = (dealer_value, hand_representation)

        action = blackjack_strategy.get(action_key, "?")

    recommendations.append(action_mapping.get(action, "Hit" if action == "?" else action))

    return recommendations


def bet_strategy(true_count, base_bet):
    if true_count > 4:
        return f"Betting 2x base bet (€{2 * base_bet})"
    elif true_count >= 2:
        return f"Betting 1.5x base bet (€{1.5 * base_bet})"
    elif true_count < 0:
        return "Betting half base bet (Consider not playing)"
    else:
        return f"Betting base bet (€{base_bet})"


# Helper function to interpret the dealer's card correctly
def get_dealer_card_value(card):
    if card in ["J", "Q", "K"]:
        return "10"
    return card


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
