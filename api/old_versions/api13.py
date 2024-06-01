# TODO: Implement system to detect splitted hands
# TODO: Fix 2x Ace being counted as "12", instead of 2 (and thus split)
# TODO: Implement OCR for Saldo and Current bet
# TODO: Make player_regions Scalable (e.g. 1920x1080, now 2560x1440)
# TODO: Finetune accuracy for both models (adjust parameters)
# TODO: Refine player regions
# TODO: Clean up code
# TODO: Make an UI

import os
import csv
import time
import queue
import tempfile
import threading
import numpy as np

import display1  # dispay1.py

from mss import mss
from PIL import Image
from roboflow import Roboflow
from matplotlib.path import Path
from screeninfo import get_monitors
from collections import defaultdict

from class_mapping import class_mapping
from dealer_class_mapping2 import dealer_class_mapping

global dealer_value

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

action_mapping = {  # No need to map D/H, D/S, P/H and R/H
    "H": "Hit",
    "S": "Stand",
    "P": "Split",
}

base_bet = 10

# Initialize the card counter variables
running_count = 0
true_count = 0
deck_count = 8  # 8-deck shoe at the start

csv_file_path = 'Blackjack cheat sheet table.csv'  # Path to Blackjack cheat sheet file

blackjack_strategy = {}  # Initialize a dictionary to hold the strategy

with open(csv_file_path, newline='') as csvfile:  # Read the CSV file and populate the strategy dictionary
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        dealer_card, player_hand, action = row
        blackjack_strategy[(dealer_card, player_hand)] = action

# Initialize a dictionary to store the cards for each player
player_cards = defaultdict(lambda: {"cards": ["-", "-"], "confidences": [0.0, 0.0]})

update_queue = queue.Queue()  # A queue for thread-safe communication

detected_card_values = []  # Initialize a list to store the values of detected cards

card_value_counts = defaultdict(int)  # Initialize outside to maintain counts across rounds

# Global variables for tracking card detections
first_cards_detected = set()
second_cards_detected = set()
players_received_first_card = set()

counted_cards_this_round = set()  # Initialize outside to maintain unique counts across rounds

rounds_observed = 0  # Initialize a counter for the number of rounds observed

api_key = "WBy7jG6AiiqjzifOfiNH"

# Initialize for player cards
project_id_players = "dey022"  # Project ID for player cards
rf_players = Roboflow(api_key=api_key)
project_players = rf_players.workspace().project(project_id_players)
model_players = project_players.version(1).model

# Initialize for dealer cards
project_id_dealer = "carddetection-v1hqz"  # Project ID for dealer cards
rf_dealer = Roboflow(api_key=api_key)
project_dealer = rf_dealer.workspace().project(project_id_dealer)
model_dealer = project_dealer.version(17).model


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


def get_card_name(class_label):  # Function to get card name from class label
    return class_mapping.get(class_label, "Unknown")


def get_card_value(card_name):  # Mapping function for card names to Blackjack values
    value_mapping = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "Jack": 10, "Queen": 10, "King": 10, "Ace": 1  # Count Ace as 1 for simplicity
    }
    card_value = card_name.split(' ')[0]  # Extract the card's face value (ignore suit for now, e.g., " of Spades")
    return value_mapping.get(card_value, 0)  # Return 0 if card name is not recognized


def capture_screen_and_track_cards(monitor_number=0):  # Continuously capture screen and perform object detection
    global player_cards, card_value_counts, counted_cards_this_round, true_count, rounds_observed

    detection_start_time = time.time()
    minimum_detection_duration = 2  # Minimum duration in seconds before checking if all cards have been detected
    initial_cards_received = defaultdict(bool)  # Initialize flag for each player

    while True:
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

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:  # Save image to a temporary file
            temp_file_path = temp_file.name
            img.save(temp_file_path)

        # Dealer's card detection area
        left = 1548
        upper = 12
        width, height = 1000, 800
        right = left + width  # Calculate the right boundary
        lower = upper + height  # Calculate the lower boundary

        dealer_area = img.crop((left, upper, right, lower))  # Crop to dealer's area
        dealer_cards = capture_dealer_cards(dealer_area, model_dealer)  # Pass cropped image to dealer's model
        print(f"Dealer's cards: {dealer_cards}")

        # Predict using object detection model
        predictions = model_players.predict(temp_file_path, confidence=70, overlap=100).json()['predictions']
        print(f"{len(predictions)} predictions")

        # Process each prediction
        for prediction in predictions:
            x, y, class_label, confidence = prediction['x'], prediction['y'], prediction['class'], prediction[
                'confidence']
            card_name = get_card_name(class_label)

            update_count(card_name)

            for i, region in enumerate(player_regions):
                if region.contains_point([x, y]):
                    detected_card = {'x': x, 'y': y, 'confidence': confidence, 'card_name': card_name}
                    if not is_duplicate_or_nearby_card(detected_card, player_cards[i]['cards']):
                        add_or_update_player_card(detected_card, player_cards[i], card_name)

        # Calculate true_count before making decisions
        calculate_true_count()
        process_player_decisions_and_print_info(initial_cards_received, dealer_cards)

        # Delete temporary file
        os.unlink(temp_file_path)

        # Check if the minimum detection duration has elapsed before considering to break the loop
        current_time = time.time()
        if (current_time - detection_start_time) > minimum_detection_duration:
            if second_cards_detected and players_received_first_card.issubset(second_cards_detected):
                print_all_cards()
                break

    # Update the GUI with the detected dealer cards
    if dealer_cards:
        card_info = {'dealer_card': dealer_cards[0]}  # Format the data as a dictionary
    else:
        card_info = {'dealer_card': "No card detected"}  # Handle the case where no card is detected

    display1.update_dealer_card_display(card_info)
    rounds_observed += 1


def process_player_decisions_and_print_info(initial_cards_received, dealer_cards):
    global player_cards, true_count, base_bet, rounds_observed

    dealer_up_card = dealer_cards[0] if dealer_cards else "Unknown"
    players_cards_data = []

    for player_index, player_data in sorted(player_cards.items(), key=lambda x: x[0]):
        cards = player_data['cards']

        # Calculate hand value and update initial card receipt status
        if len(cards) == 2 and not initial_cards_received[player_index]:
            initial_cards_received[player_index] = True

        # Once all players have received their initial two cards, print all cards
        if all(initial_cards_received.values()):
            print_all_cards()

        # Make decision recommendations based on the current state
        if initial_cards_received[player_index] or len(cards) > 2:
            decision_recommendations = blackjack_decision(cards, dealer_up_card, true_count, base_bet)
            previous_recommendation = player_data.get('recommendation')

            if previous_recommendation != decision_recommendations:
                player_data['recommendation'] = decision_recommendations
                print_player_cards(player_index, cards, decision_recommendations)
            elif not previous_recommendation:
                player_data['recommendation'] = decision_recommendations
                print_player_cards(player_index, cards, decision_recommendations)

        # Prepare player's cards data for GUI update
        players_cards_data.append({'player_index': player_index, 'cards': cards})

    # Update the GUI with the detected player cards
    display1.update_player_cards_display(players_cards_data)

    if rounds_observed > 3:
        betting_strategy = bet_strategy(true_count, base_bet)
        print(f"\nBetting strategy for next round: {betting_strategy}")
    else:
        print(
            f"\nAccumulating card count data, betting strategy recommendations will start after {3 - rounds_observed} more rounds.")


def is_duplicate_or_nearby_card(detected_card, existing_cards):
    return detected_card['card_name'] in existing_cards


def add_or_update_player_card(detected_card, player_info, card_name):
    # This assumes that 'cards' is a list of card names and 'confidences' is a list of confidence values.
    if "-" in player_info['cards']:
        replace_index = player_info['cards'].index("-")
        player_info['cards'][replace_index] = card_name
        player_info['confidences'][replace_index] = detected_card['confidence']
    else:
        player_info['cards'].append(card_name)
        player_info['confidences'].append(detected_card['confidence'])
    # Ensure unique identification for counted cards, might need adjustment if spatial data is to be included
    counted_cards_this_round.add(card_name)


# Function to print player's cards and recommendation
def print_player_cards(player_index, cards, recommendation):
    hand_value = calculate_hand_value(cards)
    cards_info = " // ".join([f"[{i + 1}] {card}" for i, card in enumerate(cards)])
    print(
        f"P{player_index + 1}: {cards_info}. Card value is {hand_value}. Recommended action: {', '.join(recommendation)}.")


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

    display1.update_dealer_card_display(dealer_cards)  # Update the GUI with the first dealer card
    return dealer_cards


def is_soft_hand(cards):
    # Check if the hand is a soft hand (contains an Ace counted as 11)
    values = [get_card_value(card.split(' ')[0]) for card in cards]
    return 1 in values and sum(values) + 10 <= 21


def is_pair_hand(cards):  # Check if the hand is a pair
    if len(cards) != 2:
        return False
    return get_card_value(cards[0].split(' ')[0]) == get_card_value(cards[1].split(' ')[0])


def get_hand_representation(cards):
    # Check for pairs of tens or face cards
    if len(cards) == 2 and all(card.split(' ')[0] in ['10', 'Jack', 'Queen', 'King'] for card in cards):
        return "10,10"  # Represent as a pair of tens for strategy lookup

    if is_soft_hand(cards):  # Existing logic for soft hands
        non_ace_total = sum([get_card_value(card.split(' ')[0]) for card in cards if card.split(' ')[0] != 'Ace'])
        return f"A,{non_ace_total}"

    elif is_pair_hand(cards):  # Check if the hand is a pair
        value = get_card_value(cards[0].split(' ')[0])
        return f"{value},{value}"

    else:  # For other hands, return the total value
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


def get_dealer_card_value(card):  # Helper function to interpret the dealer's card correctly
    if card in ["J", "Q", "K"]:
        return "10"
    return card


def calculate_hand_value(cards):
    total, aces = 0, 0

    for card in cards:  # Calculate the total value and count the aces
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

        cards_info = []
        for i, (card, conf) in enumerate(zip(cards, confidences), start=1):
            cards_info.append(f"[{i}] {card} (C: {conf * 100:.2f}%)")

        print(f"P{player_index + 1}: {' // '.join(cards_info)}")

    formatted_card_counts = [f"{value} => {count}x" for value, count in
                             sorted(card_value_counts.items(), key=lambda item: item[0])]
    total_cards = sum(card_value_counts.values())  # Sum up the counts of all cards for the total
    print(f"Card counts ({total_cards}):\n", "\n".join(formatted_card_counts))


def background_processing():
    while True:
        reset_for_new_round()  # Prepare for a new round
        capture_screen_and_track_cards(monitor_number=0)


def reset_for_new_round():
    global player_cards, first_cards_detected, second_cards_detected, players_received_first_card, card_value_counts, counted_cards_this_round
    player_cards.clear()
    first_cards_detected.clear()
    second_cards_detected.clear()
    card_value_counts.clear()
    players_received_first_card.clear()
    counted_cards_this_round.clear()  # Reset the set of counted cards for the new round
    print("Reset for new round.")


def start_gui():
    display1.initialize_gui()  # Initialize the GUI
    display1.window.after(10, display1.update_gui_from_queue)  # Schedule the first call to update GUI from the queue
    display1.run_gui()  # Start the Tkinter event loop


if __name__ == "__main__":
    # Start background processing in a separate thread
    threading.Thread(target=background_processing, daemon=True).start()

    start_gui()  # Call the function to start the GUI after the setup is complete
