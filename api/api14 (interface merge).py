# TODO: Implement system to detect splitted hands
# TODO: Fix 2x Ace being counted as "12", instead of 2 (and thus split)
# TODO: Make player_regions Scalable (e.g. 1920x1080, now 2560x1440)
# TODO: Finetune accuracy for both models (adjust parameters)
# TODO: Refine player regions
# TODO: Clean up code

# UI:
# TODO: Fix player count
# TODO: Count total per-card amount
# TODO: Manually add/remove counted cards
# TODO: Implement advice per player according to default cheat sheet in
# TODO: Implement OCR for Saldo and Current bet

import os
import csv
import cv2
import time
import queue
import tempfile
import threading
import numpy as np
import tkinter as tk

from mss import mss
from roboflow import Roboflow
from PIL import Image, ImageTk
from tkinter import PhotoImage
from matplotlib.path import Path
from collections import defaultdict
from screeninfo import get_monitors

from class_mapping import class_mapping
from dealer_class_mapping2 import dealer_class_mapping

global dealer_value, dealer_up_card

window = None
canvas = None
round_label = None
reset_button = None
dealer_value_label = None
dealer_card_label = None

round_count = 0
base_bet = 10
running_count = 0
true_count = 0
deck_count = 8  # 8-deck shoe at the start
rounds_observed = 0  # Initialize a counter for the number of rounds observed
CARD_WIDTH = 73
CARD_HEIGHT = 98
CARD_SPACING = 10

CARD_FOLDER = "cards"
DEFAULT_CARD_IMAGE_PATH = "cards/red card.jpg"

player_card_labels = []
player_decision_labels = []
card_counter_labels = []
detected_card_values = []

first_cards_detected = set()
second_cards_detected = set()
players_received_first_card = set()
counted_cards_this_round = set()

card_value_counts = defaultdict(int)  # Initialize outside to maintain counts across rounds

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

value_mapping = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "Jack": 10, "Queen": 10, "King": 10, "Ace": 1  # Count Ace as 1 for simplicity
}

card_values = ["Ace", "10", "9", "8", "7", "6", "5", "4", "3", "2"]

card_counters = {str(value): 0 for value in set(value_mapping.values())}

action_mapping = {  # No need to map D/H, D/S, P/H and R/H
    "H": "Hit",
    "S": "Stand",
    "P": "Split",
}

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


def handle_card_detection(card_name):
    global counted_cards_this_round

    # Simplify the card name to its basic value for counting purposes
    card_key = card_name.split(' ')[0]
    if card_key in ["Jack", "Queen", "King"]:  # Normalize face cards to "10"
        card_key = "10"

    # Check if this card (in its simplified form) has already been counted in this round
    if card_key not in counted_cards_this_round:
        update_card_counter(card_key, 1)  # Update the counter for this card
        counted_cards_this_round.add(card_key)  # Mark this card as counted for the current round


def update_card_counter(card_name, increment=1):
    # Extract just the value from the card name (e.g., '7 of Diamonds' -> '7')
    card_value_name = card_name.split(' ')[0]  # Assumes card_name format is "Value of Suit"

    # Convert the card name to its value using value_mapping
    if card_value_name in value_mapping:
        card_value = value_mapping[card_value_name]
    else:
        print(f"Warning: Card value '{card_value_name}' not found in value mapping.")
        return

    # Convert card value back to string for display purposes
    card_value_str = str(card_value)

    # Increment the card counter
    if card_value_str in card_counters:
        card_counters[card_value_str] += increment
        # Find the corresponding label for the card value and update its text
        for label in card_counter_labels:
            if label.cget("text").startswith(card_value_str + ":"):  # Check if the label starts with the card value
                new_text = f"{card_value_str}: {card_counters[card_value_str]}x"
                label.config(text=new_text)
                break
    else:
        print(f"Warning: Card value '{card_value_str}' not found in card counters.")


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

            handle_card_detection(card_name)  # Update the card counter

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

    update_dealer_card_display(card_info)
    rounds_observed += 1


def process_player_decisions_and_print_info(initial_cards_received, dealer_cards):
    global player_cards, true_count, base_bet, rounds_observed, players_cards_data

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
    update_player_cards_display(players_cards_data, dealer_up_card, true_count, base_bet)

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

    update_dealer_card_display(dealer_cards)  # Update the GUI with the first dealer card
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
        # reset_for_new_round()  # Prepare for a new round
        capture_screen_and_track_cards(monitor_number=0)


def reset_for_new_round():
    global player_cards, players_cards_data, player_card_labels, dealer_card_label, first_cards_detected, second_cards_detected, players_received_first_card, card_value_counts, counted_cards_this_round, round_count, player_decision_labels
    player_cards.clear()
    players_cards_data.clear()

    empty_image = PhotoImage()
    for label in player_card_labels:
        label.config(image=empty_image)  # Set the label to use the empty image
        label.image = empty_image  # Keep a reference to prevent garbage collection

    # Reset the dealer card label to either an empty image or a default card image
    if dealer_card_label:  # Check if the dealer_card_label exists
        dealer_card_label.config(image=empty_image)  # Set the dealer card to use the empty image
        dealer_card_label.image = empty_image  # Keep a reference to prevent garbage collection
        # If you have a default dealer card image, load it here instead of setting an empty image

    # Reset each player's decision label text
    for decision_label in player_decision_labels:
        decision_label.config(text="")  # Clear the decision text

    round_count += 1
    round_label.config(text=f"Round: {round_count}")  # Update the round label with the new round count

    first_cards_detected.clear()
    second_cards_detected.clear()
    card_value_counts.clear()
    players_received_first_card.clear()
    counted_cards_this_round.clear()  # Reset the set of counted cards for the new round
    print("Reset for new round.")


def clear_player_cards():
    global player_card_labels

    for label in player_card_labels:
        label.destroy()  # This removes the label from the canvas
    player_card_labels.clear()  # Clear the list for the next round


def initialize_gui():
    global window, canvas, dealer_value_label, player_card_labels, card_counter_labels, canvas, round_label, reset_button, round_count
    window = tk.Tk()
    window.title("Blackjack AI - API v14")
    window.geometry("1200x800")

    canvas = tk.Canvas(window)
    canvas.pack(fill=tk.BOTH, expand=True)

    dealer_value_label = tk.Label(canvas, text="Dealer has: ")
    dealer_value_label.place(relx=1.0, rely=0.0, x=-50, y=0, anchor='ne')  # Adjusted for top-right with padding

    round_label = tk.Label(window, text=f"Round: {round_count}")
    round_label.place(x=10, y=5)

    reset_button = tk.Button(window, text="Reset Round", command=reset_for_new_round)
    reset_button.pack()
    reset_button.place(x=100, y=5)

    start_y = 750  # Start position from the bottom of an 800px tall window
    label_height = 20  # Estimated height of each label
    spacing = 5  # Spacing between labels

    for card_value in card_values:  # No need to reverse since we're starting from the bottom now
        text = f"{card_value}: 0x"
        label = tk.Label(canvas, text=text, anchor='e')
        label.place(relx=0.95, y=start_y, anchor="se")  # Adjust relx and anchor if needed
        card_counter_labels.append(label)
        start_y -= (label_height + spacing)  # Move up for the next label


def check_for_updates():
    global cards_displayed

    if not update_queue.empty():
        data = update_queue.get()
        if 'dealer_card' in data:
            update_dealer_card_display(data['dealer_card'])
    # Cancel any previously scheduled executions and schedule the function to run again
    window.after(10, check_for_updates)


def update_dealer_card_display(dealer_cards):
    global dealer_card_label, window, CARD_FOLDER, DEFAULT_CARD_IMAGE_PATH

    # Check if dealer_cards is not empty and then proceed; otherwise, use a default value or image
    if dealer_cards and len(dealer_cards) > 0:
        card_value = dealer_cards[0]  # Assuming the first element is the dealer's card value
    else:
        # Handle the case where no dealer card is detected by setting a default
        card_value = "No card detected"

    if dealer_card_label is None:
        dealer_card_label = tk.Label(window)  # Create the label if it doesn't exist
        dealer_card_label.place(relx=1.0, rely=0.0, x=-10, y=20, anchor='ne')  # Place it in the top-right corner

    if card_value != "No card detected":
        # Correct the file name format here based on the actual card value
        # Assuming card_value is a string like 'Jack', '2', etc.
        card_image_path = f"{CARD_FOLDER}/{card_value.lower()}_of_spades.png"  # Ensure lowercase for consistency
    else:
        card_image_path = DEFAULT_CARD_IMAGE_PATH

    try:
        img = Image.open(card_image_path)
        img = img.resize((100, 150))  # Resize the image to fit the label
        imgtk = ImageTk.PhotoImage(image=img)
        dealer_card_label.config(image=imgtk)
        dealer_card_label.image = imgtk  # Keep a reference to avoid garbage collection
    except FileNotFoundError:
        print(f"Image file not found: {card_image_path}")


def cv2_to_pil(image):
    # Convert an OpenCV image (BGR) to a PIL image (RGB)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def generate_card_image_path(card):
    card = card.replace(" ", "_")
    return os.path.join(CARD_FOLDER, f"{card}.png")


def update_player_cards_display(player_data_list, dealer_up_card, true_count, base_bet):
    global player_card_labels, canvas

    clear_player_cards()  # Clear the displayed cards

    start_y = 10  # Fixed starting y-position for the cards

    # Calculate total width required for all players' cards
    total_width = 7 * (CARD_WIDTH + CARD_SPACING)  # 7 columns
    total_width += 8  # 1px border around each column
    column_width = (total_width - 8) // 7  # Width of each column without borders

    # Iterate over each player's data
    for i, player_data in enumerate(player_data_list):
        # Calculate starting x-position for the player's cards column
        start_x = i * (column_width + CARD_SPACING) + CARD_SPACING

        # Draw border around the column
        canvas.create_rectangle(start_x, start_y, start_x + column_width, start_y + window.winfo_height(),
                                outline="black", width=1)

        cards = player_data.get('cards', ['-'])  # Default card if no cards detected
        player_number = i + 1  # Player index (1-indexed)

        # Create label for player number and position it at the bottom of the column
        player_label = tk.Label(canvas, text=f"Player {player_number}")
        player_label.place(x=start_x + column_width // 2, y=window.winfo_height() - 10, anchor="s")

        card_display_y = start_y

        for card in cards:  # Iterate over each card for the player
            card_image_path = generate_card_image_path(card)  # Generate file path for the card image

            if os.path.exists(card_image_path):  # Check if the card image exists, otherwise use default image
                card_image = cv2.imread(card_image_path)
            else:
                print(f"Card image not found: {card_image_path}. Using default image.")
                card_image = cv2.imread(DEFAULT_CARD_IMAGE_PATH)

            card_image_rgb = cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB

            pil_image = Image.fromarray(card_image_rgb)  # Convert RGB image to PIL Image

            resized_image = pil_image.resize((CARD_WIDTH, CARD_HEIGHT))  # Resize the image

            photo_img = ImageTk.PhotoImage(resized_image)  # Convert PIL image to PhotoImage

            # Create image label and position it
            card_label = tk.Label(canvas, image=photo_img)
            card_label.image = photo_img  # Keep a reference to prevent garbage collection
            card_label.place(x=start_x, y=start_y)

            player_card_labels.append(card_label)  # Store the label in a list

            # Update starting y-position for the next card (stacked upwards)
            start_y += CARD_HEIGHT + 10
        #
        # Create label for player number and position it at the bottom of the displayed cards
        player_label = tk.Label(canvas, text=f"Player {player_number}")
        player_label.place(x=start_x + column_width // 2, y=card_display_y, anchor="n")

        # Get the blackjack decision for this player
        decision = blackjack_decision(cards, dealer_up_card, true_count, base_bet)[
            0]  # Assuming decision returns a list

        # Example of adding a decision label to the list
        decision_label = tk.Label(canvas, text=f"Decision: {decision}")
        # Position the label as required
        decision_label.place(x=start_x + column_width // 2, y=card_display_y + 20, anchor="n")
        player_decision_labels.append(decision_label)  # Add to the global list
        #

        start_y = 10  # Reset starting y-position for the next player's cards


def update_gui_from_queue():
    try:
        while not update_queue.empty():
            data = update_queue.get_nowait()  # Use get_nowait() to avoid blocking
            if isinstance(data, dict):
                if 'dealer_card' in data:
                    update_dealer_card_display(data['dealer_card'])
                elif 'players_cards' in data:
                    update_player_cards_display(data['players_cards'], dealer_up_card, true_count, base_bet)
    finally:
        window.after(10, update_gui_from_queue)


def start_gui():
    initialize_gui()  # Initialize the GUI
    window.after(10, update_gui_from_queue)  # Schedule the first call to update GUI from the queue
    window.mainloop()


if __name__ == "__main__":
    # Start background processing in a separate thread
    threading.Thread(target=background_processing, daemon=True).start()

    start_gui()  # Call the function to start the GUI after the setup is complete
