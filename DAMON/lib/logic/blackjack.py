from collections import defaultdict
import os
import cv2
import csv
import time
import tempfile
import tkinter as tk
from PIL import Image, ImageTk

from ..common import constants
from .utils import Utils
from .card_handler import CardHandler
from .card_utils import CardUtils
from .monitor_utils import MonitorUtils
from .decision_making import DecisionMaking


class BlackjackLogic:
    def __init__(self, gui):
        self.round_count = 0
        self.rounds_observed = 0
        self.dealer_value = 0
        self.minimum_detection_duration = 2
        self.dealer_up_card = None
        self.dealer_value = None
        self.dealer_card_label = None
        self.base_bet = constants.BASE_BET
        self.deck_count = constants.DECK_COUNT
        self.blackjack_strategy = {}
        self.cards_info = []
        self.recommendations = []
        self.players_cards_data = []
        self.player_cards_labels = []
        self.players_decision_labels = []
        self.first_card_detected = set()
        self.second_card_detected = set()
        self.players_received_first_card = set()
        self.utils = Utils()
        self.gui = gui
        self.card_handler = CardHandler()
        self.card_utils = CardUtils()
        self.monitor_utils = MonitorUtils()
        self.decision_making = DecisionMaking()
        self.model_players = self.utils.initialize_player_model()
        self.model_dealer = self.utils.initialize_dealer_model()
        self.card_value_counts = defaultdict(int)
        self.player_cards = defaultdict(lambda: {"cards": ["-", "-"], "confidences": [0.0, 0.0]})
        self.detection_timers = defaultdict(lambda: {"first_card": None, "second_card": None})
        self.locked_first_cards = set()
        self.locked_second_cards = set()
        self.player_regions = []

    def initialize_screenshot(self):
        self.detection_start_time = time.time()
        self.captured_screenshot = self.monitor_utils.capture_screen()

    def populate_blackjack_strategy(self):  # Read the CSV file and populate the strategy dictionary
        with open(constants.CSV_FILE_PATH, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                dealer_card, player_hand, action = row
                self.blackjack_strategy[(dealer_card, player_hand)] = action

    def set_monitor(self, monitor):
        self.monitor_utils.set_monitor(monitor)
        self.initialize_screenshot()  # Initialize screenshot after monitor is set

    def capture_screen_and_track_cards(self):  # Continuously capture screen and perform object detection
        initial_cards_received = defaultdict(bool)
        current_resolution = self.monitor_utils.get_current_resolution()
        scale_x, scale_y = self.monitor_utils.get_scaling_factors(constants.BASE_RESOLUTION, current_resolution)
        self.player_regions = self.monitor_utils.scale_player_regions(constants.BASE_PLAYER_REGIONS, scale_x, scale_y)

        while True:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file_path = temp_file.name
                self.captured_screenshot.save(temp_file_path)

            # Crop to dealer's area
            dealer_area = self.captured_screenshot.crop(
                (constants.DEALER_AREA_LEFT,
                 constants.DEALER_AREA_UPPER,
                 constants.DEALER_AREA_RIGHT,
                 constants.DEALER_AREA_LOWER))

            # Pass cropped image to dealer's model
            dealer_cards = self.card_handler.capture_dealer_cards(dealer_area, self.model_dealer)
            print(f"Dealer's cards: {dealer_cards}")

            predictions = self.model_players.predict(temp_file_path,
                                                     confidence=constants.PREDICTION_CONFIDENCE_PLAYERS,
                                                     overlap=constants.PREDICTION_OVERLAP_PLAYERS).json()['predictions']

            print(f"{len(predictions)} predictions")

            for prediction in predictions:
                x, y, class_label, confidence = prediction['x'], prediction['y'], prediction['class'], prediction[
                    'confidence']
                card_name = self.card_utils.get_card_name(class_label)

                self.card_utils.update_card_counter(card_name, 1)
                self.card_handler.handle_card_detection(card_name)

                for i, region in enumerate(self.player_regions):
                    if region.contains_point([x, y]):
                        detected_card = {'x': x, 'y': y, 'confidence': confidence, 'card_name': card_name}
                        self.handle_card_detection(i, detected_card)

            self.card_utils.true_count = self.card_utils.calculate_true_count()
            self.process_player_decisions_and_print_info(initial_cards_received, dealer_cards)

            # Delete temporary file
            os.unlink(temp_file_path)

            if (time.time() - self.detection_start_time) > self.minimum_detection_duration:
                if self.second_card_detected and self.players_received_first_card.issubset(self.second_card_detected):
                    self.card_handler.print_all_cards(self.player_cards, self.card_utils.card_counters)
                    break

        # Update the GUI with the detected dealer cards
        self.update_dealer_card_display(dealer_cards)
        self.rounds_observed += 1

    def handle_card_detection(self, player_index, detected_card):
        current_time = time.time()
        card_slot = "first_card" if player_index not in self.locked_first_cards else "second_card"

        if card_slot == "first_card" and player_index in self.locked_first_cards:
            return

        if card_slot == "second_card" and player_index in self.locked_second_cards:
            return

        timer = self.detection_timers[player_index][card_slot]

        if timer is None:
            self.detection_timers[player_index][card_slot] = {"time": current_time, "card": detected_card}
        else:
            elapsed_time = current_time - timer["time"]
            if elapsed_time > 2:  # 2-second delay to ensure highest confidence card is selected
                if detected_card["confidence"] > timer["card"]["confidence"]:
                    self.detection_timers[player_index][card_slot] = {"time": current_time, "card": detected_card}
                self.player_cards[player_index]["cards"][0 if card_slot == "first_card" else 1] = \
                    self.detection_timers[player_index][card_slot]["card"]["card_name"]
                self.player_cards[player_index]["confidences"][0 if card_slot == "first_card" else 1] = \
                    self.detection_timers[player_index][card_slot]["card"]["confidence"]
                self.detection_timers[player_index][card_slot] = None

                if card_slot == "first_card":
                    self.locked_first_cards.add(player_index)
                    self.players_received_first_card.add(player_index)
                else:
                    self.locked_second_cards.add(player_index)
                    self.second_card_detected.add(player_index)

    def blackjack_decision(self, player_cards, dealer_up_card, true_count, base_bet):
        dealer_value = self.card_utils.get_dealer_card_value(dealer_up_card) if dealer_up_card != "Unknown" else "0"
        hand_value = self.card_utils.calculate_hand_value(player_cards)

        # Directly handle values where the strategy is universally to stand
        if hand_value >= 17:
            action = "S"  # Stand on 17, 18, 19, 20, 21
        else:
            hand_representation = self.card_utils.get_hand_representation(player_cards)

            if self.card_utils.is_pair_hand(player_cards):
                pair_value = str(self.card_utils.get_card_value(player_cards[0].split(' ')[0]))
                pair_value = "10" if pair_value in ["J", "Q", "K"] else pair_value
                action_key = (dealer_value, f"Pair {pair_value}")
            else:
                action_key = (dealer_value, hand_representation)

            action = self.blackjack_strategy.get(action_key, "?")

        self.recommendations.append(constants.ACTION_MAPPING.get(action, "Hit" if action == "?" else action))
        return self.recommendations

    def process_player_decisions_and_print_info(self, initial_cards_received, dealer_cards):
        dealer_up_card = dealer_cards[0] if dealer_cards else "Unknown"

        for player_index, player_data in sorted(self.player_cards.items(), key=lambda x: x[0]):
            cards = player_data['cards']

            # Calculate hand value and update initial card receipt status
            if len(cards) == 2 and not initial_cards_received[player_index]:
                initial_cards_received[player_index] = True

            # Once all players have received their initial two cards, print all cards
            if all(initial_cards_received.values()):
                self.print_all_cards()

            # Make decision recommendations based on the current state
            if initial_cards_received[player_index] or len(cards) > 2:
                decision_recommendations = self.blackjack_decision(cards, dealer_up_card, self.card_utils.true_count,
                                                                   constants.BASE_BET)
                previous_recommendation = player_data.get('recommendation')

                if previous_recommendation != decision_recommendations:
                    player_data['recommendation'] = decision_recommendations
                    self.card_utils.print_player_cards(player_index, cards, decision_recommendations)
                elif not previous_recommendation:
                    player_data['recommendation'] = decision_recommendations
                    self.card_utils.print_player_cards(player_index, cards, decision_recommendations)

            # Prepare player's cards data for GUI update
            self.players_cards_data.append({'player_index': player_index, 'cards': cards})

        # Update the GUI with the detected player cards
        self.update_player_cards_display(self.players_cards_data, dealer_up_card, self.card_utils.true_count,
                                         constants.BASE_BET)

        if self.rounds_observed > 3:
            betting_strategy = self.decision_making.bet_strategy(self.card_utils.true_count, constants.BASE_BET)
            print(f"\nBetting strategy for next round: {betting_strategy}")
        else:
            print(
                f"\nAccumulating card count data, betting strategy recommendations will start after {3 - self.rounds_observed} more rounds.")

    # Print all cards in console
    def print_all_cards(self):
        self.cards_info.clear()  # Clear previous card info
        for player_index in sorted(self.player_cards):
            cards = self.player_cards[player_index]['cards']
            confidences = self.player_cards[player_index]['confidences']

            card_info = []
            for i, (card, conf) in enumerate(zip(cards, confidences), start=1):
                card_info.append(f"[{i}] {card} (C: {conf * 100:.2f}%)")

            print(f"P{player_index + 1}: {' // '.join(card_info)}")

        formatted_card_counts = [f"{value} => {count}x" for value, count in
                                 sorted(self.card_value_counts.items(), key=lambda item: item[0])]
        total_cards = sum(self.card_value_counts.values())  # Sum up the counts of all cards for the total
        print(f"Card counts ({total_cards}):\n", "\n".join(formatted_card_counts))

    def update_player_cards_display(self, player_data_list, dealer_up_card, true_count, base_bet):
        self.clear_player_cards()  # Clear the displayed cards

        start_y = 10  # Fixed starting y-position for the cards
        total_width = 7 * (constants.CARD_WIDTH + constants.CARD_SPACING) + 8  # Total width for all columns
        column_width = (total_width - 8) // 7  # Width of each column without borders

        for i in range(7):  # Iterate over 7 players to create columns
            start_x = i * (column_width + constants.CARD_SPACING) + constants.CARD_SPACING
            self.gui.canvas.create_rectangle(start_x, start_y, start_x + column_width,
                                             start_y + self.gui.winfo_height(),
                                             outline="black", width=1)
            # Check if player data is available for the current index
            player_data = next((data for data in player_data_list if data['player_index'] == i), None)
            cards = player_data['cards'] if player_data else ['-', '-']  # Ensure two card slots
            player_number = i + 1  # Player index (1-indexed)

            card_display_y = start_y
            for card in cards:
                if card != '-':  # Only create labels for detected cards
                    photo_img = self.get_card_image(card)
                    card_label = tk.Label(self.gui.canvas, image=photo_img)
                    card_label.image = photo_img  # Keep a reference to prevent garbage collection
                    card_label.place(x=start_x, y=card_display_y)
                    self.player_cards_labels.append(card_label)  # Store the label in a list
                    card_display_y += constants.CARD_HEIGHT + 10  # Update starting y-position for the next card

            # Reset starting y-position for the next player's cards
            start_y = 10
            self.create_label(f"Player {player_number}", start_x + column_width // 2,
                              self.gui.winfo_height() - 10,
                              anchor="s")

            decision = self.blackjack_decision(cards, dealer_up_card, true_count, base_bet)[0] if player_data else "-"
            decision_label = self.create_label(f"Decision: {decision}", start_x + column_width // 2,
                                               card_display_y + 20, anchor="n")
            self.players_decision_labels.append(decision_label)  # Add to the global list

    def update_dealer_card_display(self, dealer_cards):
        # Check if dealer_cards is not empty and then proceed; otherwise, use a default value or image
        card_value = dealer_cards[0] if dealer_cards else "No card detected"

        if self.dealer_card_label is None:
            self.dealer_card_label = tk.Label(self.gui)  # Create the label if it doesn't exist
            self.dealer_card_label.place(relx=1.0,
                                         rely=0.0,
                                         x=-10,
                                         y=20,
                                         anchor='ne')  # Place it in the top-right corner

        if card_value != "No card detected":
            # Correct the file name format here based on the actual card value
            # Assuming card_value is a string like 'Jack', '2', etc.
            card_image_path = f"{constants.CARD_FOLDER_PATH}/{card_value.lower()}_of_spades.png"  # Ensure lowercase for consistency
        else:
            card_image_path = constants.DEFAULT_CARD_IMAGE_PATH

        try:
            img = Image.open(card_image_path)
            img = img.resize((100, 150))  # Resize the image to fit the label
            imgtk = ImageTk.PhotoImage(image=img)
            self.dealer_card_label.config(image=imgtk)
            self.dealer_card_label.image = imgtk  # Keep a reference to avoid garbage collection
        except FileNotFoundError:
            print(f"Image file not found: {card_image_path}")

    def get_card_image(self, card):
        card_image_path = self.utils.generate_card_image_path(card)  # Generate file path for the card image
        if not os.path.exists(card_image_path):  # Check if the card image exists
            print(f"Card image not found: {card_image_path}. Using default image.")
            card_image_path = constants.DEFAULT_CARD_IMAGE_PATH
        # Load the card image
        card_image = cv2.imread(card_image_path)
        # Convert BGR image to RGB and then to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB))
        # Resize the image and convert to PhotoImage
        return ImageTk.PhotoImage(pil_image.resize((constants.CARD_WIDTH, constants.CARD_HEIGHT)))

    def create_label(self, text, x, y, anchor="n"):
        label = tk.Label(self.gui.canvas, text=text)
        label.place(x=x, y=y, anchor=anchor)
        return label

    # Logic to reset some stuff for a new round
    def reset_for_new_round(self):
        self.player_cards.clear()
        self.players_cards_data.clear()

        empty_image = tk.PhotoImage()
        for label in self.player_cards_labels:
            label.config(image=empty_image)  # Set the label to use the empty image
            label.image = empty_image  # Keep a reference to prevent garbage collection

        # Reset the dealer card label to either an empty image or a default card image
        if self.dealer_card_label:  # Check if the dealer_card_label exists
            self.dealer_card_label.config(
                image=empty_image)  # Set the dealer card to use the empty image
            self.dealer_card_label.image = empty_image  # Keep a reference to prevent garbage collection
            # If you have a default dealer card image, load it here instead of setting an empty image

        # Reset each player's decision label text
        for decision_label in self.players_decision_labels:
            decision_label.config(text="")  # Clear the decision text

        self.round_count += 1
        self.gui.round_label.config(
            text=f"Round: {self.round_count}")  # Update the round label with the new round count

        self.players_received_first_card.clear()
        self.first_card_detected.clear()
        self.second_card_detected.clear()
        self.locked_first_cards.clear()
        self.locked_second_cards.clear()
        self.card_value_counts.clear()
        self.card_utils.counted_cards_this_round.clear()  # Reset the set of counted cards for the new round
        print("Reset for new round.")

    def clear_player_cards(self):
        for label in self.player_cards_labels:
            label.destroy()  # This removes the label from the canvas
        self.player_cards_labels.clear()  # Clear the list for the next round
