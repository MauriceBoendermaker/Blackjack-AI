import csv
import time
from collections import defaultdict
import os
import tempfile
from tkinter import font
import tkinter as tk
import threading

import cv2
from PIL import Image, ImageEnhance, ImageOps, ImageTk, ImageDraw, ImageFont

from ..common import constants, card_mappings
from .utils import Utils
from .card_handler import CardHandler
from .card_utils import CardUtils
from .monitor_utils import MonitorUtils
from .decision_making import DecisionMaking


class BlackjackLogic:
    class DetectionState:
        WAITING_FOR_FIRST_CARD = "WAITING_FOR_FIRST_CARD"
        FIRST_CARD_DETECTED = "FIRST_CARD_DETECTED"
        WAITING_FOR_SECOND_CARD = "WAITING_FOR_SECOND_CARD"
        SECOND_CARD_DETECTED = "SECOND_CARD_DETECTED"

    def __init__(self, gui):
        self.gui = gui
        self.round_count = 0
        self.rounds_observed = 0
        self.dealer_value = 0
        self.minimum_detection_duration = 2
        self.dealer_up_card = None
        self.dealer_value = None
        self.dealer_card_label = None
        self.base_bet = constants.BASE_BET
        self.deck_count = constants.DECK_COUNT
        self.detected_card = {}
        self.blackjack_strategy = {}
        self.load_strategy()
        self.create_dealer_card_placeholder()
        self.lock = threading.Lock()
        self.cards_info = []
        self.recommendations = []
        self.players_cards_data = []
        self.player_cards_labels = []
        self.players_decision_labels = defaultdict(list)
        self.first_card_detected = set()
        self.second_card_detected = set()
        self.players_received_first_card = set()
        self.utils = Utils()
        self.card_handler = CardHandler()
        self.card_utils = CardUtils()
        self.monitor_utils = MonitorUtils()
        self.decision_making = DecisionMaking()
        self.model_players = self.utils.initialize_player_model()
        self.model_dealer = self.utils.initialize_dealer_model()
        self.card_value_counts = defaultdict(int)
        self.initial_cards_received = defaultdict(bool)
        self.player_cards = defaultdict(lambda: {"cards": ["-", "-"], "confidences": [0.0, 0.0]})
        self.detection_timers = defaultdict(lambda: {"first_card": None, "second_card": None})
        self.detection_start_time = time.time()
        self.locked_cards = defaultdict(set)  # Track locked card indices separately
        self.manually_replaced_cards = defaultdict(set)  # Track manually replaced card indices separately
        self.player_regions = []
        self.detection_states = defaultdict(
            lambda: BlackjackLogic.DetectionState.WAITING_FOR_FIRST_CARD)  # Track detection states for each player

    def draw_predictions(self, image, predictions, output_path):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for prediction in predictions:
            if 'x' in prediction and 'y' in prediction and 'width' in prediction and 'height' in prediction:
                x, y = prediction['x'], prediction['y']
                width, height = prediction['width'], prediction['height']
                bbox = [x, y, x + width, y + height]
                label = prediction['class']
                draw.rectangle(bbox, outline="red", width=2)
                draw.text((x, y - 10), label, fill="red", font=font)
            else:
                print(f"Prediction missing coordinates: {prediction}")

    def initialize_screenshot(self):
        self.captured_screenshot = self.monitor_utils.capture_screen()

    def load_strategy(self):
        strategy = {}
        with open(constants.CSV_FILE_PATH, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                dealer_card, player_hand, action = row
                dealer_card = dealer_card.strip().upper()  # Normalize dealer card value
                player_hand = player_hand.strip().upper()  # Normalize player hand representation
                action = action.strip().upper()  # Normalize action
                strategy[(dealer_card, player_hand)] = action
        self.blackjack_strategy = strategy
        print("Loaded strategy:", self.blackjack_strategy)

    def set_monitor(self, monitor):
        self.monitor_utils.set_monitor(monitor)
        self.initialize_screenshot()

    def capture_screen_and_track_cards(self):
        current_resolution = self.monitor_utils.get_current_resolution()
        scale_x, scale_y = self.monitor_utils.get_scaling_factors(constants.BASE_RESOLUTION, current_resolution)
        self.player_regions = self.monitor_utils.scale_player_regions(constants.BASE_PLAYER_REGIONS, scale_x, scale_y)

        while True:
            self.initialize_screenshot()

            # Crop to dealer's area
            dealer_area = self.captured_screenshot.crop((constants.DEALER_AREA_LEFT,
                                                         constants.DEALER_AREA_UPPER,
                                                         constants.DEALER_AREA_RIGHT,
                                                         constants.DEALER_AREA_LOWER))

            # Save the cropped dealer area image for debugging
            debug_image_path = "dealer_area_current_view.jpg"
            dealer_area.save(debug_image_path)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file_path = temp_file.name
                dealer_area.save(temp_file_path)

            predictions_dealer = self.model_dealer.predict(temp_file_path,
                                                           confidence=constants.PREDICTION_CONFIDENCE_DEALER,
                                                           overlap=constants.PREDICTION_OVERLAP_DEALER).json()[
                'predictions']

            # Draw predictions on the dealer area image
            self.draw_predictions(dealer_area, predictions_dealer, "dealer_area_with_predictions.jpg")

            # Debug: Print the raw predictions
            print(f"Raw dealer predictions: {predictions_dealer}")

            dealer_card = []
            for prediction in predictions_dealer:
                class_label = prediction.get('class')
                card_name = card_mappings.dealer_class_mapping.get(class_label, "Unknown")
                dealer_card.append(card_name)

            # Debug: Print the identified dealer cards
            print(f"Dealer card: {dealer_card}")

            # Update the dealer's card information
            self.dealer_up_card = dealer_card[0] if dealer_card else "Unknown"

            # Use the updated dealer card in your game logic
            print(f"Updated dealer's card: {self.dealer_up_card}")

            # Update the dealer card display
            self.update_dealer_card_display(dealer_card)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file_path = temp_file.name
                self.captured_screenshot.save(temp_file_path)

            predictions_players = self.model_players.predict(temp_file_path,
                                                             confidence=constants.PREDICTION_CONFIDENCE_PLAYERS,
                                                             overlap=constants.PREDICTION_OVERLAP_PLAYERS).json()[
                'predictions']

            # Draw predictions on the full screenshot
            full_screenshot_with_predictions_path = "full_screenshot_with_predictions.jpg"
            self.draw_predictions(self.captured_screenshot, predictions_players, full_screenshot_with_predictions_path)

            print(f"{len(predictions_players)} predictions")

            for player_index, region in enumerate(self.player_regions):
                self.process_player_predictions(predictions_players, player_index, region)

            self.card_utils.calculate_true_count()
            self.process_player_decisions_and_print_info(self.initial_cards_received, self.dealer_up_card)

            os.unlink(temp_file_path)

            self.gui.update()

    def process_player_predictions(self, predictions, player_index, region):
        state = self.detection_states[player_index]

        if state == BlackjackLogic.DetectionState.SECOND_CARD_DETECTED:
            return

        best_card = None
        best_confidence = 0

        for prediction in predictions:
            x, y, class_label, confidence = prediction['x'], prediction['y'], prediction['class'], prediction[
                'confidence']
            if region.contains_point([x, y]):
                card_name = self.card_utils.get_card_name(class_label)
                if confidence > best_confidence and card_name not in self.player_cards[player_index]['cards']:
                    best_card = {'x': x, 'y': y, 'confidence': confidence, 'card_name': card_name}
                    best_confidence = confidence

        if best_card:
            detected_card = best_card
            print(f"Detected card for player {player_index} in state {state}: {detected_card}")

            if state == BlackjackLogic.DetectionState.WAITING_FOR_FIRST_CARD:
                self.lock_and_update_player_card(player_index, detected_card, card_index=0)
                self.detection_states[player_index] = BlackjackLogic.DetectionState.FIRST_CARD_DETECTED
                print(f"Transitioned to FIRST_CARD_DETECTED for player {player_index}")

            elif state == BlackjackLogic.DetectionState.FIRST_CARD_DETECTED:
                self.lock_and_update_player_card(player_index, detected_card, card_index=1)
                self.detection_states[player_index] = BlackjackLogic.DetectionState.SECOND_CARD_DETECTED
                print(f"Transitioned to SECOND_CARD_DETECTED for player {player_index}")

    def lock_and_update_player_card(self, player_index, detected_card, card_index):
        self.locked_cards[player_index].add(card_index)
        self.detection_timers[player_index][f"card_{card_index}"] = time.time()
        print(f"Card {card_index} locked for player {player_index}")

        if not self.card_utils.is_duplicate_or_nearby_card(detected_card, self.player_cards[player_index]['cards']):
            self.card_handler.add_or_update_player_card(detected_card, self.player_cards[player_index],
                                                        detected_card['card_name'])
            self.card_handler.print_all_cards(self.player_cards,
                                              self.card_utils.card_counters)  # Update and print cards info
            self.update_gui()  # Ensure the GUI is updated

    def update_if_higher_confidence(self, player_index, detected_card):
        if detected_card['confidence'] > max(self.player_cards[player_index]['confidences']):
            replace_index = self.player_cards[player_index]['confidences'].index(
                min(self.player_cards[player_index]['confidences']))
            self.player_cards[player_index]['cards'][replace_index] = detected_card['card_name']
            self.player_cards[player_index]['confidences'][replace_index] = detected_card['confidence']
            self.card_handler.print_all_cards(self.player_cards, self.card_utils.card_counters)

    def blackjack_decision(self, player_cards, dealer_up_card, true_count, base_bet):
        self.recommendations.clear()  # Clear previous recommendations
        if dealer_up_card is None or dealer_up_card == "Unknown":
            dealer_value = "A"  # Default to Ace if unknown
        else:
            dealer_value = self.card_utils.get_dealer_card_value(dealer_up_card)
            if dealer_value in ["10", "Jack", "Queen", "King"]:  # Normalize face cards to 10
                dealer_value = "10"
            elif dealer_value == "1":  # Ensure "1" is converted to "A"
                dealer_value = "A"

        hand_representation = self.card_utils.get_hand_representation(player_cards)

        if self.card_utils.is_pair_hand(player_cards):
            pair_value = str(self.card_utils.get_card_value(player_cards[0].split(' ')[0]))
            pair_value = "10" if pair_value in ["Jack", "Queen", "King"] else pair_value
            action_key = (dealer_value, f"{pair_value},{pair_value}")
        else:
            action_key = (dealer_value, hand_representation)

        # Normalize keys
        action_key = (str(action_key[0]).strip().upper(), str(action_key[1]).strip().upper())

        print(f"Looking up action for key: {action_key}")  # Debug print
        action = self.blackjack_strategy.get(action_key, "?")

        if action == "?":
            print(
                f"Warning: Strategy not found for dealer up card {dealer_value} and player hand {hand_representation}")
            print(f"Current strategy keys: {list(self.blackjack_strategy.keys())}")

        mapped_action = constants.ACTION_MAPPING.get(action, action)
        color = self.get_action_color(action)

        self.recommendations.append((mapped_action, color))
        return self.recommendations

    def get_colored_action(self, action):
        if action in constants.ACTION_MAPPING:
            action_text = constants.ACTION_MAPPING[action]
        else:
            action_text = action

        color = self.get_action_color(action)
        return action_text, color

    def get_action_color(self, action):
        return constants.ACTION_COLORS.get(action, "black")

    def process_player_decisions_and_print_info(self, initial_cards_received, dealer_cards):
        dealer_up_card = dealer_cards[0] if dealer_cards else "Unknown"

        for player_index, player_data in sorted(self.player_cards.items(), key=lambda x: x[0]):
            cards = player_data['cards']

            if len(cards) == 2 and not initial_cards_received[player_index]:
                initial_cards_received[player_index] = True

            if all(initial_cards_received.values()):
                self.print_all_cards()

            if initial_cards_received[player_index] or len(cards) > 2:
                if dealer_up_card == "Unknown":
                    decision_recommendations = [("Waiting for the dealer card", "black")]
                else:
                    decision_recommendations = self.blackjack_decision(cards, dealer_up_card,
                                                                       self.card_utils.true_count, constants.BASE_BET)

                previous_recommendation = player_data.get('recommendation')

                if previous_recommendation != decision_recommendations:
                    player_data['recommendation'] = decision_recommendations
                    self.card_utils.print_player_cards(player_index, cards, decision_recommendations)
                elif not previous_recommendation:
                    player_data['recommendation'] = decision_recommendations
                    self.card_utils.print_player_cards(player_index, cards, decision_recommendations)

            self.players_cards_data.append({'player_index': player_index, 'cards': cards})

        self.update_player_cards_display(self.players_cards_data, dealer_up_card, self.card_utils.true_count,
                                         constants.BASE_BET)

        if self.rounds_observed > 3:
            betting_strategy = self.decision_making.bet_strategy(self.card_utils.true_count, constants.BASE_BET)
            print(f"\nBetting strategy for next round: {betting_strategy}")
        else:
            print(
                f"\nAccumulating card count data, betting strategy recommendations will start after {3 - self.rounds_observed} more rounds.")

    def print_all_cards(self):
        self.cards_info.clear()
        for player_index in sorted(self.player_cards):
            cards = self.player_cards[player_index]['cards']
            confidences = self.player_cards[player_index]['confidences']

            card_info = []
            for i, (card, conf) in enumerate(zip(cards, confidences), start=1):
                card_info.append(f"[{i}] {card} (C: {conf * 100:.2f}%)")

            self.cards_info.append(f"P{player_index + 1}: {' // '.join(card_info)}")
            print(f"P{player_index + 1}: {' // '.join(card_info)}")

        formatted_card_counts = [f"{value} => {count}x" for value, count in
                                 sorted(self.card_value_counts.items(), key=lambda item: item[0])]
        total_cards = sum(self.card_value_counts.values())
        print(f"Card counts ({total_cards}):\n", "\n".join(formatted_card_counts))

    def update_player_cards_display(self, player_data_list, dealer_up_card, true_count, base_bet):
        self.clear_player_cards()

        start_y = 50  # Increase the starting y position for more padding
        total_width = 7 * (constants.CARD_WIDTH + constants.CARD_SPACING) + 8
        column_width = (total_width - 8) // 7

        for i in range(7):
            start_x = i * (column_width + constants.CARD_SPACING) + constants.CARD_SPACING

            player_data = next((data for data in player_data_list if data['player_index'] == i), None)
            cards = player_data['cards'] if player_data else ['-', '-']
            player_number = i + 1

            card_display_y = start_y

            for j, card in enumerate(cards):
                if card == '-':
                    card = "default"

                photo_img = self.get_card_image(card)
                card_label = tk.Label(self.gui.canvas, image=photo_img, bg="white")
                card_label.image = photo_img
                card_label.place(x=start_x, y=card_display_y)  # Remove padx and pady
                card_label.bind("<Button-1>", lambda e, pi=i, ci=j: self.on_card_click(pi, ci))
                self.player_cards_labels.append(card_label)
                card_display_y += constants.CARD_HEIGHT + 20  # Increase spacing between cards

            start_y = 50  # Reset the starting y position for the next player
            self.create_label(f"Player {player_number}", start_x + column_width // 2, self.gui.winfo_height() - 20,
                              anchor="s")

            decision = self.blackjack_decision(cards, dealer_up_card, true_count, base_bet)[0] if player_data else (
                "-", "black")
            self.create_colored_labels(f"Decision: ", decision[0], decision[1], start_x + column_width // 2,
                                       card_display_y + 5, "n")

    def create_colored_labels(self, prefix, text, color, x, y, anchor="n"):
        player_number = int(
            x // (constants.CARD_WIDTH + constants.CARD_SPACING))  # Determine player number based on x position

        # Clear previous labels for the specific player
        if player_number in self.players_decision_labels:
            for label in self.players_decision_labels[player_number]:
                label.destroy()
        self.players_decision_labels[player_number] = []

        # Define a larger font
        large_font = font.Font(family="Helvetica", size=14, weight="bold")  # Increase font size

        # Create a single label to hold both the prefix and the decision text
        combined_text = f"{prefix} {text}"
        label_combined = tk.Label(self.gui.canvas, text=combined_text, fg=color, bg="white", font=large_font)
        label_combined.place(x=x, y=y, anchor=anchor)  # Remove padx and pady
        self.players_decision_labels[player_number].append(label_combined)

    def on_card_click(self, player_index, card_index):
        self.open_card_selection_window(player_index, card_index)

    def on_dealer_card_click(self):
        self.open_card_selection_window("dealer", 0)

    def open_card_selection_window(self, player_index, card_index):
        selection_window = tk.Toplevel(self.gui)
        selection_window.title("Select Card")

        available_cards = self.card_utils.get_all_card_names()

        for i, card in enumerate(available_cards):
            card_image_path = self.utils.generate_card_image_path(card)
            try:
                img = Image.open(card_image_path)
                img = img.resize((60, 90))  # Resize the image to fit the button
                imgtk = ImageTk.PhotoImage(image=img)
                card_button = tk.Button(selection_window, image=imgtk,
                                        command=lambda c=card: self.replace_card(player_index, card_index, c))
                card_button.image = imgtk  # Keep a reference to avoid garbage collection
                card_button.grid(row=i // 10, column=i % 10)  # Arrange buttons in a grid
            except FileNotFoundError:
                print(f"Image file not found: {card_image_path}")

        # Add the default card image as an option
        default_img = Image.open(constants.DEFAULT_CARD_IMAGE_PATH).resize((60, 90))
        default_imgtk = ImageTk.PhotoImage(image=default_img)
        default_button = tk.Button(selection_window, image=default_imgtk,
                                   command=lambda: self.replace_card(player_index, card_index, "-"))
        default_button.image = default_imgtk  # Keep a reference to avoid garbage collection
        default_button.grid(row=len(available_cards) // 10, column=0)

    def replace_card(self, player_index, card_index, card_name):
        if player_index == "dealer":
            if card_name == "-":
                self.dealer_up_card = None
                self.update_dealer_card_display([])
            else:
                normalized_card_name = self.card_utils.get_dealer_card_value(card_name.split(' ')[0])
                self.dealer_up_card = normalized_card_name
                self.update_dealer_card_display([card_name])
            print(
                f"Dealer card replaced with: {self.dealer_up_card} (normalized: {normalized_card_name})")  # Debug print
        else:
            if card_name == "-":
                self.player_cards[player_index]['cards'][card_index] = "-"
                self.player_cards[player_index]['confidences'][card_index] = 0.0
                self.locked_cards[player_index].discard(card_index)
                self.manually_replaced_cards[player_index].discard(card_index)
            else:
                self.player_cards[player_index]['cards'][card_index] = card_name
                self.player_cards[player_index]['confidences'][
                    card_index] = 1.0  # Assuming full confidence for manual replacement
                self.manually_replaced_cards[player_index].add(card_index)  # Mark card index as manually replaced
                self.locked_cards[player_index].add(card_index)  # Also lock the card index

            print(f"Manually replaced card {card_index} for player {player_index} with {card_name}")

        # Update GUI
        self.update_gui()

        # Close all Toplevel windows (the card selection window)
        for widget in self.gui.winfo_children():
            if isinstance(widget, tk.Toplevel):
                widget.destroy()

    def update_dealer_card_display(self, dealer_cards):
        if dealer_cards:
            card_face = dealer_cards[0]  # Get the face value (e.g., '2', 'Q')
            card_image_path = None

            # Check for the existence of any card with the given face value
            for suit in ['hearts', 'diamonds', 'spades', 'clubs']:  # Assuming suits are named like this
                potential_path = f"{constants.CARD_FOLDER_PATH}/{card_face.lower()}_of_{suit}.png"
                if os.path.exists(potential_path):
                    card_image_path = potential_path
                    break

            if card_image_path is None:
                card_value = "No card detected"
                card_image_path = constants.DEFAULT_CARD_IMAGE_PATH
            else:
                card_value = card_face
        else:
            card_value = "No card detected"
            card_image_path = constants.DEFAULT_CARD_IMAGE_PATH

        # Debug: Print the image path being used
        print(f"Card image path: {card_image_path}")

        try:
            img = Image.open(card_image_path)
            img = img.resize((100, 150))

            # Apply green color filter
            img = img.convert("L")  # Convert to grayscale
            img = ImageOps.colorize(img, black="grey", white="white")

            imgtk = ImageTk.PhotoImage(image=img)
            self.dealer_card_label.config(image=imgtk)
            self.dealer_card_label.image = imgtk
            print(f"Dealer card display updated with: {card_value}")  # Debug print
        except FileNotFoundError:
            print(f"Image file not found: {card_image_path}")
        except Exception as e:
            print(f"Error updating dealer card display: {e}")

    def create_dealer_card_placeholder(self):
        placeholder_img = self.get_card_image("default")
        self.dealer_card_label = tk.Label(self.gui.canvas, image=placeholder_img, bg="white")
        self.dealer_card_label.image = placeholder_img
        self.dealer_card_label.place(relx=0.5, rely=0.1, anchor="center")
        self.dealer_card_label.bind("<Button-1>", lambda e: self.on_dealer_card_click())

    def get_card_image(self, card):
        if card == "default":
            card_image_path = constants.DEFAULT_CARD_IMAGE_PATH
        else:
            card_image_path = self.utils.generate_card_image_path(card)

        if not os.path.exists(card_image_path):
            print(f"Card image not found: {card_image_path}. Using default image.")
            card_image_path = constants.DEFAULT_CARD_IMAGE_PATH
        card_image = cv2.imread(card_image_path)
        pil_image = Image.fromarray(cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB))
        return ImageTk.PhotoImage(pil_image.resize((constants.CARD_WIDTH, constants.CARD_HEIGHT)))

    def create_label(self, text, x, y, anchor="n"):
        label = tk.Label(self.gui.canvas, text=text)
        label.place(x=x, y=y, anchor=anchor)
        return label

    def update_gui(self):
        self.players_cards_data.clear()
        for player_index, player_data in sorted(self.player_cards.items(), key=lambda x: x[0]):
            self.players_cards_data.append({'player_index': player_index, 'cards': player_data['cards']})
        self.update_player_cards_display(self.players_cards_data, self.dealer_up_card, self.card_utils.true_count,
                                         constants.BASE_BET)

    def reset_for_new_round(self):
        self.player_cards.clear()
        self.players_cards_data.clear()

        empty_image = tk.PhotoImage()
        for label in self.player_cards_labels:
            label.config(image=empty_image)
            label.image = empty_image

        if self.dealer_card_label:
            self.dealer_card_label.config(image=empty_image)
            self.dealer_card_label.image = empty_image

        for decision_label in self.players_decision_labels:
            decision_label.config(text="")

        self.round_count += 1
        self.gui.round_label.config(text=f"Round: {self.round_count}")

        self.first_card_detected.clear()
        self.second_card_detected.clear()
        self.card_value_counts.clear()
        self.players_received_first_card.clear()
        self.card_utils.counted_cards_this_round.clear()
        print("Reset for new round.")

    def clear_player_cards(self):
        with self.lock:
            # Collect all changes in a temporary list
            changes = []
            for player_number, label_list in self.players_decision_labels.items():
                for label in label_list:
                    changes.append((player_number, label))

            # Apply changes after iteration
            for player_number, label in changes:
                self.players_decision_labels[player_number].remove(label)
                label.destroy()

            # Clear the recommendations
            self.recommendations.clear()
