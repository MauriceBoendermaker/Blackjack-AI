import csv
import time
from collections import defaultdict
import os
import tempfile
import tkinter as tk

import cv2
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
        self.detected_card = {}
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
        self.initial_cards_received = defaultdict(bool)
        self.player_cards = defaultdict(lambda: {"cards": ["-", "-"], "confidences": [0.0, 0.0]})
        self.detection_timers = defaultdict(lambda: {"first_card": None, "second_card": None})
        self.detection_start_time = time.time()
        self.locked_first_cards = set()
        self.locked_second_cards = set()
        self.player_regions = []

    def initialize_screenshot(self):
        self.captured_screenshot = self.monitor_utils.capture_screen()

    def populate_blackjack_strategy(self):
        with open(constants.CSV_FILE_PATH, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                dealer_card, player_hand, action = row
                self.blackjack_strategy[(dealer_card, player_hand)] = action

    def set_monitor(self, monitor):
        self.monitor_utils.set_monitor(monitor)
        self.initialize_screenshot()

    def capture_screen_and_track_cards(self):
        current_resolution = self.monitor_utils.get_current_resolution()
        scale_x, scale_y = self.monitor_utils.get_scaling_factors(constants.BASE_RESOLUTION, current_resolution)
        self.player_regions = self.monitor_utils.scale_player_regions(constants.BASE_PLAYER_REGIONS, scale_x, scale_y)

        while True:
            self.initialize_screenshot()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file_path = temp_file.name
                self.captured_screenshot.save(temp_file_path)

            predictions = self.model_players.predict(temp_file_path,
                                                     confidence=constants.PREDICTION_CONFIDENCE_PLAYERS,
                                                     overlap=constants.PREDICTION_OVERLAP_PLAYERS).json()['predictions']

            print(f"{len(predictions)} predictions")

            for player_index, region in enumerate(self.player_regions):
                self.process_player_predictions(predictions, player_index, region)

            self.card_utils.calculate_true_count()
            self.process_player_decisions_and_print_info(self.initial_cards_received, self.dealer_up_card)

            os.unlink(temp_file_path)

            self.gui.update_idletasks()
            self.gui.update()

    def process_player_predictions(self, predictions, player_index, region):
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
            if player_index not in self.locked_first_cards:
                self.lock_and_update_player_card(player_index, detected_card, is_first_card=True)
            elif player_index in self.locked_first_cards and player_index not in self.locked_second_cards:
                self.lock_and_update_player_card(player_index, detected_card, is_first_card=False)
            elif player_index in self.locked_second_cards:
                self.update_if_higher_confidence(player_index, detected_card)

    def lock_and_update_player_card(self, player_index, detected_card, is_first_card):
        if is_first_card:
            self.locked_first_cards.add(player_index)
            self.detection_timers[player_index]["first_card"] = time.time()
        else:
            self.locked_second_cards.add(player_index)
            self.detection_timers[player_index]["second_card"] = time.time()

        if not self.card_utils.is_duplicate_or_nearby_card(detected_card, self.player_cards[player_index]['cards']):
            self.card_handler.add_or_update_player_card(detected_card, self.player_cards[player_index],
                                                        detected_card['card_name'])
            self.card_handler.print_all_cards(self.player_cards,
                                              self.card_utils.card_counters)  # Update and print cards info

    def update_if_higher_confidence(self, player_index, detected_card):
        if detected_card['confidence'] > max(self.player_cards[player_index]['confidences']):
            replace_index = self.player_cards[player_index]['confidences'].index(
                min(self.player_cards[player_index]['confidences']))
            self.player_cards[player_index]['cards'][replace_index] = detected_card['card_name']
            self.player_cards[player_index]['confidences'][replace_index] = detected_card['confidence']
            self.card_handler.print_all_cards(self.player_cards, self.card_utils.card_counters)

    def update_gui(self):
        self.update_player_cards_display(self.players_cards_data, self.dealer_up_card, self.card_utils.true_count,
                                         constants.BASE_BET)

    def blackjack_decision(self, player_cards, dealer_up_card, true_count, base_bet):
        dealer_value = self.card_utils.get_dealer_card_value(dealer_up_card) if dealer_up_card != "Unknown" else "0"
        hand_value = self.card_utils.calculate_hand_value(player_cards)

        if hand_value >= 17:
            action = "S"
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

    def process_player_decisions_and_print_info(self, initial_cards_received, dealer_up_card):
        dealer_up_card = dealer_up_card[0] if dealer_up_card else "Unknown"

        for player_index, player_data in sorted(self.player_cards.items(), key=lambda x: x[0]):
            cards = player_data['cards']

            if len(cards) == 2 and not initial_cards_received[player_index]:
                initial_cards_received[player_index] = True

            if all(initial_cards_received.values()):
                self.print_all_cards()

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

            for i, (card, conf) in enumerate(zip(cards, confidences), start=1):
                self.cards_info.append(f"[{i}] {card} (C: {conf * 100:.2f}%)")

            self.cards_info.append(f"P{player_index + 1}: {' // '.join(self.cards_info)}")
            print(f"P{player_index + 1}: {' // '.join(self.cards_info)}")

        formatted_card_counts = [f"{value} => {count}x" for value, count in
                                 sorted(self.card_value_counts.items(), key=lambda item: item[0])]
        total_cards = sum(self.card_value_counts.values())
        print(f"Card counts ({total_cards}):\n", "\n".join(formatted_card_counts))

    def update_player_cards_display(self, player_data_list, dealer_up_card, true_count, base_bet):
        self.clear_player_cards()

        start_y = 10
        total_width = 7 * (constants.CARD_WIDTH + constants.CARD_SPACING) + 8
        column_width = (total_width - 8) // 7

        for i in range(7):
            start_x = i * (column_width + constants.CARD_SPACING) + constants.CARD_SPACING

            player_data = next((data for data in player_data_list if data['player_index'] == i), None)
            cards = player_data['cards'] if player_data else ['-', '-']
            player_number = i + 1

            card_display_y = start_y

            for card in cards:
                if card != '-':
                    photo_img = self.get_card_image(card)
                    card_label = tk.Label(self.gui.canvas, image=photo_img)
                    card_label.image = photo_img
                    card_label.place(x=start_x, y=card_display_y)
                    self.player_cards_labels.append(card_label)
                    card_display_y += constants.CARD_HEIGHT + 10

            start_y = 10
            self.create_label(f"Player {player_number}", start_x + column_width // 2,
                              self.gui.winfo_height() - 10, anchor="s")

            decision = self.blackjack_decision(cards, dealer_up_card, true_count, base_bet)[0] if player_data else "-"
            decision_label = self.create_label(f"Decision: {decision}", start_x + column_width // 2,
                                               card_display_y + 20, anchor="n")
            self.players_decision_labels.append(decision_label)

    def update_dealer_card_display(self, dealer_cards):
        card_value = dealer_cards[0] if dealer_cards else "No card detected"

        if self.dealer_card_label is None:
            self.dealer_card_label = tk.Label(self.gui)
            self.dealer_card_label.place(relx=1.0, rely=0.0, x=-10, y=20, anchor='ne')

        if card_value != "No card detected":
            card_image_path = f"{constants.CARD_FOLDER_PATH}/{card_value.lower()}_of_spades.png"
        else:
            card_image_path = constants.DEFAULT_CARD_IMAGE_PATH

        try:
            img = Image.open(card_image_path)
            img = img.resize((100, 150))
            imgtk = ImageTk.PhotoImage(image=img)
            self.dealer_card_label.config(image=imgtk)
            self.dealer_card_label.image = imgtk
        except FileNotFoundError:
            print(f"Image file not found: {card_image_path}")

    def get_card_image(self, card):
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
        for label in self.player_cards_labels:
            label.destroy()
        self.player_cards_labels.clear()
