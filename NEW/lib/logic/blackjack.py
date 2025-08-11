import os
import cv2
import csv
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk

from tkinter import font
from collections import defaultdict

from PIL import Image, ImageTk

from .utils import Utils
from .card_utils import get_card_utils
from .card_handler import CardHandler
from .monitor_utils import MonitorUtils
from .decision_making import DecisionMaking
from ..common import constants, card_mappings

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
        self.card_utils = get_card_utils()
        self.monitor_utils = MonitorUtils()
        self.decision_making = DecisionMaking()
        self.model_players = self.utils.initialize_player_model()
        self.model_dealer = self.utils.initialize_dealer_model()
        self.card_value_counts = defaultdict(int)
        self.initial_cards_received = defaultdict(bool)
        self.player_cards = defaultdict(lambda: {"cards": ["-", "-"], "confidences": [0.0, 0.0]})
        self.detection_timers = defaultdict(lambda: {"first_card": None, "second_card": None})
        self.detection_start_time = time.time()
        self.locked_cards = defaultdict(set)
        self.manually_replaced_cards = defaultdict(set)
        self.player_regions = []
        self.player_region_rects = []
        self.detection_states = defaultdict(lambda: BlackjackLogic.DetectionState.WAITING_FOR_FIRST_CARD)
        self.player_decisions = {}
        self.card_image_cache = {}
        self.recommendation_cache = {}
        self.recommendation_executor = ThreadPoolExecutor(max_workers=2)
        self.default_card_imgtk = None
        self.last_dealer_displayed = None
        self.last_recommendation_time = {}
        self.recommendation_inflight = {}

        try:
            default_img = Image.open(constants.DEFAULT_CARD_IMAGE_PATH).resize((60, 90))
            self.default_card_imgtk = ImageTk.PhotoImage(default_img)
        except Exception:
            self.default_card_imgtk = None

        self.create_dealer_card_placeholder()

    def fetch_second_recommendation(self, player_cards, dealer_card, player_number):
        cache_key = (tuple(player_cards), dealer_card)
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]
        try:
            counts = {'2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, 'J': 0, 'Q': 0, 'K': 0, 'A': 0}
            for card in player_cards:
                v = card.split(' ')[0]
                if v in counts:
                    counts[v] += 1
                elif v in ["Jack", "Queen", "King"]:
                    counts['10'] += 1
            params = {
                'a': counts['2'],'b': counts['3'],'c': counts['4'],'d': counts['5'],'e': counts['6'],
                'f': counts['7'],'g': counts['8'],'h': counts['9'],'i': counts['10'],'j': counts['A'],
                'k': 0,'l': 1.5,'m': 1,'n': 1,'o': 0,'p': 1,'q': 1,'r': 0,'s': 0,'t': 1,'u': 6,'v': 44
            }
            r = requests.get("https://wizardofodds.com/calculators-js/blackjack/calculate/", params=params, timeout=5)
            data = r.json()
        except Exception:
            return "Error", {}
        if data.get("Error"):
            return "Error", {}
        actions = ["Surrender", "Stand", "Hit", "Double", "Split"]
        avail = {a: data[a] for a in actions if data.get(f"Has{a}", False)}
        if not avail:
            return "No action", {}
        best = sorted(avail.items(), key=lambda x: x[1], reverse=True)
        result = (f"{best[0][0]} ({best[0][1]:.3f})", dict(best))
        self.recommendation_cache[cache_key] = result
        return result

    def draw_predictions(self, image, predictions, output_path):
        if not constants.DEBUG_MODE:
            return
        img = image.copy()
        for p in predictions:
            if {'x','y','width','height'}.issubset(p.keys()):
                x, y, w, h = p['x'], p['y'], p['width'], p['height']
                x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
                cv = cv2.cvtColor(cv2.cvtColor(cv2.imread(constants.DEFAULT_CARD_IMAGE_PATH), cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR)
        if output_path:
            try:
                image.save(output_path)
            except Exception:
                pass

    def initialize_screenshot(self):
        self.captured_screenshot = self.monitor_utils.capture_screen()

    def get_current_player_decisions(self):
        return self.player_decisions

    def load_strategy(self):
        strategy = {}
        with open(constants.CSV_FILE_PATH, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                dealer_card, player_hand, action = row
                dealer_card = dealer_card.strip().upper()
                player_hand = player_hand.strip().upper()
                action = action.strip().upper()
                strategy[(dealer_card, player_hand)] = action
        self.blackjack_strategy = strategy

    def set_monitor(self, monitor):
        self.monitor_utils.set_monitor(monitor)
        self.initialize_screenshot()
        current_resolution = self.monitor_utils.get_current_resolution()
        scale_x, scale_y = self.monitor_utils.get_scaling_factors(constants.BASE_RESOLUTION, current_resolution)
        self.player_regions = self.monitor_utils.scale_player_regions(constants.BASE_PLAYER_REGIONS, scale_x, scale_y)
        self.player_region_rects = [self.monitor_utils.path_bounding_rect(p) for p in self.player_regions]

    def capture_screen_and_track_cards(self):
        self.initialize_screenshot()
        dealer_area = self.captured_screenshot.crop(
            (constants.DEALER_AREA_LEFT, constants.DEALER_AREA_UPPER, constants.DEALER_AREA_RIGHT, constants.DEALER_AREA_LOWER)
        )
        try:
            dealer_area.save(constants.INPUT_DEALER_PATH)
            predictions_dealer = self.model_dealer.predict(
                constants.INPUT_DEALER_PATH,
                confidence=constants.PREDICTION_CONFIDENCE_DEALER,
                overlap=constants.PREDICTION_OVERLAP_DEALER
            ).json()['predictions']
        except Exception:
            predictions_dealer = []
        dealer_card = []
        for prediction in predictions_dealer:
            class_label = prediction.get('class')
            card_name = card_mappings.dealer_class_mapping.get(class_label, "Unknown")
            dealer_card.append(card_name)
        self.dealer_up_card = dealer_card[0] if dealer_card else "Unknown"
        if self.dealer_up_card != self.last_dealer_displayed:
            self.update_dealer_card_display(dealer_card)
            self.last_dealer_displayed = self.dealer_up_card
        predictions_players = []
        for idx, rect in enumerate(self.player_region_rects):
            l, t, r, b = rect
            crop = self.captured_screenshot.crop((l, t, r, b))
            path = f"INPUT_player_{idx}.jpg"
            try:
                crop.save(path)
                preds = self.model_players.predict(
                    path,
                    confidence=constants.PREDICTION_CONFIDENCE_PLAYERS,
                    overlap=constants.PREDICTION_OVERLAP_PLAYERS
                ).json()['predictions']
                for p in preds:
                    p2 = dict(p)
                    p2['x'] = p['x'] + l
                    p2['y'] = p['y'] + t
                    predictions_players.append(p2)
            except Exception:
                continue
        for player_index, region in enumerate(self.player_regions):
            self.process_player_predictions(predictions_players, player_index, region)
        self.card_utils.calculate_true_count()
        self.process_player_decisions_and_print_info(self.initial_cards_received, self.dealer_up_card)

    def process_player_predictions(self, predictions, player_index, region):
        state = self.detection_states[player_index]
        if state == BlackjackLogic.DetectionState.SECOND_CARD_DETECTED:
            return
        best_card = None
        best_confidence = 0
        for prediction in predictions:
            x, y = prediction['x'], prediction['y']
            class_label = prediction['class']
            confidence = prediction['confidence']
            if region.contains_point([x, y]):
                card_name = self.card_utils.get_card_name(class_label)
                card_index = 0 if state == BlackjackLogic.DetectionState.WAITING_FOR_FIRST_CARD else 1
                if confidence > best_confidence and card_index not in self.locked_cards[player_index]:
                    best_card = {'x': x, 'y': y, 'confidence': confidence, 'card_name': card_name}
                    best_confidence = confidence
        if best_card:
            self.card_handler.handle_card_detection(best_card['card_name'])
            if state == BlackjackLogic.DetectionState.WAITING_FOR_FIRST_CARD:
                self.lock_and_update_player_card(player_index, best_card, card_index=0)
                self.detection_states[player_index] = BlackjackLogic.DetectionState.FIRST_CARD_DETECTED
            elif state == BlackjackLogic.DetectionState.FIRST_CARD_DETECTED:
                self.lock_and_update_player_card(player_index, best_card, card_index=1)
                self.detection_states[player_index] = BlackjackLogic.DetectionState.SECOND_CARD_DETECTED

    def lock_and_update_player_card(self, player_index, detected_card, card_index):
        existing_card = self.player_cards[player_index]['cards'][card_index]
        if existing_card == detected_card['card_name']:
            return
        self.locked_cards[player_index].add(card_index)
        self.detection_timers[player_index][f"card_{card_index}"] = time.time()
        identity_key = (player_index, card_index)
        if identity_key not in self.card_utils.counted_cards_this_round:
            self.card_handler.add_or_update_player_card(
                detected_card, self.player_cards[player_index],
                detected_card['card_name'], player_index, card_index
            )
            self.card_utils.counted_cards_this_round.add(identity_key)
        self.card_handler.print_all_cards(self.player_cards)
        self.update_gui()

    def update_if_higher_confidence(self, player_index, detected_card):
        if detected_card['confidence'] > max(self.player_cards[player_index]['confidences']):
            replace_index = self.player_cards[player_index]['confidences'].index(min(self.player_cards[player_index]['confidences']))
            self.player_cards[player_index]['cards'][replace_index] = detected_card['card_name']
            self.player_cards[player_index]['confidences'][replace_index] = detected_card['confidence']
            self.card_handler.print_all_cards(self.player_cards)

    def blackjack_decision(self, player_cards, dealer_up_card, true_count, base_bet):
        self.recommendations.clear()
        if dealer_up_card is None or dealer_up_card == "Unknown":
            dealer_value = "A"
        else:
            dealer_value = self.card_utils.get_dealer_card_value(dealer_up_card)
            if dealer_value in ["10", "Jack", "Queen", "King"]:
                dealer_value = "10"
            elif dealer_value == "1":
                dealer_value = "A"
        hand_representation = self.card_utils.get_hand_representation(player_cards)
        if self.card_utils.is_pair_hand(player_cards):
            pair_value = str(self.card_utils.get_card_value(player_cards[0].split(' ')[0]))
            pair_value = "10" if pair_value in ["Jack", "Queen", "King"] else pair_value
            action_key = (dealer_value, f"{pair_value},{pair_value}")
        else:
            action_key = (dealer_value, hand_representation)
        action_key = (str(action_key[0]).strip().upper(), str(action_key[1]).strip().upper())
        action = self.blackjack_strategy.get(action_key, "?")
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
            if initial_cards_received[player_index] or len(cards) > 2:
                if dealer_up_card == "Unknown":
                    decision_recommendations = [("Waiting for the dealer card", "black")]
                else:
                    decision_recommendations = self.blackjack_decision(cards, dealer_up_card, self.card_utils.true_count, constants.BASE_BET)
                previous_recommendation = player_data.get('recommendation')
                mapped_decision = decision_recommendations[0][0] if decision_recommendations else "-"
                second_recommendation = None
                key = (tuple(cards), dealer_up_card)
                now_ms = int(time.time() * 1000)
                last_ms = self.last_recommendation_time.get((player_index + 1, key), 0)
                inflight = self.recommendation_inflight.get((player_index + 1, key), False)
                if key in self.recommendation_cache:
                    second_recommendation = self.recommendation_cache[key][0]
                else:
                    if not inflight and now_ms - last_ms >= constants.RECOMMENDATION_DEBOUNCE_MS:
                        self.recommendation_inflight[(player_index + 1, key)] = True
                        self.last_recommendation_time[(player_index + 1, key)] = now_ms
                        def fetch_and_store(pi=player_index + 1, k=key, c=cards, d=dealer_up_card):
                            rec, _ = self.fetch_second_recommendation(c, d, pi)
                            self.recommendation_cache[k] = (rec, {})
                            self.recommendation_inflight[(pi, k)] = False
                        self.recommendation_executor.submit(fetch_and_store)
                if second_recommendation is None:
                    second_recommendation = "Loading..."
                self.player_decisions[player_index + 1] = {"decision": mapped_decision, "second": second_recommendation}
                if previous_recommendation != decision_recommendations or not previous_recommendation:
                    player_data['recommendation'] = decision_recommendations
                    self.card_utils.print_player_cards(player_index, cards, decision_recommendations)
                self.players_cards_data.append({'player_index': player_index, 'cards': cards})
        self.update_player_cards_display(self.players_cards_data, dealer_up_card, self.card_utils.true_count, constants.BASE_BET)
        if self.rounds_observed > 3:
            betting_strategy = self.decision_making.bet_strategy(self.card_utils.true_count, constants.BASE_BET)
            pass

    def print_all_cards(self):
        self.cards_info.clear()
        for player_index in sorted(self.player_cards):
            cards = self.player_cards[player_index]['cards']
            confidences = self.player_cards[player_index]['confidences']
            card_info = []
            for i, (card, conf) in enumerate(zip(cards, confidences), start=1):
                card_info.append(f"[{i}] {card} (C: {conf * 100:.2f}%)")
            self.cards_info.append(f"P{player_index + 1}: {' // '.join(card_info)}")
        self.card_utils.print_card_counts()

    def update_player_cards_display(self, player_data_list, dealer_up_card, true_count, base_bet):
        start_y = 50
        total_width = 7 * (constants.CARD_WIDTH + constants.CARD_SPACING) + 8
        column_width = (total_width - 8) // 7
        while len(self.player_cards_labels) < 14:
            ph = tk.Label(self.gui.canvas, image=self.default_card_imgtk, bg="white")
            ph.place(x=0, y=0)
            self.player_cards_labels.append(ph)
        for i in range(7):
            start_x = i * (column_width + constants.CARD_SPACING) + constants.CARD_SPACING
            player_data = next((data for data in player_data_list if data['player_index'] == i), None)
            cards = player_data['cards'] if player_data else ['-', '-']
            card_display_y = start_y
            for j in range(2):
                card = cards[j] if cards[j] != '-' else "default"
                photo_img = self.get_cached_card_image(card)
                idx = i * 2 + j
                lbl = self.player_cards_labels[idx]
                lbl.config(image=photo_img)
                lbl.image = photo_img
                lbl.place(x=start_x, y=card_display_y)
                lbl.bind("<Button-1>", lambda e, pi=i, ci=j: self.on_card_click(pi, ci))
                card_display_y += constants.CARD_HEIGHT + 20
            player_number = i + 1
            if player_number in self.players_decision_labels and self.players_decision_labels[player_number]:
                pass
            else:
                large_font = font.Font(family="Helvetica", size=14, weight="bold")
                l1 = tk.Label(self.gui.canvas, text="", fg="black", bg="white", font=large_font)
                l1.place(x=start_x + column_width // 2, y=card_display_y + 5, anchor="n")
                l2 = tk.Label(self.gui.canvas, text="Optimal: ", fg="blue", bg="white", font=large_font)
                l2.place(x=start_x + column_width // 2, y=card_display_y + 35, anchor="n")
                self.players_decision_labels[player_number] = [l1, l2]
            decision = self.blackjack_decision(cards, dealer_up_card, true_count, base_bet)[0] if player_data else ("-", "black")
            label1 = self.players_decision_labels[player_number][0]
            if label1.cget("text") != decision[0] or label1.cget("fg") != decision[1]:
                label1.config(text=decision[0], fg=decision[1])
            label2 = self.players_decision_labels[player_number][1]
            cache_key = (tuple(cards), dealer_up_card)
            second_text = self.recommendation_cache.get(cache_key, ("Loading...", {}))[0]
            if label2.cget("text") != f"Optimal: {second_text}":
                label2.config(text=f"Optimal: {second_text}")
            self.create_label(f"Player {player_number}", start_x + column_width // 2, self.gui.winfo_height() - 20, anchor="s")

    def create_colored_labels(self, prefix, text, color, x, y, anchor="n"):
        player_number = int(x // (constants.CARD_WIDTH + constants.CARD_SPACING))
        combined_text = f"{prefix} {text}"
        large_font = font.Font(family="Helvetica", size=14, weight="bold")
        if player_number in self.players_decision_labels and self.players_decision_labels[player_number]:
            label = self.players_decision_labels[player_number][0]
            if label.cget("text") != combined_text or label.cget("fg") != color:
                label.config(text=combined_text, fg=color)
        else:
            label = tk.Label(self.gui.canvas, text=combined_text, fg=color, bg="white", font=large_font)
            label.place(x=x, y=y, anchor=anchor)
            self.players_decision_labels[player_number] = [label]

    def create_second_decision_label(self, text, x, y, player_number):
        combined_text = f"Optimal: {text}"
        large_font = font.Font(family="Helvetica", size=14, weight="bold")
        if len(self.players_decision_labels.get(player_number, [])) > 1:
            label = self.players_decision_labels[player_number][1]
            if label.cget("text") != combined_text:
                label.config(text=combined_text)
        else:
            label = tk.Label(self.gui.canvas, text=combined_text, fg="blue", bg="white", font=large_font)
            label.place(x=x, y=y, anchor="n")
            if player_number not in self.players_decision_labels:
                self.players_decision_labels[player_number] = []
            self.players_decision_labels[player_number].append(label)

    def on_card_click(self, player_index, card_index):
        self.open_card_selection_window(player_index, card_index)

    def on_dealer_card_click(self):
        self.open_card_selection_window("dealer", 0)

    def open_card_selection_window(self, player_index, card_index):
        selection_window = tk.Toplevel(self.gui)
        selection_window.title("Select Card")
        suits_order = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
        suit_cards = {suit: [] for suit in suits_order}
        for card in self.card_utils.get_all_card_names():
            try:
                suit = card.split(" of ")[1]
                if suit in suit_cards:
                    suit_cards[suit].append(card)
            except IndexError:
                continue
        for row_index, suit in enumerate(suits_order):
            for col_index, card in enumerate(suit_cards[suit]):
                imgtk = self.get_cached_card_image(card)
                if imgtk:
                    card_button = tk.Button(selection_window, image=imgtk, command=lambda c=card: self.replace_card(player_index, card_index, c))
                    card_button.image = imgtk
                    card_button.grid(row=row_index, column=col_index)
        if self.default_card_imgtk:
            default_button = tk.Button(selection_window, image=self.default_card_imgtk, command=lambda: self.replace_card(player_index, card_index, "-"))
            default_button.image = self.default_card_imgtk
            default_button.grid(row=len(suits_order), column=0)

    def replace_card(self, player_index, card_index, card_name):
        for widget in self.gui.winfo_children():
            if isinstance(widget, tk.Toplevel):
                widget.destroy()
        def do_replacement():
            if player_index == "dealer":
                if card_name == "-":
                    self.dealer_up_card = None
                    self.gui.after(0, lambda: self.update_dealer_card_display([]))
                else:
                    normalized_card = self.card_utils.get_dealer_card_value(card_name.split(' ')[0])
                    self.dealer_up_card = normalized_card
                    self.gui.after(0, lambda: self.update_dealer_card_display([card_name]))
                return
            player = self.player_cards[player_index]
            if card_name == "-":
                player['cards'][card_index] = "-"
                player['confidences'][card_index] = 0.0
                self.locked_cards[player_index].discard(card_index)
                self.manually_replaced_cards[player_index].discard(card_index)
            else:
                player['cards'][card_index] = card_name
                player['confidences'][card_index] = 1.0
                self.manually_replaced_cards[player_index].add(card_index)
                self.locked_cards[player_index].add(card_index)
            def finish_update():
                self.update_gui()
            self.gui.after(0, finish_update)
        threading.Thread(target=do_replacement, daemon=True).start()

    def refresh_player_card_image(self, player_index, card_index, card_name):
        try:
            photo_img = self.get_cached_card_image(card_name)
            label_index = player_index * 2 + card_index
            if 0 <= label_index < len(self.player_cards_labels):
                card_label = self.player_cards_labels[label_index]
                card_label.config(image=photo_img)
                card_label.image = photo_img
        except Exception:
            pass

    def update_dealer_card_display(self, dealer_cards):
        if dealer_cards:
            card_face = dealer_cards[0]
            card_image_path = constants.DEFAULT_CARD_IMAGE_PATH
            for suit in ['hearts', 'diamonds', 'spades', 'clubs']:
                potential_path = f"{constants.CARD_FOLDER_PATH}/{card_face.lower()}_of_{suit}.png"
                if os.path.exists(potential_path):
                    card_image_path = potential_path
                    break
            card_value = card_face
        else:
            card_value = "No card detected"
            card_image_path = constants.DEFAULT_CARD_IMAGE_PATH
        try:
            img = Image.open(card_image_path)
            img = img.resize((100, 150))
            imgtk = ImageTk.PhotoImage(image=img)
            self.dealer_card_label.config(image=imgtk)
            self.dealer_card_label.image = imgtk
        except Exception:
            pass

    def create_dealer_card_placeholder(self):
        placeholder_img = self.get_card_image("default")
        self.dealer_card_label = tk.Label(self.gui.canvas, image=placeholder_img, bg="white")
        self.dealer_card_label.image = placeholder_img
        self.dealer_card_label.place(relx=0.5, rely=0.07, anchor="center")
        self.dealer_card_label.bind("<Button-1>", lambda e: self.on_dealer_card_click())

    def get_card_image(self, card):
        if card == "default":
            card_image_path = constants.DEFAULT_CARD_IMAGE_PATH
        else:
            card_image_path = self.utils.generate_card_image_path(card)
        if not os.path.exists(card_image_path):
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
        self.update_player_cards_display(self.players_cards_data, self.dealer_up_card, self.card_utils.true_count, constants.BASE_BET)

    def reset_for_new_round(self):
        self.player_cards.clear()
        self.players_cards_data.clear()
        self.first_card_detected.clear()
        self.second_card_detected.clear()
        self.card_value_counts.clear()
        self.players_received_first_card.clear()
        self.card_utils.counted_cards_this_round.clear()
        self.round_count += 1

    def reset_gui_elements(self):
        empty_image = tk.PhotoImage()
        for label in self.player_cards_labels:
            label.config(image=empty_image)
            label.image = empty_image
        if self.dealer_card_label:
            self.dealer_card_label.config(image=empty_image)
            self.dealer_card_label.image = empty_image
        for player_index in self.players_decision_labels:
            for label in self.players_decision_labels[player_index]:
                label.config(text="")
        self.gui.round_label.config(text=f"Round: {self.round_count}")

    def clear_player_cards(self):
        with self.lock:
            changes = []
            for player_number, label_list in self.players_decision_labels.items():
                for label in label_list:
                    changes.append((player_number, label))
            for player_number, label in changes:
                self.players_decision_labels[player_number].remove(label)
                label.destroy()
            self.recommendations.clear()

    def get_cached_card_image(self, card_name):
        if card_name in self.card_image_cache:
            return self.card_image_cache[card_name]
        try:
            card_image_path = self.utils.generate_card_image_path(card_name)
            img = Image.open(card_image_path).resize((60, 90))
            imgtk = ImageTk.PhotoImage(img)
            self.card_image_cache[card_name] = imgtk
            return imgtk
        except Exception:
            return self.default_card_imgtk
