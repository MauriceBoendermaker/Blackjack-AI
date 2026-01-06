import os
import cv2
import csv
import json
import math
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk

from tkinter import font
from collections import defaultdict, deque

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
        self.create_dealer_card_placeholder()
        self.lock = threading.Lock()
        self.cards_info = []
        self.recommendations = []
        self.players_cards_data = []
        self.player_cards_labels = []
        self.players_decision_labels = defaultdict(list)
        self.player_number_labels = {}  # Track player number labels to prevent duplicates
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
        self.detection_states = defaultdict(lambda: BlackjackLogic.DetectionState.WAITING_FOR_FIRST_CARD)
        self.player_decisions = {}
        self.card_image_cache = {}
        self.card_thumbnail_cache = {}
        self.recommendation_cache = {}
        self.recommendation_executor = ThreadPoolExecutor(max_workers=2)
        self.last_recommendation_time = {}
        self.recommendation_inflight = {}
        self.default_card_imgtk = None

        self.dealer_locked = False
        self.dealer_history = deque(maxlen=5)
        self.dealer_missing_frames = 0
        self.last_dealer_displayed = None

        # Performance monitoring
        self.cycle_times = deque(maxlen=30)  # Track last 30 cycle times
        self.last_cycle_time = time.time()
        self.frame_count = 0

        try:
            default_img = Image.open(constants.DEFAULT_CARD_IMAGE_PATH).resize((60, 90))
            self.default_card_imgtk = ImageTk.PhotoImage(default_img)
        except Exception as e:
            print(f"Failed to load default image: {e}")

        # Preload all card images in background for better performance
        self.preload_card_images()

        # Load cached recommendations from disk
        self.load_recommendation_cache()

    def _select_stable_dealer_label(self, predictions):
        from ..common.card_mappings import dealer_class_mapping2
        candidates = []
        for p in predictions:
            cls = str(p.get('class'))
            if cls == 'cuttingcard':
                continue
            name = dealer_class_mapping2.get(cls, "Unknown")
            if name == "Unknown":
                continue
            candidates.append((name, float(p.get('confidence', 0.0))))
        if not candidates:
            self.dealer_missing_frames += 1
            if self.dealer_missing_frames < 8:
                return self.dealer_up_card
            return None
        best = max(candidates, key=lambda x: x[1])[0]
        self.dealer_history.append(best)
        self.dealer_missing_frames = 0
        most = max(set(self.dealer_history), key=self.dealer_history.count)
        if self.dealer_history.count(most) >= 2:
            return most
        return self.dealer_up_card

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
            # Reduced timeout from 5s to 2s for faster failure recovery
            r = requests.get("https://wizardofodds.com/calculators-js/blackjack/calculate/", params=params, timeout=2)
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
        current_resolution = self.monitor_utils.get_current_resolution()
        scale_x, scale_y = self.monitor_utils.get_scaling_factors(constants.BASE_RESOLUTION, current_resolution)
        self.player_regions = self.monitor_utils.scale_player_regions(constants.BASE_PLAYER_REGIONS, scale_x, scale_y)

        self.initialize_screenshot()

        d_left = int(constants.DEALER_AREA_LEFT * scale_x)
        d_top = int(constants.DEALER_AREA_UPPER * scale_y)
        d_right = int((constants.DEALER_AREA_LEFT + constants.DEALER_AREA_WIDTH) * scale_x)
        d_bottom = int((constants.DEALER_AREA_UPPER + constants.DEALER_AREA_HEIGHT) * scale_y)
        dealer_area = self.captured_screenshot.crop((d_left, d_top, d_right, d_bottom))
        dealer_area.save(constants.INPUT_DEALER_PATH)

        predictions_dealer = []
        if not self.dealer_locked:
            try:
                predictions_dealer = self.model_dealer.predict(
                    constants.INPUT_DEALER_PATH,
                    confidence=constants.PREDICTION_CONFIDENCE_DEALER,
                    overlap=constants.PREDICTION_OVERLAP_DEALER
                ).json()['predictions']
            except Exception:
                predictions_dealer = []

        dealer_label = self._select_stable_dealer_label(predictions_dealer)
        if dealer_label is not None:
            self.dealer_up_card = dealer_label
            if self.dealer_up_card != self.last_dealer_displayed:
                self.update_dealer_card_display([self.dealer_up_card])
                self.last_dealer_displayed = self.dealer_up_card

        self.captured_screenshot.save(constants.INPUT_FULL_PATH)
        try:
            predictions_players = self.model_players.predict(
                constants.INPUT_FULL_PATH,
                confidence=constants.PREDICTION_CONFIDENCE_PLAYERS,
                overlap=constants.PREDICTION_OVERLAP_PLAYERS
            ).json()['predictions']
        except Exception:
            predictions_players = []

        for player_index, region in enumerate(self.player_regions):
            self.process_player_predictions(predictions_players, player_index, region)

        self.card_utils.calculate_true_count()
        self.process_player_decisions_and_print_info(self.initial_cards_received, self.dealer_up_card)

        # Log performance metrics
        self.log_performance_metrics()

        # Return detection activity state for adaptive sleep timing
        # Check how many players are actively being dealt cards
        active_dealing = sum(1 for state in self.detection_states.values()
                           if state in [BlackjackLogic.DetectionState.WAITING_FOR_SECOND_CARD,
                                       BlackjackLogic.DetectionState.FIRST_CARD_DETECTED])
        all_complete = all(state == BlackjackLogic.DetectionState.SECOND_CARD_DETECTED
                          for state in self.detection_states.values())

        if active_dealing > 0:
            return "active_dealing"  # Cards being dealt, check frequently
        elif all_complete or self.dealer_up_card:
            return "round_complete"  # Round finished, can slow down
        else:
            return "waiting"  # Waiting for cards, slower checks

    def process_player_predictions(self, predictions, player_index, region):
        state = self.detection_states[player_index]
        if state == BlackjackLogic.DetectionState.SECOND_CARD_DETECTED:
            return

        best_card = None
        best_confidence = 0.0

        for p in predictions:
            x, y = p['x'], p['y']
            w, h = p['width'], p['height']
            cx = x + w / 2.0
            cy = y + h / 2.0
            if not region.contains_point([cx, cy]):
                continue
            class_label = str(p['class'])
            if not self.card_utils.is_valid_player_class(class_label):
                continue
            confidence = float(p['confidence'])
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
        self.gui.after(0, self.update_gui)

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
        # Get canvas dimensions for semicircular layout
        try:
            canvas_width = self.gui.canvas.winfo_width()
            canvas_height = self.gui.canvas.winfo_height()
        except:
            canvas_width = 800
            canvas_height = 600

        # Semicircle parameters for player positioning (bowl shape at bottom)
        # Dynamic sizing based on canvas dimensions
        CENTER_X = canvas_width / 2
        CENTER_Y = canvas_height + (canvas_height * 0.3)  # Center below canvas for upward bowl
        RADIUS = min(canvas_width * 0.45, canvas_height * 0.6)  # Scale with canvas size
        ARC_SPAN_DEGREES = 90  # 90-degree arc for natural bowl shape
        NUM_PLAYERS = 7

        # Create player card labels if needed
        while len(self.player_cards_labels) < 14:
            ph = tk.Label(self.gui.canvas, image=self.default_card_imgtk, bg="white")
            ph.place(x=0, y=0)
            self.player_cards_labels.append(ph)

        # Position each player in semicircular arc
        for i in range(NUM_PLAYERS):
            # Calculate angle for this player
            if NUM_PLAYERS > 1:
                t = i / (NUM_PLAYERS - 1)  # 0 to 1
                # Start from 90° (left) to 90° (right), creating upward bowl from center below
                angle_deg = 90 - ARC_SPAN_DEGREES/2 + t * ARC_SPAN_DEGREES
            else:
                angle_deg = 90  # Straight up

            angle_rad = math.radians(angle_deg)

            # Calculate base position (center of player's space)
            # For a center below canvas, this creates an upward-opening bowl
            base_x = CENTER_X + RADIUS * math.cos(angle_rad)
            base_y = CENTER_Y - RADIUS * math.sin(angle_rad)

            # Get player data
            player_data = next((data for data in player_data_list if data['player_index'] == i), None)
            cards = player_data['cards'] if player_data else ['-', '-']

            # Position two cards for this player
            for j in range(2):
                card = cards[j] if cards[j] != '-' else "default"
                photo_img = self.get_cached_card_image(card)
                idx = i * 2 + j
                lbl = self.player_cards_labels[idx]

                # Vertical offset for second card
                card_offset_y = j * (constants.CARD_HEIGHT + 5)

                # Center cards horizontally
                card_x = base_x - constants.CARD_WIDTH / 2
                card_y = base_y - constants.CARD_HEIGHT + card_offset_y

                lbl.config(image=photo_img)
                lbl.image = photo_img
                lbl.place(x=card_x, y=card_y)
                lbl.bind("<Button-1>", lambda e, pi=i, ci=j: self.on_card_click(pi, ci))

            # Position decision labels below cards
            player_number = i + 1
            decision_y = base_y + constants.CARD_HEIGHT + 10

            if player_number not in self.players_decision_labels or not self.players_decision_labels[player_number]:
                large_font = font.Font(family="Helvetica", size=12, weight="bold")
                l1 = tk.Label(self.gui.canvas, text="", fg="black", bg="white", font=large_font)
                l1.place(x=base_x, y=decision_y, anchor="n")
                l2 = tk.Label(self.gui.canvas, text="Optimal: ", fg="blue", bg="white", font=large_font)
                l2.place(x=base_x, y=decision_y + 25, anchor="n")
                self.players_decision_labels[player_number] = [l1, l2]
            else:
                # Update position of existing labels
                self.players_decision_labels[player_number][0].place(x=base_x, y=decision_y, anchor="n")
                self.players_decision_labels[player_number][1].place(x=base_x, y=decision_y + 25, anchor="n")

            # Update decision text
            decision = self.blackjack_decision(cards, dealer_up_card, true_count, base_bet)[0] if player_data else ("-", "black")
            label1 = self.players_decision_labels[player_number][0]
            if label1.cget("text") != decision[0] or label1.cget("fg") != decision[1]:
                label1.config(text=decision[0], fg=decision[1])

            label2 = self.players_decision_labels[player_number][1]
            cache_key = (tuple(cards), dealer_up_card)
            second_text = self.recommendation_cache.get(cache_key, ("Loading...", {}))[0]
            if label2.cget("text") != f"Optimal: {second_text}":
                label2.config(text=f"Optimal: {second_text}")

            # Player number label at bottom (create once and reuse)
            if player_number not in self.player_number_labels:
                player_label = tk.Label(self.gui.canvas, text=f"Player {player_number}", bg="white")
                self.player_number_labels[player_number] = player_label
            else:
                player_label = self.player_number_labels[player_number]

            # Ensure text is set and position the label
            player_label.config(text=f"Player {player_number}")
            player_label.place(x=base_x, y=decision_y + 50, anchor="n")

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
        """Open card selection window with Excel-like table layout"""
        selection_window = tk.Toplevel(self.gui)
        selection_window.title("Select Card")
        selection_window.configure(bg='#f8f9fa')

        # Make window modal
        selection_window.grab_set()

        # Set large fixed size to fit everything
        window_width = 1100
        window_height = 750

        # Center on screen
        x = (selection_window.winfo_screenwidth() // 2) - (window_width // 2)
        y = (selection_window.winfo_screenheight() // 2) - (window_height // 2)
        selection_window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        selection_window.resizable(False, False)

        # Header
        header_frame = tk.Frame(selection_window, bg='#ffffff', height=60)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        title = tk.Label(header_frame, text="Select a Card",
                        font=('Inter', 16, 'bold'),
                        bg='#ffffff', fg='#212529')
        title.pack(pady=18)

        # Main content frame
        content_frame = tk.Frame(selection_window, bg='#f8f9fa')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Excel-like table: All 52 cards displayed left to right in rows by suit
        # 4 suits × 13 values = 52 cards total
        suits_order = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
        values_order = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']

        # Create table headers (values)
        headers_frame = tk.Frame(content_frame, bg='#f8f9fa')
        headers_frame.pack(fill=tk.X, pady=(0, 5))

        # Empty space for suit column
        tk.Label(headers_frame, text="Suit", font=('Inter', 11, 'bold'),
                bg='#f8f9fa', fg='#212529', width=8, anchor='w').pack(side=tk.LEFT, padx=5)

        # Value headers (displayed left to right)
        for value in values_order:
            tk.Label(headers_frame, text=value,
                    font=('Inter', 10, 'bold'),
                    bg='#f8f9fa', fg='#212529',
                    width=6).pack(side=tk.LEFT, padx=1)

        # Scrollable frame for cards
        canvas = tk.Canvas(content_frame, bg='#f8f9fa', highlightthickness=0)
        scrollbar = tk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f8f9fa')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create rows for each suit
        suit_symbols = {'Spades': '♠', 'Hearts': '♥', 'Diamonds': '♦', 'Clubs': '♣'}
        for suit in suits_order:
            row_frame = tk.Frame(scrollable_frame, bg='#ffffff',
                               highlightbackground='#dee2e6', highlightthickness=1)
            row_frame.pack(fill=tk.X, pady=2, padx=5)

            # Suit label (like Excel row header)
            suit_label = tk.Label(row_frame, text=f"{suit_symbols[suit]} {suit}",
                                  font=('Inter', 11, 'bold'),
                                  bg='#ffffff', fg='#212529',
                                  width=8, anchor='w')
            suit_label.pack(side=tk.LEFT, padx=5, pady=8)

            # Card buttons for each value (left to right)
            for value in values_order:
                card_name = f"{value} of {suit}"
                imgtk = self.get_cached_card_image(card_name, thumbnail=False)

                if imgtk:
                    card_button = tk.Button(row_frame, image=imgtk,
                                          command=lambda c=card_name, w=selection_window: self.replace_card(player_index, card_index, c, w),
                                          relief='flat',
                                          bg='#ffffff',
                                          activebackground='#e9ecef',
                                          cursor='hand2',
                                          borderwidth=1)
                    card_button.image = imgtk
                    card_button.pack(side=tk.LEFT, padx=2, pady=3)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Button frame at bottom
        button_frame = tk.Frame(selection_window, bg='#ffffff', height=60)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)
        button_frame.pack_propagate(False)

        cancel_btn = tk.Button(button_frame, text="Cancel",
                               command=selection_window.destroy,
                               bg='#6c757d', fg='white',
                               font=('Inter', 11, 'normal'),
                               relief='flat',
                               cursor='hand2',
                               padx=30, pady=10)
        cancel_btn.pack(side=tk.LEFT, padx=20, pady=15)

        if self.default_card_imgtk:
            default_btn = tk.Button(button_frame, text="Reset to Default",
                                   command=lambda: self.replace_card(player_index, card_index, "-", selection_window),
                                   bg='#0d6efd', fg='white',
                                   font=('Inter', 11, 'normal'),
                                   relief='flat',
                                   cursor='hand2',
                                   padx=30, pady=10)
            default_btn.pack(side=tk.RIGHT, padx=20, pady=15)

    def replace_card(self, player_index, card_index, card_name, selection_window=None):
        # Only destroy the specific card selection window, not all Toplevel windows
        if selection_window and selection_window.winfo_exists():
            selection_window.destroy()

        def do_replacement():
            if player_index == "dealer":
                if card_name == "-":
                    self.dealer_locked = False
                    self.dealer_up_card = None
                    self.gui.after(0, lambda: self.update_dealer_card_display([]))
                else:
                    self.dealer_locked = True
                    self.dealer_up_card = card_name.split(" of ")[0]
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
            face = dealer_cards[0]
            if " of " in str(face):
                display_name = face
            else:
                display_name = f"{face} of Hearts" if face not in (None, "Unknown") else None
        else:
            display_name = None
        if display_name:
            try:
                path = self.utils.generate_card_image_path(display_name)
            except Exception:
                path = constants.DEFAULT_CARD_IMAGE_PATH
        else:
            path = constants.DEFAULT_CARD_IMAGE_PATH
        try:
            img = Image.open(path).resize((100, 150))
            imgtk = ImageTk.PhotoImage(image=img)

            def _apply():
                self.dealer_card_label.config(image=imgtk)
                self.dealer_card_label.image = imgtk

            self.gui.after(0, _apply)
        except Exception as e:
            print(f"Error updating dealer card display: {e}")

    def create_dealer_card_placeholder(self):
        placeholder_img = self.get_card_image("default")
        self.dealer_card_label = tk.Label(self.gui.canvas, image=placeholder_img,
                                         bg="white", borderwidth=2, relief="raised")
        self.dealer_card_label.image = placeholder_img
        # Use absolute positioning for better control - updated after canvas is sized
        self.gui.after(100, self.position_dealer_card)
        self.dealer_card_label.bind("<Button-1>", lambda e: self.on_dealer_card_click())

    def position_dealer_card(self):
        """Position dealer card at top center of canvas"""
        try:
            canvas_width = self.gui.canvas.winfo_width()
            if canvas_width > 1:  # Canvas is properly sized
                dealer_x = canvas_width // 2 - 50  # Center (card is 100px wide)
                dealer_y = 120  # Below dealer label
                self.dealer_card_label.place(x=dealer_x, y=dealer_y)
            else:
                # Canvas not sized yet, use relative positioning
                self.dealer_card_label.place(relx=0.5, rely=0.15, anchor="center")
        except Exception as e:
            print(f"Error positioning dealer card: {e}")
            self.dealer_card_label.place(relx=0.5, rely=0.15, anchor="center")

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
        self.dealer_locked = False
        self.dealer_history.clear()
        self.dealer_missing_frames = 0
        self.dealer_up_card = None
        self.last_dealer_displayed = None
        self.round_count += 1
        print("Reset for new round.")

        # Save recommendation cache periodically
        if self.round_count % 5 == 0:  # Save every 5 rounds
            self.save_recommendation_cache()

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
        # Player number labels remain visible during reset
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

    def get_cached_card_image(self, card_name, thumbnail=False):
        cache = self.card_thumbnail_cache if thumbnail else self.card_image_cache
        if card_name in cache:
            return cache[card_name]
        try:
            card_image_path = self.utils.generate_card_image_path(card_name)
            size = (60, 90) if thumbnail else (60, 90)  # Larger thumbnails for bigger window
            img = Image.open(card_image_path).resize(size)
            imgtk = ImageTk.PhotoImage(img)
            cache[card_name] = imgtk
            return imgtk
        except Exception:
            return self.default_card_imgtk

    def preload_card_images(self):
        """Preload all card images in a background thread to improve performance"""
        def _preload():
            try:
                all_cards = self.card_utils.get_all_card_names()
                total = len(all_cards)
                for idx, card_name in enumerate(all_cards):
                    # Preload both full-size and thumbnail
                    self.get_cached_card_image(card_name, thumbnail=False)
                    self.get_cached_card_image(card_name, thumbnail=True)

                    # Update status every 10 cards
                    if (idx + 1) % 10 == 0 or (idx + 1) == total:
                        progress = f"Loading card images: {idx + 1}/{total}"
                        self.gui.after(0, lambda p=progress: self.gui.set_status(p))

                self.gui.after(0, lambda: self.gui.set_status("Card images loaded successfully"))
            except Exception as e:
                print(f"Error preloading card images: {e}")
                self.gui.after(0, lambda: self.gui.set_status(f"Error loading images: {e}"))

        # Run in background thread
        threading.Thread(target=_preload, daemon=True).start()

    def load_recommendation_cache(self):
        """Load recommendation cache from disk for faster lookups"""
        cache_file = os.path.join(os.path.dirname(__file__), '..', '..', 'recommendation_cache.json')
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    # Convert string keys back to tuples
                    for key_str, value in cached_data.items():
                        try:
                            key = eval(key_str)  # Convert string representation back to tuple
                            self.recommendation_cache[key] = tuple(value)
                        except:
                            continue
                print(f"Loaded {len(self.recommendation_cache)} cached recommendations")
        except Exception as e:
            print(f"Failed to load recommendation cache: {e}")

    def save_recommendation_cache(self):
        """Save recommendation cache to disk for future use"""
        cache_file = os.path.join(os.path.dirname(__file__), '..', '..', 'recommendation_cache.json')
        try:
            # Convert tuple keys to strings for JSON serialization
            cache_dict = {str(k): v for k, v in self.recommendation_cache.items()}
            with open(cache_file, 'w') as f:
                json.dump(cache_dict, f, indent=2)
            print(f"Saved {len(cache_dict)} recommendations to cache")
        except Exception as e:
            print(f"Failed to save recommendation cache: {e}")

    def log_performance_metrics(self):
        """Track and log performance metrics for debugging"""
        current_time = time.time()
        cycle_time = current_time - self.last_cycle_time
        self.last_cycle_time = current_time
        self.cycle_times.append(cycle_time)
        self.frame_count += 1

        # Calculate FPS (frames per second)
        if len(self.cycle_times) > 0:
            avg_cycle_time = sum(self.cycle_times) / len(self.cycle_times)
            fps = 1.0 / avg_cycle_time if avg_cycle_time > 0 else 0

            # Log warning if cycle time is too slow
            if cycle_time > 0.8:
                print(f"⚠️  Slow cycle detected: {cycle_time:.2f}s (target: <0.8s)")

            # Update FPS counter display every 5 frames
            if self.frame_count % 5 == 0:
                try:
                    # Update status bar
                    status_msg = f"FPS: {fps:.1f} | Avg Cycle: {avg_cycle_time*1000:.0f}ms | Frame: {self.frame_count}"
                    self.gui.after(0, lambda msg=status_msg: self.gui.set_status(msg))

                    # Update FPS display label
                    cycle_ms = avg_cycle_time * 1000
                    self.gui.after(0, lambda f=fps, c=cycle_ms: self.gui.update_fps_display(f, c))
                except Exception as e:
                    print(f"Error updating FPS display: {e}")
