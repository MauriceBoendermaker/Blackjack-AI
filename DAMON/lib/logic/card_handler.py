import os
import tempfile

from ..common import constants
from ..common.card_mappings import dealer_class_mapping
from .card_utils import CardUtils


class CardHandler:
    def __init__(self):
        self.dealer_cards = []
        self.cards_info = []
        self.card_utils = CardUtils()

    def handle_card_detection(self, card_name):
        if isinstance(card_name, int):
            card_value = card_name
        else:
            card_value = self.card_utils.get_card_value(card_name)

        self.card_utils.update_count(card_name)
        self.card_utils.print_card_counts()

    def convert_int_to_card_name(self, value):
        # Example conversion logic; customize as needed
        if value == 11:
            return "Ace"
        elif value <= 10:
            return str(value)
        else:
            return "Unknown"

    def capture_dealer_cards(self, image, model):
        # Save the cropped image for debugging
        image.save(constants.OUTPUT_DEBUG_IMAGE_PATH)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image.save(temp_file_path)

        # Proceed with the prediction
        predictions = model.predict(temp_file_path, confidence=constants.PREDICTION_CONFIDENCE_DEALER,
                                    overlap=constants.PREDICTION_OVERLAP_DEALER).json()['predictions']
        os.unlink(temp_file_path)  # Delete the temp file after prediction

        for prediction in predictions:
            class_label = prediction['class']
            card_name = dealer_class_mapping.get(class_label, "Unknown")
            self.dealer_cards.append(card_name)

        return self.dealer_cards

    def print_all_cards(self, player_cards):
        self.cards_info.clear()  # Clear previous card info
        for player_index in sorted(player_cards):
            cards = player_cards[player_index]['cards']
            confidences = player_cards[player_index]['confidences']

            card_info = []
            for i, (card, conf) in enumerate(zip(cards, confidences), start=1):
                card_info.append(f"[{i}] {card} (C: {conf * 100:.2f}%)")

            self.cards_info.append(f"P{player_index + 1}: {' // '.join(card_info)}")
            print(f"P{player_index + 1}: {' // '.join(card_info)}")

        # Print card counts using the method from CardUtils
        self.card_utils.print_card_counts()

    def add_or_update_player_card(self, detected_card, player_info, card_name):
        if "-" in player_info['cards']:
            replace_index = player_info['cards'].index("-")
            player_info['cards'][replace_index] = card_name
            player_info['confidences'][replace_index] = detected_card['confidence']
        else:
            player_info['cards'].append(card_name)
            player_info['confidences'].append(detected_card['confidence'])

        print(f"Player {player_info} card updated: {player_info['cards']}")
        self.card_utils.update_count(card_name)  # Ensure card count is updated for each detected card
        self.card_utils.print_card_counts()  # Print updated card counts
