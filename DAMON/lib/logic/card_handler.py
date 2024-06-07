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
        # Simplify the card name to its basic value for counting purposes
        card_key = card_name.split(' ')[0]
        if card_key in ["Jack", "Queen", "King"]:  # Normalize face cards to "10"
            card_key = "10"

        # Check if this card (in its simplified form) has already been counted in this round
        if card_key not in self.card_utils.counted_cards_this_round:
            self.card_utils.update_card_counter(card_key, 1)  # Update the counter for this card
            self.card_utils.counted_cards_this_round.add(card_key)  # Mark this card as counted for the current round

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

    def print_all_cards(self, player_cards, card_value_counts):
        self.cards_info.clear()  # Clear previous card info
        for player_index in sorted(player_cards):
            cards = player_cards[player_index]['cards']
            confidences = player_cards[player_index]['confidences']

            card_info = []
            for i, (card, conf) in enumerate(zip(cards, confidences), start=1):
                card_info.append(f"[{i}] {card} (C: {conf * 100:.2f}%)")

            self.cards_info.append(f"P{player_index + 1}: {' // '.join(card_info)}")
            print(f"P{player_index + 1}: {' // '.join(card_info)}")

        formatted_card_counts = [f"{value} => {count}x" for value, count in
                                 sorted(card_value_counts.items(), key=lambda item: item[0])]
        total_cards = sum(card_value_counts.values())  # Sum up the counts of all cards for the total
        print(f"Card counts ({total_cards}):\n", "\n".join(formatted_card_counts))

    def add_or_update_player_card(self, detected_card, player_info, card_name):
        if "-" in player_info['cards']:
            replace_index = player_info['cards'].index("-")
            player_info['cards'][replace_index] = card_name
            player_info['confidences'][replace_index] = detected_card['confidence']
        else:
            player_info['cards'].append(card_name)
            player_info['confidences'].append(detected_card['confidence'])

        print(f"Player {player_info} card updated: {player_info['cards']}")
        # self.update_gui()  # Ensure the GUI is updated
