from ..common import constants
from ..common.card_mappings import class_mapping


class CardUtils:
    def __init__(self):
        self.true_count = 0
        self.running_count = 0
        self.hand_value = None
        self.card_counters = {str(value): 0 for value in set(constants.VALUE_MAPPING.values())}
        self.card_counter_labels = {}
        self.counted_cards_this_round = set()

    # Get card name from class label
    def get_card_name(self, class_label):
        return class_mapping.get(class_label, "Unknown")

    def get_all_card_names(self):
        suits = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
        values = constants.VALUE_MAPPING.keys()
        return [f"{value} of {suit}" for suit in suits for value in values]

    def update_count(self, card_name):
        card_value = self.get_card_value(card_name)
        if card_value >= 2 and card_value <= 6:
            self.running_count += 1
        elif card_value == 10 or card_name == "Ace":
            self.running_count -= 1

    def update_card_counter(self, card_name, increment):
        # Extract just the value from the card name (e.g., '7 of Diamonds' -> '7')
        card_value_name = card_name.split(' ')[0]  # Assumes card_name format is "Value of Suit"

        # Convert the card name to its value using value_mapping
        if card_value_name in constants.VALUE_MAPPING:
            card_value = constants.VALUE_MAPPING[card_value_name]
        else:
            print(f"Warning: Card value '{card_value_name}' not found in value mapping.")
            return

        # Convert card value back to string for display purposes
        card_value_str = str(card_value)

        # Increment the card counter
        if card_value_str in self.card_counters:
            self.card_counters[card_value_str] += increment
            # Find the corresponding label for the card value and update its text
            for label in self.card_counter_labels:
                if label.cget("text").startswith(card_value_str + ":"):  # Check if the label starts with the card value
                    new_text = f"{card_value_str}: {self.card_counters[card_value_str]}x"
                    label.config(text=new_text)
                    break
        else:
            print(f"Warning: Card value '{card_value_str}' not found in card counters.")

    def calculate_true_count(self):
        decks_remaining = (constants.DECK_COUNT * 52 - len(self.counted_cards_this_round)) / 52
        true_count = self.running_count / decks_remaining if decks_remaining > 0 else self.running_count
        return true_count

    # Get card value
    def get_card_value(self, card_name):
        card_value = card_name.split(' ')[0]  # Extract the card's face value (ignore suit for now, e.g., " of Spades")
        return constants.VALUE_MAPPING.get(card_value, 0)  # Return 0 if card name is not recognized

    # Helper function to interpret the dealer's card correctly
    def get_dealer_card_value(self, card):
        if card in ["J", "Q", "K"]:
            return "10"
        return card

    # Check if card is duplicate or nearby
    def is_duplicate_or_nearby_card(self, detected_card, existing_cards):
        return detected_card['card_name'] in existing_cards

    # Print player cards in console
    def print_player_cards(self, player_index, cards, recommendation):
        hand_value = self.calculate_hand_value(cards)
        cards_info = " // ".join([f"[{i + 1}] {card}" for i, card in enumerate(cards)])
        recommendation_text = ", ".join([action[0] for action in recommendation])  # Extract the action text
        print(
            f"P{player_index + 1}: {cards_info}. Card value is {hand_value}. Recommended action: {recommendation_text}.")

    def get_hand_representation(self, cards):
        if self.is_pair_hand(cards):
            pair_value = str(self.get_card_value(cards[0].split(' ')[0]))
            pair_value = "10" if pair_value in ["J", "Q", "K"] else pair_value
            return f"Pair {pair_value}"

        if self.is_soft_hand(cards):
            non_ace_total = sum(
                [self.get_card_value(card.split(' ')[0]) for card in cards if card.split(' ')[0] != 'Ace'])
            return f"A,{non_ace_total}"

        hand_total = self.calculate_hand_value(cards)
        return str(hand_total)

    def calculate_hand_value(self, cards):
        total, aces = 0, 0

        for card in cards:  # Calculate the total value and count the aces
            # Extract the value part (e.g., "Ace" from "Ace of Diamonds")
            card_value = self.get_card_value(card.split(' ')[0])
            if card_value == 1:  # Ace is counted as 1 initially
                aces += 1
            else:
                total += card_value
        # Adjust for Aces after calculating the total
        for _ in range(aces):
            # If adding 11 keeps the total 21 or under, use 11 for Ace; otherwise, use 1
            total += 11 if total + 11 <= 21 else 1
        return total

    # Check if the hand is a soft hand (contains an Ace counted as 11)
    def is_soft_hand(self, cards):
        values = [self.get_card_value(card.split(' ')[0]) for card in cards]
        return 1 in values and sum(values) + 10 <= 21

    # Check if hand is a pair
    def is_pair_hand(self, cards):
        if len(cards) != 2:
            return False
        return self.get_card_value(cards[0].split(' ')[0]) == self.get_card_value(cards[1].split(' ')[0])
