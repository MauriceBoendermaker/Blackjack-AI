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
        card_value_name = card_name.split(' ')[0]
        if card_value_name in constants.VALUE_MAPPING:
            card_value = constants.VALUE_MAPPING[card_value_name]
        else:
            print(f"Warning: Card value '{card_value_name}' not found in value mapping.")
            return
        card_value_str = str(card_value)
        if card_value_str in self.card_counters:
            self.card_counters[card_value_str] += increment
            for label in self.card_counter_labels:
                if label.cget("text").startswith(card_value_str + ":"):
                    new_text = f"{card_value_str}: {self.card_counters[card_value_str]}x"
                    label.config(text=new_text)
                    break
        else:
            print(f"Warning: Card value '{card_value_str}' not found in card counters.")

    def calculate_true_count(self):
        decks_remaining = (constants.DECK_COUNT * 52 - len(self.counted_cards_this_round)) / 52
        true_count = self.running_count / decks_remaining if decks_remaining > 0 else self.running_count
        return true_count

    def get_card_value(self, card):
        card_value = card.split(' ')[0]
        if card_value in ["Jack", "Queen", "King"]:
            return 10
        elif card_value == "Ace":
            return 11  # Typically, an Ace is worth 11 unless it causes a bust, then it's worth 1
        elif card_value == "-":
            return 0  # Placeholder value for no card detected
        return int(card_value)

    def get_dealer_card_value(self, card):
        card_value = card.split(' ')[0].capitalize()  # Capitalize the first letter
        if card_value in ["Jack", "Queen", "King"]:
            return "10"
        elif card_value == "Ace":
            return "A"
        return card_value

    def is_duplicate_or_nearby_card(self, detected_card, existing_cards):
        return detected_card['card_name'] in existing_cards

    def print_player_cards(self, player_index, cards, recommendation):
        hand_value = self.calculate_hand_value(cards)
        cards_info = " // ".join([f"[{i + 1}] {card}" for i, card in enumerate(cards)])
        recommendation_text = ", ".join([action[0] for action in recommendation])
        print(
            f"P{player_index + 1}: {cards_info}. Card value is {hand_value}. Recommended action: {recommendation_text}.")

    def get_hand_representation(self, cards):
        if self.is_pair_hand(cards):
            pair_value = str(self.get_card_value(cards[0].split(' ')[0]))
            pair_value = "10" if pair_value in ["Jack", "Queen", "King"] else pair_value
            return f"{pair_value},{pair_value}"

        if self.is_soft_hand(cards):
            non_ace_total = sum(
                [self.get_card_value(card.split(' ')[0]) for card in cards if card.split(' ')[0] != 'Ace'])
            return f"A,{non_ace_total}"

        hand_total = self.calculate_hand_value(cards)
        return str(hand_total)

    def calculate_hand_value(self, cards):
        total = 0
        ace_count = 0
        for card in cards:
            card_value = self.get_card_value(card)
            if card_value == 11:  # Ace handling
                ace_count += 1
            total += card_value

        # Adjust for Aces if total is above 21
        while total > 21 and ace_count:
            total -= 10
            ace_count -= 1

        return total

    def is_soft_hand(self, cards):
        values = [self.get_card_value(card.split(' ')[0]) for card in cards]
        return 11 in values and sum(values) + 10 <= 21

    @staticmethod
    def is_pair_hand(player_cards):
        return len(player_cards) == 2 and player_cards[0].split(' ')[0] == player_cards[1].split(' ')[0]
