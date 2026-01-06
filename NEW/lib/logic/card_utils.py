from ..common import constants
from ..common.card_mappings import class_mapping


class CardUtils:
    def __init__(self, gui=None):
        self.gui = gui
        self.true_count = 0
        self.hand_value = None
        self.card_counter_labels = {}
        self.counted_cards_this_round = set()
        self.running_count = 0
        self.card_counters = {str(k): 0 for k in range(1, 11)}
        self.card_counters.update({'Ace': 0})

    def get_card_name(self, class_label):
        return class_mapping.get(class_label, "Unknown")

    def is_valid_player_class(self, class_label):
        name = class_mapping.get(class_label, "Unknown")
        if name in ("Unknown", "Continue", "End Red", "GG", "G"):
            return False
        return name.endswith(("Hearts", "Diamonds", "Spades", "Clubs"))

    def is_valid_player_card_name(self, card_name):
        if card_name in ("Unknown", "Continue", "End Red", "GG", "G"):
            return False
        return card_name.endswith(("Hearts", "Diamonds", "Spades", "Clubs"))

    def get_all_card_names(self):
        suits = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
        values = constants.VALUE_MAPPING.keys()
        return [f"{value} of {suit}" for suit in suits for value in values]

    def update_count(self, card_name):
        card_value_name = card_name.split(' ')[0]
        print(f"[update_count] base: {card_value_name}")
        card_value = self.get_card_value(card_name)
        if card_value == 0:
            return
        if 2 <= card_value <= 6:
            self.running_count += 1
        elif card_value in [10, 11]:
            self.running_count -= 1
        if card_value_name == "Ace":
            card_key = "Ace"
        elif card_value_name in ["Jack", "Queen", "King"]:
            card_key = "10"
        else:
            card_key = card_value_name
        self.card_counters[card_key] += 1
        print(f"Updated counter for {card_key}: {self.card_counters[card_key]}x")
        for label_key, label in self.card_counter_labels.items():
            display_base = label_key.split()[0]
            mapped_value = constants.VALUE_MAPPING.get(display_base, None)
            if mapped_value and str(mapped_value) == card_key:
                try:
                    top = label.winfo_toplevel()
                    top.after(0, lambda l=label, k=label_key, v=self.card_counters[card_key]: l.config(text=f"{k}: {v}x"))
                except Exception as e:
                    print(f"Failed to update label for {label_key}: {e}")

    def update_card_counter(self, card_name, increment):
        card_value_name = card_name.split(' ')[0]
        if card_value_name in constants.VALUE_MAPPING:
            card_value = constants.VALUE_MAPPING[card_value_name]
        else:
            print(f"Warning: Card value '{card_value_name}' not found in value mapping.")
            return
        if card_value_name == "Ace":
            card_value_str = "Ace"
        elif card_value_name in ["Jack", "Queen", "King"]:
            card_value_str = "10"
        else:
            card_value_str = card_value_name
        self.card_counters[card_value_str] += increment
        print(f"Updated counter for {card_value_str}: {self.card_counters[card_value_str]}x")

        # Update old-style labels (legacy GUI)
        for label_key, label in self.card_counter_labels.items():
            label_base = label_key.split(' ')[0]
            if (label_base == card_value_name or constants.VALUE_MAPPING.get(label_base) == card_value):
                try:
                    root = label.winfo_toplevel()
                    root.after(0, lambda l=label, k=label_key, v=self.card_counters[card_value_str]: l.config(text=f"{k}: {v}x"))
                except Exception as e:
                    print(f"Failed to update label for {label_key}: {e}")

        # Update modern GUI widgets
        if self.gui and hasattr(self.gui, 'card_counter_widgets'):
            # Direct update for the specific card value
            count = self.card_counters[card_value_str]
            for widget_key, widget_info in self.gui.card_counter_widgets.items():
                # Match the widget key with the updated counter
                if widget_key == card_value_name or widget_key == card_value_str:
                    try:
                        # Direct update without after() for immediate visibility
                        widget_info['var'].set(f"{count}x")
                        print(f"Updated modern widget for {widget_key}: {count}x")
                    except Exception as e:
                        print(f"Failed to update modern widget for {widget_key}: {e}")

    def calculate_true_count(self):
        decks_remaining = (constants.DECK_COUNT * 52 - len(self.counted_cards_this_round)) / 52
        true_count = self.running_count / decks_remaining if decks_remaining > 0 else self.running_count
        return true_count

    def get_card_value(self, card_name):
        if card_name == '-':
            return 0
        if isinstance(card_name, int):
            return card_name
        face_values = {'Jack': 10, 'Queen': 10, 'King': 10, 'Ace': 11}
        card_value_name = card_name.split()[0]
        if card_value_name in face_values:
            return face_values[card_value_name]
        else:
            try:
                return int(card_value_name)
            except ValueError:
                raise ValueError(f"Invalid card value: {card_value_name}")

    def print_card_counts(self):
        print("Current Running Count:", self.running_count)
        print("Card counts:")
        for value, count in sorted(self.card_counters.items()):
            print(f" {value} => {count}x")

    def get_dealer_card_value(self, card):
        v = str(card).strip().upper()
        if v in {"J", "Q", "K"}:
            return "10"
        if v == "A":
            return "A"
        if v in {"10", "9", "8", "7", "6", "5", "4", "3", "2"}:
            return v
        card_value = v.split(' ')[0].capitalize()
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
        print(f"P{player_index + 1}: {cards_info}. Card value is {hand_value}. Recommended action: {recommendation_text}.")

    def get_hand_representation(self, cards):
        if self.is_pair_hand(cards):
            pair_value = str(self.get_card_value(cards[0].split(' ')[0]))
            pair_value = "10" if pair_value in ["Jack", "Queen", "King"] else pair_value
            return f"{pair_value},{pair_value}"
        if self.is_soft_hand(cards):
            non_ace_total = sum([self.get_card_value(card.split(' ')[0]) for card in cards if card.split(' ')[0] != 'Ace'])
            return f"A,{non_ace_total}"
        hand_total = self.calculate_hand_value(cards)
        return str(hand_total)

    def calculate_hand_value(self, cards):
        total = 0
        ace_count = 0
        for card in cards:
            card_value = self.get_card_value(card)
            if card_value == 11:
                ace_count += 1
            total += card_value
        while total > 21 and ace_count:
            total -= 10
            ace_count -= 1
        return total

    def is_soft_hand(self, cards):
        values = [self.get_card_value(card.split(' ')[0]) for card in cards if card != '-']
        return 11 in values and sum(values) + 10 <= 21

    @staticmethod
    def is_pair_hand(player_cards):
        return len(player_cards) == 2 and player_cards[0].split(' ')[0] == player_cards[1].split(' ')[0]

    def update_label_safe(self, label, text):
        if self.gui:
            print(f"[UI Thread] Scheduling label update: {text}")
            self.gui.after(0, lambda: label.config(text=text))
        else:
            print("[Warning] GUI not attached. Label not updated.")

    def refresh_card_counter_widgets(self):
        """Refresh all card counter widgets to reflect current counts"""
        if not self.gui or not hasattr(self.gui, 'card_counter_widgets'):
            print("[Warning] GUI or card_counter_widgets not available")
            return

        for card_value, widget_info in self.gui.card_counter_widgets.items():
            # Map display card value to internal counter key
            if card_value == "Ace":
                counter_key = "Ace"
            elif card_value in ["Jack", "Queen", "King"]:
                counter_key = "10"
            else:
                counter_key = card_value

            count = self.card_counters.get(counter_key, 0)
            widget_info['var'].set(f"{count}x")

        print("[Card Counters] Refreshed all counter widgets")


_card_utils_instance = None


def get_card_utils(gui=None):
    global _card_utils_instance
    if _card_utils_instance is None:
        _card_utils_instance = CardUtils(gui)
    elif gui:
        _card_utils_instance.gui = gui
    return _card_utils_instance
