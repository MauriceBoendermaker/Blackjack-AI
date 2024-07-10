import random
from collections import defaultdict

# Constants
NUM_DECKS = 8
CARDS_PER_DECK = 52
INITIAL_CARDS_COUNT = {
    '2': 4 * NUM_DECKS, '3': 4 * NUM_DECKS, '4': 4 * NUM_DECKS, '5': 4 * NUM_DECKS,
    '6': 4 * NUM_DECKS, '7': 4 * NUM_DECKS, '8': 4 * NUM_DECKS, '9': 4 * NUM_DECKS,
    '10': 16 * NUM_DECKS, 'A': 4 * NUM_DECKS
}
DEALER_HIT_SOFT_17 = True  # Set to False if dealer stands on soft 17


# Helper functions
def calculate_probabilities(cards_count):
    total_cards = sum(cards_count.values())
    probabilities = {card: count / total_cards for card, count in cards_count.items()}
    return probabilities


def update_cards_count(cards_count, card_drawn):
    cards_count[card_drawn] -= 1
    return cards_count


def calculate_hand_total(hand):
    total = 0
    aces = 0
    for card in hand:
        if card in ['10', 'J', 'Q', 'K']:
            total += 10
        elif card == 'A':
            total += 11
            aces += 1
        else:
            total += int(card)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def optimal_decision(player_hand, dealer_card, cards_count):
    probabilities = calculate_probabilities(cards_count)
    player_total = calculate_hand_total(player_hand)

    if len(player_hand) == 2:
        if player_hand[0] == player_hand[1]:
            return optimal_split(player_hand, dealer_card, probabilities)
        if player_total in [9, 10, 11]:
            return optimal_double(player_hand, dealer_card, probabilities)

    if player_total <= 11:
        return "Hit"
    if player_total >= 17:
        return "Stand"
    if player_total >= 12 and dealer_card in ['4', '5', '6']:
        return "Stand"
    return "Hit"


def optimal_split(player_hand, dealer_card, probabilities):
    card = player_hand[0]
    if card in ['A', '8']:
        return "Split"
    if card == '10':
        return "Stand"
    if card == '9':
        if dealer_card in ['2', '3', '4', '5', '6', '8', '9']:
            return "Split"
        else:
            return "Stand"
    if card == '7':
        if dealer_card in ['2', '3', '4', '5', '6', '7']:
            return "Split"
        else:
            return "Hit"
    if card == '6':
        if dealer_card in ['2', '3', '4', '5', '6']:
            return "Split"
        else:
            return "Hit"
    if card == '5':
        return "Double Down" if dealer_card in ['2', '3', '4', '5', '6', '7', '8', '9'] else "Hit"
    if card == '4':
        return "Split" if dealer_card in ['5', '6'] else "Hit"
    if card == '3' or card == '2':
        return "Split" if dealer_card in ['2', '3', '4', '5', '6', '7'] else "Hit"
    return "Stand"


def optimal_double(player_hand, dealer_card, probabilities):
    player_total = calculate_hand_total(player_hand)
    if player_total == 11:
        return "Double Down"
    if player_total == 10 and dealer_card not in ['10', 'A']:
        return "Double Down"
    if player_total == 9 and dealer_card in ['3', '4', '5', '6']:
        return "Double Down"
    return "Hit"


def draw_card(cards_count):
    card = random.choices(
        population=list(cards_count.keys()),
        weights=list(cards_count.values()),
        k=1
    )[0]
    cards_count = update_cards_count(cards_count, card)
    return card, cards_count


# Simulation of a round
def simulate_round():
    cards_count = INITIAL_CARDS_COUNT.copy()
    player_hand = []
    dealer_hand = []

    # Initial draw
    for _ in range(2):
        card, cards_count = draw_card(cards_count)
        player_hand.append(card)
    card, cards_count = draw_card(cards_count)
    dealer_hand.append(card)
    card, cards_count = draw_card(cards_count)
    dealer_hand.append(card)

    player_total = calculate_hand_total(player_hand)
    dealer_card = dealer_hand[0]

    while True:
        decision = optimal_decision(player_hand, dealer_card, cards_count)
        print(f"Player hand: {player_hand}, Dealer card: {dealer_card}, Decision: {decision}")
        if decision == "Stand":
            break
        elif decision == "Hit":
            card, cards_count = draw_card(cards_count)
            player_hand.append(card)
            player_total = calculate_hand_total(player_hand)
            if player_total > 21:
                print(f"Player busts with hand {player_hand}")
                break
        elif decision == "Double Down":
            card, cards_count = draw_card(cards_count)
            player_hand.append(card)
            player_total = calculate_hand_total(player_hand)
            print(f"Player doubles down and draws {card}, final hand: {player_hand}")
            break
        elif decision == "Split":
            print(f"Player splits {player_hand}")
            break

    # Dealer's turn
    while calculate_hand_total(dealer_hand) < 17 or (
            DEALER_HIT_SOFT_17 and calculate_hand_total(dealer_hand) == 17 and 'A' in dealer_hand):
        card, cards_count = draw_card(cards_count)
        dealer_hand.append(card)

    print(
        f"Final hands - Player: {player_hand} (total: {calculate_hand_total(player_hand)}), Dealer: {dealer_hand} (total: {calculate_hand_total(dealer_hand)})")


simulate_round()
