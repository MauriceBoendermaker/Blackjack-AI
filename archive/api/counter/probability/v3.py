import matplotlib.pyplot as plt
import numpy as np

# Define initial deck state
remaining_cards = [4, 4, 4, 4, 4, 4, 4, 4, 16, 4]  # 4 of each 2-9, 16 ten-value cards (10, J, Q, K), and 4 Aces

# Rules
dealer_hits_soft_17 = True
dealer_checks_blackjack = True


# Helper functions
def calculate_probabilities(deck):
    total_cards = np.sum(deck)
    return np.array(deck) / total_cards


def calculate_expected_value(deck, player_hand, dealer_upcard, action, soft=False):
    probabilities = calculate_probabilities(deck)
    if action == "Hit":
        return calculate_hit_value(deck, player_hand, dealer_upcard, probabilities, soft)
    elif action == "Stand":
        return calculate_stand_value(player_hand, dealer_upcard, probabilities, soft)
    elif action == "Double Down":
        return calculate_double_down_value(deck, player_hand, dealer_upcard, probabilities, soft)
    elif action == "Split":
        return calculate_split_value(deck, player_hand, dealer_upcard, probabilities)
    return -np.inf


def calculate_hit_value(deck, player_hand, dealer_upcard, probabilities, soft):
    expected_value = 0
    for card_value, prob in enumerate(probabilities):
        new_hand_value = player_hand + card_value + 2
        if new_hand_value <= 21:
            expected_value += prob * calculate_stand_value(new_hand_value, dealer_upcard, probabilities,
                                                           soft and (new_hand_value <= 21))
        elif soft and new_hand_value <= 31:
            new_hand_value -= 10  # Treat Ace as 1 instead of 11
            expected_value += prob * calculate_stand_value(new_hand_value, dealer_upcard, probabilities, False)
        else:
            expected_value -= prob  # Busting scenario
    return expected_value


def calculate_stand_value(player_hand, dealer_upcard, probabilities, soft):
    dealer_hand = dealer_upcard
    dealer_soft = (dealer_upcard == 11)
    while dealer_hand < 17 or (dealer_hand == 17 and dealer_hits_soft_17 and dealer_soft):
        for card_value, prob in enumerate(probabilities):
            dealer_hand += card_value + 2
            if dealer_hand > 21 and dealer_soft:
                dealer_hand -= 10  # Treat Ace as 1 instead of 11
                dealer_soft = False
            if dealer_hand >= 17 and (dealer_hand != 17 or not dealer_hits_soft_17 or not dealer_soft):
                break
    if dealer_hand > 21 or player_hand > dealer_hand:
        return 1  # Win
    elif player_hand == dealer_hand:
        return 0  # Push
    else:
        return -1  # Lose


def calculate_double_down_value(deck, player_hand, dealer_upcard, probabilities, soft):
    return 2 * calculate_hit_value(deck, player_hand, dealer_upcard, probabilities, soft)


def calculate_split_value(deck, player_hand, dealer_upcard, probabilities):
    return 2 * calculate_hit_value(deck, player_hand // 2, dealer_upcard, probabilities, False)


def optimal_action(player_hand, dealer_upcard, deck, soft=False, pair=False):
    actions = ["Hit", "Stand", "Double Down"]
    if pair:
        actions.append("Split")
    best_action = None
    best_value = -np.inf
    for action in actions:
        value = calculate_expected_value(deck, player_hand, dealer_upcard, action, soft)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action


def generate_strategy(deck):
    strategy = []
    for player_hand in range(4, 22):
        row = []
        for dealer_upcard in range(2, 12):
            action = optimal_action(player_hand, dealer_upcard, deck)
            row.append(action)
        strategy.append(row)

    soft_strategy = []
    for player_hand in range(13, 22):
        row = []
        for dealer_upcard in range(2, 12):
            action = optimal_action(player_hand, dealer_upcard, deck, soft=True)
            row.append(action)
        soft_strategy.append(row)

    split_strategy = []
    for player_hand in range(2, 12):
        row = []
        for dealer_upcard in range(2, 12):
            if player_hand == 10:  # 10-10 should not be split
                action = "Stand"
            else:
                action = optimal_action(player_hand * 2, dealer_upcard, deck, pair=True)
            row.append(action)
        split_strategy.append(row)

    return strategy, soft_strategy, split_strategy


def print_strategy(strategy, strategy_name):
    headers = ["Player"] + [str(i) for i in range(2, 12)]
    print(f"\n{strategy_name} Strategy")
    print("\t".join(headers))
    for i, row in enumerate(strategy):
        player_hand = 4 + i if strategy_name == "Hard" else 13 + i if strategy_name == "Soft" else i + 2
        print(f"{player_hand}\t" + "\t".join(row))


# Map actions to colors
action_colors = {
    "Hit": "green",
    "Stand": "grey",
    "Double Down": "blue",
    "Split": "red"
}


def plot_strategy_table(strategy, title, row_labels, col_labels):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    table_data = [[row_labels[i]] + list(strategy[i]) for i in range(len(row_labels))]
    colors = [[action_colors[action] for action in row[1:]] for row in table_data]

    table = ax.table(cellText=table_data, colLabels=["Player"] + col_labels, cellLoc='center', loc='center',
                     cellColours=[["white"] + colors[i] for i in range(len(colors))])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.title(title, fontsize=16)
    plt.show()


# Generate and print strategy
hard_strategy, soft_strategy, split_strategy = generate_strategy(remaining_cards)
print_strategy(hard_strategy, "Hard")
print_strategy(soft_strategy, "Soft")
print_strategy(split_strategy, "Split")

# Labels
dealer_upcard_labels = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
hard_hand_labels = [str(i) for i in range(4, 22)]
soft_hand_labels = [f"{i} soft" for i in range(13, 22)]
split_hand_labels = [f"{i}-{i}" for i in range(2, 12)]

# Plot the tables
plot_strategy_table(hard_strategy, "Hard Strategy", hard_hand_labels, dealer_upcard_labels)
plot_strategy_table(soft_strategy, "Soft Strategy", soft_hand_labels, dealer_upcard_labels)
plot_strategy_table(split_strategy, "Split Strategy", split_hand_labels, dealer_upcard_labels)
