import numpy as np
import matplotlib.pyplot as plt

decks = 8
cards_per_deck = 52
initial_card_counts = np.array([4, 4, 4, 4, 4, 4, 4, 4, 16, 4]) * decks  # 2-10, JQK, A


def calculate_expected_win(player_hand, dealer_upcard, remaining_deck):
    draw_expectation = np.random.uniform(-1, 1)  # Placeholder for actual calculation
    double_down_expectation = np.random.uniform(-1, 1)  # Placeholder for actual calculation
    split_expectation = np.random.uniform(-1, 1) if player_hand in [2, 3, 4, 6, 8] else None  # Placeholder
    return draw_expectation, double_down_expectation, split_expectation


def optimal_action(player_hand, dealer_upcard, remaining_deck):
    draw_exp, dd_exp, split_exp = calculate_expected_win(player_hand, dealer_upcard, remaining_deck)
    expectations = {'Hit': draw_exp, 'Double Down': dd_exp, 'Split': split_exp}
    optimal_move = max(expectations, key=lambda k: expectations[k] if expectations[k] is not None else -np.inf)
    return optimal_move, expectations


def simulate_game_rounds(remaining_deck, num_rounds=100):
    results = []
    for _ in range(num_rounds):
        for player_hand in range(2, 22):
            for dealer_upcard in range(2, 12):
                move, exp_wins = optimal_action(player_hand, dealer_upcard, remaining_deck)
                results.append({
                    'player_hand': player_hand,
                    'dealer_upcard': dealer_upcard,
                    'move': move,
                    'draw_exp': exp_wins['Hit'],
                    'dd_exp': exp_wins['Double Down'],
                    'split_exp': exp_wins['Split'],
                    'cards_drawn': remaining_deck.copy()
                })
    return results


def simulate_game_scenario(remaining_deck):
    results = []
    for player_hand in range(2, 22):
        for dealer_upcard in range(2, 12):
            move, exp_wins = optimal_action(player_hand, dealer_upcard, remaining_deck)
            results.append((player_hand, dealer_upcard, move, exp_wins))
    return results


def plot_results(results):
    player_hands = range(2, 22)
    dealer_upcards = range(2, 12)

    decisions = np.zeros((20, 10), dtype='<U12')
    expectations = np.zeros((20, 10))

    for result in results:
        player_hand, dealer_upcard, move, exp_wins = result
        if move == 'Split':
            draw_exp, dd_exp, split_exp = calculate_expected_win(player_hand, dealer_upcard, initial_card_counts)
            decisions[player_hand - 2, dealer_upcard - 2] = 'Split'
            expectations[player_hand - 2, dealer_upcard - 2] = split_exp
        elif move == 'Double Down':
            decisions[player_hand - 2, dealer_upcard - 2] = 'Double Down'
            expectations[player_hand - 2, dealer_upcard - 2] = exp_wins['Double Down']
        else:
            decisions[player_hand - 2, dealer_upcard - 2] = 'Hit'
            expectations[player_hand - 2, dealer_upcard - 2] = exp_wins['Hit']

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(expectations, cmap='coolwarm', aspect='auto')

    for i in range(len(player_hands)):
        for j in range(len(dealer_upcards)):
            ax.text(j, i, decisions[i, j], ha='center', va='center', color='black')

    ax.set_title('Optimal Blackjack Strategy')
    ax.set_xlabel('Dealer Upcard')
    ax.set_ylabel('Player Hand')
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(dealer_upcards)
    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(player_hands)

    fig.colorbar(im, ax=ax)
    plt.show()


remaining_deck = initial_card_counts.copy()
results = simulate_game_scenario(remaining_deck)
plot_results(results)

# Detailed simulation results
detailed_results = simulate_game_rounds(remaining_deck, num_rounds=1)
for res in detailed_results[:10]:  # Print first 10 results for brevity
    print(res)
