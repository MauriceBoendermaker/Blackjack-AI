import random

from collections import defaultdict
import matplotlib.pyplot as plt

# Constants
SOFT = 1
FIRST = 2
SECOND = 4
DD_POSS = 8
SPLIT_POSS = 16
NO_BJ = 32


class BJHand:
    def __init__(self, card_sum, flags):
        self.card_sum = card_sum
        self.flags = flags
        self.win_expectation_stand = -100
        self.win_expectation_opt = -100
        self.optimal_move = ""

    def as_string(self):
        if self.card_sum == 22:
            return "bust"
        if self.is_blackjack():
            return "BJ"
        if self.flags > 0:
            return f"{self.card_sum}f{self.flags}"
        return f"{self.card_sum}"

    def has_flag(self, weight):
        return (self.flags & weight) == weight

    def is_blackjack(self):
        return self.has_flag(SECOND) and self.card_sum == 21 and not self.has_flag(NO_BJ)

    def draw(self, card):
        if self.card_sum == 0:
            return BJHand(card, self.flags | FIRST | (SOFT if card == 11 else 0))

        if self.has_flag(FIRST):
            new_flags = SECOND | ((self.card_sum == 11 or card == 11) and SOFT or 0)
            new_sum = self.card_sum + card
            if new_sum == 21 and self.has_flag(NO_BJ):
                new_flags |= NO_BJ
            if new_sum == 22:
                new_sum = 12
            if self.has_flag(DD_POSS):
                if 9 <= new_sum <= 11:
                    new_flags |= DD_POSS
                if 19 <= new_sum <= 20 and (new_flags & SOFT) == SOFT:
                    new_flags |= DD_POSS
            if self.has_flag(SPLIT_POSS) and self.card_sum == card:
                new_flags |= SPLIT_POSS
            return BJHand(new_sum, new_flags)

        aces_as_eleven = self.has_flag(SOFT) + (card == 11)
        new_sum = self.card_sum + card
        if new_sum >= 22 and aces_as_eleven:
            aces_as_eleven -= 1
            new_sum -= 10
        if new_sum > 22:
            new_sum = 22
        return BJHand(new_sum, SOFT if aces_as_eleven > 0 else 0)

    def prob_hand_after_draw(self, card_probabilities):
        result = ProbabilityDistribution()
        for card, prob in card_probabilities.items():
            new_hand = self.draw(card)
            result.add_probability(new_hand.as_string(), prob)
        return result

    def translate_flags(self):
        flags = []
        if self.has_flag(SOFT):
            flags.append("SOFT")
        if self.has_flag(FIRST):
            flags.append("FIRST")
        if self.has_flag(SECOND):
            flags.append("SECOND")
        if self.has_flag(DD_POSS):
            flags.append("DD_POSS")
        if self.has_flag(SPLIT_POSS):
            flags.append("SPLIT_POSS")
        if self.has_flag(NO_BJ):
            flags.append("NO_BJ")
        return " | ".join(flags) if flags else "No Flags"

    def __str__(self):
        return f"Card Sum: {self.card_sum}, Flags: {self.translate_flags()}"


class ProbabilityDistribution:
    def __init__(self):
        self.distribution = defaultdict(float)

    def add_probability(self, hand_str, prob):
        self.distribution[hand_str] += prob

    def normalize(self):
        total_prob = sum(self.distribution.values())
        for key in self.distribution:
            self.distribution[key] /= total_prob

    def get_probabilities(self):
        return self.distribution

    def __str__(self):
        return str(self.distribution)


def default_blackjack_strategy(player_sum, dealer_card, soft, pair):
    if pair:
        if player_sum == 22 or player_sum == 11:
            return "Split"
        if player_sum == 20:
            return "Stand"
        if player_sum == 18:
            return "Split" if dealer_card < 7 or dealer_card == 8 else "Stand"
        if player_sum == 16 or player_sum == 14:
            return "Split" if dealer_card < 7 else "Hit"
        if player_sum == 12:
            return "Split" if dealer_card < 7 else "Hit"
        if player_sum == 10:
            return "Split" if dealer_card < 10 else "Hit"
        if player_sum == 8:
            return "Split" if dealer_card < 7 else "Hit"
        if player_sum == 6 or player_sum == 4:
            return "Split" if dealer_card < 8 else "Hit"
    if soft:
        if player_sum >= 19:
            return "Stand"
        if player_sum == 18:
            return "Double Down" if dealer_card < 7 else "Stand"
        if player_sum == 17 or player_sum == 16:
            return "Double Down" if 3 <= dealer_card <= 6 else "Hit"
        if player_sum == 15 or player_sum == 14:
            return "Double Down" if 4 <= dealer_card <= 6 else "Hit"
        if player_sum == 13 or player_sum == 12:
            return "Double Down" if 5 <= dealer_card <= 6 else "Hit"
    else:
        if player_sum >= 17:
            return "Stand"
        if player_sum >= 13:
            return "Stand" if dealer_card < 7 else "Hit"
        if player_sum == 12:
            return "Stand" if 4 <= dealer_card <= 6 else "Hit"
        if player_sum == 11:
            return "Double Down"
        if player_sum == 10:
            return "Double Down" if dealer_card < 10 else "Hit"
        if player_sum == 9:
            return "Double Down" if 3 <= dealer_card <= 6 else "Hit"
    return "Hit"


class BlackjackAnalysis:
    def __init__(self, card_probabilities, rule_variant, bj_check, dealer_hits_soft_17):
        self.card_probabilities = card_probabilities
        self.rule_variant = rule_variant
        self.bj_check = bj_check
        self.dealer_hits_soft_17 = dealer_hits_soft_17
        self.hands_analyzed = set()
        self.bank_card = None
        self.bj_direct_prob = 0

    def expected_win_stand(self, hand):
        if hand.as_string() not in self.hands_analyzed:
            self.analyze(hand)
        return hand.win_expectation_stand

    def expected_win_opt(self, hand):
        if hand.as_string() not in self.hands_analyzed:
            self.analyze(hand)
        return hand.win_expectation_opt

    def optimal_move(self, hand):
        if hand.as_string() not in self.hands_analyzed:
            self.analyze(hand)
        return hand.optimal_move

    def expected_win_draw(self, hand):
        result = 0
        new_prob = hand.prob_hand_after_draw(self.card_probabilities)
        for new_hand_str, prob in new_prob.get_probabilities().items():
            new_hand = self.create_hand_from_string(new_hand_str)
            result += self.expected_win_opt(new_hand) * prob
        return result

    def expected_win_draw1(self, hand):
        result = 0
        new_prob = hand.prob_hand_after_draw(self.card_probabilities)
        for new_hand_str, prob in new_prob.get_probabilities().items():
            new_hand = self.create_hand_from_string(new_hand_str)
            result += self.expected_win_stand(new_hand) * prob
        return result

    def analyze(self, hand):
        print(f"Analyzing hand: {hand} (raw: {hand.as_string()})")
        self.hands_analyzed.add(hand.as_string())

        sum_prob = 0
        for bank_hand_str, prob in self.bank_hand_distribution().get_probabilities().items():
            bank_hand = self.create_hand_from_string(bank_hand_str)
            sum_prob += self.calculate_win(bank_hand, hand) * prob
        hand.win_expectation_stand = sum_prob
        hand.win_expectation_opt = sum_prob
        hand.optimal_move = "Stand"

        draw_expectation = self.expected_win_draw(hand)
        if self.bj_check and hand.card_sum == 0:
            draw_expectation = self.bj_direct_prob * (
                    1 - 2 * self.card_probabilities[10] * self.card_probabilities[11]) * (-1) + (
                                       1 - self.bj_direct_prob) * draw_expectation

        if draw_expectation > hand.win_expectation_opt:
            hand.win_expectation_opt = draw_expectation
            hand.optimal_move = "Draw"

        double_expectation = -float('inf')
        if hand.has_flag(DD_POSS | SECOND):
            double_hand = BJHand(hand.card_sum - 10 if hand.has_flag(SOFT) else hand.card_sum, SECOND)
            double_expectation = 2 * self.expected_win_draw1(double_hand)
            if double_expectation > hand.win_expectation_opt:
                hand.win_expectation_opt = double_expectation
                hand.optimal_move = "Double Down"

        split_expectation = -float('inf')
        if hand.has_flag(SPLIT_POSS | SECOND):
            split_card = 11 if hand.has_flag(SOFT) else hand.card_sum // 2
            if self.rule_variant in [0, 3]:
                if split_card == 11:
                    split_hand = BJHand(11, FIRST | SOFT | NO_BJ)
                    split_expectation = 2 * self.expected_win_draw1(split_hand)
                else:
                    flags_new = FIRST
                    if self.rule_variant in [2, 3]:
                        flags_new |= DD_POSS
                    if split_card == 10:
                        flags_new |= NO_BJ
                    split_hand = BJHand(split_card, flags_new)
                    split_expectation = 2 * self.expected_win_draw(split_hand)
            else:
                repeat_prob = self.card_probabilities[split_card]
                if split_card == 11:
                    flags_new = FIRST | SOFT | NO_BJ
                    sum_expectation = self.expected_win_draw1(BJHand(11, flags_new))
                    sum_expectation -= self.expected_win_stand(BJHand(12, SOFT)) * self.card_probabilities[11]
                    split_expectation = 2 / (1 - 2 * repeat_prob) * sum_expectation
                else:
                    flags_new = FIRST
                    if self.rule_variant == 2:
                        flags_new |= DD_POSS
                    if split_card == 10:
                        flags_new |= NO_BJ
                    sum_expectation = self.expected_win_draw(BJHand(split_card, flags_new))
                    sum_expectation -= self.expected_win_opt(BJHand(2 * split_card, SECOND | (
                        DD_POSS if split_card == 5 and self.rule_variant == 2 else 0))) * self.card_probabilities[
                                           split_card]
                    split_expectation = 2 / (1 - 2 * repeat_prob) * sum_expectation

            if split_expectation > hand.win_expectation_opt:
                hand.win_expectation_opt = split_expectation
                hand.optimal_move = "Split"

        default_move = default_blackjack_strategy(hand.card_sum, self.bank_card, hand.has_flag(SOFT),
                                                  hand.has_flag(SPLIT_POSS | SECOND))
        print(
            f"\nAnalysis result: {hand} (raw: {hand.as_string()}) -> {hand.optimal_move} +++ Default strategy: {default_move}")
        print(
            f"\nExplanation: Draw Expectation: {draw_expectation}, Double Down Expectation: {double_expectation if hand.has_flag(DD_POSS | SECOND) else 'N/A'}, Split Expectation: {split_expectation if hand.has_flag(SPLIT_POSS | SECOND) else 'N/A'}")

        # Additional debug information
        print(f"Initial Stand Expectation: {sum_prob}")
        print(f"Calculated Draw Expectation: {draw_expectation}")
        print(f"Calculated Double Down Expectation: {double_expectation if hand.has_flag(DD_POSS | SECOND) else 'N/A'}")
        print(f"Calculated Split Expectation: {split_expectation if hand.has_flag(SPLIT_POSS | SECOND) else 'N/A'}")
        print(f"Optimal Move: {hand.optimal_move}, with Expected Win: {hand.win_expectation_opt}")

    def bank_hand_distribution(self):
        result = ProbabilityDistribution()
        hand = BJHand(self.bank_card, SOFT if self.bank_card == 11 else 0 + FIRST)
        result.add_probability(hand.as_string(), 1.0)
        final_result = ProbabilityDistribution()

        while result.get_probabilities():
            new_result = ProbabilityDistribution()
            for hand_str, prob in result.get_probabilities().items():
                hand = self.create_hand_from_string(hand_str)
                if hand.card_sum <= 16 or (self.dealer_hits_soft_17 and hand.has_flag(SOFT) and hand.card_sum == 17):
                    for new_hand_str, new_prob in hand.prob_hand_after_draw(
                            self.card_probabilities).get_probabilities().items():
                        new_result.add_probability(new_hand_str, prob * new_prob)
                else:
                    final_result.add_probability(hand_str, prob)
            result = new_result

        final_distribution = ProbabilityDistribution()
        for hand_str, prob in final_result.get_probabilities().items():
            hand = self.create_hand_from_string(hand_str)
            if not hand.is_blackjack():
                hand.flags = 0
            final_distribution.add_probability(hand.as_string(), prob)
        return final_distribution

    def calculate_win(self, bank_hand, player_hand):
        if player_hand.card_sum > 21:
            return -1
        if player_hand.is_blackjack():
            if bank_hand.is_blackjack():
                return 0
            return 1.5
        if bank_hand.is_blackjack():
            return -1
        if bank_hand.card_sum > 21:
            return 1
        if player_hand.card_sum > bank_hand.card_sum:
            return 1
        if player_hand.card_sum < bank_hand.card_sum:
            return -1
        return 0

    def create_hand_from_string(self, hand_str):
        if hand_str == "bust":
            return BJHand(22, 0)
        if hand_str == "BJ":
            return BJHand(21, SECOND)
        parts = hand_str.split('f')
        card_sum = int(parts[0])
        flags = int(parts[1]) if len(parts) > 1 else 0
        return BJHand(card_sum, flags)

    def draw_expectation(self, hand):
        draw_probs = hand.prob_hand_after_draw(self.card_probabilities)
        expectation = 0
        for new_hand_str, prob in draw_probs.items():
            new_hand = BJHand.from_string(new_hand_str)
            new_hand_stand_expectation = self.stand_expectation(new_hand)
            print(f"New Hand: {new_hand}, Probability: {prob}, Stand Expectation: {new_hand_stand_expectation}")
            expectation += new_hand_stand_expectation * prob
        return expectation


def main():
    # Example card probabilities for one deck
    card_probabilities = {
        2: 4 / 52,
        3: 4 / 52,
        4: 4 / 52,
        5: 4 / 52,
        6: 4 / 52,
        7: 0 / 52,
        8: 0 / 52,
        9: 0 / 52,
        10: 0 / 52,
        11: 0 / 52  # Ace
    }

    # Create initial hands
    player_hand = BJHand(0, DD_POSS | SPLIT_POSS)
    dealer_hand = 8  # Example: dealer shows a 7

    # Initialize analysis
    analysis = BlackjackAnalysis(card_probabilities, 2, True, True)
    analysis.bank_card = dealer_hand

    # Perform analysis
    analysis.analyze(player_hand)
    optimal_move = analysis.optimal_move(player_hand)

    print(f"Optimal move for player hand {player_hand} with dealer showing {dealer_hand}: {optimal_move}")


if __name__ == "__main__":
    main()

# Example data
hands = ['Bust', '21', '20', '19', '18', '17', '16', '15', '14', '13', '12', '11', '10', '9', '8', '7', '6', '5', '4',
         '3', '2', '1']
new_strategy = ['Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Stand',
                'Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Draw']
default_strategy = ['Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Stand', 'Hit', 'Hit', 'Hit', 'Hit', 'Hit',
                    'Double Down', 'Double Down', 'Hit', 'Hit', 'Hit', 'Hit', 'Hit', 'Hit', 'Hit', 'Hit', 'Hit']

# Visualizing comparison
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(hands, new_strategy, label='New Strategy', marker='o')
ax.plot(hands, default_strategy, label='Default Strategy', marker='x')
ax.set_xlabel('Player Hand')
ax.set_ylabel('Strategy')
ax.set_title('Comparison of Blackjack Strategies')
ax.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
