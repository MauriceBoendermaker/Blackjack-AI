from ..common import constants


class DecisionMaking:
    def __init__(self):
        pass

    def bet_strategy(self, true_count, base_bet):
        if true_count > 4:
            return f"Betting 2x base bet (€{2 * base_bet})"
        elif true_count >= 2:
            return f"Betting 1.5x base bet (€{1.5 * base_bet})"
        elif true_count < 0:
            return "Betting half base bet (Consider not playing)"
        else:
            return f"Betting base bet (€{base_bet})"
