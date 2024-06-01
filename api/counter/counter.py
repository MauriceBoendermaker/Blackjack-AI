def update_count(card, count):
    """Update the count based on the card value."""
    if card in ['2', '3', '4', '5', '6']:
        return count + 1
    elif card in ['10', 'J', 'Q', 'K', 'A']:
        return count - 1
    return count


def calculate_true_count(running_count, decks_remaining):
    """Calculate the true count."""
    if decks_remaining > 0:
        return running_count / decks_remaining
    return 0


def get_advice(true_count):
    """Provide betting advice based on the true count."""
    if true_count > 2:
        return "High chance of a high card. Consider standing on a lower total."
    elif true_count < -1:
        return "Low chance of a high card. Consider hitting on a higher total."
    else:
        return "Count is neutral. Follow basic blackjack strategy."


def main():
    decks = int(input("Enter the number of decks being used: "))
    count = 0
    cards_seen = 0

    while True:
        card = input("Enter card seen (or 'quit' to exit): ").upper()
        if card == 'QUIT':
            break
        count = update_count(card, count)
        cards_seen += 1
        decks_remaining = decks - (cards_seen / 52)
        true_count = calculate_true_count(count, decks_remaining)
        advice = get_advice(true_count)
        print(f"Current count: {count}, True count: {true_count:.2f}, Advice: {advice}")


if __name__ == "__main__":
    main()
