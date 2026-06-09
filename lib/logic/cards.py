"""Pure card and hand helpers. No Tkinter, no I/O — fully unit-testable."""

from ..common import constants


def rank_of(card_name: str) -> str:
    """'Ace of Spades' -> 'Ace'; 'King' -> 'King'; '10' -> '10'."""
    return card_name.split(" ")[0]


def card_value(card_name: str) -> int:
    """Blackjack value of a card name or bare rank. Ace counts as 11. '-' is 0."""
    if not card_name or card_name == "-":
        return 0
    rank = rank_of(card_name)
    try:
        return constants.VALUE_MAPPING[rank]
    except KeyError:
        raise ValueError(f"Invalid card: {card_name!r}")


def all_card_names():
    return [f"{rank} of {suit}" for suit in constants.CARD_SUITS for rank in constants.CARD_RANKS]


def hand_value(cards) -> int:
    """Best total: aces drop from 11 to 1 while the hand would bust."""
    cards = [c for c in cards if c and c != "-"]
    total = sum(card_value(c) for c in cards)
    aces = sum(1 for c in cards if rank_of(c) == "Ace")
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def is_soft(cards) -> bool:
    """True when an ace is currently counted as 11 (min total + 10 <= 21)."""
    cards = [c for c in cards if c and c != "-"]
    if not any(rank_of(c) == "Ace" for c in cards):
        return False
    min_total = sum(1 if rank_of(c) == "Ace" else card_value(c) for c in cards)
    return min_total + 10 <= 21


def is_pair(cards) -> bool:
    """Two cards of equal strategy rank (10/J/Q/K all count as 'ten' pairs)."""
    cards = [c for c in cards if c and c != "-"]
    if len(cards) != 2:
        return False
    return strategy_rank(cards[0]) == strategy_rank(cards[1])


def strategy_rank(card_name: str) -> str:
    """Rank as used by the strategy table: A, 2-9, or 10 (covers J/Q/K)."""
    rank = rank_of(card_name)
    if rank == "Ace":
        return "A"
    if rank in ("10", "Jack", "Queen", "King"):
        return "10"
    return rank


def hand_key(cards) -> str:
    """Strategy-table key for a hand: 'A,A', '8,8', 'A,7', or a hard total like '16'."""
    cards = [c for c in cards if c and c != "-"]
    if is_pair(cards):
        r = strategy_rank(cards[0])
        return f"{r},{r}"
    if is_soft(cards):
        non_ace = sum(card_value(c) for c in cards if rank_of(c) != "Ace")
        # Additional aces beyond the one counted as 11 contribute 1 each.
        extra_aces = sum(1 for c in cards if rank_of(c) == "Ace") - 1
        return f"A,{non_ace + extra_aces}"
    return str(hand_value(cards))


def describe_hand(cards) -> str:
    """Human label: 'Blackjack!', 'Pair of Aces', 'Soft 17', 'Hard 16', 'Bust (23)'."""
    cards = [c for c in cards if c and c != "-"]
    if not cards:
        return ""
    total = hand_value(cards)
    if total == 21 and len(cards) == 2:
        return "Blackjack!"
    if is_pair(cards):
        r = rank_of(cards[0])
        label = {"Jack": "Jacks", "Queen": "Queens", "King": "Kings", "Ace": "Aces"}.get(r, f"{r}s")
        return f"Pair of {label} ({total})"
    if total > 21:
        return f"Bust ({total})"
    if is_soft(cards):
        return f"Soft {total}"
    return f"Hard {total}" if len(cards) > 1 else f"{total}"


def dealer_strategy_rank(dealer_card: str) -> str | None:
    """Dealer up-card ('King', '10 of Hearts', 'A'...) -> strategy column A/2-10."""
    if not dealer_card:
        return None
    v = rank_of(str(dealer_card).strip())
    aliases = {"J": "Jack", "Q": "Queen", "K": "King", "A": "Ace"}
    v = aliases.get(v, v)
    if v.capitalize() in ("Jack", "Queen", "King"):
        return "10"
    if v.capitalize() == "Ace":
        return "A"
    if v in {"2", "3", "4", "5", "6", "7", "8", "9", "10"}:
        return v
    return None


def hilo_delta(card_name: str) -> int:
    """Hi-Lo count contribution: 2-6 -> +1, 7-9 -> 0, 10/J/Q/K/A -> -1."""
    v = card_value(card_name)
    if 2 <= v <= 6:
        return 1
    if v >= 10:  # 10, faces, and Ace (11)
        return -1
    return 0
