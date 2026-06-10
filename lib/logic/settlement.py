"""Round settlement: dealer final total vs every hand (V2 Feature 1).

Pure logic, no Tkinter, no I/O. The engine calls settle_round() at round end
with what detection saw; outcomes feed the session P&L, the bankroll
auto-update, and the rounds table.

Honesty rules baked in:
  * A round only settles when the dealer hand is COMPLETE — total >= 17, or
    every tracked hand busted (dealer doesn't draw then). Anything else
    (stream cut off, cards missed) yields outcome None for the round.
  * Bet sizing per hand is unknown to detection (a 3-card hand could be a
    double or a hit), so units are 1 per hand; blackjack pays rules.bj_pays.
    Euro P&L = units x the user's "bet placed" amount.
"""

from ..common import constants
from . import cards


def dealer_final(up_rank, extras):
    """(total, is_blackjack, completed) for the dealer's finished hand.
    `extras` are the playout ranks detected after the up-card locked."""
    if not up_rank:
        return 0, False, False
    hand = [up_rank] + list(extras)
    total = cards.hand_value(hand)
    is_bj = len(hand) == 2 and total == 21
    completed = total >= 17
    return total, is_bj, completed


def settle_hand(hand_names, dealer_total, dealer_bj, natural_allowed=True,
                bj_pays=None):
    """(outcome, units) for one hand against a completed dealer hand.

    outcome: 'blackjack' | 'win' | 'push' | 'lose'; units are per 1 unit bet.
    """
    if bj_pays is None:
        bj_pays = constants.RULES["bj_pays"]
    hand = [c for c in hand_names if c and c != "-"]
    if len(hand) < 2:
        return None, 0.0
    total = cards.hand_value(hand)
    natural = natural_allowed and len(hand) == 2 and total == 21

    if total > 21:
        return "lose", -1.0  # player bust loses even to a dealer bust/BJ
    if dealer_bj:
        return ("push", 0.0) if natural else ("lose", -1.0)
    if natural:
        return "blackjack", float(bj_pays)
    if dealer_total > 21:
        return "win", 1.0
    if total > dealer_total:
        return "win", 1.0
    if total < dealer_total:
        return "lose", -1.0
    return "push", 0.0


def settle_round(seats, dealer_rank, dealer_extras, bj_pays=None):
    """Settle every seat of a finished round.

    `seats` is a list of dicts with keys: index, cards (names), split (bool),
    hand_of (per-card hand tags). Returns None when the dealer hand is
    incomplete, else {"dealer_total", "dealer_bj", "seats": [...]} where each
    seat entry carries per-hand outcomes and the seat's net units."""
    total, is_bj, completed = dealer_final(dealer_rank, dealer_extras)
    live_hands = []
    for seat in seats:
        names = [c for c in seat["cards"] if c and c != "-"]
        if len(names) < 2:
            continue
        if seat.get("split"):
            hand_of = seat.get("hand_of") or [0] * len(names)
            for h in (0, 1):
                hand = [n for n, tag in zip(names, hand_of) if tag == h]
                if len(hand) >= 2:
                    live_hands.append((seat["index"], hand, False))
        else:
            live_hands.append((seat["index"], names, True))
    if not live_hands:
        return None

    all_bust = all(cards.hand_value(h) > 21 for _, h, _ in live_hands)
    if not completed and not all_bust:
        return None  # dealer hand incomplete — refuse to guess

    out = {"dealer_total": total, "dealer_bj": is_bj, "seats": []}
    by_seat = {}
    for idx, hand, natural_ok in live_hands:
        outcome, units = settle_hand(hand, total, is_bj, natural_ok, bj_pays)
        entry = by_seat.setdefault(idx, {"index": idx, "hands": [], "net_units": 0.0})
        entry["hands"].append({"cards": hand, "outcome": outcome, "units": units})
        entry["net_units"] += units
    out["seats"] = [by_seat[k] for k in sorted(by_seat)]
    return out
