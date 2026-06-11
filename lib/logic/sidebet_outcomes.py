"""Side-bet OUTCOME evaluation from detected cards (pure logic, no I/O).

The EV engines in sidebets.py price a bet BEFORE the deal; this module
resolves what a placed bet actually DID, from what detection saw: the
seat's first two suited card names, the dealer up-card (rank-only or full
name) and the dealer's playout draws. Feeds the per-seat outcome badges
and the side-bet settlement into the session P&L.

Honesty rule (same as settlement.py): an outcome that cannot be decided
from the observed cards is reported as result "unknown" — never guessed.
The one systematic blind spot is the dealer up-card's SUIT when the
rank-only dealer model is active: the flush/suited tiers of the three-card
bets become undecidable exactly when the player's two cards share a suit.
Undecidable tier ambiguity is resolved by evaluating the classifier under
all four candidate dealer suits — when every suit agrees, the outcome is
decided anyway.
"""

from ..common import constants
from . import cards
from .settlement import dealer_final
from .sidebets import _BJ_VALUE, _POKER_IDX, _THREE_CARD_CLASSIFIERS

_RED = ("Hearts", "Diamonds")
_SUITS = ("Spades", "Hearts", "Diamonds", "Clubs")

_TIER_LABELS = {
    "perfect": "Perfect Pair", "colored": "Colored Pair", "mixed": "Mixed Pair",
    "suited_trips": "Suited Trips", "straight_flush": "Straight Flush",
    "trips": "Trips", "straight": "Straight", "flush": "Flush",
    "777": "7-7-7", "suited_21": "Suited 21", "21": "21", "20": "20", "19": "19",
    "suited_777": "Suited 7-7-7", "suited_678": "Suited 6-7-8", "678": "6-7-8",
    "qh_pair_dealer_bj": "Q♥ Pair + Dealer BJ", "qh_pair": "Q♥ Pair",
    "matched_20": "Matched 20", "suited_20": "Suited 20", "any_20": "Any 20",
}


def tier_label(tier):
    if isinstance(tier, int):  # Bust It tiers are the bust-hand length
        return f"Bust in {tier}{'+' if tier >= 8 else ''}"
    return _TIER_LABELS.get(tier, tier)


def _props(name):
    """('Queen', 'Hearts'|None) from a full name or bare rank."""
    name = str(name)
    if " of " in name:
        rank, _, suit = name.partition(" of ")
        return rank, suit
    return cards.rank_of(name), None


def _win(tier, paytable):
    return {"result": "win", "tier": tier, "pays": float(paytable[tier])}


_LOSS = {"result": "lose", "tier": None, "pays": None}


def _unknown(reason):
    return {"result": "unknown", "tier": None, "pays": None, "reason": reason}


def _perfect_pairs(c1, c2, paytable):
    (r1, s1), (r2, s2) = _props(c1), _props(c2)
    if r1 != r2:
        return dict(_LOSS)
    if s1 is None or s2 is None:
        return _unknown("player card suit unknown")
    if s1 == s2:
        return _win("perfect", paytable)
    if (s1 in _RED) == (s2 in _RED):
        return _win("colored", paytable)
    return _win("mixed", paytable)


def _three_card(bet_key, c1, c2, dealer_up, paytable):
    classify = _THREE_CARD_CLASSIFIERS[bet_key]
    (r1, s1), (r2, s2) = _props(c1), _props(c2)
    rd, sd = _props(dealer_up)
    if s1 is None or s2 is None:
        return _unknown("player card suit unknown")
    p1, p2, pd = _POKER_IDX[r1], _POKER_IDX[r2], _POKER_IDX[rd]
    v1, v2, vd = _BJ_VALUE[r1], _BJ_VALUE[r2], _BJ_VALUE[rd]
    # Unknown dealer suit: decide by consensus over all four candidates —
    # when the player's two cards are off-suit, no flush tier is possible
    # and every candidate agrees anyway.
    candidates = (sd,) if sd is not None else _SUITS
    tiers = {classify(p1, p2, pd, s1, s2, suit, v1, v2, vd, r1, r2, rd)
             for suit in candidates}
    if len(tiers) > 1:
        return _unknown("dealer up-card suit unknown")
    tier = tiers.pop()
    return _win(tier, paytable) if tier else dict(_LOSS)


def _bust_it(dealer_up, dealer_extras, paytable):
    if not dealer_up:
        return _unknown("no dealer up-card")
    total, _, completed = dealer_final(dealer_up, dealer_extras)
    if total > 21:
        return _win(min(1 + len(dealer_extras), 8), paytable)
    if completed:
        return dict(_LOSS)
    # Evolution's dealer always plays out — an incomplete hand here means
    # detection missed draws (or the round ended early). Refuse to guess.
    return _unknown("dealer hand incomplete")


def _lucky_ladies(c1, c2, dealer_up, dealer_extras, paytable):
    (r1, s1), (r2, s2) = _props(c1), _props(c2)
    if cards.hand_value([c1, c2]) != 20:
        return dict(_LOSS)
    if s1 is None or s2 is None:
        return _unknown("player card suit unknown")
    qh_pair = r1 == r2 == "Queen" and s1 == s2 == "Hearts"
    if qh_pair:
        # The top tier needs the dealer-blackjack verdict, which is decided
        # once the dealer has a second card (BJ = exactly 2 cards = 21).
        if not dealer_up or not dealer_extras:
            return _unknown("dealer hand incomplete")
        _, is_bj, _ = dealer_final(dealer_up, dealer_extras)
        return _win("qh_pair_dealer_bj" if is_bj else "qh_pair", paytable)
    if r1 == r2 and s1 == s2:
        return _win("matched_20", paytable)
    if s1 == s2:
        return _win("suited_20", paytable)
    return _win("any_20", paytable)


def seat_outcomes(first_two, dealer_up, dealer_extras, side_bets=None):
    """{bet_key: {result, tier, pays, label, ...}} for one seat's first two
    cards. Bets that can't resolve yet (no dealer up-card for the 3-card
    bets) are omitted; undecidable ones come back result='unknown'."""
    if side_bets is None:
        side_bets = constants.SIDE_BETS
    first_two = [c for c in (first_two or []) if c and c != "-"]
    if len(first_two) < 2:
        return {}
    c1, c2 = first_two[0], first_two[1]
    extras = list(dealer_extras or [])
    out = {}
    for key, cfg in side_bets.items():
        if not cfg.get("enabled"):
            continue
        paytable = cfg["paytable"]
        if key == "perfect_pairs":
            result = _perfect_pairs(c1, c2, paytable)
        elif key in _THREE_CARD_CLASSIFIERS:
            if not dealer_up:
                continue  # needs the up-card; not resolvable yet
            result = _three_card(key, c1, c2, dealer_up, paytable)
        elif key == "bust_it":
            result = _bust_it(dealer_up, extras, paytable)
        elif key == "lucky_ladies":
            result = _lucky_ladies(c1, c2, dealer_up, extras, paytable)
        else:
            continue
        result["key"] = key
        result["label"] = cfg.get("label", key)
        out[key] = result
    return out
