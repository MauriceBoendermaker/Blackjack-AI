"""Hi-Lo index-play deviations: the Illustrious 18 and Fab 4 (Don Schlesinger,
*Blackjack Attack*), multi-deck indices.

These are the classic human-playable true-count deviations from basic
strategy. The exact-EV engine (ev_engine.py) subsumes them — it sees the real
composition instead of a one-number count — so this layer serves as the
human-readable explanation ("the count says...") next to the exact advice,
and as a cross-check in tests: near each index threshold with a neutral
composition, the EV engine must flip to the same action.

Index values verified against wizardofodds.com/games/blackjack/card-counting/high-low/
and the Blackjack Apprenticeship S17/H17 chart PDFs. Insurance (the #1 index,
TC >= +3) is not in these tables — ev_engine.insurance_advice() makes that
call exactly from the tens fraction.

Convention: play `above` when true count >= index, else `below`. For the
negative-index entries basic strategy is Stand — keep standing at or above
the index and hit only when the count falls below it.
"""

from ..common import constants

# (hand_key, dealer) -> (action at/above index, index, action below index)
# hand_key follows cards.hand_key(): hard totals "9".."16", pairs "10,10".
I18_S17 = {
    ("16", "10"): ("S", 0, "H"),
    ("15", "10"): ("S", 4, "H"),
    ("10,10", "5"): ("P", 5, "S"),
    ("10,10", "6"): ("P", 4, "S"),
    ("10", "10"): ("D", 4, "H"),
    ("12", "3"): ("S", 2, "H"),
    ("12", "2"): ("S", 3, "H"),
    ("11", "A"): ("D", 1, "H"),
    ("9", "2"): ("D", 1, "H"),
    ("10", "A"): ("D", 4, "H"),
    ("9", "7"): ("D", 3, "H"),
    ("16", "9"): ("S", 5, "H"),
    ("13", "2"): ("S", -1, "H"),
    ("12", "4"): ("S", 0, "H"),
    ("12", "5"): ("S", -2, "H"),
    ("12", "6"): ("S", -1, "H"),
    ("13", "3"): ("S", -2, "H"),
}

# Verified H17 differences: 11 vs A is a basic-strategy double (no index),
# 10 vs A drops to +3, and H17 adds the stand deviations 16 vs A and 15 vs A.
I18_H17 = {k: v for k, v in I18_S17.items() if k != ("11", "A")}
I18_H17[("10", "A")] = ("D", 3, "H")
I18_H17[("16", "A")] = ("S", 3, "H")
I18_H17[("15", "A")] = ("S", 5, "H")

# Fab 4 late-surrender indices: surrender when TC >= index. Only consulted
# when the table offers surrender (standard Evolution tables do not).
FAB4_S17 = {
    ("14", "10"): 3,
    ("15", "10"): 0,
    ("15", "9"): 2,
    ("15", "A"): 1,
}
FAB4_H17 = {**FAB4_S17, ("15", "A"): -1}

INSURANCE_INDEX = 3  # take insurance at TC >= +3 (annotation only; see ev_engine)


def index_advice(hand_key, dealer_rank, true_count,
                 s17=None, surrender=None, two_cards=True):
    """Look up the index play for a hand. Returns None when no index applies,
    else {"action", "index", "triggered", "source"}.

    Fab 4 surrender takes precedence over the I18 entry for the same hand
    (e.g. 15 vs 10) when surrender is offered and the hand is two cards.
    Double/split/surrender deviations only apply to two-card hands.
    """
    if s17 is None:
        s17 = constants.RULES["s17"]
    if surrender is None:
        surrender = constants.RULES["surrender"]
    key = (str(hand_key).upper(), str(dealer_rank).upper())

    if surrender and two_cards:
        fab = (FAB4_S17 if s17 else FAB4_H17).get(key)
        if fab is not None:
            triggered = true_count >= fab
            return {"action": "R" if triggered else "H", "index": fab,
                    "triggered": triggered, "source": "Fab4"}

    entry = (I18_S17 if s17 else I18_H17).get(key)
    if entry is None:
        return None
    above, index, below = entry
    if above in ("D", "P") and not two_cards:
        return None
    triggered = true_count >= index
    return {"action": above if triggered else below, "index": index,
            "triggered": triggered, "source": "I18"}
