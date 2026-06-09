"""Exact side-bet EV engines (Feature 3).

Every common side bet's pre-deal EV is an exact combinatorial function of the
remaining-shoe composition:

  * Perfect Pairs           — 2-card enumeration over the 52-cell composition
  * 21+3 / Hot 3 / L.Lucky  — 3-card enumeration (player's two + dealer up)
  * Lucky Ladies            — 2-card enumeration x conditional dealer-BJ prob
  * Bust It                 — exact dealer-draw recursion over rank counts

The 52-cell composition may hold fractional expected counts (unknown-suit
dealer removals are spread across suits — see shoe.py); the enumerations treat
counts as expected multiplicities, which is exact for all cross-cell terms and
a negligible approximation for the same-cell pair terms.

Baselines verified against wizardofodds.com (8 decks, full shoe): Perfect
Pairs 25/12/6 -> -4.10%, Evolution 21+3 -> -3.7039%, Hot 3 -> -5.40%,
Evolution Bust It (S17, dealer always plays out) -> -6.1842%, Lucky Lucky
PT1 -> -2.63%, Lucky Ladies table A -> -24.05%. See tests/test_sidebets.py.
"""

from functools import lru_cache

from ..common import constants
from .shoe import RANKS, SUITS

# Poker rank index (Ace low *and* high for straights) and blackjack value.
_POKER_IDX = {r: i + 1 for i, r in enumerate(RANKS)}        # A=1 ... K=13
_BJ_VALUE = {r: constants.VALUE_MAPPING[r] for r in RANKS}  # Ace = 11

_CELLS = [(r, s) for r in RANKS for s in SUITS]


def _cell_arrays(comp52):
    """Parallel lists (count, poker_idx, bj_value, suit, rank) for cells > 0."""
    out = []
    for cell in _CELLS:
        n = comp52.get(cell, 0.0)
        if n > 0:
            r, s = cell
            out.append((n, _POKER_IDX[r], _BJ_VALUE[r], s, r))
    return out


def _best_total(values):
    """Best blackjack total for 2-3 cards; aces drop 11 -> 1 while busting."""
    total = sum(values)
    aces = sum(1 for v in values if v == 11)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def _is_straight(p1, p2, p3):
    a, b, c = sorted((p1, p2, p3))
    if (a, b, c) == (1, 12, 13):  # Q-K-A (ace high)
        return True
    return b == a + 1 and c == b + 1


# ----------------------------------------------------------- 2-card bets

def ev_perfect_pairs(comp52, paytable):
    """Player's first two cards: perfect (identical), colored, mixed pair."""
    red = ("Hearts", "Diamonds")
    n_total = sum(comp52.values())
    if n_total < 2:
        return None
    pair_div = n_total * (n_total - 1) / 2.0
    win = 0.0
    p_win = 0.0
    by_rank = {}
    for (r, s), n in comp52.items():
        if n > 0:
            by_rank.setdefault(r, []).append((s, n))
    for r, suited in by_rank.items():
        for i, (s1, n1) in enumerate(suited):
            same = n1 * (n1 - 1) / 2.0 / pair_div
            p_win += same
            win += same * paytable["perfect"]
            for s2, n2 in suited[i + 1:]:
                p = n1 * n2 / pair_div
                p_win += p
                colored = (s1 in red) == (s2 in red)
                win += p * paytable["colored" if colored else "mixed"]
    return win - (1.0 - p_win)


def ev_lucky_ladies(comp52, paytable):
    """Player's two cards total 20; Queen-of-Hearts tiers, with the QH-pair +
    dealer-blackjack tier resolved via the conditional dealer-BJ probability
    over the two dealer cards drawn from the remainder."""
    n_total = sum(comp52.values())
    if n_total < 4:
        return None
    pair_div = n_total * (n_total - 1) / 2.0
    cells = [(cell, n) for cell, n in comp52.items() if n > 0]
    ev = 0.0
    p_win = 0.0

    def dealer_bj_prob(removed_qh):
        n = n_total - 2
        aces = sum(c for (r, s), c in comp52.items() if r == "Ace")
        tens = sum(c for (r, s), c in comp52.items() if _BJ_VALUE[r] == 10) - removed_qh
        return 2.0 * aces * tens / (n * (n - 1))

    for i, ((r1, s1), n1) in enumerate(cells):
        for (r2, s2), n2 in cells[i:]:
            if (r1, s1) == (r2, s2):
                p = n1 * (n1 - 1) / 2.0 / pair_div
            else:
                p = n1 * n2 / pair_div
            if p <= 0 or _best_total((_BJ_VALUE[r1], _BJ_VALUE[r2])) != 20:
                continue
            qh_pair = r1 == r2 == "Queen" and s1 == s2 == "Hearts"
            if qh_pair:
                p_bj = dealer_bj_prob(removed_qh=2)
                ev += p * (p_bj * paytable["qh_pair_dealer_bj"]
                           + (1 - p_bj) * paytable["qh_pair"])
            elif r1 == r2 and s1 == s2:
                ev += p * paytable["matched_20"]
            elif s1 == s2:
                ev += p * paytable["suited_20"]
            else:
                ev += p * paytable["any_20"]
            p_win += p
    return ev - (1.0 - p_win)


# ----------------------------------------------------------- 3-card bets

def _classify_21p3(p1, p2, p3, s1, s2, s3, v1, v2, v3, r1, r2, r3):
    trips = p1 == p2 == p3
    flush = s1 == s2 == s3
    straight = _is_straight(p1, p2, p3)
    if trips and flush:
        return "suited_trips"
    if straight and flush:
        return "straight_flush"
    if trips:
        return "trips"
    if straight:
        return "straight"
    if flush:
        return "flush"
    return None


def _classify_hot3(p1, p2, p3, s1, s2, s3, v1, v2, v3, r1, r2, r3):
    if r1 == r2 == r3 == "7":
        return "777"
    total = _best_total((v1, v2, v3))
    if total == 21:
        return "suited_21" if s1 == s2 == s3 else "21"
    if total == 20:
        return "20"
    if total == 19:
        return "19"
    return None


def _classify_lucky_lucky(p1, p2, p3, s1, s2, s3, v1, v2, v3, r1, r2, r3):
    suited = s1 == s2 == s3
    ranks = sorted((r1, r2, r3))
    if ranks == ["7", "7", "7"]:
        return "suited_777" if suited else "777"
    if ranks == ["6", "7", "8"]:
        return "suited_678" if suited else "678"
    total = _best_total((v1, v2, v3))
    if total == 21:
        return "suited_21" if suited else "21"
    if total == 20:
        return "20"
    if total == 19:
        return "19"
    return None


_THREE_CARD_CLASSIFIERS = {
    "21+3": _classify_21p3,
    "hot3": _classify_hot3,
    "lucky_lucky": _classify_lucky_lucky,
}


def ev_three_card(comp52, bet_key, paytable):
    """Shared engine for 21+3 / Hot 3 / Lucky Lucky: enumerate the player's
    two cards (unordered) x the dealer up-card over the composition."""
    classify = _THREE_CARD_CLASSIFIERS[bet_key]
    cells = _cell_arrays(comp52)
    n_total = sum(c[0] for c in cells)
    if n_total < 3:
        return None
    pair_div = n_total * (n_total - 1) / 2.0
    third_div = n_total - 2.0
    ev = 0.0
    p_win = 0.0
    for i, (n1, p1, v1, s1, r1) in enumerate(cells):
        for j in range(i, len(cells)):
            n2, p2, v2, s2, r2 = cells[j]
            if i == j:
                p_pair = n1 * (n1 - 1) / 2.0 / pair_div
            else:
                p_pair = n1 * n2 / pair_div
            if p_pair <= 0:
                continue
            for n3, p3, v3, s3, r3 in cells:
                # subtract the player's two cards from the dealer draw pool
                m = n3
                if p3 == p1 and s3 == s1:
                    m -= 1
                if p3 == p2 and s3 == s2:
                    m -= 1
                if m <= 0:
                    continue
                cls = classify(p1, p2, p3, s1, s2, s3, v1, v2, v3, r1, r2, r3)
                if cls is None:
                    continue
                p = p_pair * m / third_div
                ev += p * paytable[cls]
                p_win += p
    return ev - (1.0 - p_win)


# ------------------------------------------------------------- Bust It

@lru_cache(maxsize=200_000)
def _bust_len_dist(comp, total, soft, ncards, s17):
    """P(dealer busts with exactly k cards) for k=3..8 (8 = 8+), plus
    'no bust', continuing from (total, soft, ncards). comp is the 10-bucket
    tuple from ev_engine (index 0 = Ace ... 9 = ten)."""
    if total > 21:
        out = [0.0] * 7  # k=3..8+, no-bust
        out[min(ncards, 8) - 3] = 1.0
        return tuple(out)
    if total >= 18 or (total == 17 and (s17 or not soft)):
        return (0.0,) * 6 + (1.0,)
    n = sum(comp)
    if n == 0:
        return (0.0,) * 6 + (1.0,)
    acc = [0.0] * 7
    for idx in range(10):
        c = comp[idx]
        if not c:
            continue
        if idx == 0 and total + 11 <= 21:
            t2, sft2 = total + 11, True
        else:
            t2 = total + (1 if idx == 0 else idx + 1)
            sft2 = soft
            if t2 > 21 and sft2:
                t2, sft2 = t2 - 10, False
        sub = _bust_len_dist(comp[:idx] + (comp[idx] - 1,) + comp[idx + 1:],
                             t2, sft2, ncards + 1, s17)
        p = c / n
        for k in range(7):
            acc[k] += p * sub[k]
    return tuple(acc)


def ev_bust_it(comp10, paytable, s17=None):
    """Evolution Bust It: dealer busts -> pays by bust-hand length. The dealer
    always plays out the hand (Infinite Blackjack rule), so no BJ/peek terms.
    comp10 = the ev_engine 10-bucket composition (pre-deal, dealer cards
    still in the shoe)."""
    if s17 is None:
        s17 = constants.RULES["s17"]
    if sum(comp10) < 10:
        return None
    dist = _bust_len_dist(tuple(comp10), 0, False, 0, bool(s17))
    ev = sum(dist[k - 3] * paytable[k] for k in range(3, 9))
    return ev - dist[6]


# ------------------------------------------------------------- top level

def evaluate_all(comp52, comp10, side_bets=None):
    """EV per enabled side bet. Returns [{key, label, ev}] (ev may be None
    when the shoe is too depleted to evaluate)."""
    if side_bets is None:
        side_bets = constants.SIDE_BETS
    out = []
    for key, cfg in side_bets.items():
        if not cfg.get("enabled"):
            continue
        paytable = cfg["paytable"]
        if key == "perfect_pairs":
            ev = ev_perfect_pairs(comp52, paytable)
        elif key in _THREE_CARD_CLASSIFIERS:
            ev = ev_three_card(comp52, key, paytable)
        elif key == "bust_it":
            ev = ev_bust_it(comp10, paytable)
        elif key == "lucky_ladies":
            ev = ev_lucky_ladies(comp52, paytable)
        else:
            continue
        out.append({"key": key, "label": cfg.get("label", key), "ev": ev})
    return out
