"""Risk-aware bet sizing: fractional Kelly from the true count (Feature 7).

Bet variation supplies ~70-80% of a counter's total edge, so the ramp matters
more than any single playing decision. The standard estimate for Hi-Lo shoe
games puts the player edge at roughly (base table edge + ~0.5% per true-count
point); the Kelly-optimal wager for a roughly even-money bet is
bankroll * edge / variance, and betting a FRACTION of Kelly (half by default)
gives up little growth for far less ruin risk. Blackjack hand variance is
~1.3 units^2 (doubles/splits/naturals included).

All knobs live in constants.BETTING and persist with the table profile.

When the ramp designer (lib/logic/ramp_optimizer.py) has installed a per-TC
bet table (BETTING["bet_table"], {str(floored_tc): bet_eur}), suggest()
follows the table instead of the formula — a 0 entry means sit out.
"""

import math

from ..common import constants


def estimate_edge(true_count, betting=None) -> float:
    b = betting or constants.BETTING
    return b["base_edge"] + b["edge_per_tc"] * true_count


def ramp_bet(true_count, betting=None):
    """(floored_tc, bet) from the installed per-TC bet table, clamping
    out-of-range counts to the edge buckets; None when no table is set
    (the formula path applies) or the table is unreadable."""
    b = betting or constants.BETTING
    table = b.get("bet_table")
    if not isinstance(table, dict) or not table:
        return None
    buckets = {}
    for key, val in table.items():
        try:
            buckets[int(key)] = max(0.0, float(val))
        except (TypeError, ValueError):
            continue
    if not buckets:
        return None
    tc = int(math.floor(true_count))
    tc = max(min(buckets), min(max(buckets), tc))
    return tc, buckets[tc]


def multi_seat_factor(seats, betting=None) -> float:
    """Per-seat Kelly shrink for k simultaneous seats (V3 E5).

    Hands at one table share the dealer, so their outcomes are positively
    correlated (covariance ~0.479 units^2, Wizard of Odds) — k seats at
    full single-hand size over-bet the bankroll. The k-hand optimum per
    seat is v / (v + (k-1)c) of the single-hand bet: ~73.5% each at two
    seats (~1.47x total action), ~58% at three."""
    b = betting or constants.BETTING
    k = max(1, int(seats))
    if k == 1:
        return 1.0
    v = float(b["variance"])
    c = float(b.get("covariance") or 0.0)
    return v / (v + (k - 1) * c)


def suggest(true_count, betting=None, exact_edge=None, seats=1) -> dict:
    """{"edge", "bet", "sit_out", "text", "capped"} for the current count.
    When the exact pre-deal EV is available (V2 Feature 3) it replaces the
    linear true-count estimate — the text says which one it used. "capped"
    is True iff the Kelly wager was clamped DOWN by the table max.
    An installed bet_table (ramp designer) overrides the formula: the bet
    comes from the floored-TC bucket, 0 = sit out (bet 0.0).
    `seats` is the number of simultaneous seats the bet rides on — the
    PER-SEAT wager shrinks by multi_seat_factor (covariance-aware Kelly,
    V3 E5); applies to the formula and the installed ramp alike."""
    b = betting or constants.BETTING
    exact = exact_edge is not None
    edge = exact_edge if exact else estimate_edge(true_count, b)
    tag = "exact" if exact else "TC est."
    table_min = max(1.0, float(b["table_min"]))
    table_max = float(b["table_max"]) if b["table_max"] else float("inf")
    k = max(1, int(seats))
    factor = multi_seat_factor(k, b)
    seats_note = f", {k} seats" if k > 1 else ""

    ramp = ramp_bet(true_count, b)
    if ramp is not None:
        tc_bucket, table_bet = ramp
        if table_bet <= 0:
            return {"edge": edge, "bet": 0.0, "sit_out": True,
                    "text": (f"Sit out (ramp TC {tc_bucket:+d}) — "
                             f"edge {edge:+.2%} ({tag})"),
                    "capped": False}
        table_bet *= factor  # the designed table assumed one seat
        # Two decimals, not whole euros: the designer chip-rounds its
        # table (0.50 steps are legal) and the installed values must
        # survive verbatim at one seat.
        bet = round(min(max(table_bet, table_min), table_max), 2)
        return {"edge": edge, "bet": float(bet), "sit_out": False,
                "text": (f"Bet €{bet:g} (ramp TC {tc_bucket:+d}"
                         f"{seats_note}; edge {edge:+.2%} {tag})"),
                "capped": table_bet > table_max}

    if edge <= 0:
        sit_out = edge < b["base_edge"]  # worse than off-the-top: count is negative
        text = f"Min bet (€{table_min:g}) — edge {edge:+.2%} ({tag})"
        if sit_out:
            text += ", consider sitting out"
        return {"edge": edge, "bet": table_min, "sit_out": sit_out, "text": text,
                "capped": False}

    kelly = b["bankroll"] * b["kelly_fraction"] * edge / b["variance"] * factor
    capped = kelly > table_max  # Kelly wanted more than the table allows
    bet = round(min(max(kelly, table_min), table_max))
    frac = {1.0: "full", 0.5: "1/2", 0.25: "1/4"}.get(b["kelly_fraction"],
                                                      f"{b['kelly_fraction']:g}x")
    text = f"Bet €{bet:g} (edge {edge:+.2%} {tag}, {frac} Kelly{seats_note})"
    if bet >= table_max < float("inf"):
        text += " — table max"
    return {"edge": edge, "bet": float(bet), "sit_out": False, "text": text,
            "capped": capped}


def side_bet_stake(ev, variance, betting=None) -> float:
    """Fractional-Kelly stake (EUR) for one side bet; 0 when not +EV.

    Independent-Kelly approximation: ignores the correlation with the
    simultaneous main bet and between side bets sharing the same cards
    (21+3 and Perfect Pairs both ride the player's first two) — fine at
    these stake sizes, where the huge paytable variance (10^2-10^3 units^2)
    keeps the Kelly fraction tiny by construction."""
    b = betting or constants.BETTING
    if not ev or not variance or ev <= 0 or variance <= 0:
        return 0.0
    return round(b["bankroll"] * b["kelly_fraction"] * ev / variance, 2)


def bet_behind_hint(true_count, betting=None) -> str:
    """Bet Behind is the main game by proxy: same edge, someone else's plays."""
    edge = estimate_edge(true_count, betting)
    if edge > 0:
        return (f"Bet behind is +EV ({edge:+.2%}) — pick seats that follow "
                "basic strategy")
    return f"Bet behind is -EV right now ({edge:+.2%})"
