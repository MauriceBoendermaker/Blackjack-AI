"""Risk-aware bet sizing: fractional Kelly from the true count (Feature 7).

Bet variation supplies ~70-80% of a counter's total edge, so the ramp matters
more than any single playing decision. The standard estimate for Hi-Lo shoe
games puts the player edge at roughly (base table edge + ~0.5% per true-count
point); the Kelly-optimal wager for a roughly even-money bet is
bankroll * edge / variance, and betting a FRACTION of Kelly (half by default)
gives up little growth for far less ruin risk. Blackjack hand variance is
~1.3 units^2 (doubles/splits/naturals included).

All knobs live in constants.BETTING and persist with the table profile.
"""

from ..common import constants


def estimate_edge(true_count, betting=None) -> float:
    b = betting or constants.BETTING
    return b["base_edge"] + b["edge_per_tc"] * true_count


def suggest(true_count, betting=None) -> dict:
    """{"edge", "bet", "sit_out", "text"} for the current count."""
    b = betting or constants.BETTING
    edge = estimate_edge(true_count, b)
    table_min = max(1.0, float(b["table_min"]))
    table_max = float(b["table_max"]) if b["table_max"] else float("inf")

    if edge <= 0:
        sit_out = edge < b["base_edge"]  # worse than off-the-top: count is negative
        text = f"Min bet (€{table_min:g}) — edge {edge:+.2%}"
        if sit_out:
            text += ", consider sitting out"
        return {"edge": edge, "bet": table_min, "sit_out": sit_out, "text": text}

    kelly = b["bankroll"] * b["kelly_fraction"] * edge / b["variance"]
    bet = round(min(max(kelly, table_min), table_max))
    frac = {1.0: "full", 0.5: "1/2", 0.25: "1/4"}.get(b["kelly_fraction"],
                                                      f"{b['kelly_fraction']:g}x")
    text = f"Bet €{bet:g} (edge {edge:+.2%}, {frac} Kelly)"
    if bet >= table_max < float("inf"):
        text += " — table max"
    return {"edge": edge, "bet": float(bet), "sit_out": False, "text": text}


def bet_behind_hint(true_count, betting=None) -> str:
    """Bet Behind is the main game by proxy: same edge, someone else's plays."""
    edge = estimate_edge(true_count, betting)
    if edge > 0:
        return (f"Bet behind is +EV ({edge:+.2%}) — pick seats that follow "
                "basic strategy")
    return f"Bet behind is -EV right now ({edge:+.2%})"
