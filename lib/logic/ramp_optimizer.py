"""Bet-ramp designer / optimizer (V3 Feature 5). Pure logic, no Tkinter.

Turns the live formula ramp (bankroll x kelly_fraction x edge / variance,
clamped) into a solved, personal INTEGER ramp: a per-true-count bet table
searched under the user's actual constraints — bankroll, target lifetime
risk of ruin, table min/max, chip step, max spread — weighted by the TC
frequency distribution measured from their own recorded rounds
(bankroll.TC_FREQUENCIES is the 8-deck model fallback).

Search strategy: the Kelly-optimal ramp shape is proportional to edge
(bet ~ bankroll * edge / variance), so the solution space is effectively
one-dimensional — a Kelly scale. The optimizer scans a dense geometric
grid of scales, rounds every candidate to the chip step and table limits
(monotone in TC by construction), keeps the best EV ramp whose closed-form
ruin meets the target, then hill-climbs single-bucket chip steps to repair
what integer rounding gave up. Candidates are priced with the same closed
forms bankroll.py uses everywhere else (lifetime_ror_exp & friends), so
the designer's numbers agree with the Bankroll & Risk window by
construction.

The winning table installs as constants.BETTING["bet_table"]
({str(floored_tc): bet_eur}); betting.suggest() follows it from the next
snapshot on, which also makes bankroll.model_round_stats — and through it
every RoR readout — reflect the installed ramp automatically.
"""

import math

from ..common import constants
from . import bankroll, betting

#: Bucket range for designed ramps; matches bankroll.TC_FREQUENCIES.
TC_MIN, TC_MAX = -5, 5

#: Kelly-scale scan grid (fractions of full Kelly, geometric, dense).
_SCAN_POINTS = 96
_SCAN_LO, _SCAN_HI = 0.02, 4.0

#: Hill-climb iteration cap — the space is tiny, this is a backstop.
_CLIMB_LIMIT = 200


# ------------------------------------------------------------ frequencies

def bucket_frequencies(tcs, lo=TC_MIN, hi=TC_MAX):
    """Normalized {floored_tc: frequency} from raw true counts, clamped to
    [lo, hi] (tail mass collapses into the edge buckets). None when empty."""
    if not tcs:
        return None
    counts = {tc: 0 for tc in range(lo, hi + 1)}
    for tc in tcs:
        counts[max(lo, min(hi, int(math.floor(tc))))] += 1
    n = len(tcs)
    return {tc: c / n for tc, c in counts.items()}


def frequency_source(store, session_only=False, min_rounds=100):
    """(freqs, label) — measured from the store when it holds enough rounds,
    else the standard 8-deck model. Tolerates store=None / broken DB."""
    tcs = []
    if store is not None:
        try:
            tcs = store.pre_deal_tcs(session_only=session_only)
        except Exception:
            tcs = []
    if len(tcs) >= min_rounds:
        return (bucket_frequencies(tcs),
                f"measured from {len(tcs)} recorded rounds")
    return (dict(bankroll.TC_FREQUENCIES),
            f"8-deck ~50%-pen model ({len(tcs)} rounds recorded — "
            f"needs {min_rounds}+ to switch to your data)")


# ----------------------------------------------------------------- ramps

def chip_round(value, chip_step):
    """Nearest multiple of the chip step (ties round up)."""
    if chip_step <= 0:
        return float(value)
    return math.floor(value / chip_step + 0.5) * chip_step


def formula_ramp(betting_cfg=None, lo=TC_MIN, hi=TC_MAX):
    """The live formula's per-TC bets — what the user plays today, for the
    side-by-side comparison. Honors an installed bet_table if one exists
    (the comparison then shows table-vs-table)."""
    b = betting_cfg or constants.BETTING
    return {tc: betting.suggest(tc, b)["bet"] for tc in range(lo, hi + 1)}


def ramp_metrics(ramp, freqs, betting_cfg=None, bankroll_eur=None,
                 rounds_per_hour=60.0):
    """Closed-form per-round and risk numbers for a candidate ramp.

    A bet of 0 means the round is not played (wong-out) — it contributes
    nothing to mean or variance but still consumes table time, so EV/hour
    stays honest about sitting out."""
    b = betting_cfg or constants.BETTING
    bank = float(bankroll_eur if bankroll_eur is not None else b["bankroll"])
    v = float(b["variance"])
    mu = 0.0
    var = 0.0
    for tc, f in freqs.items():
        bet = float(ramp.get(tc, 0.0))
        if bet <= 0:
            continue
        edge = betting.estimate_edge(tc, b)
        mu += f * edge * bet
        var += f * v * bet * bet
    sigma = math.sqrt(var)
    di, score = bankroll.di_score(mu, sigma)
    bets = [x for x in ramp.values() if x > 0]
    return {
        "mu": mu,
        "sigma": sigma,
        "ev_hr": mu * rounds_per_hour,
        "ror": bankroll.lifetime_ror_exp(mu, sigma, bank),
        "ror_classic": bankroll.lifetime_ror(mu, sigma, bank),
        "n0": (var / (mu * mu)) if mu > 0 else None,
        "di": di,
        "score": score,
        "ce": bankroll.certainty_equivalent(mu, sigma, bank),
        "min_bet": min(bets) if bets else 0.0,
        "max_bet": max(bets) if bets else 0.0,
        "play_share": sum(f for tc, f in freqs.items()
                          if ramp.get(tc, 0.0) > 0),
    }


def _build_ramp(scale, freqs, b, bank, chip_step, table_min, table_max,
                max_spread, sit_out_negative):
    """One candidate: full-Kelly bets x scale, chip-rounded, clamped,
    monotone non-decreasing in TC, spread-capped relative to the ramp's
    own SMALLEST placed bet (a wong-out ramp that never bets the table
    minimum may spread from its real floor — anchoring the cap to
    table_min would forfeit that EV)."""
    v = float(b["variance"])
    ramp = {}
    prev = 0.0
    for tc in sorted(freqs):
        edge = betting.estimate_edge(tc, b)
        if edge <= 0:
            bet = 0.0 if sit_out_negative else table_min
        else:
            kelly = bank * scale * edge / v
            bet = min(max(chip_round(kelly, chip_step), table_min), table_max)
        bet = max(bet, prev)  # rounding must never dip the ramp
        ramp[tc] = bet
        prev = bet
    positive = [x for x in ramp.values() if x > 0]
    if positive and max_spread:
        cap = min(positive) * max_spread
        if chip_step > 0:  # down to the chip grid: the cap is a hard limit
            cap = math.floor(cap / chip_step + 1e-9) * chip_step
        ramp = {tc: min(bet, cap) for tc, bet in ramp.items()}
    return ramp


def _spread_ok(ramp, max_spread):
    bets = [x for x in ramp.values() if x > 0]
    if not bets or not max_spread:
        return True
    return max(bets) <= min(bets) * max_spread + 1e-9


def optimize(betting_cfg=None, freqs=None, *, target_ror=0.05,
             chip_step=5.0, max_spread=20.0, sit_out_negative=True,
             rounds_per_hour=60.0, bankroll_eur=None):
    """Search ramp space for the best-EV integer ramp meeting the RoR target.

    Returns {"ramp", "metrics", "feasible", "scale", "evaluated"}; when no
    candidate meets the target (bankroll too small for the table minimum),
    feasible=False and the lowest-ruin candidate is returned instead so the
    UI can show WHY rather than nothing. Deterministic — no randomness."""
    b = betting_cfg or constants.BETTING
    bank = float(bankroll_eur if bankroll_eur is not None else b["bankroll"])
    table_min = max(1.0, float(b["table_min"]))
    table_max = float(b["table_max"]) if b["table_max"] else float("inf")
    if freqs is None:
        freqs = dict(bankroll.TC_FREQUENCIES)

    def evaluate(ramp):
        return ramp_metrics(ramp, freqs, b, bank, rounds_per_hour)

    best = None          # feasible with max mu
    fallback = None      # min ror, when nothing is feasible
    evaluated = 0
    ratio = (_SCAN_HI / _SCAN_LO) ** (1.0 / (_SCAN_POINTS - 1))
    seen = set()
    for i in range(_SCAN_POINTS):
        scale = _SCAN_LO * ratio ** i
        ramp = _build_ramp(scale, freqs, b, bank, chip_step, table_min,
                           table_max, max_spread, sit_out_negative)
        key = tuple(sorted(ramp.items()))
        if key in seen or not _spread_ok(ramp, max_spread):
            continue
        seen.add(key)
        m = evaluate(ramp)
        evaluated += 1
        if fallback is None or m["ror"] < fallback[1]["ror"]:
            fallback = (ramp, m, scale)
        if m["ror"] <= target_ror and (best is None or m["mu"] > best[1]["mu"]):
            best = (ramp, m, scale)

    feasible = best is not None
    ramp, metrics, scale = best if feasible else fallback

    # Hill-climb: single-bucket chip steps the proportional shape can't
    # reach (rounding gave some buckets away; the RoR slack may fit one
    # more chip on the most frequent bucket).
    if feasible:
        for _ in range(_CLIMB_LIMIT):
            improved = None
            for tc in sorted(ramp):
                for delta in (chip_step, -chip_step):
                    cand = dict(ramp)
                    bet = cand[tc] + delta
                    if bet <= 0:
                        bet = 0.0
                        if not sit_out_negative:
                            continue
                    if bet > 0 and (bet < table_min - 1e-9
                                    or bet > table_max + 1e-9):
                        continue  # spread is guarded by _spread_ok below
                    cand[tc] = bet
                    cells = sorted(cand)
                    if any(cand[a] > cand[c] + 1e-9 for a, c in
                           zip(cells, cells[1:])):
                        continue  # keep the ramp monotone
                    if not _spread_ok(cand, max_spread):
                        continue
                    m = evaluate(cand)
                    evaluated += 1
                    if m["ror"] <= target_ror and m["mu"] > metrics["mu"] + 1e-12:
                        if improved is None or m["mu"] > improved[1]["mu"]:
                            improved = (cand, m)
            if improved is None:
                break
            ramp, metrics = improved

    return {"ramp": ramp, "metrics": metrics, "feasible": feasible,
            "scale": scale, "evaluated": evaluated}


# ------------------------------------------------------------- validation

def synth_outcomes(ramp, freqs, betting_cfg=None, n=20_000, seed=12345):
    """Synthetic per-round EUR outcomes for the seeded Monte Carlo — a
    normal approximation per TC bucket mixed by frequency. Validation
    readout only; the optimizer itself is closed-form."""
    import numpy as np
    b = betting_cfg or constants.BETTING
    rng = np.random.default_rng(seed)
    tcs = sorted(freqs)
    probs = np.array([freqs[tc] for tc in tcs], dtype=np.float64)
    probs = probs / probs.sum()
    bets = np.array([float(ramp.get(tc, 0.0)) for tc in tcs])
    edges = np.array([betting.estimate_edge(tc, b) for tc in tcs])
    sd = math.sqrt(float(b["variance"]))
    picks = rng.choice(len(tcs), size=n, p=probs)
    bet = bets[picks]
    out = bet * edges[picks] + rng.standard_normal(n) * sd * bet
    return out.tolist()


# ------------------------------------------------------------ installing

def to_bet_table(ramp) -> dict:
    """JSON/settings shape for constants.BETTING['bet_table']."""
    return {str(int(tc)): float(bet) for tc, bet in sorted(ramp.items())}


def from_bet_table(table) -> dict:
    """{int_tc: bet} from a stored bet table; {} for junk."""
    out = {}
    if not isinstance(table, dict):
        return out
    for key, val in table.items():
        try:
            out[int(key)] = max(0.0, float(val))
        except (TypeError, ValueError):
            continue
    return out
