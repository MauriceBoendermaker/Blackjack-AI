"""Bankroll risk math: ruin probabilities, Kelly mapping, Monte Carlo
(V2 Feature 2). Formulas verified against primary sources — Schlesinger's
Blackjack Attack (via the Wizard of Odds session-RoR calculator and Eric
Farmer's derivation): see FEATURES_V2.md cheat-sheet.

Conventions: mu and sigma are the MEAN and SD of one round's result in the
same currency as the bankroll, for the ramp actually played.
"""

import math

from ..common import constants
from . import betting

# Rough floored-true-count frequency model for an 8-deck, ~50%-penetration
# shoe (more zero-concentrated than the published 6-deck tables; ~7% of
# rounds at TC >= +2). Used only when there aren't enough recorded rounds.
TC_FREQUENCIES = {
    -5: 0.01, -4: 0.02, -3: 0.05, -2: 0.10, -1: 0.18, 0: 0.36,
    1: 0.14, 2: 0.07, 3: 0.04, 4: 0.02, 5: 0.01,
}


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def lifetime_ror(mu, sigma, bankroll):
    """Schlesinger/BJA simple lifetime ruin: ((1-r)/(1+r))^(B/sigma), r=mu/sigma."""
    if mu <= 0:
        return 1.0
    if sigma <= 0 or bankroll <= 0:
        return 0.0 if bankroll > 0 else 1.0
    r = mu / sigma
    if r >= 1:
        return 0.0
    return ((1 - r) / (1 + r)) ** (bankroll / sigma)


def lifetime_ror_exp(mu, sigma, bankroll):
    """First-order (diffusion) form: exp(-2*EV*B/Var)."""
    if mu <= 0:
        return 1.0
    if sigma <= 0 or bankroll <= 0:
        return 0.0 if bankroll > 0 else 1.0
    return math.exp(max(-700.0, -2.0 * mu * bankroll / (sigma * sigma)))


def trip_ror(mu, sigma, bankroll, n_rounds):
    """BJA p.132 short-term (trip) ruin over n_rounds — Brownian first passage.
    Converges to the lifetime value as n_rounds grows."""
    if sigma <= 0 or n_rounds <= 0 or bankroll <= 0:
        return 0.0 if bankroll > 0 else 1.0
    sq = sigma * math.sqrt(n_rounds)
    a = _phi((-bankroll - mu * n_rounds) / sq)
    expo = math.exp(max(-700.0, min(700.0, -2.0 * mu * bankroll / (sigma * sigma))))
    b = expo * _phi((-bankroll + mu * n_rounds) / sq)
    return min(1.0, a + b)


def kelly_fixed_ror(fraction):
    """Ruin for a fixed (never resized) ramp at f x full Kelly: e^(-2/f).
    Full Kelly 13.53%, half 1.83%, quarter 0.034%."""
    if fraction <= 0:
        return 0.0
    return math.exp(-2.0 / fraction)


def bankroll_for_ror(target_ror, mu, sigma):
    """Bankroll required for a target lifetime RoR (exponential form inverse)."""
    if mu <= 0 or not (0.0 < target_ror < 1.0) or sigma <= 0:
        return None
    return -math.log(target_ror) * sigma * sigma / (2.0 * mu)


def di_score(mu, sigma):
    """Schlesinger's Desirability Index (1000*mu/sigma) and SCORE (= DI^2)."""
    if sigma <= 0:
        return 0.0, 0.0
    di = 1000.0 * mu / sigma
    return di, di * di


def certainty_equivalent(mu, sigma, bankroll):
    """Risk-adjusted value of one round for a full-Kelly (log-utility)
    bankroll: CE = mu - sigma^2 / (2B). At Kelly-optimal sizing CE = mu/2."""
    if bankroll <= 0:
        return 0.0
    return mu - sigma * sigma / (2.0 * bankroll)


# ------------------------------------------------------------ round stats

def empirical_round_stats(outcomes):
    """(mu, sigma) from recorded per-round EUR results; None if too few."""
    if len(outcomes) < 30:
        return None
    n = len(outcomes)
    mu = sum(outcomes) / n
    var = sum((x - mu) ** 2 for x in outcomes) / (n - 1)
    return mu, math.sqrt(var)


def model_round_stats(betting_cfg=None, tc_frequencies=None, seats=1):
    """(mu, sigma) from the TC-frequency model x the configured ramp.
    Play-all assumption; the variance term uses the per-hand blackjack
    variance and ignores the (small) between-TC spread of means.

    `seats` models k simultaneous seats (V3 E5): the per-seat bet shrinks
    by the covariance-aware Kelly factor and the round variance is
    k·v + k(k-1)·c per bet² — hands at one table share the dealer."""
    b = betting_cfg or constants.BETTING
    freqs = tc_frequencies or TC_FREQUENCIES
    k = max(1, int(seats))
    c = float(b.get("covariance") or 0.0)
    round_var = k * float(b["variance"]) + k * (k - 1) * c
    mu = 0.0
    var = 0.0
    for tc, f in freqs.items():
        edge = betting.estimate_edge(tc, b)
        bet = betting.suggest(tc, b, seats=k)["bet"]
        mu += f * k * edge * bet
        var += f * round_var * bet * bet
    return mu, math.sqrt(var)


def monte_carlo(outcomes, bankroll, n_rounds, trials=10_000, seed=12345):
    """Bootstrap-resample per-round EUR outcomes into `trials` futures of
    `n_rounds` each. Returns ruin probability, P(profit), and quantiles of
    the final result and the maximum drawdown."""
    import numpy as np
    outcomes = np.asarray(list(outcomes), dtype=np.float64)
    if outcomes.size == 0 or n_rounds <= 0 or trials <= 0:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(outcomes, size=(trials, n_rounds), replace=True)
    paths = np.cumsum(samples, axis=1)
    running_max = np.maximum.accumulate(np.maximum(paths, 0.0), axis=1)
    drawdown = (running_max - paths).max(axis=1)
    ruined = (paths <= -bankroll).any(axis=1)
    final = paths[:, -1]

    def q(arr, p):
        return float(np.quantile(arr, p))

    return {
        "trials": trials,
        "n_rounds": n_rounds,
        "ruin": float(ruined.mean()),
        "p_profit": float((final > 0).mean()),
        "final_p10": q(final, 0.10),
        "final_p50": q(final, 0.50),
        "final_p90": q(final, 0.90),
        "drawdown_p50": q(drawdown, 0.50),
        "drawdown_p90": q(drawdown, 0.90),
    }
