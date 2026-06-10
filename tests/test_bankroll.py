"""V2 Feature 2: risk-of-ruin math and Monte Carlo.

Run:  .venv\\Scripts\\python -m unittest tests.test_bankroll -v
"""

import math
import unittest

from lib.logic import bankroll as br


class RuinFormulas(unittest.TestCase):
    MU, SIGMA = 0.5, 25.0  # EUR per round — a realistic counter's ramp

    def test_kelly_fixed_ror_landmarks(self):
        self.assertAlmostEqual(br.kelly_fixed_ror(1.0), math.exp(-2), places=12)
        self.assertAlmostEqual(br.kelly_fixed_ror(1.0), 0.13534, places=5)
        self.assertAlmostEqual(br.kelly_fixed_ror(0.5), 0.01832, places=5)
        self.assertAlmostEqual(br.kelly_fixed_ror(0.25), 0.000335, places=6)

    def test_lifetime_forms_agree_for_small_edge(self):
        for bank in (500, 2000, 10_000):
            simple = br.lifetime_ror(self.MU, self.SIGMA, bank)
            expo = br.lifetime_ror_exp(self.MU, self.SIGMA, bank)
            self.assertAlmostEqual(simple, expo, delta=0.01)

    def test_no_edge_means_certain_ruin(self):
        self.assertEqual(br.lifetime_ror(0.0, self.SIGMA, 1000), 1.0)
        self.assertEqual(br.lifetime_ror(-0.5, self.SIGMA, 1000), 1.0)
        self.assertEqual(br.lifetime_ror_exp(-0.5, self.SIGMA, 1000), 1.0)

    def test_bigger_bankroll_means_less_ruin(self):
        small = br.lifetime_ror(self.MU, self.SIGMA, 500)
        big = br.lifetime_ror(self.MU, self.SIGMA, 5000)
        self.assertGreater(small, big)

    def test_trip_converges_to_lifetime(self):
        life = br.lifetime_ror_exp(self.MU, self.SIGMA, 2000)
        trip_short = br.trip_ror(self.MU, self.SIGMA, 2000, 100)
        trip_long = br.trip_ror(self.MU, self.SIGMA, 2000, 2_000_000)
        self.assertLess(trip_short, life)          # less time, less ruin
        self.assertAlmostEqual(trip_long, life, places=4)

    def test_bankroll_for_ror_inverts(self):
        need = br.bankroll_for_ror(0.05, self.MU, self.SIGMA)
        self.assertAlmostEqual(br.lifetime_ror_exp(self.MU, self.SIGMA, need),
                               0.05, places=10)
        self.assertIsNone(br.bankroll_for_ror(0.05, -1.0, self.SIGMA))

    def test_di_score_and_ce(self):
        di, score = br.di_score(self.MU, self.SIGMA)
        self.assertAlmostEqual(di, 20.0)
        self.assertAlmostEqual(score, 400.0)
        # CE at Kelly-optimal bankroll (B = sigma^2/mu... here just sanity):
        ce = br.certainty_equivalent(self.MU, self.SIGMA, 10_000)
        self.assertLess(ce, self.MU)
        self.assertGreater(ce, 0)


class RoundStats(unittest.TestCase):
    def test_empirical_needs_enough_rounds(self):
        self.assertIsNone(br.empirical_round_stats([1.0] * 29))
        mu, sigma = br.empirical_round_stats([10.0, -10.0] * 20)
        self.assertAlmostEqual(mu, 0.0)
        self.assertAlmostEqual(sigma, 10.0, delta=0.2)

    def test_model_stats_finite(self):
        mu, sigma = br.model_round_stats()
        self.assertTrue(math.isfinite(mu))
        self.assertGreater(sigma, 0)
        self.assertAlmostEqual(sum(br.TC_FREQUENCIES.values()), 1.0, places=9)


class MonteCarlo(unittest.TestCase):
    def test_deterministic_and_sane(self):
        outcomes = [10.0, -10.0, -10.0, 12.0, -8.0, 15.0]  # slight +EV
        a = br.monte_carlo(outcomes, bankroll=200, n_rounds=500, trials=2000)
        b = br.monte_carlo(outcomes, bankroll=200, n_rounds=500, trials=2000)
        self.assertEqual(a, b)  # seeded -> reproducible
        rich = br.monte_carlo(outcomes, bankroll=100_000, n_rounds=500, trials=2000)
        self.assertEqual(rich["ruin"], 0.0)
        poor = br.monte_carlo(outcomes, bankroll=20, n_rounds=500, trials=2000)
        self.assertGreater(poor["ruin"], a["ruin"])
        self.assertGreaterEqual(a["drawdown_p90"], a["drawdown_p50"])
        self.assertGreaterEqual(a["final_p90"], a["final_p50"])

    def test_empty_input(self):
        self.assertIsNone(br.monte_carlo([], 1000, 100))


if __name__ == "__main__":
    unittest.main()
