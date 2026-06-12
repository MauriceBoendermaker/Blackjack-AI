"""V3 Feature 5: bet-ramp designer / optimizer.

Run:  .venv\\Scripts\\python -m unittest tests.test_ramp_optimizer -v
"""

import math
import unittest

from lib.logic import bankroll, betting, ramp_optimizer

BASE = {
    "bankroll": 20_000.0, "kelly_fraction": 0.5, "base_edge": -0.005,
    "edge_per_tc": 0.005, "variance": 1.33, "table_min": 10, "table_max": 5000,
}


def cfg(**over):
    return {**BASE, **over}


class Frequencies(unittest.TestCase):
    def test_buckets_floor_clamp_and_normalize(self):
        freqs = ramp_optimizer.bucket_frequencies(
            [0.2, 0.9, 1.4, -7.3, 9.9], lo=-5, hi=5)
        self.assertAlmostEqual(sum(freqs.values()), 1.0)
        self.assertAlmostEqual(freqs[0], 2 / 5)   # 0.2 and 0.9 floor to 0
        self.assertAlmostEqual(freqs[1], 1 / 5)
        self.assertAlmostEqual(freqs[-5], 1 / 5)  # -7.3 clamps into the edge
        self.assertAlmostEqual(freqs[5], 1 / 5)   # 9.9 clamps into the edge
        self.assertIn(3, freqs)                   # empty buckets exist as 0.0
        self.assertEqual(freqs[3], 0.0)

    def test_empty_returns_none(self):
        self.assertIsNone(ramp_optimizer.bucket_frequencies([]))

    def test_frequency_source_falls_back_to_model(self):
        freqs, label = ramp_optimizer.frequency_source(None)
        self.assertEqual(freqs, bankroll.TC_FREQUENCIES)
        self.assertIn("model", label)


class Metrics(unittest.TestCase):
    def test_single_bucket_matches_direct_formula(self):
        b = cfg()
        freqs = {3: 1.0}
        ramp = {3: 100.0}
        m = ramp_optimizer.ramp_metrics(ramp, freqs, b, rounds_per_hour=60)
        edge = betting.estimate_edge(3, b)
        self.assertAlmostEqual(m["mu"], edge * 100.0)
        self.assertAlmostEqual(m["sigma"], math.sqrt(1.33) * 100.0)
        self.assertAlmostEqual(m["ev_hr"], edge * 100.0 * 60)
        self.assertAlmostEqual(
            m["ror"], bankroll.lifetime_ror_exp(m["mu"], m["sigma"], 20_000))
        self.assertAlmostEqual(
            m["ce"], bankroll.certainty_equivalent(m["mu"], m["sigma"], 20_000))
        self.assertAlmostEqual(m["n0"], (m["sigma"] / m["mu"]) ** 2)

    def test_sit_out_buckets_cost_nothing_but_count_in_play_share(self):
        b = cfg()
        freqs = {-1: 0.5, 3: 0.5}
        played = ramp_optimizer.ramp_metrics({-1: 0.0, 3: 100.0}, freqs, b)
        self.assertAlmostEqual(played["play_share"], 0.5)
        only = ramp_optimizer.ramp_metrics({3: 100.0}, {3: 0.5}, b)
        self.assertAlmostEqual(played["mu"], only["mu"])
        self.assertAlmostEqual(played["sigma"], only["sigma"])


class Optimize(unittest.TestCase):
    def test_deterministic(self):
        a = ramp_optimizer.optimize(cfg(), target_ror=0.05)
        b = ramp_optimizer.optimize(cfg(), target_ror=0.05)
        self.assertEqual(a["ramp"], b["ramp"])
        self.assertEqual(a["metrics"], b["metrics"])

    def test_meets_target_and_constraints(self):
        b = cfg()
        result = ramp_optimizer.optimize(b, target_ror=0.05, chip_step=5.0,
                                         max_spread=20.0)
        self.assertTrue(result["feasible"])
        m = result["metrics"]
        self.assertLessEqual(m["ror"], 0.05 + 1e-12)
        bets = [x for x in result["ramp"].values() if x > 0]
        self.assertLessEqual(max(bets), min(bets) * 20.0 + 1e-9)
        for bet in bets:
            self.assertGreaterEqual(bet, 10.0)
            self.assertLessEqual(bet, 5000.0)
            # Chip-rounded: every positive bet is a multiple of the step.
            self.assertAlmostEqual(bet % 5.0, 0.0)
        tcs = sorted(result["ramp"])
        for lo, hi in zip(tcs, tcs[1:]):
            self.assertLessEqual(result["ramp"][lo], result["ramp"][hi])

    def test_sit_out_zeroes_negative_edges(self):
        result = ramp_optimizer.optimize(cfg(), target_ror=0.05,
                                         sit_out_negative=True)
        for tc, bet in result["ramp"].items():
            if betting.estimate_edge(tc, BASE) <= 0:
                self.assertEqual(bet, 0.0)
        played = ramp_optimizer.optimize(cfg(), target_ror=0.05,
                                         sit_out_negative=False)
        for bet in played["ramp"].values():
            self.assertGreaterEqual(bet, 10.0)

    def test_beats_the_formula_ramp_it_replaces(self):
        # The designed ramp must dominate the live formula at equal-or-less
        # risk — otherwise the feature has no reason to exist.
        b = cfg()
        freqs = dict(bankroll.TC_FREQUENCIES)
        formula = ramp_optimizer.formula_ramp(b)
        f_m = ramp_optimizer.ramp_metrics(formula, freqs, b)
        result = ramp_optimizer.optimize(b, freqs,
                                         target_ror=max(f_m["ror"], 1e-6),
                                         sit_out_negative=False)
        self.assertTrue(result["feasible"])
        self.assertGreaterEqual(result["metrics"]["mu"], f_m["mu"] - 1e-12)

    def test_infeasible_flagged_with_lowest_ruin_candidate(self):
        # Tiny bankroll vs a big table minimum: nothing meets 0.1% ruin.
        b = cfg(bankroll=200.0, table_min=100)
        result = ramp_optimizer.optimize(b, target_ror=0.001,
                                         sit_out_negative=False)
        self.assertFalse(result["feasible"])
        self.assertGreater(result["metrics"]["ror"], 0.001)

    def test_chip_round(self):
        self.assertEqual(ramp_optimizer.chip_round(12.4, 5.0), 10.0)
        self.assertEqual(ramp_optimizer.chip_round(12.5, 5.0), 15.0)
        self.assertEqual(ramp_optimizer.chip_round(12.4, 0.0), 12.4)


class BetTable(unittest.TestCase):
    def test_round_trip(self):
        ramp = {-1: 0.0, 0: 10.0, 3: 150.0}
        table = ramp_optimizer.to_bet_table(ramp)
        self.assertEqual(set(table), {"-1", "0", "3"})
        self.assertEqual(ramp_optimizer.from_bet_table(table), ramp)

    def test_junk_filtered(self):
        self.assertEqual(ramp_optimizer.from_bet_table(None), {})
        self.assertEqual(ramp_optimizer.from_bet_table("x"), {})
        self.assertEqual(
            ramp_optimizer.from_bet_table({"2": "abc", "x": 5, "3": 20}),
            {3: 20.0})


class SynthOutcomes(unittest.TestCase):
    def test_seeded_and_plausible(self):
        b = cfg()
        freqs = {3: 1.0}
        ramp = {3: 100.0}
        a = ramp_optimizer.synth_outcomes(ramp, freqs, b, n=20_000, seed=9)
        c = ramp_optimizer.synth_outcomes(ramp, freqs, b, n=20_000, seed=9)
        self.assertEqual(a, c)
        mean = sum(a) / len(a)
        edge = betting.estimate_edge(3, b)
        # SE of the mean ~ sqrt(1.33)*100/sqrt(20k) ~ 0.8 EUR; 4 SE bound.
        self.assertAlmostEqual(mean, edge * 100.0, delta=3.3)


if __name__ == "__main__":
    unittest.main()
