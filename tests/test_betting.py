"""Feature 7: fractional-Kelly bet sizing.

Run:  .venv\\Scripts\\python -m unittest tests.test_betting -v
"""

import unittest

from lib.logic import betting

BASE = {
    "bankroll": 1000.0, "kelly_fraction": 0.5, "base_edge": -0.005,
    "edge_per_tc": 0.005, "variance": 1.33, "table_min": 10, "table_max": 5000,
}


def cfg(**over):
    return {**BASE, **over}


class KellySizing(unittest.TestCase):
    def test_edge_estimate(self):
        self.assertAlmostEqual(betting.estimate_edge(0, BASE), -0.005)
        self.assertAlmostEqual(betting.estimate_edge(1, BASE), 0.0)
        self.assertAlmostEqual(betting.estimate_edge(3, BASE), 0.01)

    def test_negative_edge_means_min_bet(self):
        s = betting.suggest(0, BASE)
        self.assertEqual(s["bet"], 10)
        self.assertFalse(s["sit_out"])  # off-the-top edge, not worse
        s = betting.suggest(-2, BASE)
        self.assertTrue(s["sit_out"])
        self.assertIn("sitting out", s["text"])

    def test_positive_edge_scales_with_kelly(self):
        # TC +3 -> edge 1%; half-Kelly on 1000 over var 1.33 = ~3.76 -> table min.
        s = betting.suggest(3, BASE)
        self.assertEqual(s["bet"], 10)
        # Bigger bankroll: 100k * 0.5 * 0.01 / 1.33 = ~376
        s = betting.suggest(3, cfg(bankroll=100_000))
        self.assertAlmostEqual(s["bet"], 376, delta=1)
        # Quarter Kelly halves it.
        s = betting.suggest(3, cfg(bankroll=100_000, kelly_fraction=0.25))
        self.assertAlmostEqual(s["bet"], 188, delta=1)

    def test_table_max_clamps(self):
        s = betting.suggest(10, cfg(bankroll=10_000_000))
        self.assertEqual(s["bet"], 5000)
        s = betting.suggest(10, cfg(bankroll=10_000_000, table_max=0))
        self.assertGreater(s["bet"], 5000)  # 0 = no max

    def test_bet_behind_hint(self):
        self.assertIn("+EV", betting.bet_behind_hint(3, BASE))
        self.assertIn("-EV", betting.bet_behind_hint(0, BASE))


class MultiSeatKelly(unittest.TestCase):
    """V3 E5: covariance-aware per-seat sizing for simultaneous seats."""

    CFG = dict(BASE, bankroll=100_000, covariance=0.479)

    def test_factor(self):
        self.assertEqual(betting.multi_seat_factor(1, self.CFG), 1.0)
        # Two hands sharing a dealer: v/(v+c) = 1.33/1.809 ~ 73.5% each.
        self.assertAlmostEqual(betting.multi_seat_factor(2, self.CFG),
                               1.33 / (1.33 + 0.479), places=12)
        self.assertAlmostEqual(betting.multi_seat_factor(3, self.CFG),
                               1.33 / (1.33 + 2 * 0.479), places=12)

    def test_per_seat_bet_shrinks(self):
        single = betting.suggest(3, self.CFG)
        double = betting.suggest(3, self.CFG, seats=2)
        self.assertAlmostEqual(
            double["bet"], single["bet"] * 1.33 / (1.33 + 0.479), delta=1)
        self.assertIn("2 seats", double["text"])
        self.assertNotIn("seats", single["text"])
        # Total action still grows: 2 x 73.5% ~ 1.47x of one hand.
        self.assertGreater(2 * double["bet"], single["bet"])

    def test_ramp_table_also_shrinks(self):
        cfg = dict(self.CFG, bet_table={"3": 200.0})
        single = betting.suggest(3.0, cfg)
        double = betting.suggest(3.0, cfg, seats=2)
        self.assertAlmostEqual(single["bet"], 200.0)
        self.assertAlmostEqual(double["bet"],
                               round(200.0 * 1.33 / (1.33 + 0.479), 2))

    def test_half_euro_ramp_values_survive(self):
        # The designer chip-rounds at 0.50 steps — installed values must
        # not be silently re-rounded to whole euros.
        cfg = dict(self.CFG, bet_table={"2": 12.5}, table_min=1)
        self.assertEqual(betting.suggest(2.0, cfg)["bet"], 12.5)

    def test_min_bet_branch_unaffected(self):
        self.assertEqual(betting.suggest(0, self.CFG, seats=3)["bet"],
                         betting.suggest(0, self.CFG)["bet"])

    def test_missing_covariance_key_is_factor_one(self):
        cfg = {k: v for k, v in self.CFG.items() if k != "covariance"}
        self.assertEqual(betting.multi_seat_factor(2, cfg), 1.0)


class RampTable(unittest.TestCase):
    """V3 Feature 5: an installed per-TC bet table overrides the formula."""

    TABLE = {"-1": 0.0, "0": 10.0, "2": 50.0, "4": 200.0}

    def test_follows_floored_tc_bucket(self):
        c = cfg(bet_table=self.TABLE)
        self.assertEqual(betting.suggest(2.7, c)["bet"], 50.0)
        self.assertEqual(betting.suggest(0.4, c)["bet"], 10.0)
        self.assertIn("ramp TC +2", betting.suggest(2.7, c)["text"])

    def test_clamps_out_of_range_counts_to_edge_buckets(self):
        c = cfg(bet_table=self.TABLE)
        self.assertEqual(betting.suggest(9.0, c)["bet"], 200.0)
        s = betting.suggest(-6.0, c)
        self.assertEqual(s["bet"], 0.0)

    def test_zero_means_sit_out(self):
        s = betting.suggest(-1.0, cfg(bet_table=self.TABLE))
        self.assertTrue(s["sit_out"])
        self.assertEqual(s["bet"], 0.0)
        self.assertIn("Sit out", s["text"])

    def test_table_limits_still_clamp(self):
        s = betting.suggest(4.0, cfg(bet_table=self.TABLE, table_max=100))
        self.assertEqual(s["bet"], 100.0)
        self.assertTrue(s["capped"])
        s = betting.suggest(0.0, cfg(bet_table=self.TABLE, table_min=25))
        self.assertEqual(s["bet"], 25.0)

    def test_junk_or_missing_table_uses_formula(self):
        for table in (None, {}, "junk", {"x": "y"}):
            s = betting.suggest(3, cfg(bet_table=table, bankroll=100_000))
            self.assertAlmostEqual(s["bet"], 376, delta=1)


class SettingsIntegration(unittest.TestCase):
    def test_betting_persists_via_settings(self):
        from lib.common import constants, settings
        before = settings.snapshot()
        try:
            settings.apply({"betting": {"bankroll": 2500, "kelly_fraction": 0.25,
                                        "junk": "x"}})
            self.assertEqual(constants.BETTING["bankroll"], 2500)
            self.assertEqual(constants.BETTING["kelly_fraction"], 0.25)
            self.assertNotIn("junk", constants.BETTING)
            self.assertEqual(settings.snapshot()["betting"]["bankroll"], 2500)
        finally:
            settings.apply(before)

    def test_bet_table_round_trips_and_filters_junk(self):
        from lib.common import constants, settings
        before = settings.snapshot()
        try:
            settings.apply({"betting": {"bet_table": {
                "0": 10, "3": 75.0, "bad": 5, "2": "junk", "99": 10}}})
            self.assertEqual(constants.BETTING["bet_table"],
                             {"0": 10.0, "3": 75.0})
            self.assertEqual(settings.snapshot()["betting"]["bet_table"],
                             {"0": 10.0, "3": 75.0})
            # A profile saved without the key must not wipe a live table.
            settings.apply({"betting": {"bankroll": 500}})
            self.assertEqual(constants.BETTING["bet_table"],
                             {"0": 10.0, "3": 75.0})
            # Non-dict junk clears it instead of crashing.
            settings.apply({"betting": {"bet_table": "garbage"}})
            self.assertEqual(constants.BETTING["bet_table"], {})
        finally:
            settings.apply(before)


if __name__ == "__main__":
    unittest.main()
