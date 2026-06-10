"""Cross-validation: the human TC >= +3 insurance index vs the exact 1/3-tens call.

Production surfaces ONLY the exact call (ev_engine.insurance_advice: take iff
the unseen-tens fraction exceeds 1/3); deviations.INSURANCE_INDEX is an
annotation of what an unaided human Hi-Lo counter would do instead. This test
quantifies that human-proxy loss on a seeded simulation: deal many eight-deck
shoes to 50-75% penetration through the real CardCounter and, at every dealt
ace (the dealer-up-card moment, with the ace already counted), compare

    heur  = (true_count >= INSURANCE_INDEX)        # human index play
    exact = insurance_advice(per_rank)["take"]     # production call

No dependence on recorded data (output/session.db) — CI-safe and repeatable.
Measured with SEED 20260610: 7949 decision points, 5.4% overall disagreement,
38.8% inside the |TC-3| <= 1 boundary band vs 2.9% outside it, and a mean
foregone edge of +0.028/unit on the exact-take/heuristic-decline points.

Run:  .venv\\Scripts\\python.exe -m pytest tests\\test_insurance_crossval.py -q
"""

import random
import unittest

from lib.logic import ev_engine
from lib.logic.counting import CardCounter
from lib.logic.deviations import INSURANCE_INDEX

# Physical shoe ranks (the counter folds J/Q/K into the "10" bucket itself).
RANKS = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10",
         "Jack", "Queen", "King"]

SEED = 20260610          # fixed seed -> deterministic decision set
DECKS = 8
SHOES = 400
PEN_RANGE = (0.50, 0.75)  # deal each shoe to a 50-75% penetration cutoff
NEAR_BAND = 1.0           # |tc - INSURANCE_INDEX| <= 1 is "near the boundary"


def _simulate():
    """Replay SHOES seeded shoes through the real CardCounter.

    Returns (points, conservation_errors). Each point is recorded at a dealt
    ace — the moment a dealer up-card ace would trigger the insurance prompt —
    after the ace itself was counted, exactly as the engine counts the up-card
    before publish_snapshot() computes the table-level insurance advice.
    """
    rng = random.Random(SEED)
    points = []
    conservation_errors = 0
    for _ in range(SHOES):
        shoe = [rank for rank in RANKS for _ in range(4 * DECKS)]
        rng.shuffle(shoe)
        cutoff = int(len(shoe) * rng.uniform(*PEN_RANGE))
        counter = CardCounter(deck_count=DECKS)
        for seen, card in enumerate(shoe[:cutoff], start=1):
            counter.count_card(card)
            if card != "Ace":
                continue
            snap = counter.snapshot()
            # Counter -> composition plumbing must conserve the shoe.
            comp = ev_engine.comp_from_per_rank(snap["per_rank"], DECKS)
            if sum(comp) != 52 * DECKS - seen:
                conservation_errors += 1
            info = ev_engine.insurance_advice(snap["per_rank"], deck_count=DECKS)
            points.append({
                "tc": snap["true"],          # the counter's own TC convention
                "p_ten": info["p_ten"],
                "ev": info["ev"],            # per unit of insurance bet
                "exact": info["take"],
                "heur": snap["true"] >= float(INSURANCE_INDEX),
            })
    return points, conservation_errors


class InsuranceCrossValidation(unittest.TestCase):
    """TC >= +3 human index vs the exact 1/3-tens production call."""

    @classmethod
    def setUpClass(cls):
        cls.points, cls.conservation_errors = _simulate()
        cls.n = len(cls.points)
        cls.disagree = [p for p in cls.points if p["exact"] != p["heur"]]
        cls.near = [p for p in cls.points
                    if abs(p["tc"] - INSURANCE_INDEX) <= NEAR_BAND]
        cls.far = [p for p in cls.points
                   if abs(p["tc"] - INSURANCE_INDEX) > NEAR_BAND]
        cls.near_dis = [p for p in cls.near if p["exact"] != p["heur"]]
        cls.far_dis = [p for p in cls.far if p["exact"] != p["heur"]]
        # Exact says take (+EV) but the human index declines: edge left behind.
        cls.foregone = [p["ev"] for p in cls.points
                        if p["exact"] and not p["heur"]]
        # Human index takes a -EV bet the exact call declines: edge paid.
        cls.incurred = [-p["ev"] for p in cls.points
                        if p["heur"] and not p["exact"]]

    # ------------------------------------------------------------- coverage

    def test_simulation_covers_the_decision_space(self):
        """The seeded sample is big, balanced, and internally consistent."""
        self.assertEqual(
            self.conservation_errors, 0,
            "comp_from_per_rank lost cards vs the dealt shoe "
            f"({self.conservation_errors} of {self.n} decision points)")
        self.assertGreaterEqual(
            self.n, 5000,
            f"only {self.n} dealt-ace decision points; sample too thin to "
            "measure a disagreement rate")
        self.assertGreaterEqual(
            len(self.near), 250,
            f"only {len(self.near)} points inside the |TC-{INSURANCE_INDEX}|"
            f" <= {NEAR_BAND} band; boundary comparison would be noise")
        for key in ("exact", "heur"):
            takes = sum(1 for p in self.points if p[key])
            self.assertTrue(
                0 < takes < self.n,
                f"{key} call never flipped ({takes}/{self.n} takes) — "
                "simulation did not exercise both sides of the decision")

    # ---------------------------------------------------------- disagreement

    def test_overall_disagreement_rate_small_but_nonzero(self):
        """The index is a good-but-imperfect proxy: rate in (0, 0.15)."""
        rate = len(self.disagree) / self.n
        self.assertGreater(
            len(self.disagree), 0,
            f"heuristic and exact call never disagreed over {self.n} points — "
            "the cross-validation is vacuous")
        self.assertLess(
            rate, 0.15,
            f"TC >= +{INSURANCE_INDEX} disagreed with the exact 1/3-tens call "
            f"on {len(self.disagree)}/{self.n} = {rate:.3%} of decisions "
            "(expected a few percent for a one-number Hi-Lo proxy)")

    def test_disagreement_concentrates_at_the_boundary(self):
        """Mistakes cluster where TC sits near the index, not far from it."""
        near_rate = len(self.near_dis) / len(self.near)
        far_rate = len(self.far_dis) / len(self.far)
        detail = (f"near band |TC-{INSURANCE_INDEX}| <= {NEAR_BAND}: "
                  f"{len(self.near_dis)}/{len(self.near)} = {near_rate:.3%}; "
                  f"far band: {len(self.far_dis)}/{len(self.far)} = "
                  f"{far_rate:.3%}")
        self.assertGreaterEqual(
            near_rate, 0.15,
            f"expected heavy disagreement near the TC boundary — {detail}")
        self.assertLessEqual(
            far_rate, 0.08,
            f"index should be near-perfect away from the boundary — {detail}")
        self.assertGreater(
            near_rate, 4.0 * far_rate,
            f"disagreement does not concentrate at the boundary — {detail}")

    # -------------------------------------------------------------- EV loss

    def test_ev_foregone_when_heuristic_declines_a_good_bet(self):
        """Mean +EV left behind on exact-take/heuristic-decline points is tiny."""
        self.assertGreater(
            len(self.foregone), 30,
            f"only {len(self.foregone)} exact-take/heuristic-decline points; "
            "EV-foregone estimate would be noise")
        mean_foregone = sum(self.foregone) / len(self.foregone)
        mean_incurred = (sum(self.incurred) / len(self.incurred)
                         if self.incurred else 0.0)
        detail = (f"{len(self.foregone)} declines of a +EV bet, mean edge "
                  f"foregone {mean_foregone:+.4f}/unit (max "
                  f"{max(self.foregone):+.4f}); other direction: "
                  f"{len(self.incurred)} -EV takes, mean edge paid "
                  f"{mean_incurred:+.4f}/unit")
        self.assertGreater(
            mean_foregone, 0.0,
            f"foregone edge must be strictly positive by construction — {detail}")
        self.assertLess(
            mean_foregone, 0.06,
            "human proxy loss per declined unit should stay in the few-cents "
            f"range — {detail}")
        self.assertLess(
            max(self.foregone), 0.30,
            f"single worst missed edge implausibly large — {detail}")


if __name__ == "__main__":
    unittest.main()
