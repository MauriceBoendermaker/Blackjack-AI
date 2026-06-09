"""Feature 2: Illustrious 18 + Fab 4 index deviations.

Two layers of tests:
 1. Table/lookup semantics (positive and negative indices, H17 differences,
    Fab 4 gating on surrender, two-card gating, Fab4-over-I18 precedence).
 2. The cross-check from FEATURES.md: near each index threshold, with a
    Hi-Lo-neutral constructed composition, the exact-EV engine must flip to
    the same deviation action. This ties the human count wisdom and the
    composition math together — a strong mutual-correctness test.

Run:  .venv\\Scripts\\python -m unittest tests.test_deviations -v
"""

import unittest

from lib.logic import deviations
from lib.logic.deviations import index_advice
from lib.logic.ev_engine import ACE, TEN, Rules, evaluate, full_shoe, insurance_ev

# Hi-Lo tags by composition index (A,2,3,4,5,6,7,8,9,T).
_HILO = {ACE: -1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 0, TEN: -1}


def comp_at_tc(target_tc, removed):
    """Construct a realistic composition near `target_tc`: remove the
    hand/up-card `removed` (composition indices), then deal out 2..9 for
    positive counts (2-6 drive the count; 7-9 deplete alongside, as in a real
    high shoe) or tens+aces in their natural 4:1 ratio for negative counts.

    Returns (comp, actual_tc)."""
    comp = list(full_shoe(8))
    rc = 0
    for idx in removed:
        comp[idx] -= 1
        rc += _HILO[idx]

    def tc():
        return rc / (sum(comp) / 52.0)

    pos_cycle = [1, 2, 3, 4, 5, 6, 7, 8]          # ranks 2..9
    neg_cycle = [TEN, TEN, TEN, TEN, ACE]          # natural 16:4 per deck
    cycle = pos_cycle if target_tc > 0 else neg_cycle
    i = 0
    while (tc() < target_tc) if target_tc > 0 else (tc() > target_tc):
        idx = cycle[i % len(cycle)]
        i += 1
        if comp[idx] == 0:
            if all(comp[j] == 0 for j in cycle):
                break
            continue
        comp[idx] -= 1
        rc += _HILO[idx]
    return tuple(comp), tc()


class IndexTable(unittest.TestCase):
    def test_positive_index(self):
        dev = index_advice("16", "10", true_count=1.0, s17=True, surrender=False)
        self.assertEqual((dev["action"], dev["triggered"]), ("S", True))
        dev = index_advice("16", "10", true_count=-0.5, s17=True, surrender=False)
        self.assertEqual((dev["action"], dev["triggered"]), ("H", False))

    def test_negative_index_means_stop_standing(self):
        # 13 v 2: book is Stand; hit only below TC -1.
        dev = index_advice("13", "2", true_count=0.0, s17=True, surrender=False)
        self.assertEqual((dev["action"], dev["triggered"]), ("S", True))
        dev = index_advice("13", "2", true_count=-2.0, s17=True, surrender=False)
        self.assertEqual((dev["action"], dev["triggered"]), ("H", False))

    def test_h17_differences(self):
        # 11 v A: index +1 in S17, plain basic strategy (no index) in H17.
        self.assertIsNotNone(index_advice("11", "A", 0.0, s17=True, surrender=False))
        self.assertIsNone(index_advice("11", "A", 0.0, s17=False, surrender=False))
        # 10 v A drops from +4 to +3.
        self.assertFalse(index_advice("10", "A", 3.5, s17=True, surrender=False)["triggered"])
        self.assertTrue(index_advice("10", "A", 3.5, s17=False, surrender=False)["triggered"])
        # H17-only stand deviations exist.
        self.assertIsNotNone(index_advice("16", "A", 3.0, s17=False, surrender=False))
        self.assertIsNone(index_advice("16", "A", 3.0, s17=True, surrender=False))

    def test_fab4_gating_and_precedence(self):
        # No surrender offered -> 15 v 10 falls back to the I18 stand index.
        dev = index_advice("15", "10", 1.0, s17=True, surrender=False)
        self.assertEqual(dev["source"], "I18")
        # Surrender offered -> Fab 4 takes precedence: surrender at TC >= 0.
        dev = index_advice("15", "10", 1.0, s17=True, surrender=True)
        self.assertEqual((dev["source"], dev["action"]), ("Fab4", "R"))
        dev = index_advice("15", "10", -1.0, s17=True, surrender=True)
        self.assertEqual(dev["action"], "H")
        # 3+ card hands can't surrender or double.
        dev = index_advice("15", "10", 1.0, s17=True, surrender=True, two_cards=False)
        self.assertEqual(dev["source"], "I18")
        self.assertIsNone(index_advice("10", "10", 9.0, s17=True,
                                       surrender=False, two_cards=False))

    def test_no_index_for_most_hands(self):
        self.assertIsNone(index_advice("18", "10", 5.0, s17=True, surrender=False))
        self.assertIsNone(index_advice("A,7", "10", 5.0, s17=True, surrender=False))
        self.assertIsNone(index_advice("8,8", "10", 5.0, s17=True, surrender=False))


class EvEngineCrossCheck(unittest.TestCase):
    """At ~1.5-2 TC beyond each index the exact-EV engine must agree with the
    deviation; well below it, with basic strategy. Indices are derived for
    US peek games, so the engine runs peek rules here."""

    RULES = Rules(peek=True)
    # (name, hand indices, dealer idx, index value, action above, action below)
    CASES = [
        ("16v10@0", (TEN, 5), TEN, 0, "S", "H"),
        ("15v10@4", (TEN, 4), TEN, 4, "S", "H"),
        ("12v2@3", (TEN, 1), 1, 3, "S", "H"),
        ("12v4@0", (TEN, 1), 3, 0, "S", "H"),
        ("13v2@-1", (TEN, 2), 1, -1, "S", "H"),
        ("TTv6@4", (TEN, TEN), 5, 4, "P", "S"),
        ("9v2@1", (4, 3), 1, 1, "D", "H"),
        ("10v10@4", (5, 3), TEN, 4, "D", "H"),
    ]
    MARGIN = 2.0

    def test_deviations_match_exact_ev(self):
        for name, hand, dealer, index, above, below in self.CASES:
            removed = list(hand) + [dealer]
            with self.subTest(case=name, side="above"):
                comp, tc = comp_at_tc(index + self.MARGIN, removed)
                result = evaluate(tuple(sorted(hand)), dealer, comp, self.RULES)
                self.assertEqual(result["best"], above,
                                 f"{name}: at TC {tc:+.1f} expected {above}, "
                                 f"engine says {result['best']} {result['evs']}")
            with self.subTest(case=name, side="below"):
                comp, tc = comp_at_tc(index - self.MARGIN, removed)
                result = evaluate(tuple(sorted(hand)), dealer, comp, self.RULES)
                self.assertEqual(result["best"], below,
                                 f"{name}: at TC {tc:+.1f} expected {below}, "
                                 f"engine says {result['best']} {result['evs']}")

    def test_insurance_index_consistency(self):
        # The TC >= +3 insurance index should roughly match the exact 1/3
        # tens-fraction threshold on count-neutral low-card-stripped shoes.
        comp, tc = comp_at_tc(4.5, [TEN, 5, TEN])
        self.assertGreater(insurance_ev(comp)[1], 0, f"TC {tc:+.1f}")
        comp, tc = comp_at_tc(1.0, [TEN, 5, TEN])
        self.assertLess(insurance_ev(comp)[1], 0, f"TC {tc:+.1f}")


if __name__ == "__main__":
    unittest.main()
