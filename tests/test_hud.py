"""V2 Feature 4: overlay HUD snapshot formatting (pure part — no Tk).

Run:  .venv\\Scripts\\python -m unittest tests.test_hud -v
"""

import unittest

from lib.interfaces.hud import format_hud_lines


def snap(**over):
    base = {
        "count": {"true": 2.34, "running": 14, "decks_remaining": 5.97,
                  "cards_seen": 106, "per_rank": {}},
        "bet": "Bet €40 (edge +0.62% exact, 1/2 Kelly)",
        "insurance": None,
        "seats": [
            {"index": 0, "cards": ["10 of Hearts", "6 of Clubs"], "mine": True,
             "advice": "Hit", "optimal": "Optimal: Stand (-0.48) ≠ book"},
            {"index": 2, "cards": ["8 of Hearts", "8 of Clubs"], "mine": False,
             "advice": "Split", "optimal": "Optimal: Hit (-0.57)"},
            {"index": 4, "cards": [], "mine": False, "advice": "", "optimal": ""},
        ],
        "side_bets": [{"key": "bust_it", "label": "Bust It", "ev": 0.012},
                      {"key": "hot3", "label": "Hot 3", "ev": -0.05},
                      {"key": "21+3", "label": "21+3", "ev": None}],
        "session_pnl": {"units": 3.5, "eur": 87.5, "rounds": 12},
    }
    base.update(over)
    return base


class FormatHudLines(unittest.TestCase):
    def test_count_and_bet(self):
        lines = format_hud_lines(snap())
        self.assertEqual(lines["count"], "TC +2.3   RC +14   6.0 decks")
        self.assertIn("€40", lines["bet"])

    def test_owned_seats_take_priority(self):
        lines = format_hud_lines(snap())
        self.assertEqual(len(lines["seats"]), 1)  # only the owned seat
        self.assertIn("P1 ★: Stand (-0.48)", lines["seats"][0])

    def test_active_seats_when_none_owned(self):
        s = snap()
        for seat in s["seats"]:
            seat["mine"] = False
        lines = format_hud_lines(s)
        self.assertEqual(len(lines["seats"]), 2)  # the two seats with cards

    def test_insurance_and_sidebets_and_pnl(self):
        s = snap(insurance={"text": "Insurance: TAKE (+0.021/unit)",
                            "color": "#2fbf71", "take": True})
        lines = format_hud_lines(s)
        self.assertIn("TAKE", lines["insurance"])
        self.assertIn("Bust It +1.2%", lines["sidebets"])
        self.assertNotIn("Hot 3", lines["sidebets"])  # only +EV bets flagged
        self.assertIn("€+87.50", lines["pnl"])

    def test_guardrail_banner_line(self):
        # V3 E5: a breached session guardrail outranks everything.
        s = snap(guardrails={"enabled": True, "breached": True,
                             "kind": "stop_win",
                             "text": "STOP-WIN reached (€+120.00) — bank it"})
        self.assertIn("STOP-WIN", format_hud_lines(s)["guardrail"])
        s = snap(guardrails={"enabled": True, "breached": False,
                             "kind": None, "text": ""})
        self.assertEqual(format_hud_lines(s)["guardrail"], "")
        self.assertEqual(format_hud_lines(snap())["guardrail"], "")

    def test_quiet_table_collapses(self):
        s = snap(insurance=None, side_bets=[], session_pnl={"rounds": 0})
        for seat in s["seats"]:
            seat["cards"] = []
        lines = format_hud_lines(s)
        self.assertEqual(lines["insurance"], "")
        self.assertEqual(lines["seats"], [])
        self.assertEqual(lines["sidebets"], "")
        self.assertEqual(lines["pnl"], "")


if __name__ == "__main__":
    unittest.main()
