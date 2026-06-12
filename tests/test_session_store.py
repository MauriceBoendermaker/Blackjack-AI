"""Feature 8: session persistence and analytics.

Run:  .venv\\Scripts\\python -m unittest tests.test_session_store -v
"""

import tempfile
import time
import unittest
from pathlib import Path

from lib.logic.counting import CardCounter
from lib.logic.session_store import SessionStore


def snapshot(round_number=1, tc=0.0, dealer="King", insurance=None, side_bets=None):
    return {
        "round": round_number,
        "dealer": {"card": dealer, "locked": True, "extras": ["7"]},
        "seats": [{"index": 0, "cards": ["10 of Hearts", "6 of Clubs"],
                   "total": "Hard 16", "advice": "Hit",
                   "optimal": "Optimal: Hit (-0.572)", "split": False}],
        "count": {"running": int(tc * 6), "true": tc, "decks_remaining": 6.0,
                  "cards_seen": 100},
        "insurance": insurance,
        "side_bets": side_bets or [],
    }


class StoreRoundTrip(unittest.TestCase):
    def setUp(self):
        self.store = SessionStore(Path(tempfile.mkdtemp()) / "session.db")

    def test_record_and_stats(self):
        self.store.record_round(snapshot(1, tc=-1.0))
        self.store.record_round(snapshot(2, tc=3.0,
                                         side_bets=[{"key": "bust_it", "ev": 0.02}]))
        self.store.record_round(snapshot(3, tc=2.5, dealer="Ace",
                                         insurance={"take": True, "ev": 0.01}))
        stats = self.store.stats()
        self.assertEqual(stats["rounds"], 3)
        self.assertAlmostEqual(stats["avg_tc"], (-1.0 + 3.0 + 2.5) / 3, places=6)
        self.assertEqual(stats["rounds_tc_2_plus"], 2)
        self.assertEqual(stats["dealer_ace_rounds"], 1)
        self.assertEqual(stats["insurance_take_rounds"], 1)
        self.assertEqual(stats["plus_ev_sidebet_rounds"], 1)

    def test_empty_rounds_skipped(self):
        empty = snapshot()
        empty["seats"] = []
        empty["dealer"]["card"] = None
        self.store.record_round(empty)
        self.assertEqual(self.store.stats()["rounds"], 0)

    def test_session_scoping(self):
        self.store.record_round(snapshot())
        other = SessionStore(self.store.path)  # same DB, new session id
        other.record_round(snapshot())
        self.assertEqual(other.stats(session_only=True)["rounds"], 1)
        self.assertEqual(other.stats(session_only=False)["rounds"], 2)

    def test_export_csv(self):
        self.store.record_round(snapshot())
        out = Path(tempfile.mkdtemp()) / "rounds.csv"
        self.assertEqual(self.store.export_csv(out), 1)
        text = out.read_text(encoding="utf-8")
        self.assertIn("true_count", text)
        self.assertIn("10 of Hearts", text)


class RampDesignerQueries(unittest.TestCase):
    """V3 Feature 5: TC distribution + table pace measured from rounds."""

    def setUp(self):
        self.store = SessionStore(Path(tempfile.mkdtemp()) / "session.db")

    def _insert(self, ts, tc, session_id=None):
        import sqlite3
        with sqlite3.connect(self.store.path) as con:
            con.execute(
                "INSERT INTO rounds (ts, session_id, true_count) VALUES (?,?,?)",
                (ts, session_id or self.store.session_id, tc))

    def test_pre_deal_tcs(self):
        self.store.record_round(snapshot(1, tc=-1.0))
        self.store.record_round(snapshot(2, tc=2.5))
        self._insert(time.time(), 4.0, session_id="other-session")
        self.assertEqual(self.store.pre_deal_tcs(), [-1.0, 2.5, 4.0])
        self.assertEqual(self.store.pre_deal_tcs(session_only=True),
                         [-1.0, 2.5])

    def test_rounds_per_hour_gap_aware(self):
        t0 = 1_000_000.0
        # 12 rounds at one a minute, then a 2-hour break, then one more.
        for i in range(12):
            self._insert(t0 + i * 60.0, 0.0)
        self._insert(t0 + 11 * 60.0 + 7200.0, 0.0)
        rph = self.store.rounds_per_hour()
        self.assertAlmostEqual(rph, 60.0, places=6)  # the gap doesn't count

    def test_rounds_per_hour_needs_data(self):
        self.assertIsNone(self.store.rounds_per_hour())
        t0 = 1_000_000.0
        for i in range(5):
            self._insert(t0 + i * 60.0, 0.0)
        self.assertIsNone(self.store.rounds_per_hour())  # < 10 intervals

    def test_sessions_do_not_chain(self):
        # The interval between the last round of one app run and the first
        # of the next is dead time, not play time.
        t0 = 1_000_000.0
        for i in range(11):
            self._insert(t0 + i * 30.0, 0.0)
        self._insert(t0 + 12 * 30.0, 0.0, session_id="other-session")
        rph = self.store.rounds_per_hour()
        self.assertAlmostEqual(rph, 120.0, places=6)


class ShoeStatePersistence(unittest.TestCase):
    def setUp(self):
        self.store = SessionStore(Path(tempfile.mkdtemp()) / "session.db")

    def test_counter_state_round_trip(self):
        c = CardCounter(deck_count=8)
        for name in ("10 of Hearts", "6 of Clubs", "King", "Ace of Spades"):
            c.count_card(name)
        self.store.save_shoe_state(c.get_state(), round_number=7,
                                   cutting_card_seen=True)
        info = self.store.load_recent_shoe_state()
        self.assertIsNotNone(info)
        self.assertEqual(info["round_number"], 7)
        self.assertTrue(info["cutting_card_seen"])

        restored = CardCounter(deck_count=8)
        restored.apply_state(info["state"])
        self.assertEqual(restored.running_count, c.running_count)
        self.assertEqual(restored.cards_seen, c.cards_seen)
        self.assertEqual(restored.per_rank, c.per_rank)
        self.assertEqual(restored.suit_seen, c.suit_seen)
        self.assertEqual(restored.rank_seen_nosuit, c.rank_seen_nosuit)

    def test_stale_or_empty_state_ignored(self):
        c = CardCounter(deck_count=8)
        self.store.save_shoe_state(c.get_state(), 1, False)  # nothing seen
        self.assertIsNone(self.store.load_recent_shoe_state())
        c.count_card("King")
        self.store.save_shoe_state(c.get_state(), 1, False)
        self.assertIsNone(self.store.load_recent_shoe_state(max_age_s=0.0))
        self.assertIsNotNone(self.store.load_recent_shoe_state(max_age_s=60))

    def test_engine_records_round_on_reset(self):
        from lib.logic.engine import DetectionEngine
        eng = DetectionEngine(log=lambda *a, **k: None)
        eng.store = self.store
        eng.replace_card(0, 0, "10 of Hearts")
        eng.replace_card(0, 1, "6 of Clubs")
        eng.replace_dealer("King")
        eng.new_round()
        eng.flush_advice(timeout=60)  # persist job runs on the advice pool
        self.assertEqual(self.store.stats(session_only=False)["rounds"], 1)
        info = self.store.load_recent_shoe_state()
        self.assertIsNotNone(info)
        self.assertEqual(info["state"]["cards_seen"], 3)


if __name__ == "__main__":
    unittest.main()
