"""Reliability pack (Core-3): records, betting telemetry, engine inputs.

Covers the paytable fingerprint (canonical hash, schema migration, recording,
CSV export, mid-shoe change flag), the once-per-round bet-capped counter, the
OCR bet clamp, and explicit hand routing for manual cards on split seats.

Run:  .venv\\Scripts\\python -m unittest tests.test_reliability_records -v
"""

import copy
import csv
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from lib.common import constants
from lib.logic import betting
from lib.logic import settlement
from lib.logic.session_store import SessionStore

# The rounds schema as it stood before the paytable_hash migration (and the
# settlement-era columns) — what an old session.db looks like on reopen.
OLD_SCHEMA = """
CREATE TABLE rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    round_number INTEGER,
    running_count INTEGER,
    true_count REAL,
    decks_remaining REAL,
    cards_seen INTEGER,
    dealer_card TEXT,
    dealer_extras TEXT,
    seats TEXT,
    insurance TEXT,
    side_bets TEXT
);
"""


def _engine(log=None):
    from lib.logic.engine import DetectionEngine
    eng = DetectionEngine(log=log or (lambda *a, **k: None))
    eng.store = None  # no DB writes from tests (re-set per test when needed)
    return eng


def _snapshot(round_number=1):
    return {
        "round": round_number,
        "dealer": {"card": "King", "locked": True, "extras": ["7"]},
        "seats": [{"index": 0, "cards": ["10 of Hearts", "7 of Clubs"],
                   "total": "Hard 17", "advice": "Stand",
                   "optimal": "Optimal: Stand (-0.123)", "split": False,
                   "book_action": "S", "optimal_action": "S"}],
        "count": {"running": 0, "true": 0.0, "decks_remaining": 6.0,
                  "cards_seen": 10},
        "insurance": None,
        "side_bets": [],
    }


class ConstantsGuard(unittest.TestCase):
    """Snapshot/restore every runtime-mutable constant a test may touch."""

    def setUp(self):
        self._rules = dict(constants.RULES)
        self._betting = dict(constants.BETTING)
        self._side = copy.deepcopy(constants.SIDE_BETS)
        self._ocr = dict(constants.OCR)

    def tearDown(self):
        constants.RULES.clear()
        constants.RULES.update(self._rules)
        constants.BETTING.clear()
        constants.BETTING.update(self._betting)
        constants.SIDE_BETS.clear()
        constants.SIDE_BETS.update(self._side)
        constants.OCR.clear()
        constants.OCR.update(self._ocr)


class PaytableHash(ConstantsGuard):
    def test_format(self):
        digest = settlement.paytable_hash()
        self.assertEqual(len(digest), 16)
        int(digest, 16)  # valid hex

    def test_stable_across_dict_ordering(self):
        before = settlement.paytable_hash()
        reordered = {k: constants.SIDE_BETS[k]
                     for k in reversed(list(constants.SIDE_BETS))}
        constants.SIDE_BETS.clear()
        constants.SIDE_BETS.update(reordered)
        self.assertEqual(settlement.paytable_hash(), before)

    def test_int_and_str_paytable_keys_hash_identically(self):
        before = settlement.paytable_hash()
        constants.SIDE_BETS["bust_it"]["paytable"] = {
            str(k): v
            for k, v in constants.SIDE_BETS["bust_it"]["paytable"].items()}
        self.assertEqual(settlement.paytable_hash(), before)

    def test_ignores_disabled_bets_and_cosmetic_keys(self):
        before = settlement.paytable_hash()
        constants.SIDE_BETS["lucky_lucky"]["paytable"]["777"] = 999  # disabled
        constants.SIDE_BETS["hot3"]["label"] = "Hot Three"           # cosmetic
        self.assertEqual(settlement.paytable_hash(), before)

    def test_changes_on_payout_relevant_edits(self):
        before = settlement.paytable_hash()
        constants.SIDE_BETS["hot3"]["paytable"]["21"] = 5
        self.assertNotEqual(settlement.paytable_hash(), before)
        constants.SIDE_BETS["hot3"]["paytable"]["21"] = 4
        self.assertEqual(settlement.paytable_hash(), before)  # exact round trip
        constants.SIDE_BETS["perfect_pairs"]["enabled"] = False
        after_disable = settlement.paytable_hash()
        self.assertNotEqual(after_disable, before)
        constants.RULES["bj_pays"] = 1.2
        self.assertNotEqual(settlement.paytable_hash(), after_disable)


class StoreMigrationAndRecord(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "session.db"

    def test_migration_adds_column_to_old_db(self):
        con = sqlite3.connect(self.path)
        con.executescript(OLD_SCHEMA)
        con.close()
        store = SessionStore(self.path)  # reopen -> additive migration
        with sqlite3.connect(self.path) as con:
            cols = {row[1] for row in con.execute("PRAGMA table_info(rounds)")}
        self.assertIn("paytable_hash", cols)
        store.record_round(_snapshot(), paytable_hash="cafe0123deadbeef")
        with sqlite3.connect(self.path) as con:
            stored = con.execute(
                "SELECT paytable_hash FROM rounds").fetchone()[0]
        self.assertEqual(stored, "cafe0123deadbeef")

    def test_hash_defaults_to_null(self):
        store = SessionStore(self.path)
        store.record_round(_snapshot())
        with sqlite3.connect(self.path) as con:
            stored = con.execute(
                "SELECT paytable_hash FROM rounds").fetchone()[0]
        self.assertIsNone(stored)

    def test_seats_json_carries_action_codes(self):
        store = SessionStore(self.path)
        store.record_round(_snapshot(),
                           paytable_hash=settlement.paytable_hash())
        with sqlite3.connect(self.path) as con:
            seats_json = con.execute("SELECT seats FROM rounds").fetchone()[0]
        seat = json.loads(seats_json)[0]
        self.assertEqual(seat["book_action"], "S")
        self.assertEqual(seat["optimal_action"], "S")

    def test_export_csv_includes_hash_column(self):
        store = SessionStore(self.path)
        store.record_round(_snapshot(), paytable_hash="cafe0123deadbeef")
        out = Path(tempfile.mkdtemp()) / "rounds.csv"
        self.assertEqual(store.export_csv(out), 1)
        with open(out, newline="", encoding="utf-8") as f:
            header, row = list(csv.reader(f))
        self.assertIn("paytable_hash", header)
        self.assertEqual(row[header.index("paytable_hash")],
                         "cafe0123deadbeef")


class EnginePersistsHashAndCodes(ConstantsGuard):
    def _dealt_engine(self):
        eng = _engine()
        eng.store = SessionStore(Path(tempfile.mkdtemp()) / "session.db")
        eng.replace_card(0, 0, "10 of Hearts")
        eng.replace_card(0, 1, "7 of Clubs")     # hard 17 vs 10: Stand, clearly
        eng.replace_dealer("King")
        self.assertTrue(eng.flush_advice(timeout=60))
        return eng

    def _stored_hash(self, eng):
        with sqlite3.connect(eng.store.path) as con:
            return con.execute("SELECT paytable_hash FROM rounds").fetchone()[0]

    def test_persist_job_stamps_hash_and_advice_codes(self):
        eng = self._dealt_engine()
        eng.new_round()
        self.assertTrue(eng.flush_advice(timeout=60))
        with sqlite3.connect(eng.store.path) as con:
            stored_hash, seats_json = con.execute(
                "SELECT paytable_hash, seats FROM rounds").fetchone()
        self.assertEqual(stored_hash, settlement.paytable_hash())
        seat = json.loads(seats_json)[0]
        self.assertEqual(seat["book_action"], "S")
        self.assertEqual(seat["optimal_action"], "S")

    def test_hash_captured_at_round_end_not_at_persist_time(self):
        # The persist job can queue for seconds behind EV work — a paytable
        # saved in that window must not restamp the round that was settled
        # under the old payouts.
        eng = self._dealt_engine()
        old_hash = settlement.paytable_hash()
        gate = threading.Event()
        eng._advice_pool.submit(gate.wait)  # the persist job queues behind this
        try:
            eng.new_round()
            constants.RULES["bj_pays"] = 1.2  # settings dialog lands meanwhile
            eng.refresh_settings()
        finally:
            gate.set()
        self.assertTrue(eng.flush_advice(timeout=60))
        self.assertEqual(self._stored_hash(eng), old_hash)
        self.assertNotEqual(self._stored_hash(eng), settlement.paytable_hash())


class MidShoePaytableChange(ConstantsGuard):
    def test_refresh_flags_and_warns_once_until_shoe_reset(self):
        logs = []
        eng = _engine(log=lambda msg, level=None: logs.append((level, msg)))
        eng.store = None

        def warnings():
            return [m for level, m in logs
                    if level == "WARNING" and "mid-shoe" in m]

        eng.refresh_settings()  # no paytable change -> no flag, no warning
        self.assertFalse(eng.get_snapshot()["paytable_changed_midshoe"])
        self.assertEqual(warnings(), [])

        constants.RULES["bj_pays"] = 1.2
        eng.refresh_settings()
        self.assertTrue(eng.get_snapshot()["paytable_changed_midshoe"])
        self.assertEqual(len(warnings()), 1)
        self.assertIn("round 1", warnings()[0])

        eng.refresh_settings()  # unchanged again: sticky flag, no re-log
        self.assertTrue(eng.get_snapshot()["paytable_changed_midshoe"])
        self.assertEqual(len(warnings()), 1)

        eng.new_round()         # round end does NOT clear the flag
        self.assertTrue(eng.get_snapshot()["paytable_changed_midshoe"])

        eng.reset_shoe()        # a fresh shoe starts clean under the new tables
        self.assertFalse(eng.get_snapshot()["paytable_changed_midshoe"])
        eng.refresh_settings()  # same paytables as captured at reset: no flag
        self.assertFalse(eng.get_snapshot()["paytable_changed_midshoe"])


class SuggestCappedFlag(unittest.TestCase):
    BASE = {"bankroll": 1000.0, "kelly_fraction": 0.5, "base_edge": -0.005,
            "edge_per_tc": 0.005, "variance": 1.33, "table_min": 10,
            "table_max": 5000}

    def test_capped_only_when_clamped_down_by_table_max(self):
        self.assertFalse(betting.suggest(0, self.BASE)["capped"])   # min bet
        self.assertFalse(betting.suggest(3, self.BASE)["capped"])   # kelly < max
        s = betting.suggest(10, {**self.BASE, "bankroll": 10_000_000})
        self.assertTrue(s["capped"])
        self.assertEqual(s["bet"], 5000)
        s = betting.suggest(10, {**self.BASE, "bankroll": 10_000_000,
                                 "table_max": 0})
        self.assertFalse(s["capped"])                               # no max


class BetCappedTelemetry(ConstantsGuard):
    def test_counts_once_per_round_and_survives_shoe_reset(self):
        logs = []
        # Positive off-the-top edge + huge bankroll: every suggestion capped.
        constants.BETTING.update(bankroll=10_000_000, base_edge=0.01,
                                 table_max=5000, use_exact_edge=0)
        eng = _engine(log=lambda msg, level=None: logs.append((level, msg)))
        eng.store = None
        eng.replace_card(0, 0, "10 of Hearts")  # a real round is in progress
        for _ in range(4):  # many snapshots, still zero counted rounds
            eng.publish_snapshot()
        self.assertEqual(eng.get_snapshot()["bet_capped_rounds"], 0)

        eng.new_round()
        self.assertEqual(eng.get_snapshot()["bet_capped_rounds"], 1)
        warns = [m for level, m in logs if level == "WARNING" and "capped" in m]
        self.assertEqual(warns, ["Bet capped at table max (1× this session)"])

        eng.replace_card(0, 0, "9 of Clubs")
        eng.publish_snapshot()
        eng.new_round()
        self.assertEqual(eng.get_snapshot()["bet_capped_rounds"], 2)

        eng.publish_snapshot()
        eng.new_round()  # capped suggestion but nothing dealt: not a round
        self.assertEqual(eng.get_snapshot()["bet_capped_rounds"], 2)

        eng.reset_shoe()  # session-scoped: a shoe reset keeps the tally
        self.assertEqual(eng.get_snapshot()["bet_capped_rounds"], 2)

    def test_uncapped_round_does_not_increment(self):
        constants.BETTING.update(bankroll=1000.0, base_edge=-0.005,
                                 use_exact_edge=0)
        eng = _engine()
        eng.store = None
        eng.publish_snapshot()
        eng.new_round()
        self.assertEqual(eng.get_snapshot()["bet_capped_rounds"], 0)


class OcrBetClamp(ConstantsGuard):
    def test_bet_above_table_max_clamps_with_warning(self):
        logs = []
        constants.BETTING["table_max"] = 5000
        constants.OCR["sync_bet"] = 1
        eng = _engine(log=lambda msg, level=None: logs.append((level, msg)))
        eng.store = None
        eng._apply_ocr_values({"bet": 85000.0})  # misread: 8.50 seen as 85000
        self.assertEqual(eng.bet_placed, 5000.0)
        self.assertTrue(any(level == "WARNING" and "clamped" in m
                            for level, m in logs))

    def test_legit_bet_passes_unclamped(self):
        constants.BETTING["table_max"] = 5000
        constants.OCR["sync_bet"] = 1
        eng = _engine()
        eng.store = None
        eng._apply_ocr_values({"bet": 250.0})
        self.assertEqual(eng.bet_placed, 250.0)

    def test_no_table_max_means_no_clamp(self):
        constants.BETTING["table_max"] = 0
        constants.OCR["sync_bet"] = 1
        eng = _engine()
        eng.store = None
        eng._apply_ocr_values({"bet": 85000.0})
        self.assertEqual(eng.bet_placed, 85000.0)


class SplitAwareReplaceCard(unittest.TestCase):
    @staticmethod
    def _split_seat(eng, seat=0):
        eng.replace_card(seat, 0, "8 of Hearts")
        eng.replace_card(seat, 1, "8 of Spades")
        eng.set_split(seat)

    def test_explicit_hand_index_routes_added_cards(self):
        eng = _engine()
        self._split_seat(eng)
        eng.replace_card(0, 99, "3 of Clubs", hand_index=1)
        eng.replace_card(0, 99, "10 of Clubs", hand_index=1)  # auto would say h0
        seat = eng.get_snapshot()["seats"][0]
        self.assertEqual(seat["hand_of"], [0, 1, 1, 1])
        self.assertEqual(len(seat["book_action"]), 2)  # per-hand code lists
        self.assertEqual(len(seat["optimal_action"]), 2)

    def test_none_keeps_auto_routing(self):
        # Mirrors test_split.test_split_lifecycle: balanced alternation.
        eng = _engine()
        self._split_seat(eng)
        eng.replace_card(0, 99, "3 of Clubs")   # appended -> shorter hand (h0)
        eng.replace_card(0, 99, "10 of Clubs")  # -> other hand
        seat = eng.get_snapshot()["seats"][0]
        self.assertEqual(seat["hand_of"], [0, 1, 0, 1])  # append order is fixed

    def test_out_of_range_hand_index_falls_back_to_auto(self):
        eng = _engine()
        self._split_seat(eng)
        eng.replace_card(0, 99, "3 of Clubs", hand_index=7)
        seat = eng.get_snapshot()["seats"][0]
        self.assertEqual(sorted(seat["hand_of"]), [0, 0, 1])

    def test_hand_index_ignored_without_split(self):
        eng = _engine()
        eng.replace_card(2, 0, "8 of Hearts")
        eng.replace_card(2, 99, "5 of Clubs", hand_index=1)
        self.assertEqual(eng.get_snapshot()["seats"][2]["hand_of"], [0, 0])

    def test_replace_keeps_existing_hand_tag(self):
        eng = _engine()
        self._split_seat(eng)
        eng.replace_card(0, 1, "8 of Diamonds", hand_index=0)  # replace, not add
        seat = eng.get_snapshot()["seats"][0]
        self.assertEqual(seat["cards"][1], "8 of Diamonds")
        self.assertEqual(seat["hand_of"], [0, 1])  # tag untouched


if __name__ == "__main__":
    unittest.main()
