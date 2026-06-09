"""Session persistence and analytics (Feature 8). SQLite, stdlib only.

Two tables in output/session.db:
  rounds      — one row per completed round: counts at round end, dealer
                card + playout, per-seat cards/advice, insurance and
                side-bet EVs (JSON columns for the structured parts)
  shoe_state  — the latest counter state, so a crash or restart mid-shoe
                doesn't lose the count (the real edge lives in that state)

Connections are opened per operation: writes come from the engine's worker
thread, reads from the Tk thread, and round-end frequency is far too low for
connection reuse to matter.
"""

import json
import sqlite3
import time
import uuid

from ..common import constants

SCHEMA = """
CREATE TABLE IF NOT EXISTS rounds (
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
CREATE TABLE IF NOT EXISTS shoe_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    ts REAL NOT NULL,
    round_number INTEGER,
    cutting_card_seen INTEGER,
    state TEXT NOT NULL
);
"""


class SessionStore:
    def __init__(self, path=None):
        self.path = str(path or (constants.OUTPUT_DIR / "session.db"))
        self.session_id = uuid.uuid4().hex[:12]
        constants.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path, timeout=5)

    # -------------------------------------------------------------- rounds

    def record_round(self, snapshot: dict):
        """Persist a completed round from an engine snapshot. Rounds where
        nothing was dealt are skipped."""
        seats = [s for s in snapshot["seats"] if s["cards"]]
        if not seats and not snapshot["dealer"]["card"]:
            return
        count = snapshot["count"]
        with self._conn() as con:
            con.execute(
                "INSERT INTO rounds (ts, session_id, round_number, running_count,"
                " true_count, decks_remaining, cards_seen, dealer_card,"
                " dealer_extras, seats, insurance, side_bets)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), self.session_id, snapshot["round"],
                 count["running"], count["true"], count["decks_remaining"],
                 count["cards_seen"], snapshot["dealer"]["card"],
                 json.dumps(snapshot["dealer"].get("extras", [])),
                 json.dumps([{k: s[k] for k in
                              ("index", "cards", "total", "advice", "optimal", "split")
                              if k in s} for s in seats]),
                 json.dumps(snapshot.get("insurance")),
                 json.dumps(snapshot.get("side_bets", []))))

    # ---------------------------------------------------------- shoe state

    def save_shoe_state(self, state: dict, round_number: int, cutting_card_seen: bool):
        with self._conn() as con:
            con.execute(
                "INSERT INTO shoe_state (id, ts, round_number, cutting_card_seen, state)"
                " VALUES (1,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET ts=excluded.ts,"
                " round_number=excluded.round_number,"
                " cutting_card_seen=excluded.cutting_card_seen, state=excluded.state",
                (time.time(), round_number, int(cutting_card_seen), json.dumps(state)))

    def load_recent_shoe_state(self, max_age_s: float = 2 * 3600):
        """The persisted shoe state if it is recent and non-empty, else None.
        Returns {"state", "round_number", "cutting_card_seen", "age_s"}."""
        with self._conn() as con:
            row = con.execute("SELECT ts, round_number, cutting_card_seen, state"
                              " FROM shoe_state WHERE id = 1").fetchone()
        if row is None:
            return None
        ts, round_number, cutting, state_json = row
        age = time.time() - ts
        state = json.loads(state_json)
        if age > max_age_s or not state.get("cards_seen"):
            return None
        return {"state": state, "round_number": round_number,
                "cutting_card_seen": bool(cutting), "age_s": age}

    def clear_shoe_state(self):
        with self._conn() as con:
            con.execute("DELETE FROM shoe_state WHERE id = 1")

    # --------------------------------------------------------------- stats

    def stats(self, session_only: bool = True) -> dict:
        where = "WHERE session_id = ?" if session_only else ""
        args = (self.session_id,) if session_only else ()
        with self._conn() as con:
            row = con.execute(
                f"SELECT COUNT(*), AVG(true_count), MAX(true_count), MIN(true_count),"
                f" SUM(true_count >= 2), SUM(dealer_card = 'Ace')"
                f" FROM rounds {where}", args).fetchone()
            insurance_takes = con.execute(
                f"SELECT COUNT(*) FROM rounds {where}{' AND' if where else ' WHERE'}"
                f" json_extract(insurance, '$.take') = 1", args).fetchone()[0]
            sidebet_rows = con.execute(
                f"SELECT side_bets FROM rounds {where}", args).fetchall()
        plus_ev_sidebet_rounds = 0
        for (sb_json,) in sidebet_rows:
            try:
                evs = json.loads(sb_json) or []
            except ValueError:
                continue
            if any(b.get("ev") is not None and b["ev"] > 0 for b in evs):
                plus_ev_sidebet_rounds += 1
        rounds, avg_tc, max_tc, min_tc, high_tc, ace_rounds = row
        return {
            "rounds": rounds or 0,
            "avg_tc": avg_tc or 0.0,
            "max_tc": max_tc or 0.0,
            "min_tc": min_tc or 0.0,
            "rounds_tc_2_plus": high_tc or 0,
            "dealer_ace_rounds": ace_rounds or 0,
            "insurance_take_rounds": insurance_takes or 0,
            "plus_ev_sidebet_rounds": plus_ev_sidebet_rounds,
        }

    def export_csv(self, path) -> int:
        """Write all recorded rounds to CSV; returns the row count."""
        import csv
        with self._conn() as con:
            rows = con.execute(
                "SELECT ts, session_id, round_number, running_count, true_count,"
                " decks_remaining, cards_seen, dealer_card, dealer_extras, seats,"
                " insurance, side_bets FROM rounds ORDER BY id").fetchall()
        header = ["timestamp", "session", "round", "running_count", "true_count",
                  "decks_remaining", "cards_seen", "dealer_card", "dealer_extras",
                  "seats", "insurance", "side_bets"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        return len(rows)
