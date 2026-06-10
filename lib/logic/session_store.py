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
            # Additive migrations for DBs created by older versions.
            existing = {row[1] for row in con.execute("PRAGMA table_info(rounds)")}
            for col, decl in (("settlement", "TEXT"), ("pnl_units", "REAL"),
                              ("pnl_eur", "REAL"), ("bet_eur", "REAL")):
                if col not in existing:
                    con.execute(f"ALTER TABLE rounds ADD COLUMN {col} {decl}")

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
        settle = snapshot.get("settlement")
        with self._conn() as con:
            con.execute(
                "INSERT INTO rounds (ts, session_id, round_number, running_count,"
                " true_count, decks_remaining, cards_seen, dealer_card,"
                " dealer_extras, seats, insurance, side_bets,"
                " settlement, pnl_units, pnl_eur, bet_eur)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), self.session_id, snapshot["round"],
                 count["running"], count["true"], count["decks_remaining"],
                 count["cards_seen"], snapshot["dealer"]["card"],
                 json.dumps(snapshot["dealer"].get("extras", [])),
                 json.dumps([{k: s[k] for k in
                              ("index", "cards", "total", "advice", "optimal",
                               "split", "mine")
                              if k in s} for s in seats]),
                 json.dumps(snapshot.get("insurance")),
                 json.dumps(snapshot.get("side_bets", [])),
                 json.dumps(settle) if settle else None,
                 settle.get("my_units") if settle else None,
                 settle.get("my_eur") if settle else None,
                 snapshot.get("bet_placed")))

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
            pnl_row = con.execute(
                f"SELECT COUNT(settlement), COALESCE(SUM(pnl_units), 0),"
                f" COALESCE(SUM(pnl_eur), 0) FROM rounds {where}", args).fetchone()
            settle_rows = con.execute(
                f"SELECT settlement FROM rounds {where}", args).fetchall()
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
        hand_outcomes = {"win": 0, "lose": 0, "push": 0, "blackjack": 0}
        for (st_json,) in settle_rows:
            if not st_json:
                continue
            try:
                settle = json.loads(st_json)
            except ValueError:
                continue
            for seat in (settle or {}).get("seats", []):
                for hand in seat.get("hands", []):
                    if hand.get("outcome") in hand_outcomes:
                        hand_outcomes[hand["outcome"]] += 1
        decided = hand_outcomes["win"] + hand_outcomes["blackjack"] + hand_outcomes["lose"]
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
            "settled_rounds": pnl_row[0],
            "net_units": pnl_row[1],
            "net_eur": pnl_row[2],
            "hand_outcomes": hand_outcomes,
            "win_rate": ((hand_outcomes["win"] + hand_outcomes["blackjack"]) / decided
                         if decided else 0.0),
        }

    def sample_rounds(self, limit=300):
        """Recent recorded rounds for the replay trainer."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT dealer_card, seats, true_count FROM rounds"
                " WHERE dealer_card IS NOT NULL ORDER BY id DESC LIMIT ?",
                (int(limit),)).fetchall()
        out = []
        for dealer, seats_json, tc in rows:
            try:
                seats = json.loads(seats_json) or []
            except ValueError:
                continue
            out.append({"dealer": dealer, "seats": seats, "true_count": tc})
        return out

    def settled_pnl(self, session_only=False):
        """Per-round EUR results of settled rounds with owned seats — the
        empirical sample the bankroll Monte Carlo resamples."""
        where = ("WHERE settlement IS NOT NULL AND pnl_eur IS NOT NULL"
                 + (" AND session_id = ?" if session_only else ""))
        args = (self.session_id,) if session_only else ()
        with self._conn() as con:
            rows = con.execute(f"SELECT pnl_eur FROM rounds {where}", args).fetchall()
        return [r[0] for r in rows]

    def export_csv(self, path) -> int:
        """Write all recorded rounds to CSV; returns the row count."""
        import csv
        with self._conn() as con:
            rows = con.execute(
                "SELECT ts, session_id, round_number, running_count, true_count,"
                " decks_remaining, cards_seen, dealer_card, dealer_extras, seats,"
                " insurance, side_bets, settlement, pnl_units, pnl_eur, bet_eur"
                " FROM rounds ORDER BY id").fetchall()
        header = ["timestamp", "session", "round", "running_count", "true_count",
                  "decks_remaining", "cards_seen", "dealer_card", "dealer_extras",
                  "seats", "insurance", "side_bets", "settlement", "pnl_units",
                  "pnl_eur", "bet_eur"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        return len(rows)
