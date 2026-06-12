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
from . import seat_quality
from .strategy import StrategyAdvisor

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
CREATE TABLE IF NOT EXISTS executor_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    round INTEGER,
    mode TEXT,
    phase TEXT,
    action TEXT,
    seat INTEGER,
    target_x INTEGER,
    target_y INTEGER,
    confidence REAL,
    fired INTEGER,
    verified INTEGER,
    observed TEXT,
    reason TEXT,
    crop TEXT
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
                              ("pnl_eur", "REAL"), ("bet_eur", "REAL"),
                              ("paytable_hash", "TEXT"),
                              # Ramp-vs-placed comparison inputs (V3 F6).
                              # Snapshot values at round END — like the count
                              # columns they describe the NEXT round's
                              # pre-deal call; row N pairs with row N+1.
                              ("bet_suggested", "REAL"),
                              ("bet_sit_out", "INTEGER"),
                              ("edge_exact", "REAL")):
                if col not in existing:
                    con.execute(f"ALTER TABLE rounds ADD COLUMN {col} {decl}")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=5)

    # -------------------------------------------------------------- rounds

    def record_round(self, snapshot: dict, paytable_hash=None):
        """Persist a completed round from an engine snapshot. Rounds where
        nothing was dealt are skipped. `paytable_hash` is the payout
        fingerprint in force (settlement.paytable_hash()) so analysis can
        group rounds by paytable. book_action/optimal_action are the raw
        advice codes (lists per hand for split seats)."""
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
                " settlement, pnl_units, pnl_eur, bet_eur, paytable_hash,"
                " bet_suggested, bet_sit_out, edge_exact)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), self.session_id, snapshot["round"],
                 count["running"], count["true"], count["decks_remaining"],
                 count["cards_seen"], snapshot["dealer"]["card"],
                 json.dumps(snapshot["dealer"].get("extras", [])),
                 json.dumps([{k: s[k] for k in
                              ("index", "cards", "total", "advice", "optimal",
                               "split", "mine", "book_action", "optimal_action")
                              if k in s} for s in seats]),
                 json.dumps(snapshot.get("insurance")),
                 json.dumps(snapshot.get("side_bets", [])),
                 json.dumps(settle) if settle else None,
                 settle.get("my_units") if settle else None,
                 # The booked round total incl. side bets — stats() and the
                 # Monte Carlo P&L sample must match what the session P&L
                 # and the auto-settled bankroll actually received. NULL
                 # stays NULL for rounds without owned-seat money.
                 (settle.get("my_eur", 0) or 0)
                 + (settle.get("my_side_eur", 0) or 0)
                 if settle and "my_eur" in settle else None,
                 snapshot.get("bet_placed"), paytable_hash,
                 snapshot.get("bet_suggested"),
                 (None if snapshot.get("bet_suggested") is None
                  else int(bool(snapshot.get("bet_sit_out")))),
                 snapshot.get("edge_exact")))

    # ------------------------------------------------------- executor audit

    def record_executor(self, entry: dict):
        """One executor decision/click for after-the-fact review (V4
        Feature 2 audit trail). `entry` keys mirror the executor's plan:
        round, mode, phase, action, seat, target (x, y), confidence,
        fired/verified flags, observed action, reason, crop path. Callers
        submit through the engine's io pool (single SQLite writer)."""
        target = entry.get("target") or (None, None)
        with self._conn() as con:
            con.execute(
                "INSERT INTO executor_log (ts, session_id, round, mode,"
                " phase, action, seat, target_x, target_y, confidence,"
                " fired, verified, observed, reason, crop)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), self.session_id, entry.get("round"),
                 entry.get("mode"), entry.get("phase"), entry.get("action"),
                 entry.get("seat"), target[0], target[1],
                 entry.get("confidence"),
                 int(bool(entry.get("fired"))),
                 (None if entry.get("verified") is None
                  else int(bool(entry.get("verified")))),
                 entry.get("observed"), entry.get("reason"),
                 entry.get("crop")))

    def update_executor_verify(self, round_number, phase_name, action,
                               verified, observed=None, seat=None):
        """Stamp the verification outcome onto the latest matching FIRED
        row (the click is recorded when confirmed; the outcome lands
        seconds later). fired=1 + the seat predicate keep a newer
        merely-planned row for the same action from absorbing the stamp."""
        with self._conn() as con:
            row = con.execute(
                "SELECT id FROM executor_log WHERE session_id = ? AND"
                " round = ? AND phase = ? AND action = ? AND fired = 1"
                " AND (seat = ? OR (seat IS NULL AND ? IS NULL))"
                " ORDER BY id DESC LIMIT 1",
                (self.session_id, round_number, phase_name, action,
                 seat, seat)).fetchone()
            if row is None:
                return
            con.execute(
                "UPDATE executor_log SET verified = ?, observed = ?"
                " WHERE id = ?",
                (int(bool(verified)), observed, row[0]))

    def executor_stats(self, session_only: bool = True) -> dict:
        """Ghost/assist telemetry: the honest gate before arming assist
        (AUTONOMY_PLAN §6). verified=NULL rows are still-pending or
        never-judged plans and stay out of the match rate."""
        where = "WHERE session_id = ?" if session_only else ""
        args = (self.session_id,) if session_only else ()
        with self._conn() as con:
            plans, fired, judged, matched = con.execute(
                # A confirmed decision writes a planned row AND a fired row;
                # counting fired rows as plans would double-count it.
                f"SELECT COALESCE(SUM(fired = 0), 0), COALESCE(SUM(fired), 0),"
                f" COUNT(verified), COALESCE(SUM(verified), 0)"
                f" FROM executor_log {where}", args).fetchone()
        return {"plans": plans, "fired": fired, "judged": judged,
                "matched": matched,
                "match_rate": matched / judged if judged else None}

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

    def seat_stats(self, session_only: bool = True) -> dict:
        """Per-seat accuracy drilldown from stored rounds. Pure DB read.

        Returns {seat_index: {"hands", "book_pct", "avg_units",
        "divergences"}}: settled hands counted from the settlement JSON,
        book-played % by replaying each settled hand through
        seat_quality.follows_book (the established measure — None until a
        hand was judgeable), average net units per settled hand, and the
        number of rounds whose stored book_action and optimal_action codes
        (both present) disagree. A compound CSV code ('R/H') agrees when
        the optimal action matches ANY of its alternatives: the snapshot's
        '≠ book' marker resolves the compound against the actions actually
        available under the rules (surrender off, 3+ cards), and the stored
        raw code does not say which alternative that was."""
        where = "WHERE session_id = ?" if session_only else ""
        args = (self.session_id,) if session_only else ()
        with self._conn() as con:
            rows = con.execute(
                f"SELECT dealer_card, seats, settlement FROM rounds {where}",
                args).fetchall()
        advisor = StrategyAdvisor()
        agg = {}

        def tally(idx):
            return agg.setdefault(idx, {"hands": 0, "units": 0.0, "book_n": 0,
                                        "book_yes": 0, "divergences": 0})

        for dealer, seats_json, settle_json in rows:
            try:
                seats = json.loads(seats_json) or []
            except (TypeError, ValueError):
                seats = []
            for entry in seats:
                book = entry.get("book_action")
                opt = entry.get("optimal_action")
                pairs = zip(book if isinstance(book, list) else [book],
                            opt if isinstance(opt, list) else [opt])
                if any(b and o and o not in str(b).split("/") for b, o in pairs):
                    tally(entry["index"])["divergences"] += 1
            if not settle_json:
                continue
            try:
                settle = json.loads(settle_json)
            except (TypeError, ValueError):
                continue
            if not settle:
                continue
            verdicts = seat_quality.score_settled_round(
                settle, seats, dealer, advisor)
            for idx, hand_verdicts in verdicts.items():
                t = tally(idx)
                t["book_n"] += len(hand_verdicts)
                t["book_yes"] += sum(hand_verdicts)
            for seat_entry in settle.get("seats", []):
                t = tally(seat_entry["index"])
                hands = seat_entry.get("hands", [])
                t["hands"] += len(hands)
                t["units"] += sum(h.get("units") or 0.0 for h in hands)
        return {idx: {
            "hands": t["hands"],
            "book_pct": t["book_yes"] / t["book_n"] if t["book_n"] else None,
            "avg_units": t["units"] / t["hands"] if t["hands"] else None,
            "divergences": t["divergences"],
        } for idx, t in sorted(agg.items())}

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

    def pre_deal_tcs(self, session_only=False):
        """Every recorded round's true count. The stored count is taken at
        round END (after the round's own cards), which makes each row the
        PRE-DEAL count of the round that followed — exactly the between-
        rounds distribution the ramp designer must weight bets by."""
        where = ("WHERE true_count IS NOT NULL"
                 + (" AND session_id = ?" if session_only else ""))
        args = (self.session_id,) if session_only else ()
        with self._conn() as con:
            rows = con.execute(f"SELECT true_count FROM rounds {where}",
                               args).fetchall()
        return [r[0] for r in rows]

    def rounds_per_hour(self, session_only=False, max_gap_s=900.0):
        """Gap-aware measured table pace from round timestamps (idle gaps
        over max_gap_s don't count as play time). None until 10+ intervals
        exist — callers fall back to a nominal pace."""
        where = "WHERE session_id = ?" if session_only else ""
        args = (self.session_id,) if session_only else ()
        with self._conn() as con:
            rows = con.execute(
                f"SELECT session_id, ts FROM rounds {where} ORDER BY id",
                args).fetchall()
        total_s = 0.0
        intervals = 0
        for (prev_sid, prev_ts), (sid, ts) in zip(rows, rows[1:]):
            dt = ts - prev_ts
            if sid == prev_sid and 0 < dt <= max_gap_s:
                total_s += dt
                intervals += 1
        if intervals < 10 or total_s <= 0:
            return None
        return 3600.0 * intervals / total_s

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
                " insurance, side_bets, settlement, pnl_units, pnl_eur, bet_eur,"
                " paytable_hash, bet_suggested, bet_sit_out, edge_exact"
                " FROM rounds ORDER BY id").fetchall()
        header = ["timestamp", "session", "round", "running_count", "true_count",
                  "decks_remaining", "cards_seen", "dealer_card", "dealer_extras",
                  "seats", "insurance", "side_bets", "settlement", "pnl_units",
                  "pnl_eur", "bet_eur", "paytable_hash", "bet_suggested",
                  "bet_sit_out", "edge_exact"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        return len(rows)
