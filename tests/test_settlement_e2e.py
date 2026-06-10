"""Settlement end-to-end (Test-E2E): a scripted multi-round shoe of synthetic
per-frame model predictions driven through the REAL ingest path
(_process_dealer / _process_players / _maybe_auto_new_round), respecting every
confirmation gate:

  * initial deal locks immediately (one card per seat per cycle while < 2)
  * hits confirm across EXTRA_CARD_CONFIRM_CYCLES consecutive cycles
  * the dealer up-card confirms across DEALER_CONFIRM_FRAMES agreeing frames
  * dealer playout draws confirm like hits
  * the auto round reset needs EMPTY_FRAMES_FOR_RESET consecutive empty
    player frames AND the dealer-area-empty recheck over the same window

Three rounds are replayed on one continuous shoe: (a) a plain win + lose
two-seat round, (b) a player natural paying 3:2, (c) a dealer bust round
whose reset is first held by a lingering dealer-area card. Seat A is marked
as mine with a bet placed so the session P&L books; outcomes are asserted
from the live engine state and from the persisted rounds in the store.

Run:  .venv\\Scripts\\python -m pytest tests\\test_settlement_e2e.py -q
"""

import json
import shutil
import sqlite3
import tempfile
import types
import unittest
from pathlib import Path

from lib.common import constants
from lib.common.card_mappings import DEALER_CLASS_MAP, PLAYER_CLASS_MAP
from lib.logic.engine import DetectionEngine
from lib.logic.session_store import SessionStore

# Full card name -> player-model class ("10 of Hearts" -> "a10"); keep the
# canonical class when several classes map to one card (the "b87" stray).
NAME_TO_CLASS = {}
for _cls, _name in PLAYER_CLASS_MAP.items():
    NAME_TO_CLASS.setdefault(_name, _cls)
RANK_TO_DEALER_CLASS = {rank: cls for cls, rank in DEALER_CLASS_MAP.items()}

# Dealer-crop coordinates for the up-card and the playout draws, spaced
# beyond SAME_CARD_DISTANCE_PX so position dedupe can never merge them.
DEALER_UP_XY = (480.0, 300.0)
DEALER_DRAW_XY = [(640.0, 300.0), (800.0, 300.0)]


def player_pred(name, point, confidence):
    """One synthetic player-model prediction (normalized models.py shape)."""
    return {"cx": float(point[0]), "cy": float(point[1]), "width": 60.0,
            "height": 85.0, "class": NAME_TO_CLASS[name],
            "confidence": confidence}


def dealer_pred(rank, point=DEALER_UP_XY, confidence=0.9):
    """One synthetic dealer-model prediction (bare-rank classes)."""
    return {"cx": float(point[0]), "cy": float(point[1]), "width": 60.0,
            "height": 85.0, "class": RANK_TO_DEALER_CLASS[rank],
            "confidence": confidence}


class SettlementE2E(unittest.TestCase):
    A, B = 0, 1          # seat A is mine, seat B is another player
    BET = 50.0

    def setUp(self):
        # Test-environment config: no exact pre-deal sweeps (~15 s each), no
        # side-bet EV jobs, no bankroll/settings.json writes on settle.
        # Settlement, the round lifecycle and P&L depend on none of these.
        self._betting = {k: constants.BETTING[k]
                         for k in ("use_exact_edge", "auto_bankroll")}
        self._sidebets = {k: cfg["enabled"]
                          for k, cfg in constants.SIDE_BETS.items()}
        constants.BETTING["use_exact_edge"] = 0
        constants.BETTING["auto_bankroll"] = 0
        for cfg in constants.SIDE_BETS.values():
            cfg["enabled"] = False

        self.entries = []    # (message, level) from the engine logger
        self.eng = DetectionEngine(
            log=lambda msg, level=None: self.entries.append((msg, level)))
        tmp = Path(tempfile.mkdtemp(prefix="bjai_settlement_e2e_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.eng.store = SessionStore(tmp / "session.db")
        # Real region setup: at 2560x1440 this loads the calibrated profile
        # when one exists, else the scaled defaults — exactly like production.
        self.eng.set_monitor(
            types.SimpleNamespace(x=0, y=0, width=2560, height=1440))
        self.eng.set_my_seat(self.A, True)
        self.eng.set_bet_placed(self.BET)
        self.pa = self._seat_points(self.A, 3)
        self.pb = self._seat_points(self.B, 3)

    def tearDown(self):
        constants.BETTING.update(self._betting)
        for key, enabled in self._sidebets.items():
            constants.SIDE_BETS[key]["enabled"] = enabled

    # ------------------------------------------------------------- helpers

    def _seat_points(self, seat_idx, n):
        """n points inside the live seat polygon, pairwise farther apart than
        the same-card dedupe radius, found with the engine's own
        _seat_for_point so any region source (calibrated or default) works."""
        region = self.eng.regions[seat_idx]
        left, top, right, bottom = region.bounds
        gap = constants.SAME_CARD_DISTANCE_PX * self.eng._dist_scale + 10.0
        pts = []
        y = top
        while y <= bottom and len(pts) < n:
            x = left
            while x <= right and len(pts) < n:
                if (self.eng._seat_for_point(x, y) == seat_idx
                        and all((x - px) ** 2 + (y - py) ** 2 >= gap * gap
                                for px, py in pts)):
                    pts.append((float(x), float(y)))
                x += 12.0
            y += 12.0
        self.assertEqual(len(pts), n,
                         f"could not place {n} spaced cards in seat {seat_idx}")
        return pts

    def logged(self, fragment, level=None):
        return [(m, lv) for m, lv in self.entries
                if fragment in m and (level is None or lv == level)]

    def frame(self, dealer_preds, player_preds):
        """One detection cycle: _detect's locked section + run_cycle's
        snapshot publish, with synthetic predictions instead of inference."""
        eng = self.eng
        with eng._lock:
            eng._process_dealer(list(dealer_preds))
            eng._process_players(list(player_preds))
            eng._maybe_auto_new_round(list(player_preds))
        eng.publish_snapshot()

    def deal(self, up_pred, deal_preds):
        """Feed the initial-deal frames: both seats' first two cards plus the
        dealer up-card, until the up-card confirms."""
        eng = self.eng
        self.frame([up_pred], deal_preds)
        # Initial deal locks immediately, but only one card per seat/cycle.
        for idx in (self.A, self.B):
            self.assertEqual(len(eng.seats[idx].cards), 1)
        if constants.DEALER_CONFIRM_FRAMES > 1:
            self.assertFalse(eng.dealer_locked)
        for _ in range(max(2, constants.DEALER_CONFIRM_FRAMES) - 1):
            self.frame([up_pred], deal_preds)
        for idx in (self.A, self.B):
            self.assertEqual(len(eng.seats[idx].cards), 2)
        self.assertTrue(eng.dealer_locked)
        with eng._lock:
            self.assertEqual(eng._activity(), "complete")

    def hit(self, seat_idx, hit_pred, dealer_preds, table_preds):
        """A hit card appears: it must survive EXTRA_CARD_CONFIRM_CYCLES
        consecutive cycles before it locks."""
        eng = self.eng
        before = len(eng.seats[seat_idx].cards)
        self.frame(dealer_preds, table_preds + [hit_pred])
        for _ in range(constants.EXTRA_CARD_CONFIRM_CYCLES - 1):
            self.assertEqual(len(eng.seats[seat_idx].cards), before,
                             "hit locked before its confirmation window")
            self.frame(dealer_preds, table_preds + [hit_pred])
        self.assertEqual(len(eng.seats[seat_idx].cards), before + 1)

    def dealer_draws(self, ranks, up_pred, table_preds):
        """Dealer playout cards appear after the up-card lock; they confirm
        across EXTRA_CARD_CONFIRM_CYCLES cycles like player hits."""
        eng = self.eng
        before = len(eng.dealer_extras)
        dealer_frame = [up_pred] + [dealer_pred(r, DEALER_DRAW_XY[i])
                                    for i, r in enumerate(ranks)]
        self.frame(dealer_frame, table_preds)
        for _ in range(constants.EXTRA_CARD_CONFIRM_CYCLES - 1):
            self.assertEqual(len(eng.dealer_extras), before,
                             "dealer draw locked before its window")
            self.frame(dealer_frame, table_preds)
        self.assertEqual(len(eng.dealer_extras), before + len(ranks))
        self.assertEqual(sorted(c["rank"] for c in eng.dealer_extras[before:]),
                         sorted(ranks))

    def clear_table(self, lingering_up=None):
        """Drive the auto-reset gate: EMPTY_FRAMES_FOR_RESET empty player
        frames with the dealer area clear for the same stretch. With
        `lingering_up` the dealer model keeps seeing the up-card for two
        frames first — the recheck must hold the round through them."""
        eng = self.eng
        start = eng.round_number
        if lingering_up is not None:
            for _ in range(2):
                self.frame([lingering_up], [])
            self.assertEqual(eng.round_number, start,
                             "reset fired while the dealer area was occupied")
        for _ in range(constants.EMPTY_FRAMES_FOR_RESET - 1):
            self.frame([], [])
            self.assertEqual(eng.round_number, start,
                             "reset fired before the empty-frame window")
        self.frame([], [])
        self.assertEqual(eng.round_number, start + 1, "auto reset did not fire")
        self.assertTrue(all(not s.cards for s in eng.seats))
        self.assertIsNone(eng.dealer_card)
        self.assertEqual(eng.dealer_extras, [])

    # ---------------------------------------------------------------- test

    def test_scripted_shoe_settles_three_rounds_and_books_pnl(self):
        eng, A, B, pa, pb = self.eng, self.A, self.B, self.pa, self.pb

        # ---- Round 1 (a): plain win (mine) + lose -----------------------
        deal1 = [
            player_pred("10 of Hearts", pa[0], 0.98),
            player_pred("King of Spades", pa[1], 0.92),
            player_pred("10 of Clubs", pb[0], 0.97),
            player_pred("2 of Diamonds", pb[1], 0.91),
        ]
        up1 = dealer_pred("King")
        self.deal(up1, deal1)
        # Higher-confidence card locked first; full names via PLAYER_CLASS_MAP.
        self.assertEqual([c["name"] for c in eng.seats[A].cards],
                         ["10 of Hearts", "King of Spades"])
        self.assertEqual([c["name"] for c in eng.seats[B].cards],
                         ["10 of Clubs", "2 of Diamonds"])
        self.assertEqual(eng.dealer_card, "King")
        snap = eng.get_snapshot()
        self.assertTrue(snap["seats"][A]["mine"])
        self.assertFalse(snap["seats"][B]["mine"])

        # Seat B hits a 5 -> 17; dealer draws a 9 -> 19.
        self.hit(B, player_pred("5 of Hearts", pb[2], 0.90), [up1], deal1)
        table1 = deal1 + [player_pred("5 of Hearts", pb[2], 0.90)]
        self.dealer_draws(["9"], up1, table1)

        self.clear_table()
        # Seat A: 20 vs 19 wins (+1u, mine, EUR 50); seat B: 17 loses.
        self.assertEqual(eng.session_pnl,
                         {"units": 1.0, "eur": 50.0, "rounds": 1})

        # ---- Round 2 (b): my natural pays 3:2 ----------------------------
        deal2 = [
            player_pred("Ace of Hearts", pa[0], 0.98),
            player_pred("King of Diamonds", pa[1], 0.92),
            player_pred("9 of Clubs", pb[0], 0.97),
            player_pred("9 of Diamonds", pb[1], 0.91),
        ]
        up2 = dealer_pred("7")
        self.deal(up2, deal2)
        self.dealer_draws(["10"], up2, deal2)  # dealer 7 + 10 = 17

        self.clear_table()
        # Natural 21 vs dealer 17: +1.5u at EUR 50 = +75; B's 18 also won
        # but is not mine, so only the blackjack books.
        self.assertEqual(eng.session_pnl,
                         {"units": 2.5, "eur": 125.0, "rounds": 2})

        # ---- Round 3 (c): dealer busts; reset held by lingering up-card --
        deal3 = [
            player_pred("8 of Hearts", pa[0], 0.98),
            player_pred("4 of Clubs", pa[1], 0.92),
            player_pred("10 of Spades", pb[0], 0.97),
            player_pred("8 of Diamonds", pb[1], 0.91),
        ]
        up3 = dealer_pred("6")
        self.deal(up3, deal3)
        self.hit(A, player_pred("5 of Diamonds", pa[2], 0.90), [up3], deal3)
        table3 = deal3 + [player_pred("5 of Diamonds", pa[2], 0.90)]
        self.dealer_draws(["10", "9"], up3, table3)  # 6+10+9 = 25, bust

        self.assertEqual(self.logged("delayed"), [])
        self.clear_table(lingering_up=up3)
        # Both seats beat the bust; only seat A is mine.
        self.assertEqual(eng.session_pnl,
                         {"units": 3.5, "eur": 175.0, "rounds": 3})
        # The dealer-area recheck held the reset and said so once, at INFO.
        delays = self.logged("delayed")
        self.assertEqual(len(delays), 1)
        self.assertIn(delays[0][1], (None, "INFO"))

        # ---- Whole-shoe invariants --------------------------------------
        self.assertEqual(eng.round_number, 4)
        # 7 + 6 + 8 cards crossed the table; each counted exactly once.
        self.assertEqual(eng.counter.cards_seen, 21)
        self.assertEqual(len(self.logged("Table cleared", level="WARNING")), 3)
        self.assertEqual(len(self.logged(" settled (dealer ")), 3)
        snap = eng.get_snapshot()
        self.assertEqual(snap["session_pnl"],
                         {"units": 3.5, "eur": 175.0, "rounds": 3})
        self.assertEqual(snap["round"], 4)

        # ---- Persisted rounds: settlement details + P&L columns ----------
        self.assertTrue(eng.flush_advice(timeout=240))
        with sqlite3.connect(eng.store.path) as con:
            rows = con.execute(
                "SELECT round_number, seats, settlement, pnl_units, pnl_eur,"
                " bet_eur FROM rounds ORDER BY id").fetchall()
        self.assertEqual([r[0] for r in rows], [1, 2, 3])
        self.assertEqual([r[3] for r in rows], [1.0, 1.5, 1.0])   # my units
        self.assertEqual([r[4] for r in rows], [50.0, 75.0, 50.0])
        self.assertEqual([r[5] for r in rows], [self.BET] * 3)

        s1, s2, s3 = (json.loads(r[2]) for r in rows)

        # (a) plain win + lose against dealer 19.
        self.assertEqual(s1["dealer_total"], 19)
        self.assertFalse(s1["dealer_bj"])
        by1 = {s["index"]: s for s in s1["seats"]}
        self.assertEqual(by1[A]["net_units"], 1.0)
        self.assertEqual(by1[A]["hands"][0]["outcome"], "win")
        self.assertEqual(by1[B]["net_units"], -1.0)
        self.assertEqual(by1[B]["hands"][0]["outcome"], "lose")
        self.assertEqual(by1[B]["hands"][0]["cards"],
                         ["10 of Clubs", "2 of Diamonds", "5 of Hearts"])

        # (b) natural pays 3:2 against dealer 17.
        self.assertEqual(s2["dealer_total"], 17)
        by2 = {s["index"]: s for s in s2["seats"]}
        self.assertEqual(by2[A]["hands"][0]["outcome"], "blackjack")
        self.assertEqual(by2[A]["hands"][0]["units"], 1.5)
        self.assertEqual(by2[A]["net_units"], 1.5)
        self.assertEqual(by2[B]["hands"][0]["outcome"], "win")

        # (c) dealer bust pays every live hand.
        self.assertEqual(s3["dealer_total"], 25)
        by3 = {s["index"]: s for s in s3["seats"]}
        self.assertEqual(by3[A]["hands"][0]["outcome"], "win")
        self.assertEqual(by3[A]["hands"][0]["cards"],
                         ["8 of Hearts", "4 of Clubs", "5 of Diamonds"])
        self.assertEqual(by3[B]["hands"][0]["outcome"], "win")

        # The recorded seats JSON kept the ownership flag and the cards.
        seats2 = {s["index"]: s for s in json.loads(rows[1][1])}
        self.assertTrue(seats2[A]["mine"])
        self.assertEqual(seats2[A]["cards"],
                         ["Ace of Hearts", "King of Diamonds"])

        # Store-level aggregation sees the same three rounds.
        stats = eng.store.stats(session_only=True)
        self.assertEqual(stats["settled_rounds"], 3)
        self.assertEqual(stats["net_units"], 3.5)
        self.assertEqual(stats["net_eur"], 175.0)
        self.assertEqual(stats["hand_outcomes"],
                         {"win": 4, "lose": 1, "push": 0, "blackjack": 1})


if __name__ == "__main__":
    unittest.main()
