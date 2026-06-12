"""Leak finder & coaching report (V3 Feature 6). Pure logic, no Tkinter.

Mines session.db into a ranked list of what the user's OWN mistakes actually
cost, in units per 100 owned rounds, by class:

* Play errors — owned (★) seats' settled hands replayed draw-by-draw
  against the book + the index deviations live at the recorded count
  (the seat_quality trick, pointed at your own seats and upgraded to
  return the divergence instead of a bool), costed by the exact engine
  on a shoe reconstructed at the recorded true count, grouped by pattern
  ("16 vs 10: stood — book says hit").
* Bet-discipline leaks — the bet placed (OCR/GUI) vs the ramp suggestion
  for the SAME round (the stored count of row N-1 is round N's pre-deal
  count; newly persisted bet_suggested columns remove the reconstruction
  for rounds recorded from now on).
* -EV side-bet habit — stakes settled at side bets whose pre-deal EV said
  no (settlement carries the frozen stakes; the prior row's side_bets
  column carries the EVs that priced them).
* Missed sit-outs — owned rounds played when the suggestion was to sit out.

Insurance errors are NOT mined: the user's actual insurance action is
recorded nowhere (the insurance column stores table-level advice only).

Two-step API because EV costing needs subprocesses: find_leaks() is pure
SQL/JSON and safe on any thread; cost_play_leaks() prices the play-error
groups via ev_offload (the dedicated "analysis" pool — never the live
advice or predeal workers) and must be called OFF the Tk thread.

Known staleness, accepted and labeled rather than hidden: the persisted
bet_suggested/edge_exact are snapshot values at round END, and the exact
edge in them comes from the sweep that ran in the PREVIOUS between-rounds
window — when the exact-edge model is on, the suggestion a row carries
can be one sweep older than the bet call the player actually saw. The
honest fix is freezing the suggestion at the round's first card
(the V4 discipline-guard design); until then, bet-discipline numbers are
directionally right and exact only for the linear-TC model.
"""

import html
import json
import math

from ..common import constants
from . import betting, cards, deviations, ev_engine, ev_offload, trainer
from .bankroll import certainty_equivalent
from .strategy import StrategyAdvisor

#: At most this many exact-EV evaluations per costing pass — the rest of a
#: pattern's examples reuse the pattern's mean cost.
_MAX_EV_JOBS = 60

ACTION_NAMES = {"S": "stood", "H": "hit", "D": "doubled", "P": "split",
                "R": "surrendered", "no-split": "didn't split",
                "no-double": "didn't double"}
EXPECT_NAMES = {"S": "stand", "H": "hit", "D": "double", "P": "split",
                "R": "surrender"}


# ----------------------------------------------------------- play replay

def _book_primary(action, current_len):
    """Resolve a compound CSV code to the playable primary, mirroring
    seat_quality (no surrender online; D needs exactly two cards)."""
    parts = action.split("/")
    primary = parts[0]
    if primary == "R":
        primary = parts[1] if len(parts) > 1 else "S"
    if primary == "D" and current_len > 2:
        primary = parts[1] if len(parts) > 1 else "H"
    return primary


def replay_divergence(card_names, dealer_rank, advisor, tc=0.0,
                      post_split=False):
    """The first point where the hand's draw sequence diverges from the
    book + live index play, or None when it can't be judged / didn't.

    Returns {"cards_at", "expected", "played", "hand_key", "dealer"}.
    Same charitable semantics as seat_quality.follows_book: doubles are
    indistinguishable from single hits (not flagged), impossible
    post-bust cards abstain (misread, not a mistake)."""
    hand = [c for c in card_names if c and c != "-"]
    if len(hand) < 2 or not dealer_rank:
        return None
    if post_split and (cards.rank_of(hand[0]) == "Ace"
                       and not constants.RULES["hit_split_aces"]):
        return None
    dealer = cards.dealer_strategy_rank(dealer_rank)
    current = hand[:2]
    taken = 2
    while True:
        total = cards.hand_value(current)
        if total >= 21:
            return None  # bust/21 mid-draw, or extras = misread; abstain
        action, _, _ = advisor.advice(current, dealer_rank,
                                      post_split=post_split)
        if action is None:
            return None
        expected = _book_primary(action, len(current))
        if dealer and not post_split:
            # Index plays apply at every decision point exactly like the
            # live advice line (engine passes two_cards=len(hand)==2 —
            # D/P deviations drop on 3+ cards, stand/hit ones stay).
            dev = deviations.index_advice(cards.hand_key(current), dealer,
                                          tc, two_cards=len(current) == 2)
            if dev is not None:
                expected = dev["action"]
                if expected == "R":  # unoffered online — book fallback
                    expected = _book_primary(action, len(current))

        def divergence(played):
            return {"cards_at": list(current), "expected": expected,
                    "played": played, "hand_key": cards.hand_key(current),
                    "dealer": dealer or str(dealer_rank)}

        if expected == "P":
            return divergence("no-split")  # split seats never reach P rows
        if expected == "D":
            if len(hand) == 2:
                return divergence("S")  # stood on a double spot
            if len(hand) > 3:
                return divergence("H")  # drew past the double's one card
            return None  # exactly 3 cards: double or single hit — charitable
        if expected == "S":
            if taken < len(hand):
                return divergence("H")
            return None
        # expected == "H"
        if taken >= len(hand):
            return divergence("S")
        current = current + [hand[taken]]
        taken += 1


# --------------------------------------------------------------- mining

def _owned_settled_hands(row, advisor):
    """Yield (hand_cards, post_split, tc) for the user's settled hands."""
    seats_snap = row["seats"] or []
    settle = row["settlement"]
    if not settle:
        return
    if settle.get("dealer_bj") and constants.RULES["peek"]:
        return
    mine = {s["index"]: s for s in seats_snap if s.get("mine")}
    for seat_entry in settle.get("seats", []):
        snap = mine.get(seat_entry["index"])
        if snap is None:
            continue
        split = bool(snap.get("split"))
        for hand in seat_entry.get("hands", []):
            yield hand.get("cards") or [], split


def _rows(store, session_only):
    where = "WHERE session_id = ?" if session_only else ""
    args = (store.session_id,) if session_only else ()
    with store._conn() as con:
        cols = [r[1] for r in con.execute("PRAGMA table_info(rounds)")]
        has_sugg = "bet_suggested" in cols
        select = ("SELECT session_id, true_count, dealer_card, seats,"
                  " settlement, side_bets, bet_eur, pnl_eur, cards_seen"
                  + (", bet_suggested, bet_sit_out, edge_exact" if has_sugg
                     else ", NULL, NULL, NULL")
                  + f" FROM rounds {where} ORDER BY id")
        raw = con.execute(select, args).fetchall()
    rows = []
    for r in raw:
        def js(text):
            try:
                return json.loads(text) if text else None
            except ValueError:
                return None
        rows.append({"session_id": r[0], "true_count": r[1],
                     "dealer_card": r[2], "seats": js(r[3]),
                     "settlement": js(r[4]), "side_bets": js(r[5]),
                     "bet_eur": r[6], "pnl_eur": r[7], "cards_seen": r[8],
                     "bet_suggested": r[9], "bet_sit_out": r[10],
                     "edge_exact": r[11]})
    return rows


def _chained(prev, row):
    """True when row pairs with prev for the one-row count lag: same app
    session AND same shoe (a cards_seen drop between rows is the only
    persisted shoe-reset marker — across it the previous row's count says
    nothing about this round)."""
    if prev is None or prev["session_id"] != row["session_id"]:
        return False
    if prev["cards_seen"] is None or row["cards_seen"] is None:
        return True
    return row["cards_seen"] >= prev["cards_seen"]


def find_leaks(store, session_only=False):
    """Mine the rounds table. Pure SQL/JSON — no EV engine, any thread.

    Play-error groups come back with cost_units=None; price them with
    cost_play_leaks() off the Tk thread."""
    rows = _rows(store, session_only)
    advisor = StrategyAdvisor()
    play_groups = {}
    bet = {"overbet_risk_ce": 0.0, "underbet_ev": 0.0, "overbet_neg_ev": 0.0,
           "missed_sit_outs": 0, "sit_out_cost": 0.0, "compared": 0,
           "reconstructed": 0, "overbet_rounds": 0, "underbet_rounds": 0}
    side = {}
    owned_rounds = 0

    prev = None
    for row in rows:
        # ---- play errors (this row's own settled hands; the row's TC is
        # post-round — the PRE-deal count lives on the previous row)
        chained = _chained(prev, row)
        pre_tc = (prev["true_count"] if chained else 0.0) or 0.0
        owned = row["pnl_eur"] is not None
        if owned:
            owned_rounds += 1
        for hand_cards, split in _owned_settled_hands(row, advisor):
            div = replay_divergence(hand_cards, row["dealer_card"], advisor,
                                    tc=pre_tc, post_split=split)
            if div is None:
                continue
            key = (div["hand_key"], div["dealer"], div["expected"],
                   div["played"])
            group = play_groups.setdefault(key, {
                "hand_key": div["hand_key"], "dealer": div["dealer"],
                "expected": div["expected"], "played": div["played"],
                "count": 0, "examples": [], "cost_units": None,
                "pattern": (f"{div['hand_key']} vs {div['dealer']}: "
                            f"{ACTION_NAMES.get(div['played'], div['played'])}"
                            f" — play says "
                            f"{EXPECT_NAMES.get(div['expected'], div['expected'])}"),
            })
            group["count"] += 1
            if len(group["examples"]) < 8:
                group["examples"].append({"cards": div["cards_at"],
                                          "dealer": row["dealer_card"],
                                          "tc": pre_tc, "split": split})

        # ---- bet discipline / sit-outs (needs the previous row's count)
        if owned and chained:
            placed = row["bet_eur"]
            if placed and placed > 0:
                if prev["bet_suggested"] is not None:
                    sugg = float(prev["bet_suggested"])
                    sit = bool(prev["bet_sit_out"])
                    edge = prev["edge_exact"]
                    if edge is None:
                        edge = betting.estimate_edge(prev["true_count"] or 0.0)
                else:
                    s = betting.suggest(prev["true_count"] or 0.0)
                    sugg, sit, edge = s["bet"], s["sit_out"], s["edge"]
                    bet["reconstructed"] += 1
                bet["compared"] += 1
                if sit:
                    bet["missed_sit_outs"] += 1
                    bet["sit_out_cost"] += placed * max(0.0, -edge)
                else:
                    tol = max(1.0, 0.05 * sugg)
                    delta = placed - sugg
                    if delta > tol:
                        bet["overbet_rounds"] += 1
                        if edge > 0:
                            bank = constants.BETTING["bankroll"]
                            v = constants.BETTING["variance"]
                            ce = (certainty_equivalent(sugg * edge,
                                                       math.sqrt(v) * sugg, bank)
                                  - certainty_equivalent(placed * edge,
                                                         math.sqrt(v) * placed,
                                                         bank))
                            bet["overbet_risk_ce"] += max(0.0, ce)
                        else:
                            bet["overbet_neg_ev"] += delta * -edge
                    elif delta < -tol and edge > 0:
                        bet["underbet_rounds"] += 1
                        bet["underbet_ev"] += -delta * edge

        # ---- -EV side-bet stakes (frozen stakes in settlement; EVs from
        # the previous row's side_bets column — the prices they were
        # placed against)
        settle = row["settlement"] or {}
        sb_settle = settle.get("side_bets") or {}
        if sb_settle and chained:
            evs = {item.get("key"): item.get("ev")
                   for item in (prev["side_bets"] or [])}
            for seat_bets in sb_settle.values():
                for key, entry in (seat_bets or {}).items():
                    stake = float(entry.get("stake") or 0.0)
                    ev = evs.get(key)
                    if stake > 0 and ev is not None and ev <= 0:
                        agg = side.setdefault(key, {
                            "label": entry.get("label") or key,
                            "count": 0, "staked": 0.0, "cost_eur": 0.0})
                        agg["count"] += 1
                        agg["staked"] += stake
                        agg["cost_eur"] += stake * -ev
        prev = row

    return {
        "rounds": len(rows),
        "owned_rounds": owned_rounds,
        "play": sorted(play_groups.values(),
                       key=lambda g: -g["count"]),
        "bets": bet,
        "side_bets": dict(sorted(side.items(),
                                 key=lambda kv: -kv[1]["cost_eur"])),
    }


# ------------------------------------------------------------ EV costing

def _cost_one(example, expected, played):
    """Exact EV given up by one recorded divergence, in units of the bet.
    Composition is reconstructed at the recorded TC (per-round shoe
    composition is not persisted) — same synthesis the trainer grades with."""
    hand = tuple(sorted(ev_engine.card_index(c) for c in example["cards"]))
    dealer = ev_engine.card_index(example["dealer"])
    comp = trainer._comp_for_tc(example["tc"], list(hand) + [dealer])
    rules = ev_engine.current_rules()
    # Dedicated pool: costing must never queue behind (or in front of)
    # the live advice worker or the betting-window sweep.
    result = ev_offload.run("analysis", ev_engine.evaluate, hand, dealer,
                            comp, rules, bool(example.get("split")))
    evs = result["evs"]
    if expected == "P" or played == "no-split":
        if "P" not in evs:
            return None
        alt = max(v for k, v in evs.items() if k != "P")
        return max(0.0, evs["P"] - alt)
    exp_ev = evs.get(expected)
    played_ev = evs.get(played)
    if exp_ev is None and expected == "D":
        exp_ev = evs.get("H")  # double unavailable in this state: best hit
    if exp_ev is None or played_ev is None:
        return None
    return max(0.0, exp_ev - played_ev)


def cost_play_leaks(play_groups, abort=None):
    """Fill cost_units on each play-error group (mean exact-EV loss per
    occurrence x count). BLOCKS on ev_offload — never the Tk thread.
    `abort` (callable -> bool) is checked between EV jobs so a closed
    window stops the pass instead of grinding orphaned subprocess work."""
    if not play_groups:
        return play_groups
    per_group = max(1, _MAX_EV_JOBS // len(play_groups))
    for group in play_groups:
        costs = []
        for example in group["examples"][:per_group]:
            if abort is not None and abort():
                return play_groups
            try:
                cost = _cost_one(example, group["expected"], group["played"])
            except Exception:
                cost = None
            if cost is not None:
                costs.append(cost)
        if costs:
            group["cost_per_error"] = sum(costs) / len(costs)
            group["cost_units"] = group["cost_per_error"] * group["count"]
    play_groups.sort(key=lambda g: -(g["cost_units"] or 0.0))
    return play_groups


# ------------------------------------------------------------ drill deck

def drill_items(result, cap=40):
    """Replay-trainer items from the worst play leaks (two-card decision
    states only — the replay drill quizzes the two-card spot). Weighted by
    severity: each pattern appears once per recorded occurrence, worst
    pattern first, capped."""
    items = []
    for group in result["play"]:
        for example in group["examples"]:
            if len(example["cards"]) != 2 or example.get("split"):
                continue
            reps = max(1, math.ceil(group["count"]
                                    / max(1, len(group["examples"]))))
            for _ in range(reps):
                items.append({"cards": list(example["cards"]),
                              "dealer": example["dealer"],
                              "tc": example["tc"], "optimal_text": ""})
                if len(items) >= cap:
                    return items
    return items


# ----------------------------------------------------------- HTML report

def report_html(result, session_label="all sessions"):
    """Self-contained coaching report. Pure string building."""
    owned = result["owned_rounds"] or 0
    per100 = 100.0 / owned if owned else 0.0

    def fmt_u(x):
        return f"−{abs(x):.2f}u" if x else "0u"

    rows = []
    for g in result["play"]:
        cost = g.get("cost_units")
        rows.append(
            "<tr><td>play</td>"
            f"<td>{html.escape(g['pattern'])}</td>"
            f"<td>{g['count']}×</td>"
            f"<td>{fmt_u(cost) if cost is not None else 'n/a'}</td>"
            f"<td>{fmt_u(cost * per100) if cost is not None and owned else '—'}"
            "</td></tr>")
    b = result["bets"]
    bet_lines = []
    if b["missed_sit_outs"]:
        bet_lines.append(f"{b['missed_sit_outs']} rounds played at a sit-out "
                         f"call — expected loss €{b['sit_out_cost']:.2f}")
    if b["underbet_rounds"]:
        bet_lines.append(f"{b['underbet_rounds']} underbet raise spots — "
                         f"€{b['underbet_ev']:.2f} EV given up")
    if b["overbet_rounds"]:
        bet_lines.append(f"{b['overbet_rounds']} overbet rounds — risk cost "
                         f"€{b['overbet_risk_ce']:.2f} (CE) + "
                         f"€{b['overbet_neg_ev']:.2f} at -EV")
    sb_lines = [f"{agg['label']}: {agg['count']} stakes "
                f"(€{agg['staked']:.2f}) placed at ≤ 0 EV — expected "
                f"loss €{agg['cost_eur']:.2f}"
                for agg in result["side_bets"].values()]
    recon_note = (f" ({b['reconstructed']} of {b['compared']} comparisons "
                  "reconstructed with today's ramp config — older rounds "
                  "predate the persisted suggestion)") if b["reconstructed"] \
        else ""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Leak report — {html.escape(session_label)}</title>
<style>
 body {{ font-family: Segoe UI, sans-serif; background:#1a1a2e; color:#eee;
        max-width: 860px; margin: 2em auto; }}
 h1 {{ font-size: 1.4em; }} h2 {{ font-size: 1.1em; margin-top: 1.6em; }}
 table {{ border-collapse: collapse; width: 100%; }}
 td, th {{ border-bottom: 1px solid #333; padding: 6px 10px; text-align: left; }}
 .muted {{ color: #999; font-size: 0.9em; }}
</style></head><body>
<h1>Leak report — {html.escape(session_label)}</h1>
<p class="muted">{result['rounds']} recorded rounds, {owned} with your money.
Costs are exact-EV (composition reconstructed at the recorded count).
Insurance decisions are not recorded and cannot be mined.</p>
<h2>Play errors (your seats, settled hands)</h2>
<table><tr><th>class</th><th>pattern</th><th>times</th><th>total cost</th>
<th>per 100 owned rounds</th></tr>
{''.join(rows) if rows else '<tr><td colspan="5">none found 🎉</td></tr>'}
</table>
<h2>Bet discipline</h2>
<p>{'<br>'.join(html.escape(line) for line in bet_lines) if bet_lines
    else 'no bet-discipline leaks found'}<span class="muted">{recon_note}</span></p>
<h2>-EV side-bet habit</h2>
<p>{'<br>'.join(html.escape(line) for line in sb_lines) if sb_lines
    else 'no -EV side-bet stakes found'}</p>
</body></html>"""
