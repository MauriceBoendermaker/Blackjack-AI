# Feature Roadmap V3 — 10 Top-Tier Features + 5 Enhancements

Researched 2026-06-11 (codebase gap-scan of the V3.3 app + web research on
commercial AP tooling — CVCX/CVData, Casino Vérité, BJA trainer — Evolution
variant math, multi-hand Kelly theory, and countable side bets; load-bearing
numbers cited in Sources at the bottom).

**Where V3.3 leaves the app:** perception (YOLO + winocr), exact math (EV,
side bets, insurance, Kelly, RoR/N0/SCORE/CE), and the closed loop
(settlement → bankroll → stats → trainer) are all shipped — both prior
roadmaps are 20/20 done. What the app still cannot do: it has **no idea what
phase the game is in** (AUTONOMY_PLAN.md calls this "the hard 80%"), it
watches **one table**, its models **never improve themselves end-to-end**, the
pre-deal exact edge lands **~one round late** (~14 s sweep vs a 12–15 s
betting window), advice is a number **without an explanation**, analytics
**describe but don't coach**, and only classic Evolution blackjack rules are
modeled. V3 closes those.

**Deliberately not proposed:** audio cues (omitted by explicit user choice in
V2 Feature 4), full unattended autonomy (advised against in AUTONOMY_PLAN.md
§7 — everything here stops at stage 2), shuffle tracking (not viable on
machine-shuffled ~50%-pen online shoes), and alternative counting systems
(Zen/Halves/etc. are strictly dominated by the exact-composition engine the
app already has; Hi-Lo stays as the human-readable layer).

---

# Part 1 — The 10 Features

## 1. ✅ Game-phase & turn detection engine ⭐ (the keystone) — COMPLETED

**Status: DONE** — shipped in commit `b73556b` (PhaseDetector, control templates,
vision assist).

**What:** A `PhaseDetector` (`lib/logic/phase.py`) that gives the snapshot the
one thing it lacks: *what is happening right now*. Small state machine —
`IDLE → BETTING_OPEN → DEALING → MY_TURN → WAITING → SETTLE` — driven by
vision/OCR signals the pipeline already supports: countdown-timer digits via
winocr, template-matched action buttons with enabled-state color sampling
(grey = disabled), the active-seat highlight ring, chip-on-spot detection via
`cv2.matchTemplate`, and the result banner OCR that already exists. Plus a
per-control template library (`assets/controls/`, saved crops per table
profile — the V3.2 profile store is the natural home).

**Why:** This is the single biggest gap the autonomy assessment found:
`get_snapshot()` knows *what to do* (`optimal_action`, bet size, insurance)
but not *whether it's legal to do it right now*. Even with zero automation,
phase awareness pays for itself in advisory mode:

- **Betting-window alert + countdown** on the HUD ("Bets open — €40 · 9s"),
  so a raise spot can't slip past while you watch the stream.
- **"YOUR TURN — Hit"** flash exactly when the buttons enable, and a
  **missed-decision alarm** before the timer auto-stands you.
- **Live discipline feedback**: diff the observed action (buttons gone, card
  count grew, chip appeared) against the advice → "you stood; optimal was
  hit — EV −0.08u" the moment it happens, not in post-session stats.
- **Phase-keyed engine pacing**: replace the empty-frame heuristics
  (`EMPTY_FRAMES_FOR_RESET`, dealer-quiet settlement trigger) with explicit
  phase transitions — fewer false round resets, faster settlement.

**How:** New worker step inside the existing detection cycle (the dealer/player
crops are already captured; timer/button regions are 2–3 extra small crops at
~1 Hz). Phase + confidence published into the snapshot; HUD and sidebar render
it. Template capture UI rides the existing region-editor pattern. This is
stage 0 of AUTONOMY_PLAN.md §6 — build it once, everything in Feature 2 and
half of Feature 3 stands on it.

**Effort:** ~4–6 days (the template library + MY_TURN seat attribution are the
real work — see AUTONOMY_PLAN §4.1).

## 2. ✅ Ghost-mode executor → one-key assisted execute — COMPLETED

**Status: DONE** — shipped in commit `0b31945` (ghost-mode executor + one-key
assisted execute, AUTONOMY stages 1–2).

**What:** Implement stages 1–2 of AUTONOMY_PLAN.md (the plan is the spec —
this entry just schedules it). Stage 1, **ghost mode**: the executor computes
the exact click it *would* make (button template location, chip sequence for
the bet ramp), draws it on the HUD, clicks nothing, and logs decision + click
+ screenshot crop + confidence into `SessionStore` — producing a measured
**would-click match rate** across betting, every decision type, splits,
insurance, and the awkward states. Stage 2, **one-key assist**: after the
match rate proves out, a single confirm key fires the click via CDP
(`Input.dispatchMouseEvent` through Playwright `connect_over_cdp`, OS-input
fallback), gated by the full §5 guard stack: ARM toggle, kill switch +
`FAILSAFE`, N-consecutive-frame phase certainty, seat-attribution certainty,
geometry sanity, money limits with auto-disarm, and post-action verification
with *never-reclick* on mismatch.

**Why:** It removes the aiming and remembering, not the deciding — the human
stays the actor on every irreversible money move, which is exactly where the
plan recommends stopping (a fully automated client is both an engineering
money-loss generator and the cleanest enforcement signal; the plan's risk
line stands). Ghost mode alone is pure upside: it is the only honest
benchmark of whether detection quality could ever be trusted, and it
exercises Feature 1 end-to-end with zero risk.

**How:** `lib/logic/executor.py` as a second consumer of the same snapshot
(never touches engine internals), per AUTONOMY_PLAN §3. Gate stage 2 behind a
long ghost-mode sample, per §6.

**Effort:** ~2–3 days (ghost) + ~3–5 days (assist + guards), after Feature 1.

## 3. Multi-table watcher

**What:** Watch N casino tables at once (separate browser windows / monitor
regions): one `DetectionEngine` instance per table, each bound to its own
named profile (the V3.2 profile store already bundles regions + OCR rects per
table — exactly the unit this needs), publishing into a per-table snapshot. A
new **lobby panel** ranks tables live — TC, exact edge, phase, time to next
betting window — and the HUD surfaces the hot one: "Table B: bets open, edge
+0.9% — GO". Per-table session rows in SQLite (`session_id` already exists;
add a table key).

**Why:** The biggest EV/hour lever available to an online counter. At 8-deck
~50% pen, only ~5–10% of rounds reach TC ≥ +2 (the V2 cheat-sheet number) —
on one table you wait through the other 90%. Three tables ≈ 3× the +EV
betting windows for the same wall-clock, and online nobody back-offs a player
for sitting out the bad shoes. Wonging *between* tables is the online
equivalent of back-counting, and no commercial tool does it.

**How:** The engine is already a self-contained class with snapshot-based IPC
— instantiate per table; widen the `ev_offload` pools (they're process pools;
the per-seat advice path is single-worker by design today). The real
constraint is inference latency/cost: hosted Roboflow at ~0.5–1 s per crop ×
N tables won't keep up — **Enhancement E1 (local ONNX) is the prerequisite**
beyond 2 tables. Phase detection (Feature 1) per table turns the lobby panel
from "counts" into "a betting-window scheduler."

**Effort:** ~5–7 days after E1 (engine instancing is cheap; the lobby
UI/arbitration and per-table profile binding are the work).

## 4. Simulation lab — forward Monte Carlo + counterfactual backtester

**What:** A CVCX-class simulator built from parts the app already owns, two
modes:

- **Forward sim:** an auto-player loops the *real* stack — rules profile,
  basic strategy + I18/Fab4 (or full exact play), the live Kelly ramp — over
  millions of synthetic shoes. Output: EV/100, SD, N0, RoR curve, drawdown
  quantiles, optimal-spread validation, hands-to-double — and sensitivity
  sweeps ("what if pen were 60%? what at 6 decks? what playing 2 seats?").
- **Backtester:** replay *recorded* shoes from `session.db` (every round
  already stores counts, dealer playout, seats, bets) under counterfactual
  configs: quarter vs half Kelly, 1 vs 2 seats, with/without each side bet,
  flat-bet baseline → "would-have" P&L distributions on your actual cards.

**Why:** This is the question the bankroll window's bootstrap (V2 F2) can't
answer until you have hundreds of recorded rounds, and the question CVCX
charges for: *is this exact table, ramp, and bankroll worth playing, and at
what risk?* It's also AUTONOMY_PLAN §7's zero-risk closing of the loop —
"point an auto-player at the built-in simulator and let it run millions of
hands": the same agent-engineering as autonomy, validating the entire
engine/ramp/counting stack end-to-end, at stake €0.

**How:** The trainer/replay infra plays hands already; the speed path is the
one V2 F9 documented but never needed: generate per-rules **strategy tables**
offline with the oracle-validated exact engine (overnight, once per rules
profile), then simulate table-driven at 100k+ rounds/min across `ev_offload`
workers. Results window reuses the bankroll-window layout + a fan chart.

**Effort:** ~4–6 days. Synergy: Feature 5 consumes its output directly.

## 5. ✅ Bet-ramp designer / optimizer — COMPLETED

**Status: DONE** — `lib/logic/ramp_optimizer.py` (Kelly-scale scan +
hill-climb under closed-form RoR), per-TC `BETTING["bet_table"]` followed by
`betting.suggest()` (0 = sit out), measured TC distribution + table pace via
`SessionStore.pre_deal_tcs()`/`rounds_per_hour()`, "Ramp designer" tab in the
bankroll window with current-vs-optimal metrics and one-click install/clear.

**What:** Today `betting.py` sizes each bet from a formula (bankroll ×
Kelly-fraction × edge/variance, clamped). The designer turns that into a
solved, personal **integer ramp**: inputs are bankroll, target RoR (or
maximize SCORE/CE), table min/max, chip denominations, max spread — and the
**TC/edge frequency distribution measured from your own `session.db`**
(fallback: standard 8-deck 50%-pen frequencies). It searches ramp space
(closed-form RoR per candidate + the existing Monte Carlo for the winner) and
presents current vs optimal side by side — EV/hr, RoR, N0, CE — with one
click writing the result into `BETTING` as a per-TC bet table the live ramp
then follows.

**Why:** "Optimal betting is calculated to improve results and lower risk" is
*the* headline CVCX feature, and every input it needs is already recorded
here — personalized to the exact table you actually play instead of a generic
sim. The verified Kelly-risk anchors are already in FEATURES_V2.md (full
13.5% / half 1.8% / quarter 0.03% RoR); this turns them from a reference
table into a decision.

**How:** `lib/logic/ramp_optimizer.py` — for ~10 TC buckets × chip-rounded
bet candidates the space is small enough for exhaustive search under the
closed forms in `bankroll.py`; validate the chosen ramp with the existing
seeded MC (or Feature 4's simulator when present). New tab in the bankroll
window.

**Effort:** ~3–4 days standalone; ~2 on top of Feature 4.

## 6. ✅ Leak finder & coaching report — COMPLETED

**Status: DONE** — `lib/logic/leaks.py` (draw-replay divergence detection on
★ seats incl. index plays, exact-EV costing via ev_offload, bet-discipline
and −EV side-bet mining with the one-row count lag handled; insurance is
honestly excluded — the user action is recorded nowhere), persisted
`bet_suggested`/`bet_sit_out`/`edge_exact` columns going forward, Leak
Finder window (🩺 nav), "Drill my leaks" feeding the trainer's replay drill
with a severity-weighted deck, self-contained HTML report export.

**What:** Mine `session.db` into a ranked list of what your mistakes actually
cost, in units per 100 rounds, by class:

- **Play errors** — played-vs-optimal divergence per round is already
  inferable (seat draws + advice/optimal lines are recorded; `seat_quality`
  does the draw-replay trick for other seats — apply it to ★ seats), costed
  by the exact engine in EV, grouped by pattern ("16vT: hit when index said
  stand — 3× this week, −2.1u total").
- **Bet-discipline leaks** — bet placed (OCR) vs ramp suggestion are both
  recorded; quantify over/under-betting and missed raise spots.
- **Insurance errors**, **−EV side-bet habit** (stakes placed when the panel
  said no), **missed sit-outs** (rounds played at negative exact edge).

Topped with a one-click **"Drill my leaks"** button that feeds your worst
patterns straight into the trainer's flashcard generator (it already grades
in EV-lost-per-error — this just selects *your* weak spots instead of random
indices), and an exportable session/weekly HTML report.

**Why:** Casino Vérité's error log ("which strategy errors you made and how
much expected value was lost") is the reference feature serious players buy
the suite for — this version is stronger because every error is costed by the
exact-composition engine against the actual shoe, and it closes into the
trainer. Analytics that *coach* instead of describe.

**Effort:** ~3 days (queries + one new window + trainer hook).

## 7. Luck-vs-skill dashboard (variance attribution)

**What:** The mid-losing-streak question — *is the system broken or am I
unlucky?* — answered with the data already recorded: cumulative **actual** P&L
plotted against cumulative **expected** EV (exact edge at bet time × bet, per
round) with ±1σ/±2σ envelopes (per-round variance × bet² — the math is in
`bankroll.py`), an **N0 progress bar** ("skill > noise in ~1,840 more
rounds"), per-shoe z-scores, measured hands/hour → **realized vs predicted
EV/hour**, and a one-line verdict ("running −1.3σ over 412 rounds — within
normal").

**Why:** Every serious counter tracks this; it's the difference between
abandoning a working system in a normal downswing and trusting a broken one
(a persistent EV-line/actual divergence beyond bands is also the earliest
detector of a *detection* problem — miscounted cards show up here first).
N0/SCORE/DI/CE are computed in the bankroll window today but only as static
projections — this is the realized trajectory against them.

**How:** One canvas chart + a stats strip in a new tab of the stats window;
every input is a `rounds`-table query. No new math.

**Effort:** ~2 days.

## 8. ✅ EV explainability inspector ("why this play?") — COMPLETED

**Status: DONE** — ev_engine gains `action_outcomes` (per-action P(win/push/
lose) via an outcome recursion following the EV-optimal policy, validated by
the EV=W−L / EV=2(W−L) identities against the WoO-anchored recursion),
`dealer_distribution` (17–21/BJ/bust display dist), `composition_drivers`,
and the picklable `inspect_hand` job with a fresh-shoe flip indicator.
Clicking any advice line on the felt opens `InspectorWindow` (EV bars,
outcome odds, dealer dist, drivers) with a what-if sandbox (editable hand,
up-card, per-rank remaining counts) recomputed on the advice pool.

**What:** Click any advice line → an inspector panel showing the *whole*
decision, not the conclusion: per-action EV bars (Stand/Hit/Double/Split/
Surrender — already computed, only the max is shown), the **dealer
final-total distribution** P(17/18/19/20/21/BJ/bust) from `dealer_dist()`
(already memoized — it exists every time advice is computed, it's just never
surfaced), per-action P(win/push/lose), and **composition drivers**: the
ranks whose depletion moved the call ("tens 33.1% of shoe vs 30.8% baseline →
stand 16vT flips"). Plus a **what-if sandbox**: edit hand, up-card, or
composition and recompute on the advice pool — an interactive exact-EV
calculator for arbitrary scenarios.

**Why:** Trust and training. When "Optimal: Stand −0.476" contradicts the
book, today you either believe it or you don't; the inspector shows the
*mechanism*, which is what catches real detection errors (a surprising call
with a weird composition panel = misread card, visible instantly). As a
sandbox it replaces the WoO online calculator with your own oracle-validated
engine. Cheapest trust-per-day feature in this list.

**How:** `ev_engine` already returns the full EV dict; add dealer-dist and
per-action outcome probabilities to the advice job payload (one extra return,
same subprocess), render in a Toplevel. Sandbox = same job with a
hand-edited composition.

**Effort:** ~2–3 days.

## 9. Evolution variant rule packs — Free Bet & Infinite, + a Lightning audit

**What:** Extend `constants.RULES` + `ev_engine` beyond classic blackjack:

- **Free Bet Blackjack** (Evolution runs 20+ such tables): free double on
  two-card 9/10/11, free split on non-ten pairs, **dealer 22 pushes** all
  non-BJ hands. New flags (`push_22`, `free_double`, `free_split`), exact EV
  through the existing recursion, basic-strategy CSV variant, and re-derived
  indices — push-22 materially reshapes stand/double indices, so the I18
  table is wrong there today. Published HE ≈ 1.0–1.6% rules-dependent; the
  engine computes it exactly for the live shoe, and composition-dependence
  (counting windows) comes free.
- **Infinite Blackjack**: one shared hand, many players — a one-seat profile
  preset + its **Six Card Charlie** rule in the EV recursion; Hot 3/Bust It
  side bets are already implemented.
- **Lightning Blackjack audit mode**: exact EV including the mandatory 100%
  Lightning fee and the banked multiplier state (2–25× by winning total).
  Sources disagree *wildly* — Evolution advertises 99.56% RTP under
  variant-optimal play; independent analyses claim figures as bad as ~17.6%
  house edge — and a banked multiplier genuinely changes optimal strategy.
  The point of the audit is to settle it exactly and, most likely, render a
  data-backed "don't play this" verdict (or find the narrow banked-multiplier
  states where play is justified).

**Why:** Every additional modeled variant multiplies playable tables (and
Feature 3 makes that multiplication literal). Free Bet is the high-value
target: huge table count, modest HE, and nobody publishes composition-exact
advice for it.

**How:** Rules flags thread through `dealer_dist` (22-push branch is one
terminal-state change) and the player expectimax (free-double/split EV = win
pays, loss returns the free chip); validate against the WoO Free Bet
calculator as the golden oracle, same pattern as v1.

**Effort:** ~4–5 days (Free Bet math is the bulk; Infinite ~1; Lightning
audit ~1–2).

## 10. Companion HUD on a second device (phone/tablet)

**What:** A tiny LAN-only web server inside the app (single endpoint +
WebSocket, token-gated, QR code in the sidebar to connect) pushing the
existing snapshot at ~2 Hz to a one-page mobile PWA: huge true count + exact
edge, the bet call, per-★-seat advice, insurance alert, session P&L — and the
phase countdown once Feature 1 lands ("Bets open · 7s").

**Why:** The overlay HUD solved "you watch the stream, not the app" — but it
still competes for pixels on the game monitor and needs focus-stealing
suppression tricks. A phone propped next to the screen is the zero-overlay,
zero-focus-risk version, and the code scan lists phone/notification
integration as the one delivery channel the app lacks. (Audio stays out, per
the V2 decision — a glanceable second screen is the quiet alternative.)

**How:** `lib/interfaces/web_hud.py` — stdlib `http.server` for the static
page + the `websockets` package (or SSE to stay stdlib-only) fed from the
same 120 ms snapshot poll the Tk HUD uses; renders the `hud.py` formatting
layer (already a tested pure function) as HTML.

**Effort:** ~2–3 days.

---

# Part 2 — The 5 Enhancements (to existing features)

## E1. Local-first detection: close the active-learning loop

**Enhances:** V2 F6 (self-improving detection — collector + ONNX backend
shipped, loop never closed).

**What:** The pieces exist but don't connect: `training_data.py` captures
corrected/flapping crops, `models.py` prefers `models/*.onnx` over hosted —
but nothing retrains, evaluates, or promotes. Add: (1) a fine-tune job
(ultralytics on the accumulated corpus — ~50–200 instances/class suffices for
a fixed table skin, per the V2 research; or Roboflow-side training from the
already-implemented `upload_batch()`); (2) a **golden regression set** built
from corrected crops with known truth; (3) an A/B harness scoring
old-vs-new weights on accuracy + latency over recorded crops; (4) one-click
promote/rollback of the ONNX file. While in there: validate
`DEALER_USE_PLAYER_MODEL` live and default it on, making dealer suits exact
(21+3 flush tiers and Lucky Ladies QH stop being expected-value-spread).

**Why:** Detection is the floor under every EV, corrections currently improve
nothing permanently, and local ONNX (≈3× CPU, no network) is the stated
prerequisite for multi-table (Feature 3) and tightens the phase loop
(Feature 1). **Effort:** ~2–3 days plumbing + training time.

## E2. ✅ Fit the exact pre-deal sweep inside the betting window — COMPLETED

**Status: DONE (measured: ~14 s → ~9–10 s; inside the 12–15 s window).**
The sweep now fans weight-balanced whole-up-card jobs across a sized
"predeal" process pool (`PREDEAL_WORKERS` = cores/2 capped at 6;
`ev_offload.run_many` with run()'s full fallback contract;
`predeal_ev_upcards` partials sum exactly to `predeal_ev`, hand-slice
partition supported and tested). **Two of this entry's premises were
falsified empirically and are documented in the code:** (1) persisting the
dealer cache across rounds buys nothing — deep dealer states key on the
exact composition and essentially never recur once any card leaves the shoe
(14.2 s warm vs 14.1 s cold, +860k new entries/round); (2) incremental
per-up-card refresh is void — every up-card's distribution moves whenever
any card is dealt. Hand-slicing below up-card granularity duplicates each
slice's dealer-tree build and inverts its gains mid-shoe (kept available
behind `predeal_jobs(grain=)`). The honest next tier for <5 s remains the
documented mypyc build (V2 F9).

**Enhances:** V2 F3 (exact pre-deal EV) + V2 F9 / V3.2 (`ev_offload`).

**What:** The sweep takes ~14 s on its dedicated subprocess; the betting
window is 12–15 s — so the "exact" bet call is structurally one round stale
(the shipped note says as much). Target < 5 s: (1) persist the dealer cache
across rounds within a shoe — only the shuffle truly invalidates it, today it
rebuilds per sweep; (2) incremental refresh — between rounds the composition
changes by ~2–6 cards, so re-sweep only the up-cards whose dealer
distribution actually moved, lazily refresh the rest; (3) widen the predeal
pool — the sweep is embarrassingly parallel across 10 up-cards and
`ev_offload` is already a process pool, currently `max_workers=1`; (4) if
still needed, the documented mypyc tier (2–5×) from V2 F9.

**Why:** Bet sizing is ~70–80% of a counter's edge; it should never run on
last round's shoe. Also makes Feature 4's simulator and Feature 3's
multi-table refresh cadence comfortable. **Effort:** ~1–2 days.

## E3. ✅ Anchor-based, resolution-independent calibration — COMPLETED

**Status: DONE** — `lib/logic/anchors.py` (multi-scale matchTemplate per
anchor, least-squares scale+offset fit with residual gate, fail-disabled),
4th "anchors" kind in the profile store + PNG templates beside the control
crops, AnchorEditor (⚓ Capture Anchors, control-capture pattern), engine
solves on the worker at set_monitor / on demand and remaps regions + OCR
rects + control templates (rescaled — matchTemplate has no scale
invariance) from the one calibrated resolution, throttled drift detector →
status-bar badge with one-click Re-anchor, and the executor auto-disarms on
drift (every click target is suspect).

**Enhances:** v1 F10 (region editor) + V3.2 (named profiles).

**What:** Profiles today are keyed per resolution (`regions{WxH}`), so a new
resolution/DPI/window position means re-drawing polygons by hand — concretely:
only 2560×1440 is calibrated today, and the actual game monitor (1440p @ 96
DPI) still isn't. Add 2–3 **UI anchors** per table profile (template crops of
stable elements — logo, chip tray, menu button); on startup or drift,
`cv2.matchTemplate` finds them, solves the scale+offset transform, and maps
the *one* calibrated region set to whatever the screen actually shows. A
drift detector (anchor confidence drop / position jump) warns and offers
one-click re-anchor instead of silently feeding cards to wrong seats.

**Why:** Kills the recurring mixed-DPI/multi-monitor pain class (V3.1's DPI
work, V3.2's dialog fixes, the uncalibrated second monitor) at the root, and
makes profiles portable across window sizes — which Feature 3 (N windows at
arbitrary sizes) outright requires. **Effort:** ~2 days.

## E4. Table fingerprinting → automatic profile & rules switching

**Enhances:** V3.2 (named table profiles) + v1 F9 (rules profiles).

**What:** The profile dropdown is manual; picking wrong (or forgetting) is
the silent-corruption failure mode — seat polygons land on the wrong table
layout and the count quietly dies. Evolution prints the table name and limits
on screen: OCR them (winocr already runs at 1 Hz) and auto-activate the
matching named profile — which since V3.2 bundles regions + OCR rects, and
can carry the rules/paytable preset too. Mismatch guard: "table shows min €5;
active profile says €10" or "BJ pays 6:5 here, profile says 3:2" → banner +
block start until confirmed. Unknown table → offer to create a profile.

**Why:** Turns the V3.2 profile system from "remembers calibrations" into
"can't use the wrong one," and is the on-ramp for Feature 3 (per-window
profile binding must be automatic at N tables). **Effort:** ~1–2 days.

## E5. ✅ Covariance-aware multi-seat Kelly + session guardrails — COMPLETED

**Status: DONE** — (1) `betting.multi_seat_factor` (v/(v+(k−1)c), c=0.479
WoO) shrinks the per-seat wager when k seats are starred (engine passes
`len(my_seats)`; formula and installed ramp alike; two seats ≈ 73.5% each),
`bankroll.model_round_stats(seats=k)` carries k·v+k(k−1)·c into every RoR
readout, bankroll window notes the k-seat sizing. (2) `constants.GUARDRAILS`
(fail-disabled default) → engine publishes a `guardrails` snapshot block
(stop-loss / stop-win / max-rounds vs live session P&L), HUD banner line +
status-bar banner render it, the executor auto-disarms on breach, settings
dialog/persistence wired.

**Enhances:** v1 F7 (Kelly ramp) + V2 F2 (bankroll & risk).

**What:** Two betting-layer corrections. (1) **Multi-seat sizing:** the ramp
sizes each seat as if it were alone, but hands at one table share the
dealer — correlation ≈ 0.5 — so k simultaneous seats at full single-hand size
over-bet the bankroll. Standard result: two hands at ~73% of the one-hand
optimal each (≈1.46× total action, ~+37% growth at equal risk); generalize
via Var(k hands) = k·σ² + k(k−1)·ρ·σ². Since ★ seats are already tracked,
`betting.py` just needs the k-aware multiplier — and the bankroll window's MC
should resample k-tuples. (2) **Session guardrails:** the math panel knows
your risk; nothing enforces the plan. Add stop-loss / stop-win / max-rounds
per session with a live adherence strip (extends the existing bet-capped
telemetry to ramp-adherence %), a HUD banner on breach ("stop-win +20u —
walk away"), and — once Feature 2 exists — auto-disarm of the executor on
any breach.

**Why:** Playing two seats is common online and currently mis-sized by
~37% per seat; and discipline enforcement is the cheapest RoR reduction
there is. **Effort:** ~1–2 days.

---

## Suggested order

1. **Feature 1 → Feature 2** — the phase engine, then ghost mode → one-key
   assist (the commissioned autonomy track, stopping at stage 2).
2. **E2 + E1** — bet call inside the betting window; local inference + the
   retrain loop (force multipliers for everything below).
3. **E3 + E4** — portable, self-selecting profiles (and the prerequisites
   for multi-table).
4. **Feature 3** — multi-table watcher, now affordable and auto-profiled.
5. **Feature 4 → Feature 5** — simulation lab, then the ramp designer on top.
6. **Feature 7 → Feature 6 → E5** — the analytics/coaching/discipline pack.
7. **Features 8 and 10** — quick trust/UX wins, slot anywhere between packs.
8. **Feature 9** — variant packs when table variety appeals.

## Sources

- CVCX simulator & optimal-betting features: [qfit.com/blackjack-simulation](https://www.qfit.com/blackjack-simulation.htm), [qfit.com CVData/CVCX V5](https://www.qfit.com/blackjacksoftware-cvdatav5.htm)
- Casino Vérité drills, error log, rule database: [blackjackinfo.com review](https://www.blackjackinfo.com/casino-verite-blackjack-the-ultimate-practice-tool/)
- BJA trainer drills & test-out: [blackjackapprenticeship.com/blackjack-training-drills](https://www.blackjackapprenticeship.com/blackjack-training-drills/), [AP math (N0, RoR, CE)](https://www.blackjackapprenticeship.com/math-behind-advantage-play/)
- N0 / RoR / CE definitions & calculators: [gamblingcalc.com bankroll calculator](https://gamblingcalc.com/casino/blackjack-bankroll-calculator/)
- Multi-hand Kelly (ρ ≈ 0.5; two hands ≈ 73% each): [Wizard of Odds — Kelly Criterion](https://wizardofodds.com/gambling/kelly-criterion/)
- Free Bet Blackjack rules & house edge: [Wizard of Odds](https://wizardofodds.com/games/free-bet-blackjack/), [casino.us Free Bet guide](https://www.casino.us/blackjack/academy/free-bet-blackjack/), [livedealer.org (Evolution's 20+ tables)](https://www.livedealer.org/pragmatic-play-free-bet-blackjack/)
- Lightning Blackjack fee/multipliers & disputed RTP: [Wizard of Odds](https://wizardofodds.com/games/lightning-blackjack/), [Wizard of Vegas thread](https://wizardofvegas.com/forum/questions-and-answers/advice/36630-evolution-lightning-blackjack/), [livecasinocomparer guide](https://www.livecasinocomparer.com/live-casino-software/evolution-live-casino-software/evolution-live-blackjack/lightning-blackjack/)
- Evolution side-bet house edges: [livecasinocomparer side-bet guide](https://www.livecasinocomparer.com/live-casino-games/live-dealer-blackjack/blackjack-side-bets/), [livedealer.org PP/21+3](https://www.livedealer.org/perfect-pairs-213-live-blackjack-side-bets/)
- Countable side bets (Lucky Ladies queen density): [blackjackincolor.com](https://www.blackjackincolor.com/blackjacksidebet1.htm), [Card counting — Wikipedia](https://en.wikipedia.org/wiki/Card_counting)
