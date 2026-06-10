# Feature Roadmap V2 — The Next 10

Researched 2026-06-10 (multi-agent: codebase gap-scan + web research on bankroll
math, OCR/vision, and overlay/performance tech; load-bearing formulas and claims
fact-checked against primary sources — Schlesinger/Blackjack Attack derivations,
Wizard of Odds calculators, Microsoft/Ultralytics/library docs).

**Where v1 left the app:** every advice surface is exact and oracle-validated,
but the loop is open — the app advises, then never learns what happened. It
doesn't settle rounds, doesn't know your P&L, can't say what your edge or risk
actually is, and assumes you're staring at it. V2 closes the loop (settle →
bankroll → risk), gets the advice to your eyes/ears faster (HUD, audio), and
makes the detection stack self-improving.

---

## 1. ✅ DONE — Automatic round settlement & true P&L ⭐ (the keystone)

> **Shipped 2026-06-10.** `lib/logic/settlement.py` settles every hand at round
> end against the dealer's reconstructed final total (naturals 3:2, split hands
> per hand, dealer BJ, busts); a round only settles when the dealer hand is
> complete (total ≥ 17 or all hands bust) — otherwise nothing is booked. Click
> a seat's name (★) to mark it yours: owned seats roll into the Session P&L
> row and auto-update the bankroll via the new "Bet placed (€)" field
> (`BETTING["auto_bankroll"]` toggle). Settlement, per-round P&L, and bet are
> persisted (schema auto-migrates); stats window shows net €/units, win rate,
> settled rounds. Tests: `tests/test_settlement.py` (8 cases incl. engine
> integration with dealer playout).

**What:** When the dealer's playout completes (no new dealer cards for ~2
cycles after players act), compute the dealer's final total from
`dealer_extras` + up-card, settle every seat — win / lose / push, blackjack
3:2, doubles, splits per hand, insurance, bust — and store `outcome`,
`payout_units`, and `net` per seat in the snapshot and the SQLite rounds table.
Show a running P&L line in Game Info ("Round 15 · Session: +2.5 units") and
auto-update the bankroll that feeds the Kelly ramp.

**Why:** This is the single biggest gap the code scan found: v1 *can* know the
outcome (dealer playout cards are now tracked — that's what makes this newly
possible) but never computes it. Without settlement there is no real P&L, no
bankroll auto-update, no bet-discipline tracking, no measured edge — features
2, 5, and 8 all stand on this.

**How:** Settlement function in `lib/logic/engine.py` triggered by the same
quiet-dealer condition that drives auto-new-round; needs one user input it
can't see — the bet per seat — so pair it with a small "bet placed" field
(default = the suggested bet) until OCR (Feature 5) automates it. Add an
`outcome` column set to the rounds table; stats window gains net P&L, win
rate, and EV-vs-actual variance.

**Effort:** ~2–3 days. Do this first; everything below feeds on it.

## 2. ✅ DONE — Risk-of-Ruin & bankroll panel (verified math + Monte Carlo)

> **Shipped 2026-06-10.** `lib/logic/bankroll.py` + the 🛡 Bankroll & Risk
> window: lifetime RoR (both verified forms), trip RoR (BJA p.132), the
> Kelly-fraction risk table (13.53% / 1.83% / 0.034%), bankroll-for-target-RoR
> inverse, DI/SCORE, certainty equivalent — and a seeded numpy Monte Carlo
> (10k futures) that bootstrap-resamples **your own settled rounds** once 30+
> exist (TC-frequency model × your ramp as fallback), reporting ruin %,
> P(profit), final and max-drawdown quantiles.
> Tests: `tests/test_bankroll.py` (11 cases incl. landmark values, trip→
> lifetime convergence, inverse round-trips, deterministic MC).

**What:** A "Bankroll" tab computing, live from your configured ramp and
recorded rounds:

- **Lifetime RoR** — `RoR = ((1−μ/σ)/(1+μ/σ))^(B/σ)` and the exponential form
  `exp(−2·EV·B/Var)` (μ, σ = per-round mean/SD of your actual ramp).
- **Trip RoR** over N hands — Schlesinger's short-term formula (BJA p.132, the
  one behind the Wizard of Odds session-RoR calculator).
- **Kelly fraction → risk**, verified: fixed ramp at f×Kelly gives
  `RoR = e^(−2/f)` → full Kelly **13.5%**, half **1.8%**, quarter **0.03%**;
  proportional resizing: P(ever halving) = 50% at full Kelly.
- **DI / SCORE / certainty equivalent**, and the inverse question: "bankroll
  needed for 5% RoR at this ramp."
- **Monte Carlo fan chart**: bootstrap-resample your own recorded per-round
  outcomes (10k trials × 1k rounds is sub-second in numpy) → drawdown
  quantiles, P(profit by round N). This captures side bets, deviations, and
  your actual table — which no closed form does.

**Why:** The Kelly ramp (v1 Feature 7) sizes bets but never tells you the
risk you're running or whether your bankroll matches your spread. These are
the standard professional numbers, and the session DB makes the Monte Carlo
version personal instead of generic.

**Effort:** ~2 days (closed forms are a page of math; MC is one numpy loop).

## 3. ✅ DONE — Exact pre-deal EV — replace the linear true-count model

> **Shipped 2026-06-10.** `ev_engine.predeal_ev()` sweeps all up-cards × 55
> starting hands without replacement (naturals, dealer-BJ mixing for peek and
> ENHC, split-once), sharing one evaluator per up-card so hands reuse player-
> tree memos (~14 s per refresh, down from 35 s naive). It runs on its own
> dedicated thread — never delaying seat advice — refreshed whenever the
> composition changes, so at most one round stale (<0.05% drift). The bet
> hint now reads "edge +x.xx% (exact)" and falls back to the TC estimate
> until the first sweep lands (toggle: `BETTING["use_exact_edge"]`, also in
> Settings). Negative exact edge floors the bet and advises sitting out.
> Sanity anchors: full-shoe ENHC −0.60%, all-tens shoe exactly 0.
> Tests: `tests/test_predeal.py` (6 cases, engineered exact-value comps).

**What:** Once per round, on the advice thread during the betting window,
compute the **exact pre-deal round EV** from the live composition (sum over
up-cards × 55 starting hands of P(deal) × EV(optimal play)) and feed *that*
to the Kelly ramp and a wong-in/wong-out signal — instead of
`edge ≈ −0.5% + 0.5%·TC`.

**Why (research-verified):** the linear model is wrong exactly where bets
flip: the first TC point is worth ~0.79%, not 0.5%; a true point is worth
~0.3–0.4% early in the shoe but ~0.7–0.9% deep (Snyder/Gwynn — >2× drift the
linear model can't see); flooring TC +1.9 → +1 misstates edge by ~0.5% right
at the wong threshold. Hi-Lo's betting correlation is 0.97 — exact composition
closes that plus all the nonlinearity. For 8-deck ~50%-pen online play the
standard advice is wong in at TC +2 / out at 0, and an exact "pre-deal EV > 0"
signal is strictly better than any floored-TC rule — online nobody backs you
off for sitting out.

**How:** The dealer cache is already shared across up-cards and hands, so the
sweep costs ~2–6 s pure Python with the split approximation (budget fits the
betting window). Tricks: recompute every 2–3 rounds before ~30% depth (EV
drifts <0.05%/round early), lazily refresh the cheap up-cards, reuse the
dealer cache across rounds until the shuffle. Show "Edge now: +0.41% (exact)"
next to the bet suggestion, and a SIT OUT banner when negative.

**Effort:** ~2–3 days. Synergy: feeds Features 2 and 8 with honest edges.

## 4. ✅ DONE (visual only) — Overlay HUD + audio cues

> **Shipped 2026-06-10, audio deliberately omitted (user choice).**
> `lib/interfaces/hud.py`: 🎯 toggles a compact frameless always-on-top panel
> (drag to move, ✕ to close) showing TC/RC/decks, the bet call with exact
> edge, an insurance alert, advice for your ★ seats (active seats as
> fallback), +EV side-bet flags, and the session P&L. WS_EX_NOACTIVATE +
> WS_EX_TOOLWINDOW via ctypes (no new dependency) so it never steals focus
> and stays out of alt-tab; topmost reasserted every 2 s; updates ride the
> existing main-thread snapshot poll. main.py now sets per-monitor DPI
> awareness so HUD/calibration coordinates line up on scaled displays.
> Tests: `tests/test_hud.py` (pure formatting layer).

**What:** A compact frameless always-on-top HUD (true count, edge, advice for
your seat, insurance alert, side-bet ● BET flags) you park next to the casino
stream, plus optional audio: a soft cue on "raise spot" (TC/edge threshold),
a distinct one for "take insurance", spoken or beeped action calls.

**Why:** You watch the *stream*, not the app — the gap scan flagged that a
raise spot can pass unnoticed entirely. This is the highest UX-value-per-day
feature in the list.

**How (verified recipe):** Tk `Toplevel` with `overrideredirect(True)`,
`-topmost`, `-transparentcolor`; optional click-through for opaque pixels via
pywin32 `WS_EX_LAYERED | WS_EX_TRANSPARENT` (+ `WS_EX_NOACTIVATE` so it never
steals focus). Works over browsers incl. F11/HTML5 fullscreen (borderless,
DWM-composited — the easy case). Draw text on a small opaque panel to avoid
color-key antialiasing fringe; re-assert `-topmost` on a timer; update via the
existing snapshot poll (main thread). Audio: stdlib `winsound` with
`SND_ASYNC` (non-blocking; single-channel is fine for cues). Windows toast
for away-from-screen alerts.

**Effort:** ~2 days.

## 5. ✅ DONE — OCR: balance, bet, and result banners

> **Shipped 2026-06-10.** `lib/logic/ocr.py` (Windows.Media.Ocr via `winocr`,
> verified working on this machine incl. Python 3.14): reads up to three
> per-resolution screen regions at ~1 Hz on the advice thread — balance syncs
> the bankroll ("screen is ground truth"), bet fills the bet-placed field,
> and the result banner snaps to win/lose/push/blackjack and is cross-checked
> against the computed settlement (a mismatch logs a misread-card warning).
> Amount parser handles EU and US number formats and the €→'?' OCR quirk.
> "🔡 OCR Regions" opens a drag-rectangle editor; "Disable OCR" deletes the
> profile. Toggles in `constants.OCR`. Tests: `tests/test_ocr.py` incl. a
> live render→OCR round trip.

**What:** Read 2–3 small fixed screen regions at ~1 Hz: account balance,
current bet, and the round-result banner. Auto-fill the bankroll, capture the
actual bet per round (feeds settlement + bet-discipline stats), and
cross-check the banner ("WIN/PUSH/LOSE") against Feature 1's computed
settlement — flag disagreements as detection errors.

**Why:** Closes the last manual inputs in the loop. Bankroll drift between
the app and reality currently goes unnoticed.

**How (research verdict):** **Windows.Media.Ocr** via `pip install winocr` —
OS-built-in engine, no model download, no GPU, fastest of the four candidates
and *more* accurate than Tesseract on rendered screen text (maintainer-verified
claim). Post-filter numerics with a regex (no char-whitelist support); snap
banner text to its tiny fixed vocabulary (or template-match it — at that point
the engine barely matters). EasyOCR (2.8 GB torch install) and PaddleOCR
(second DL framework, finicky Windows installs) are overkill for this.
Region selection reuses the calibration editor pattern (Feature 10 v1).

**Effort:** ~2 days incl. an OCR-region calibration step.

## 6. Self-improving detection + dealer suits + local weights

**What:** Three linked upgrades to the detection stack:

1. **Active-learning capture**: auto-save (a) every manual correction (gold
   labels — the model's proven mistakes), (b) low-confidence and
   confirmation-flapping detections, (c) confirmed locked cards as
   pre-annotated YOLO labels. One click uploads a tagged batch to Roboflow
   for review/retrain (`project.single_upload` API).
2. **Local fine-tuned weights**: fine-tune from current best weights — for a
   fixed table skin ~50–200 instances/class suffices (the 1,500/class
   Ultralytics figure is for general-domain robustness); export to **ONNX**
   (`model.export(format="onnx")` + onnxruntime, up to ~3× CPU speedup, drops
   the torch dependency). AGPL is a non-issue for a personal, non-distributed
   tool (obligations trigger only on distribution/network service).
3. **Dealer suits**: point the dealer crop at the 52-class player model —
   verified to be a one-constant swap (`PROJECT_ID_DEALER` → `dey022`) plus
   the class map and threshold retune. One real caveat: the player model has
   no `cuttingcard` class — keep the rank model as a fallback call or detect
   the cut card by color. Payoff: the last composition blur disappears
   (21+3 flush tiers, Perfect Pairs, Lucky Ladies QH become fully exact).

**Why:** Detection quality is the floor under every EV. Today misreads are
corrected and *forgotten*; this makes every correction permanently improve
the model — and local ONNX inference kills the ~0.5–1 s hosted-API latency.

**Effort:** ~1 day capture plumbing; fine-tune/export ~1–2 days; dealer-suit
swap ~half a day + live validation.

## 7. Practice & replay trainer

**What:** A trainer tab with the drills the commercial tools converge on —
deck countdown (benchmark: one deck < 30 s, perfect, 5× in a row), running
count with cancellation, true-count conversion, deviation flashcards — plus
three things **no commercial trainer has**:

- flashcards auto-generated from *your* rules profile (your indices, your
  S17/ENHC config),
- **replay of your own recorded rounds** from the session DB ("what was the
  count here? what's the right play?"),
- grading in **EV lost per error** (units of bet) via the exact engine,
  instead of right/wrong against a static chart.

**Why:** Counting speed and deviation recall decay; every serious counter
drills (Blackjack Apprenticeship, Casino Vérité are the references). The
session DB + exact engine turn a commodity trainer into a personalized one.

**Effort:** ~3 days for countdown + flashcards + replay grading.

## 8. Bet-behind seat scoring

**What:** Track every seat's play against basic strategy (the data is already
in each round record); maintain a per-seat "plays book correctly %" score and
surface it on the table ("P3: 94% book, 41 rounds"). Combine with the live
edge (Feature 3): "Bet behind P3 ✓ (edge +0.4%, sound player)". Report EV/hour
observed-vs-played for wonging decisions.

**Why:** Bet Behind is the one Evolution bet with main-game edge, and its
only quality variable is the seated player's discipline. v1's hint says
"behind basic-strategy players" without knowing who that is — settlement
(Feature 1) makes it measurable.

**Effort:** ~1–2 days on top of Feature 1.

## 9. EV-core speed: compile or sidecar the engine

**What:** Cut the ace-up worst case from ~2 s toward ~0.2–0.5 s and make the
pre-deal sweep (Feature 3) comfortable.

**How (research-ranked, cheapest first):**
1. **Algorithmic**: persist the dealer cache across decisions within a shoe
   (only the shuffle truly invalidates it), warm it for the current up-card
   family ahead of player hits.
2. **mypyc**: compiles the existing annotated module unchanged — typical
   1.5–5×; plausibly ~2 s → 0.5–1 s for near-zero effort.
3. **PyPy sidecar**: the engine is pure Python — run it in a PyPy subprocess
   over a pipe (PyPy ~3×+, dict/tuple recursion is its sweet spot; whole-app
   PyPy is blocked by tkinter/pywin32, a sidecar isn't).
4. **numba**: the hhoppe "~30×" figure is for array-style Monte Carlo, *not*
   dict-memoized recursion — getting it requires an array-based rewrite
   (10-int composition arrays, packed-int states, `typed.Dict`); realistic
   5–30× but the most work. `lru_cache` is unsupported; memos must be
   hand-rolled.
5. **Offline tables**: the oracle-validated engine can generate full custom
   index/strategy tables per rules profile overnight — making live exactness
   a refinement, not the latency-critical path.

**Effort:** 1 (half day) → 2 (half day + build setup) → 3 (~1 day) as needed.

## 10. Reliability & trust pack (the gap-scan bundle)

**What:** The highest-value hardening items the code scan confirmed:

- **Advice fallback**: if an EV job fails or exceeds ~3 s, show the book play
  with a "(book — EV delayed)" suffix instead of a stuck "Optimal: …";
  per-error logging instead of the once-per-session error flag.
- **Cutting-card confirmation**: require N consecutive sightings (like the
  up-card) before announcing reshuffle; show a dismissible badge.
- **Round-end gate**: auto-new-round currently fires after 2 empty frames — a
  stream hiccup can wipe a live round; require a longer quiet period plus
  dealer-area-empty re-check, and log resets prominently.
- **Split-aware card picker**: a Hand 1 / Hand 2 selector when correcting a
  split seat (today manual adds can land on the wrong hand).
- **Bankroll/bet input validation** (`validatecommand`, bounds), "bet capped
  at table max N times" telemetry, paytable-change guard mid-shoe (hash the
  active paytables into each round record).
- **Split-aces advice through one path**: delegate to
  `strategy.advice(post_split=True)` instead of the hard-coded label, so a
  rules change can't desync it.

**Why:** Each is small, but together they're the difference between a tool
you trust at a real table and one you babysit.

**Effort:** ~2–3 days for the bundle.

---

## Suggested order

**1 → 2 → 4** first (settle rounds, know your risk, see alerts — one
"closed-loop" release), then **5 → 8** (OCR + seat scoring complete the
loop), then **3 → 9** (exact edge + the speed to compute it), then
**6** (detection self-improvement), **10** woven between releases,
**7** (trainer) whenever variety appeals.

## Smaller improvements (worth batching in when nearby code is open)

- Per-seat advice-accuracy drilldown in the stats window (% book-matched,
  % optimal-matched, EV/decision); CSV already has the columns.
- Cross-validation test: how often does the TC ≥ +3 insurance heuristic
  disagree with the exact 1/3-tens call over recorded shoes (quantify the
  human-proxy loss).
- Batch cache refresh on settings change (recompute affected hands instead
  of dropping the whole advice cache).
- Model warm-up refresh after long idle/restore (hosted-API re-init, retry
  with backoff, local-weights integrity check).
- Settlement end-to-end test: replay a recorded shoe, assert outcomes match
  ground truth.
- Split-detection robustness tests with synthetic misaligned hands
  (today's tests set hand tags manually).
- DPI-awareness call before Tk root creation so calibration coordinates
  match on scaled displays (also a prerequisite for the HUD).

## Verified numbers cheat-sheet (for implementation)

- RoR at f×Kelly (fixed ramp): `e^(−2/f)` → 13.53% / 1.83% / 0.034% for
  full/half/quarter. Proportional Kelly: P(ever reach x·B) = `x^(2/f−1)`.
- Trip RoR: `Φ((−B−μN)/(σ√N)) + e^(−2μB/σ²)·Φ((−B+μN)/(σ√N))` (WoO adds a
  +0.5 continuity correction).
- Edge vs TC: first point ≈ +0.79% (not 0.5%); per-point value ~0.3–0.4%
  early shoe → ~0.7–0.9% deep; Hi-Lo betting correlation 0.97.
- 8-deck ~50% pen: expect roughly 5–10% of rounds at TC ≥ +2; wong in +2 /
  out 0 is the standard human rule the exact signal replaces.
- Exact pre-deal sweep: ~2–6 s pure Python (split-approximate), ~5–15 s fully
  exact; recompute every 2–3 rounds before ~30% depth.
- OCR: Windows.Media.Ocr (winocr) — fast, OS-built-in, beats Tesseract on
  screen text; EasyOCR ≈ 2.8 GB install, CPU-slow.
- Fine-tuning a fixed-skin card detector: ~50–200 instances/class (the
  1,500/class guidance is for general-domain models); ONNX export ≈ up to 3×
  CPU inference, removes torch from the runtime.
