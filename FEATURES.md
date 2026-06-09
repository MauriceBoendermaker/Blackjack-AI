# Feature Roadmap — 10 Top-Tier Improvements

Researched 2026-06-09 (multi-agent web research, claims fact-checked against primary
sources: wizardofodds.com, Blackjack Apprenticeship charts, Eliot Jacobson /
advancedadvantageplay.com, discountgambling.net, possiblywrong, open-source engines).

**Status quo:** advice is a static basic-strategy CSV lookup (`lib/logic/strategy.py`).
The old "Optimal" second advice that fetched wizardofodds.com was removed in v3.0 as
*broken by design* — recovered from git history (`git show e2df107:lib/logic/blackjack.py`),
it hardcoded `u=6, v=44` (always asked the server about a pair of 4s vs dealer 6), sent the
player's own 2–3 cards as the *entire shoe composition*, and cached results without any
shoe state. It never once computed an EV for the actual hand or the actual shoe.
**So no — deviation/composition-aware advice is currently NOT implemented.** Feature 1
brings it back, done right.

Key research outcome: **Wizard of Odds has no public API**, but its hand calculator is
server-side and its undocumented JSON endpoint works (verified live — see Appendix B).
Don't put it on the real-time path (~0.8 s latency, no SLA, can be bot-gated any time,
leaks play state). The app already tracks exact per-rank composition — the same exact
EVs can be computed **locally in milliseconds**. Use the WoO endpoint only as a golden-value
test oracle.

---

## 1. ✅ DONE — Exact composition-dependent EV engine — the "Optimal" second advice ⭐

> **Shipped 2026-06-09.** `lib/logic/ev_engine.py`: exact dealer recursion with
> no-blackjack conditioning (Bayes-corrected draw probabilities), memoized player
> expectimax, split-once model, peek/ENHC + S17/H17 + DAS rule config in
> `constants.RULES`. Per-seat "Optimal: <action> (EV)" line on the table with a
> "≠ book" deviation flag. Validated against the WoO oracle to 1e-9 on all 14
> golden cases (`tests/test_ev_engine.py`; splits within 0.01 — different split
> model). Cold worst-case decision 0.44 s, cached thereafter.

**What:** New `lib/logic/ev_engine.py` computing *exact* expected values for
Stand / Hit / Double / Split / Surrender / Insurance from the actual remaining-shoe
composition (full shoe minus `CardCounter.per_rank`), per seat, per decision.
Shown as a second advice line next to the basic-strategy one; when they differ, flag it
as a **deviation** and show both EVs (e.g. `Stand −0.476 vs Hit −0.508`).

**Why:** This is strictly stronger than any count-based deviation table. Exact-composition
play recovers ~0.19%/round over basic strategy in shoe games (flat-bet, exact combinatorial
analysis by possiblywrong); Hi-Lo with the full index set captures only ~47% of that.
A single scalar true count can't see ten-density vs ace-density vs mid-card imbalance —
the exact composition can, and the app already holds it.

**How:**
- Shoe as a hashable 10-tuple (A, 2..9, T) = full 8-deck counts minus `per_rank`.
  Suits and J/Q/K identity are irrelevant to main-game EV — the current counter buckets
  are already sufficient. The rank-only dealer detection is also sufficient.
- `dealer_dist(upcard, comp)`: memoized recursion → P(17/18/19/20/21/BJ/bust),
  conditioned on the table's peek/ENHC rule (config flag — Evolution titles vary).
- `ev_stand / ev_hit / ev_double`: memoized expectimax keyed on
  (hand_total, is_soft, composition).
- `ev_split`: CDZ⁻ approximation (play each split hand with the optimal no-split
  strategy) — what reference engines default to; exact resplit EV is the only genuinely
  expensive computation and isn't worth it in real time.
- Insurance exactly: take iff `tens_remaining / cards_remaining > 1/3` (see Feature 5).
- Performance: hand states are bounded by totals (not deck size) — a full decision is
  well under 100 ms in pure Python with `functools.cache`; clear caches per decision.
- Validation: golden tests against the WoO endpoint (Appendix B), bjstrat.net's CDCA,
  and MIT-licensed `hhoppe/blackjack`; `possibly-wrong/blackjack` (C++, GPLv3) is the
  gold standard if exact split EVs are ever wanted.

**Effort:** ~3–5 days incl. tests. The single highest-value feature in this list.

## 2. ✅ DONE — Hi-Lo index deviations — Illustrious 18 + Fab 4 annotation layer

> **Shipped 2026-06-09.** `lib/logic/deviations.py`: full I18 + Fab 4 tables
> (S17 + verified H17 differences), Fab4-over-I18 precedence when surrender is
> offered, two-card gating. Per-seat "Index: <action> (TC ±x.x >=/< index)"
> line on the table, colored when triggered. Cross-check test passes: the
> exact-EV engine flips to the same action ±2 TC around every tested index
> (8 cases) on realistically constructed shoes, and the insurance +3 index
> agrees with the exact 1/3 tens threshold (`tests/test_deviations.py`).

**What:** Ship the I18 + Fab 4 tables (Appendix A) as a static dict. UI shows an
"Index says: …" annotation whenever the current true count crosses an index, with a
badge when it deviates from the CSV action.

**Why:** The I18 capture ~80–85% of the value of *all* play-variation indices in shoe
games (Schlesinger, *Blackjack Attack*; confirmed via Wizard of Odds). It's also the
human-readable explanation for *why* the exact-EV line (Feature 1) deviates — and a
powerful cross-check: at the listed thresholds with a neutral composition, the EV engine
must flip to the same action within ~±1 TC. That's a strong correctness test.

**How:** Small pure dict in `lib/logic/deviations.py` + lookup in `StrategyAdvisor.advice`;
S17 and H17 variants behind the rules config (Feature 9). Note: standard Evolution live
blackjack is S17 and offers **no surrender**, so the Fab 4 only activate on tables that
have it.

**Effort:** ~1 day. Cheap, high trust-building value.

## 3. ✅ DONE — Real-time side-bet EV panel

> **Shipped 2026-06-09.** `lib/logic/sidebets.py`: exact pre-deal EV for
> Perfect Pairs (2-card), 21+3 / Hot 3 / Lucky Lucky (shared 3-card
> enumerator), Lucky Ladies (2-card × conditional dealer-BJ tier), and Bust It
> (exact dealer bust-length recursion, rank-only). Full-shoe baselines
> reproduce the published house edges — 21+3 −3.7039% and Bust It −6.1842%
> match WoO to 6 decimals. Paytables/enable flags in `constants.SIDE_BETS`
> (Evolution defaults on; Lucky Lucky/Ladies off). Left-panel "Side Bets"
> section shows live EV%, green "● BET" when positive; recomputed only on
> composition change (~107 ms worker-side, cached otherwise).
> Tests: `tests/test_sidebets.py` (12 cases incl. composition-response).

**What:** Per-round, at betting time, compute the exact EV of every side bet the table
offers from the tracked composition; display EV% per bet, green when +EV, with a small
fractional-Kelly stake hint.

**Why (verified numbers, 8 decks where stated):** All common side bets are exactly
computable from remaining-shoe composition — no simulation needed:

| Bet | Base house edge | Countable? | Needs |
|---|---|---|---|
| **Bust It** (Evolution: 1/2/9/50/100/250×) | 6.18% (WoO; Evolution advertises 94.12% RTP) | **Yes** — Jacobson 6-deck: trigger TC +4, 5.28% avg edge; How: 14.4% of hands at +6.1% | rank-only ✔ already tracked |
| **Hot 3** (100/20/4/2/1×) | 5.40% | indirectly (Lucky-Lucky-family: bet when 6/7/8/A-rich) | mostly ranks |
| **21+3** (Evolution 100/40/30/10/5×) | 3.70% | weakly — flush-imbalance windows; perfect play ≈0.27 u/100 (Jacobson) | **suits** |
| **Perfect Pairs** (25/12/6×) | 4.10% | effectively no | suits |
| **Lucky Lucky** (PT1) | 2.63% | **Yes** — ~22–28% of hands +EV, ~4–5.6% avg edge (How, Jacobson) | mostly ranks |
| **Lucky Ladies** (Table A) | ~24% | **Yes** — 13–17% avg edge when triggered | QH side-count + dealer-BJ tier |

- **Priority 1 — Bust It:** best fit (rank-only, fits the existing dealer pipeline) and
  genuinely countable. Exact dealer-bust-length recursion over the rank composition,
  dotted with the paytable. Its count is *reversed* vs Hi-Lo (roughly 7,8,9,T = +1;
  A = −2, 2 = −3, 3 = −2): it lights up in **negative** counts — exactly when the main
  bet is small. Complementary income.
- **Priority 2 — Hot 3 + 21+3:** one shared 3-card enumerator (player's 2 cards ×
  dealer up-card over remaining multiplicities, ~22k weighted combos, trivially fast).
  21+3's flush tiers need suit tracking (Feature 4).
- **Priority 3 — Lucky Lucky / Lucky Ladies** behind config-selectable paytables for
  non-Evolution tables — these two are the genuinely profitable ones in the literature.
  Caveat: Lucky Ladies' top tiers pay on QH-pair **+ dealer blackjack**, so it needs the
  2-card enumeration × conditional dealer-BJ probability, not 2 cards alone.
- Skip Royal Match (not on Evolution, weakest published return).

**Reality check:** Evolution shuffles the 8-deck shoe at ~50% penetration; published
frequencies come from 6-deck/deep-pen analyses, so expect +EV windows roughly half as
often. The panel still costs nothing once the enumerators exist, and exact EV strictly
dominates every published human counting system.

**Effort:** ~3–4 days for the engines + panel; paytables user-configurable (they swing
house edge wildly between variants).

## 4. ✅ DONE — True 52-card shoe model (un-fold tens, track suits)

> **Shipped 2026-06-09.** `CardCounter` now tracks card identity at three
> levels (exact rank+suit from player detections, rank-only from the dealer
> model, bucket-only from manual ±), and `lib/logic/shoe.py` builds the
> expected 52-cell (rank × suit) remaining composition: exact removals hit
> their cell, rank-only spread 1/4 per suit, bucket residual spreads
> uniformly. The dealer-suit YOLO extension remains optional future work —
> the expected-composition treatment quantifies away most of the error.
> Tests: `tests/test_shoe.py` (9 cases incl. mixed-level buckets).

**What:** Internally track the shoe as a 52-cell (rank × suit) vector instead of 10
buckets. Keep the UI fold (J/Q/K → "10") if desired. Extend dealer detection to suits.

**Why:** Enabler for Feature 3's suit-dependent bets (21+3 flush tiers, Perfect Pairs,
Lucky Ladies' QH) and removes the blur of unknown-suit dealer removals. The player YOLO
model already classifies suits (`a1..d13` → "8 of Diamonds"); only the dealer pipeline is
rank-only — run the player model on the dealer crop, or treat unknown-suit dealer cards as
uniformly distributed (small, quantifiable error) until the model is extended.

**How:** Widen `CardCounter.per_rank` to a 52-cell dict keyed by full card name (manual
± UI can stay rank-level); derive the 10-bucket view for the EV engine from it.

**Effort:** ~1–2 days. Do before or together with Feature 3.

## 5. ✅ DONE — Insurance & even-money advisor

> **Shipped 2026-06-09.** `ev_engine.insurance_advice()`: take iff unseen tens
> fraction > 1/3 (exact, verified against the WoO −35/413 oracle value); even
> money generalized via `bj_pays` (6:5 flips at p > 1/6). Table view shows an
> "Insurance: TAKE/Decline (EV/unit)" chip under the dealer ace, and seats
> holding a natural get an "Even money: TAKE/Decline (edge)" line. Tests:
> `tests/test_insurance.py` incl. headless engine-snapshot integration.

**What:** When the dealer shows an Ace, a dedicated advice chip: **Take / Decline
insurance** (and even-money on blackjack), computed exactly.

**Why:** Insurance is the single most valuable deviation in all of blackjack — worth
>30% of all play-variation gain (Schlesinger). With exact composition it's a *solved*
decision: insure iff remaining tens fraction > 1/3 — strictly better than the human
TC ≥ +3 proxy (show that too, as the explanation). The app currently gives no insurance
advice at all.

**How:** One-liner from the counter (`per_rank["10"] / cards_remaining`), surfaced in
`publish_snapshot()` + a `TableView` chip next to the dealer card when up-card == Ace.

**Effort:** half a day. Highest value-per-line-of-code in the list.

## 6. Split-hand support

**What:** After a split, a seat holds two hands: group the seat's cards into hand 1 /
hand 2 (spatial clustering within the seat polygon), render two card rows, and advise
each hand independently (already an open task in TASKS.md).

**Why:** Today the advisor silently mis-advises split seats (all cards pooled into one
"hand"). Also required for the EV engine's post-split advice to be usable.

**How:** Extend `Seat` to a list of hands; cluster detections by x-offset within the
polygon; per-hand advice lines in `TableView`; card picker gains a hand selector.

**Effort:** ~2–3 days (detection-side grouping is the fiddly part).

## 7. Risk-aware bet ramp — fractional Kelly + bankroll

**What:** Replace the hardcoded 2x/1.5x/0.5x hints (`StrategyAdvisor.bet_suggestion`)
with a real bet ramp: estimated edge at the current true count × fractional Kelly
(½ or ¼ Kelly) × user-entered bankroll, with table min/max clamping. Same math powers
**Bet Behind** advice (bet behind only at positive counts, behind seats observed playing
basic strategy) and side-bet stake hints (tiny fractions — 100:1-heavy paytables have
huge variance).

**Why:** Bet variation supplies ~70–80% of a counter's total edge — it deserves better
than three fixed multipliers. Reference point (WoO benchmark, 6-deck S17, 1–15 spread,
4.5/6 pen, I18+Fab4): +0.834% player advantage.

**How:** `edge ≈ −house_edge + 0.5% × true_count` as the standard approximation, or
exact pre-deal EV from Feature 1 computed once per round in a background thread.
Bankroll field in the left panel (auto-filled by OCR later — see TASKS.md open item).

**Effort:** ~1–2 days.

## 8. Session analytics & shoe persistence

**What:** Persist every round to SQLite/JSONL: timestamp, shoe id, round, composition
snapshot, TC at bet time, advised vs taken action, dealer result, P&L. Dashboard tab:
EV-vs-actual chart, count accuracy over the shoe, per-shoe penetration, win-rate,
variance; CSV export. Restore shoe state after an app restart mid-shoe.

**Why:** Today nothing survives a restart (shoe count lost mid-shoe — real loss of edge)
and there's no way to verify the assistant's long-run accuracy or your discipline.
Closing the loop (advice → outcome) is what turns the app from a toy into a tool.

**How:** Hook `publish_snapshot()`/round transitions in the engine; write on round end;
`lib/logic/session_store.py` + a simple stats window (reuse the logging-window pattern).

**Effort:** ~2–3 days.

## 9. Table rules & paytable profiles + settings UI

**What:** A settings dialog and per-table profiles (JSON in `output/profiles/`):
S17/H17, peek vs ENHC, DAS, resplit limits, BJ pays 3:2 / 6:5, surrender, deck count,
which side bets are offered and their paytables, base bet, model confidence thresholds,
monitor/region profile.

**Why:** Every EV in Features 1/3/5/7 is conditional on the rules — hardcoded
`constants.py` values mean silently wrong advice on a different table. Evolution titles
themselves vary (peek vs no-peek), and side-bet paytables swing house edges by multiples
(21+3: 3.70% → 6.29% just by trips paying 25 instead of 30).

**How:** `RulesConfig` dataclass threaded through `StrategyAdvisor`/`ev_engine`/side-bet
engines; settings window writes the profile; ship presets ("Evolution Classic 8-deck",
"Evolution Infinite", …).

**Effort:** ~2–3 days.

## 10. Interactive region calibration

**What:** Click "Calibrate" → live screenshot overlay → drag seat-polygon vertices and
the dealer rectangle; save per resolution/table profile (already an open task).

**Why:** The hardcoded 2560×1440 polygons are the #1 portability blocker — on any other
stream layout, cards get assigned to wrong seats and the count silently corrupts.
Detection quality is the foundation under every EV feature above.

**How:** Reuse the existing region-preview canvas (`lib/logic/region_preview.py` +
`TableView.show_preview`); draggable vertex handles; persist into the Feature 9 profile.

**Effort:** ~2 days.

---

## Suggested order

1 → 5 → 2 (one coherent "precise advice" release, ~a week)
then 4 → 3 (side bets), then 9 → 10 (portability), then 6, 7, 8.

---

## Appendix A — Illustrious 18 + Fab 4 (Hi-Lo, multi-deck)

Verified against Wizard of Odds and the Blackjack Apprenticeship S17/H17 charts.
Convention: deviate when TC ≥ index; for negative indices basic strategy is Stand —
keep standing while TC ≥ index, hit below it.

| # | Hand | Dealer | S17 index | Action | H17 difference |
|---|------|--------|-----------|--------|----------------|
| 1 | Insurance | A | +3 | Take insurance | same |
| 2 | 16 | 10 | 0 | Stand | same |
| 3 | 15 | 10 | +4 | Stand | same |
| 4 | 10,10 | 5 | +5 | Split | same |
| 5 | 10,10 | 6 | +4 | Split | same |
| 6 | 10 | 10 | +4 | Double | same |
| 7 | 12 | 3 | +2 | Stand | same |
| 8 | 12 | 2 | +3 | Stand | same |
| 9 | 11 | A | +1 | Double | always double (basic strategy) |
| 10 | 9 | 2 | +1 | Double | same |
| 11 | 10 | A | +4 | Double | +3 |
| 12 | 9 | 7 | +3 | Double | same |
| 13 | 16 | 9 | +5 | Stand | same |
| 14 | 13 | 2 | −1 | Stand ≥ −1, else hit | same |
| 15 | 12 | 4 | 0 | Stand ≥ 0, else hit | same |
| 16 | 12 | 5 | −2 | Stand ≥ −2, else hit | same |
| 17 | 12 | 6 | −1 | Stand ≥ −1, else hit | ~1 pt more negative |
| 18 | 13 | 3 | −2 | Stand ≥ −2, else hit | same |

H17 extras: stand 16vA at +3, stand 15vA at +5; A8v6 double becomes basic strategy.

Fab 4 (late surrender; surrender when TC ≥ index — inert on standard Evolution tables):
14v10 at +3 · 15v10 at 0 · 15v9 at +2 · 15vA at +1 (H17: about −1, and 17vA surrender
becomes basic strategy).

## Appendix B — Wizard of Odds calculator endpoint (test oracle ONLY)

No public API exists, but the hand calculator is server-side. Verified live 2026-06-09:

```
GET https://wizardofodds.com/calculators-js/blackjack/calculate/
  a..j = remaining counts of ranks 2,3,4,5,6,7,8,9,T(=10/J/Q/K),A
  k    = 0 composition is pre-deal (server removes dealt cards) | 1 post-deal
  l    = BJ pays (1.5/1.4/1.2/1)      m = 1 US peek | 0 ENHC
  n    = dealer hits soft 17           o = double: 0 any / 1 9-11 / 2 10-11
  p,q  = max resplits non-aces/aces    r = hit split aces      s = DAS
  t    = surrender 0/1/2               u = upcard (2-9,T,A)    v = hand, e.g. "T6"
→ {"Stand":-0.5408,"Hit":-0.5359,"Double":-1.0719,"Split":0.0,"Insurance":...}
```

Exactness verified (TT vs A, 8 decks: Insurance = exactly −35/413). robots.txt and the
site ToS don't forbid it, but it's undocumented, ~0.8 s/request, unrated, and can vanish
or get bot-gated any time — use as a throttled golden-value oracle in unit tests and an
optional manual "verify online" button, never on the detection path. (That's the lesson
of the removed v2 integration, which also sent the wrong shoe and the wrong hand.)
