# Feature Roadmap V4 — 10 Top-Tier Features + 5 Enhancements

Researched 2026-06-11 (multi-agent: 5 codebase gap auditors over every subsystem +
web-research lenses, ~35 raw candidates deduplicated to 22, **each one adversarially
verified against the actual code with file:line evidence and web fact-checked**, then
scored by a 3-judge panel: player-edge impact, engineering leverage, daily-use value).
Numbered V4 because a separate, independently produced V3 draft exists
(`FEATURES_V3_unverified_agent_draft.md` — parallel session, claims not verified);
this document does not depend on it.

**Scope note:** the advantage-play-math, vision, and simulation research lenses were
lost to API failures and cut when the run was stopped, so this roadmap deliberately
leans analytics / ergonomics / integrity. Game-phase detection, the ghost-mode
executor, multi-table, and the simulation lab remain well covered by
`AUTONOMY_PLAN.md` and are complementary to everything below.

**Where V3.3 leaves the app:** perception (YOLO + winocr), exact math (EV, side
bets, insurance, Kelly, RoR), and the closed loop (settlement → bankroll → stats →
trainer) are shipped — both prior roadmaps 20/20 done. What the verification pass
found instead: the app computes the right numbers and then **drops, hides, or never
compares** many of them. The deal-time exact edge is published every round and
thrown away at persist time; the Kelly suggestion and the placed bet sit side by
side in every snapshot and are never compared; OCR liveness shows "fresh" while
every field fails to parse; side-bet settlement money has been persisted since V3.3
and is invisible. V4 is mostly about closing those last gaps.

**⚠ Fix-now finding (pre-roadmap):** two independent verifiers flagged that
`constants.RULES` claims its defaults "match standard Evolution" while setting
`peek=False` (ENHC takes-all) — but Wizard of Odds and Live Casino Comparer both
report classic Evolution blackjack **deals a hole card and peeks under an ace**
(WoO live-dealer survey: Evolution Peek=Yes; LCC classic review: "dealer checks for
blackjack"). Peek-vs-ENHC moves every EV by up to ~0.11%. Verify against your
table's game help and flip the default. One line, real money.

**Deliberately not proposed:** audio cues (explicit user decision in V2 #4), full
unattended autonomy (advised against in `AUTONOMY_PLAN.md` §7), alternative count
systems (dominated by the exact-composition engine), and re-proposals of anything
shipped — every candidate below survived a code-level novelty check.

**Shared prerequisite slice (~0.5 day, land first):** Features 1, 3, 5 and
Enhancement E5 all need the same additive `rounds` columns —
`bet_suggested REAL`, `edge_pre_deal REAL`, `expected_eur REAL` — populated in
`record_round` from snapshot fields the engine already publishes
(`engine.py:1611–1628`). Every round played before this ships is permanently lost
to the analysis. Ship the migration immediately, even if the features wait.

---

# Part 1 — The 10 Features

## 1. Discipline guard — bet-ramp deviation alerts, discipline score, stop-loss ⭐

**What:** Visual-only discipline layer. At the round's first detected card (the
same moment `_round_stakes` freezes side-bet stakes, `engine.py:608-611` — there is
no bet-lock event until phase detection ships) the engine freezes the Kelly call,
the placed bet, and `edge_exact`, and publishes a `discipline` snapshot block
`{delta_eur, ev_given_up, risk_cost_ce, loss_streak, stop_flag}`. HUD and sidebar
render "OVERBET +€30 vs ramp" / "UNDERBET −€20 (gave up €0.42 EV)". Each settled
round persists `bet_suggested` and `ev_given_up`; the stats window gains a
discipline score (% of owned rounds within tolerance, cumulative EUR of expectation
given up). Configurable stop-loss / stop-win (EUR, vs `session_pnl`) and an
N-consecutive-losses cooldown render as a banner with countdown. No enforcement,
no audio, no new threads.

**Why:** Bet variation supplies the large majority of a shoe-game counter's edge
(the app's own `betting.py` docstring: ~70–80%), and **every number in the shipped
risk stack assumes the ramp is actually followed** — `bankroll.model_round_stats`
builds μ/σ from the configured ramp and the RoR closed forms consume them. Yet
nothing measures adherence: the engine computes suggestion and placed bet side by
side every snapshot (`engine.py:1611, 1625-1628`) and never compares them, and only
the placed side is persisted. Discipline enforcement is the cheapest RoR reduction
there is. (Market echo: TiltBreaker proved poker players paid for automated
stop-loss; abandoned with no successor.)

**How:** Engine worker owns all inputs — freeze beside `_round_stakes`; track
`loss_streak` in `_settle_round` (units already at `engine.py:976`). Cost math is
pure logic (new `discipline.py` or `betting.py`), **with the corrected formula**:
`ev_given_up = max(0, (B_sugg − B_placed) × edge)` for underbet-at-+EV /
overbet-at-−EV (signed, NOT `|Δ|×edge`), and for overbetting at +EV a
certainty-equivalent risk cost `CE(B_sugg) − CE(B_placed)` reusing
`bankroll.certainty_equivalent` (μ = B·edge, σ² ≈ 1.3·B²). Thresholds in
`constants.BETTING`, edited via `validation.attach_numeric_entry`. Rendering: new
keys in `format_hud_lines` (pure, extend `tests/test_hud.py`) + a sidebar strip;
banner carries `cooldown_until` so the Tk poll renders the countdown.

**Effort:** ~3 days. Highest-scored candidate across all three judges.

## 2. Global hotkey layer + keyboard card correction (never alt-tab)

**What:** Global hotkeys via ctypes/user32 (no new deps) that work while the casino
browser has focus: toggle "my seat" per seat, new round, new shoe
(press-again-within-2s confirm — no modal), HUD toggle, dismiss reshuffle badge,
and a correction popover: a compact always-on-top Toplevel on the game monitor
showing the seat's current cards; keystroke grammar — seat digit, then rank+suit to
append a missed card ("3 k s"), "x" deletes the seat's last card, "!" replaces.
On commit it calls `engine.replace_card` with the `expected_round` guard and snaps
focus back to the saved browser window. Bindings editable in the settings App tab
with live conflict detection (`RegisterHotKey` → 0 + `GetLastError`).

**Why:** Verified gap — every actionable control is a mouse-only Tk handler
requiring app focus, and the corrector is a modal mouse grid; during a 10–16 s
Evolution betting window a fix costs an alt-tab plus a mouse trip from the 1440p
game monitor to the 144-DPI primary. Misread cards corrupt the exact composition
that **everything** is built on (`replace_card` provably uncounts/recounts,
`engine.py:803-816`), and corrections feed the active-learning pipeline — making
them near-free protects the app's core guarantee. Also provides the global
kill-switch hotkey the AUTONOMY_PLAN executor track already assumes.

**How:** New `lib/logic/hotkeys.py`: HotkeyManager daemon thread —
`RegisterHotKey(NULL, id, mods|MOD_NOREPEAT, vk)` + `GetMessageW` pump on the same
thread (MS-documented requirement); WM_HOTKEY → `queue.Queue`; stop via
`PostThreadMessage(WM_QUIT)`. Inject a user32 facade so tests run without Win32.
GUI drains the queue inside the existing 120 ms `_poll_snapshot` and dispatches to
existing handlers — all engine calls stay on the Tk thread exactly like today's
buttons. **Verified hazard: defaults must avoid Ctrl+Alt combos** — AltGr
synthesizes MOD_CONTROL|MOD_ALT on European layouts (AltGr+5 = € on
US-International), so system-wide Ctrl+Alt bindings would intercept typing in all
apps. Use Ctrl+Shift+F1..F7 etc. Focus return works because pressing a registered
hotkey grants foreground-activation permission (Raymond Chen).

**Effort:** ~2–3 days.

## 3. EV "green line" graph — expected vs actual P&L with luck bands

**What:** A Graphs tab in the stats window: cumulative actual settled P&L (orange,
from existing `pnl_eur`) vs cumulative expected P&L (green) with ±1σ/±2σ envelopes
and a z-score verdict line ("running +1.8σ above expectation over 412 rounds —
luck, not skill; expectation covers 96% of settled rounds"). Expected P&L per round
= edge frozen at first card × bet frozen at the same instant × owned seats settled,
plus frozen side-bet stakes × exact side-bet EVs.

**Why:** The poker precedent is real (HM3's "Show All-in Adj" EV line, verified)
and the blackjack version is strictly stronger: HM3 can only luck-adjust all-in
pots, while the shipped exact pre-deal engine prices **every** round's true
expectation from the live shoe. Beyond run-bad-vs-leak psychology, a persistent
actual-vs-expected divergence outside the bands is the earliest automated detector
of a broken pipeline — miscounted cards, drifting detection, or a wrong paytable
show up here first. The deal-time edge is currently published every snapshot
(`engine.py:1627`) and dropped at persist time.

**How:** Persist via the shared prerequisite slice. Pure series math in
`lib/logic/luck.py`: cumulative σ = √Σ bet² × (n·v + n(n−1)·c) with
v = `BETTING["variance"]` (app ships 1.33; WoO per-hand variance 1.303, SD 1.142)
and **covariance c = 0.479 between simultaneous hands** (WoO; required because the
user can own multiple starred seats). Chart on plain `tk.Canvas` — two polylines,
filled polygons for bands, all dimensions via `scaling.px`; wrap the stats grid in
a `ttk.Notebook`. Report expectation *coverage* honestly (NULL when the sweep
hadn't landed) rather than pretending 100%.

**Effort:** ~2–3 days.

## 4. Windows toast notifications for away-from-screen events

**What:** Silent Action Center toasts for events that matter while deliberately not
watching: wong-in (sit_out was true, exact edge turns positive), reshuffle latch,
side bet crossing +EV (hysteresis: notify > +0.5%, re-arm < 0), engine
stall/backoff. Per-trigger toggles + cooldowns, plus a foreground gate that
suppresses toasts while the foreground window is on the game monitor. All toasts
`ToastAudio(silent=True)` — the no-audio decision holds.

**Why:** Wonging is the one shipped feature whose entire premise is NOT watching
the table during negative counts — yet the wong-in signal renders only as pixels
inside the app. The engine keeps counting regardless; a toast converts dead
supervision time into free time. Cost is honest-zero new runtime risk: the venv
already runs pywinrt 3.2.1 on Python 3.14.3 for OCR; this adds exactly two
verified cp314 wheels (`winrt-Windows.UI.Notifications`,
`winrt-Windows.Data.Xml.Dom` at 3.2.1, confirmed on PyPI).

**How:** Pure `ToastRules` class (rising-edge diffs over snapshots, fully
unit-testable, no winrt imports) fed at the end of `run_cycle`, fires submitted to
the existing `_io_pool` — engine stays Tk-free. `lib/logic/notify.py` lazy-imports
`windows_toasts` guarded like `ocr.py:21-24` (degrades silently). Works unpackaged
via `create_toast_notifier_with_id` — no Start Menu shortcut needed. The wong-in
trigger needs `sit_out` added to the snapshot (computed in `betting.py:35-39`,
currently dropped at `engine.py:1625`). Caveat documented: Windows auto-DND can
demote banners during fullscreen apps; toasts still land in Action Center, and
wong-in windows persist for minutes.

**Effort:** ~2 days (first task: 5-minute Windows-Toasts smoke test on 3.14; clean
fallback is ~40 lines of raw winrt XML).

## 5. Session flight-recorder report (shareable HTML)

**What:** One-click "Session report" in the stats window producing a single
self-contained HTML file (`output/reports/`, opened via `os.startfile`; print to
PDF from the browser): luck-adjusted P&L chart with σ band, bankroll curve, TC
histogram with bet-by-TC overlay, per-seat discipline table (straight from
`seat_stats`), a mistake ledger priced in EV-lost units, side-bet results vs their
exact EVs, and a session fingerprint (rules profile, paytable hash, penetration).

**Why:** Raw session P&L is statistically meaningless — variance dominates edge;
the AP literature's answer is EV-based accounting (bj21 "Expected Value vs.
Reality"; reported realization among experienced counters spans ~2/3 of theoretical
to above 100% — a spread wide enough that you cannot know your bucket without
measuring). This app measures it with an oracle-validated engine instead of a
linear TC approximation. Today nothing shareable or retrospective exists:
`stats_window` is a live-only label grid and `export_csv` dumps raw JSON columns.

**How:** Phase 1 is the shared persistence slice. Phase 2: pure
`report.build_report(store, session_only) -> str` with hand-rolled inline SVG (two
polylines + one bar histogram, zero deps), reusing `stats()`, `seat_stats()`,
`settled_pnl()`; mistake ledger prices stored book/optimal divergences via
`ev_offload.run("report", ...)` on a TC-reconstructed shoe (documented as
approximate until the comp column ships — see Smaller Improvements). Phase 3:
generate on a daemon thread, marshal success via `self.after`.

**Effort:** ~3–5 days total, separable phases.

## 6. LAN companion dashboard (read-only HUD on a phone)

**What:** Opt-in read-only web mirror of the HUD served from inside the app over
stdlib HTTP. A phone on the same Wi-Fi opens one token-protected URL: big TC/RC/
decks, exact edge + bet call, insurance alert, starred-seat advice, +EV side-bet
flags, session P&L — streamed via Server-Sent Events at snapshot rate, built from
the **same pure `format_hud_lines`** the overlay uses, so the two surfaces can
never disagree. Settings gain enable/port/token + a QR code for connect.

**Why:** The overlay HUD still competes for pixels on the 1440p game monitor and
needs a stack of Windows tricks to stay up (`hud.py:1-12`). A phone next to the
keyboard is the zero-overlay version: the game monitor shows only the casino,
nothing advantage-play-related is capturable by screen recording or screen share,
and no Tk mixed-DPI scaling is involved. With audio ruled out, a glanceable second
screen is the one quiet delivery channel left.

**How:** `web_dashboard.py`: `ThreadingHTTPServer` (daemon) on a worker thread;
`GET /events?t=TOKEN` SSE loop polling `engine.get_snapshot()` (lock-swapped
immutable dict, safe off-thread — same way the GUI consumes it), emitting only on
seq change; 403 without token; clean stop from `_on_close`. EventSource cannot
send custom headers — token rides the query string (acceptable for a home-LAN
read-only threat model). QR: vendor Nayuki's `qrcodegen.py` (MIT, single file,
pure Python — verified real; do not hand-roll Reed-Solomon). Zero new packages.
Out of scope, resist creep: no controls, no HTTPS, no PWA.

**Effort:** ~2–3 days.

## 7. Multi-profile bankroll segregation (per-casino ledgers)

**What:** Named bankroll ledgers — each holds bankroll, table min/max,
kelly_fraction, base_edge, and accumulates its own settled P&L; exactly one active,
switchable from a sidebar combobox (mirroring the region-profile switcher). Kelly,
RoR closed forms, and the Monte Carlo bootstrap compute against the active ledger
only; OCR balance sync and settlement write to it only; every round is stamped with
its ledger so per-site stats and the MC resample are clean.

**Why:** One global bankroll float (`constants.py:162`) is the basis for
everything. A user playing two casinos hits two verified failures: (1) OCR sync
from site A overwrites the bankroll (`engine.py:1416-1421`) and persists it, so
site B's bet sizes and ruin numbers are computed on site A's balance — silent
mechanical corruption; (2) `settled_pnl()` feeds the Monte Carlo a sample mixing
tables with different rules and paytables. Venue-sliced risk is the online analog
of Schlesinger's trip-bankroll concept (BJA p.132) the app already implements
time-sliced as `trip_ror`. Honest caveat: per-ledger Kelly is deliberately
conservative (pure Kelly sizes against the total fungible bankroll); a
single-ledger user loses nothing.

**How:** `settings.py` gains a `bankrolls` store; `constants.BETTING` remains the
live in-place view of the active ledger so `betting.py`, `bankroll.py`, engine
settlement/OCR, and all existing tests are untouched. Critical detail: fold live
BETTING values back into the active ledger before serializing or switching (the
engine worker mutates `BETTING["bankroll"]` in place); gate switching on
round-idle. `session_store`: additive `site TEXT` column; stats take an optional
site filter on `COALESCE(site,'Default')`.

**Effort:** ~3–4 days.

## 8. Visual session replayer with scrubber timeline

**What:** A read-only Round Replayer window (PT4-style): left pane lists sessions
and rounds with filter chips (my seats, EV-diverged-from-book, settled losses, +EV
side-bet rounds); right pane is a read-only `TableView` replaying the selected
round card-by-card via a `tk.Scale` scrubber, with back-computed RC/TC/decks at the
scrub position, stored advice/optimal strings, and the settlement outcome.

**Why:** The DB already stores per-round cards, dealer playout, advice/optimal
strings, and full settlement JSON, but there is no way to *see* a round —
`seat_stats` counts divergences without letting you inspect which rounds diverged.
A filterable visual replayer is the standard study surface in PT4 / GTO Wizard
(both verified). Scoped honestly: round-level replay is exact from day one;
within-round order is reconstructed (canonical no-hole-card order — matches
Evolution's actual dealing) until a new `deal_order` column accumulates data.

**How:** Pure `lib/logic/replay.py` (order reconstruction, per-step count
back-computation, snapshot synthesis for TableView); `session_store.sessions()` +
`rounds_for_session(filters)`; `table_view` gains a `read_only` flag;
`replay_window.py` wires it with `trainer_window` as the structural template.
Note (verified): "Index says" lines, `hand_of` split tags, and deal order are NOT
currently persisted — the seats-JSON keys need extending at write time.

**Effort:** ~3–5 days.

## 9. Evolution variant rules pack — Free Bet first, correctness for the rest

**What:** Composition-exact modeling of Evolution 7-seat **Free Bet Blackjack** —
free doubles on two-card hard 9–11, free splits on non-ten pairs, dealer-22 push
(player-BJ exemption as a confirmable flag) — with settlement, Kelly, and trainer
made push-22-aware and Hi-Lo indices variant-gated off. Plus a Six-Card-Charlie
flag unlocking Infinite / Infinite Free Bet in an explicit "no-edge table:
correct-play only" mode (count/Kelly UI disabled). Power Blackjack ships only as a
disabled preset with a reason (stripped 352-card shoe; countability unverified,
likely dead).

**Why:** Pointed at a Free Bet table today the app is silently wrong three ways:
EVs ignore push-22 and free-stake asymmetry, the I18 line gives indices derived for
the wrong game, and settlement books dealer-22 rounds as wins. The 7-seat Free Bet
is shoe-dealt (one review claims ~75% penetration — deeper than standard; verify at
the table) and **unmodeled composition-exactly anywhere publicly** — only static
charts exist. Counting folklore says Free Bet's count window is weak; the app's
exact `predeal_ev` settles that empirically per shoe instead of trusting folklore.
Honest framing: one new playable table family done correctly, plus correctness
coverage for two count-dead ones (Infinite returns discards to the shoe each round
— verified counting-dead; Charlie worth ~0.16%).

**How:** `ev_engine.py`: extend the dealer PMF to 7 terminal slots (17–21,
bust-22, bust-23+; classic rules sum the two bust slots so existing goldens stay
bit-identical); free-double EV = 2·P(win) − P(lose); free split spawns one
normal-stake and one free hand; `charlie_6` carries hand-card-count in the memo key
behind the flag. `settlement.py` gains the push-22 branch; `deviations.py` returns
nothing for non-classic variants. Validation anchors (no WoO calculator exists for
Free Bet): WoO's published 1.04% HE (6D, H17) and Evolution's official RTPs
(Free Bet 98.45%, Power 98.80%), plus a regression suite proving classic EVs
unchanged to 1e-12. All EV stays in `ev_offload` subprocesses.

**Effort:** ~4–6 days (Free Bet core 3–4; Charlie/Infinite ~1; Power stretch only
if live-verified).

## 10. Data portability pack — backup, merge, typed exports, profile sharing

**What:** Four pieces on one serialization layer: (1) `SessionStore.backup_to()`
via stdlib `sqlite3.Connection.backup()`, auto-triggered on the `_io_pool` every N
rounds and on close, timestamped with keep-last-N retention (point the folder at
OneDrive/Syncthing for free off-machine sync); (2) `merge_from(path)`: ATTACH a
foreign session.db, INSERT rounds deduped on (session_id, ts) with columns
intersected via `PRAGMA table_info` so pre-migration DBs import cleanly;
(3) typed exports — rounds / hands / side_bets as three flat tables, Parquet when
pyarrow imports (cp314 wheel verified on PyPI), else CSV; (4) region-profile
export/import JSON with casino/table/notes metadata.

**Why:** `output/session.db` is the product's memory and a single un-backed-up
file on one SSD: it holds the literal empirical sample the Monte Carlo bootstraps
(`settled_pnl`, `session_store.py:281-289`) — its value compounds with every
round. The shipped `export_csv` writes four columns as raw JSON strings inside CSV
cells (verified, `session_store.py:291-308`) — unusable in pandas/Excel without
custom parsing. Hours of calibration drag-work are locked inside
`region_profiles.json` with no way to move machines.

**How:** All writers stay on the engine's `_io_pool` (same thread as every SQLite
writer today — zero locking concerns); `_on_close` does one quick synchronous
backup after `controller.stop()`. The flattener is a new `_iter_flat_rows()`
generator over the same JSON shapes `record_round` writes. Profile import renames
on collision; metadata fields are additive.

**Effort:** ~3.5–5 days, cleanly separable (backup 1d is the urgent slice).

---

# Part 2 — The 5 Enhancements

## E1. HUD pack: click-through mode, opacity, Glance Mode

**What:** Three composing upgrades to the shipped overlay (V2 #4): (1)
click-through mode — OR `WS_EX_LAYERED|WS_EX_TRANSPARENT` into the HUD's exstyle so
all mouse input passes to the casino page beneath, plus an opacity slider (30–99% —
capped below 100% because Tk drops WS_EX_LAYERED at alpha 1.0, which would kill
click-through; verified interaction); (2) Glance Mode — an alternate single-line
render in 28–48pt type showing one thing by context priority: insurance alert ≻
first unresolved owned-seat action ≻ bet call ≻ dimmed TC, via a pure
`next_glance_line(snap)` beside `format_hud_lines`; (3) geometry/mode/opacity
persisted and restored clamped to the saved monitor's work area.

**Why:** Fixes the HUD's two verified failure modes with pure presentation
leverage: it eats clicks (Button-1/B1-Motion bound across the whole panel,
`hud.py:111-113`) so it must be parked off the action — but next to a fullscreen
1440p stream there is nowhere cheap to park it; and it's slow to read — six 9–14pt
rows at 60–90 cm inside a decision window that is **7 seconds on Speed Blackjack**
(auto-stand over 12 on timeout, verified). The app already computes the single
decision that matters; Glance is the rendering that matches that reality.

**How:** Extend `_apply_win_styles`; move style reassertion into the existing 2 s
topmost timer (alpha transitions can clear LAYERED). Fonts sized from the HUD's
own monitor DPI via `scaling._window_dpi` — NOT the root-scaled constants (root is
on the 144-DPI primary, HUD on the 96-DPI game monitor). One engine line: publish
the raw `suggest()` dict as `bet_info` (engine currently publishes only the display
text). Escape hatch: the main-window HUD button on the other monitor, since the
HUD itself can't receive clicks in ghost mode.

**Effort:** ~2–3 days. Highest daily-use score in the pool.

## E2. OCR runtime robustness & health pack

**What:** (1) Truthful per-field OCR health: per region (balance/bet/result) track
last attempt ts, last *successful-parse* ts, consecutive fails, and a parse-state
enum; colored sidebar dot (green/yellow/red) with a per-field tooltip. (2) The
settlement-vs-banner mismatch — the one check that catches a misread card
corrupting P&L — currently fires only into the log stream (`engine.py:980-987`);
give it a dismissable banner styled exactly like the reshuffle badge. (3) Fuzzy
banner snap via `difflib.get_close_matches` (cutoff ~0.75) so "Loose" → "lose"
(banner is cross-check-only; a false snap can't touch money). (4) Manual-edit
cooldown: after a sidebar bankroll commit, OCR honors a configurable ~10 s window
before overwriting, and post-cooldown reversals log an explicit conflict. Plus two
verified zero-value bugs: `if balance and` / `if bet and` skip legitimate €0 syncs
(`engine.py:1408, 1423`).

**Why:** OCR writes directly into bankroll, bet size (which scales every Kelly
stake), and settlement cross-checks, but its failure modes are invisible exactly
where it matters: **the shipped liveness indicator is misleading** —
`_ocr_last["ts"]` is stamped even when every field failed to parse
(`engine.py:1437`, `ocr.py:127-128`), so the UI shows "OCR 1s" indefinitely while
the bankroll silently stops tracking. Note (verified): Windows.Media.Ocr exposes
NO confidence scores — health must come from parse-state heuristics, not engine
confidence.

**How:** `read_regions` returns per-key (text, state); engine extends per-field
state and publishes `ocr_health` + `ocr_mismatch`; GUI clones the reshuffle-badge
block. All same-pipeline changes; no new dependencies.

**Effort:** ~2–3 days.

## E3. Evolution table rules presets (source-verified, with caveats)

**What:** A "Table preset" combobox atop the settings Rules tab loading complete,
source-verified rule+paytable profiles: Classic family (8D, S17, BJ 3:2, **DAS
allowed**, split once, no surrender, **hole card with ace-only peek**) and Infinite
(same but **no DAS**, six-card Charlie flagged "not modeled — costs ~0.16% of shown
EV"), plus deliberately greyed-out entries with reason tooltips for Free Bet
("push-22 not in the EV recursion; advice would be wrong") and Power ("352-card
shoe breaks the composition model") until Feature 9 ships. Each preset stores its
source URL + retrieval date + a 3-item verify-in-game-help checklist shown on
apply.

**Why:** Every EV, deviation, and Kelly stake is conditional on RULES, and the
failure mode is live today (see the fix-now finding above). One wrong toggle moves
the edge by ~0.11–0.14% (DAS 0.14%, hole-card handling 0.11% — WoO rule
variations), the same order as the entire counting edge per true count. Published
sources genuinely disagree per table (Infinite HE: 0.45% WoO-calculated vs 0.49%
Evolution-advertised vs 0.53% LCC) — which is exactly why presets must carry
sources and a checklist rather than pretend to be authoritative. Converts "the app
gave wrong advice on a variant" into "the app told me why it can't".

**How:** New `lib/common/table_presets.py` (ordered dict: settings-shaped values,
source, caveats); `settings.py` extended so side-bet entries can carry paytable
dicts (coerce numeric keys to int — `bust_it` is int-indexed); apply fills the
existing dialog vars so the user reviews before Save; `engine._settings_sig` and
`settlement.paytable_hash` already react — zero engine work. Schema-drift test:
every preset validates against `constants.RULES` and SIDE_BETS keys in CI.

**Effort:** ~1.5–2 days. Cheap, high trust value.

## E4. Shoe-composition integrity guard

**What:** (1) `CardCounter` integrity check: warn when any bucket exceeds its
physical maximum (deck_count×4, ten-bucket ×16) or `cards_seen` > deck_count×52;
`adjust_manual` logs a WARNING when a click crosses the bound (the +/- still
applies — user stays in control); `apply_state` clamps restored values and reports
what it clamped. (2) Snapshot fields: `composition_warning` (human-readable) and
`uncounted_on_table` (cards excluded from the displayed count but included in EV
after a mid-round shoe reset). (3) Sidebar badge via the reshuffle-badge pattern.

**Why:** The core guarantee is exact composition EV with real money on its output.
Every negative-direction error is already clamped, but **the upper bound is wide
open** (verified): a stuck/repeated + click or a corrupted persisted shoe state can
push `per_rank` past physical maxima, at which point `comp_from_per_rank` silently
clamps to zero remaining (`ev_engine.py:83`) and every downstream product —
pre-deal edge, Kelly, insurance, side-bet EVs — produces wrong-but-plausible
numbers with no indication. The analogous OCR failure is already loud; manual
count corruption deserves the same. Also removes the "why don't these numbers
match" confusion after mid-round resets (displayed count intentionally excludes
table cards while EV includes them — correct but invisible).

**How:** `bucket_max(key, deck_count)` in `counting.py`, checks computed inside
the existing lock in `snapshot()`/`get_state()` (10-key loop, negligible);
`publish_snapshot` folds warnings + the `_ev_count` uncounted sum; badge clones
`modern_gui.py:467-482`. No changes to `ev_engine`/`ev_offload`/subprocess code.

**Effort:** ~1.5 days. Cheapest insurance in the list.

## E5. Stats reporting pack — EUR/hour, TC-bucket audit, side-bet performance

**What:** Four deliverables on the existing rounds DB: (1) time reporting —
gap-aware session duration (split on >15 min idle), rounds/hour, realized EUR/hour,
hour-of-day / day-of-week breakdowns; (2) **TC-bucket ramp audit** — buckets keyed
on the PRE-DEAL true count via `LAG(true_count) OVER (PARTITION BY session_id ORDER
BY id)` (the stored TC is at-settlement and includes the round's own cards —
verified subtlety), showing rounds, avg bet, net EUR per bucket: "does my ramp
actually put money on the high counts?"; (3) side-bet performance tab — per bet:
staked, wins, hit rate, net, ROI, top tier hits — aggregated from the *settlement*
JSON's per-seat entries (persisted since V3.3, invisible today; note the `side_bets`
column holds pre-deal EVs — wrong source, verified); (4) live freshness — Stats and
Bankroll windows auto-refresh via a cheap `SELECT MAX(id)` change check.

**Why:** EUR/hour, not EUR/hand, is what a professional optimizes (PT4 added
hour-of-day/day-of-week filters in 2012; Evolution states Speed Blackjack rounds
are ~30% quicker — table pace genuinely varies). Every input is already persisted;
none of it is visible. The TC-bucket audit is the blackjack-native positional
report and the only way to verify from real data that the Kelly ramp and wong-out
logic concentrate stake where the edge lives.

**How:** `session_store.py`: `time_stats()` (gap-split in Python), `tc_buckets()`
(window function — venv SQLite 3.50.4 supports LAG, verified),
`sidebet_performance()` (parse settlement JSON); additive `edge REAL` column going
forward. `stats_window.py`: `ttk.Notebook` tabs, label-grid pattern + small Canvas
bar charts (DPI-safe via `scaling.px`); auto-refresh `after()` loop that only
recomputes on row-count change, cancelled on destroy; Monte Carlo stays manual.

**Effort:** ~3 days.

---

## Suggested order

1. **Day 0:** the shared persistence slice + the `peek` default fix — both
   preserve data/correctness you can't recover later.
2. **Live-play release:** 1 (discipline guard) → 2 (hotkeys) → E1 (HUD pack) —
   everything the user touches mid-hand.
3. **Trust release:** E2 (OCR health) + E4 (shoe integrity) + E3 (rules presets) —
   makes the money path visibly-wrong instead of silently-wrong.
4. **Analytics release:** E5 (stats pack) → 3 (green line) → 5 (flight recorder) —
   one data layer, three surfaces.
5. **Anytime, independent:** 4 (toasts), 10-slice-1 (backup), 6 (LAN dashboard).
6. **Bigger bets, when appetite allows:** 7 (ledgers), 8 (replayer), 9 (Free Bet
   pack — the most ambitious math work in the list).

## Smaller improvements (verified keep, didn't make the cut — batch in when nearby code is open)

- **Detection confidence & health surfacing** — per-card confidence into
  snapshots/rounds JSON + warning borders in table view; flap-rejection counters;
  `model_refresh_ts` has zero UI readers today; training-data cap (5000 files)
  warns only after collection already stopped. (~3–4 d)
- **Capture-to-advice latency & EV-degradation telemetry** — EV-timeout fallbacks
  are visible per-seat but never counted or trended; a user could be playing book
  instead of exact-EV on a large share of hands without knowing. The
  PERFORMANCE log category exists and nothing has ever emitted into it. (~2–3 d)
- **Leak report (exact-EV cost of real mistakes)** — GTO-Wizard-style regrade of
  owned-seat decisions; requires persisting the counter state per round (comp
  column), which also upgrades the replay trainer from approximate to historically
  exact. No tool grades real online-casino rounds today. (~3–4 d)
- **Speed Blackjack fast-table mode** — pacing preset (CYCLE_SLEEP_DEALING 0.25,
  EV timeout 2.0 s) + latency line; decision window is roughly 6–10 s with
  auto-stand 12+ / auto-hit ≤11 on expiry. (~2 d)
- **OCR calibration preview** — "Test OCR" button running the production read path
  inline in the region editor; rect-size validation (engine silently drops <4 px
  rects); clone-and-scale prefill from the nearest calibrated resolution (only
  2560×1440 is calibrated today — every monitor switch dead-ends OCR). (~2–3 d)
- **Filterable round browser** — Treeview grid over recorded rounds with
  composable filters + aggregate footer; pairs naturally with Feature 8. (~2–3 d)
- **Drill my leaks** — trainer decks weighted by your own costliest recorded
  mistakes (CV "Drill Errors" analog); today's replay drill samples ALL seats
  including strangers' hands. (~3 d, 1.5 d after the leak report)

## Verified numbers cheat-sheet (for implementation)

- Blackjack per-hand variance ≈ 1.303 (SD 1.142, rule-dependent ~1.30–1.36); app
  ships `BETTING["variance"]=1.33` — use the constant. **Covariance between
  simultaneous hands at the same table: 0.479**; total round variance
  n·v + n(n−1)·c (Wizard of Odds).
- Discipline cost: signed `(B*−B)×edge` only for underbet-at-+EV /
  overbet-at-−EV; overbet-at-+EV is a *risk* cost — use CE loss,
  CE = μ − σ²/2B with μ = B·edge, σ² ≈ 1.3·B².
- Rule deltas (WoO): DAS ±0.14%, ENHC −0.11%, six-card Charlie +0.16%,
  no-resplit −0.10%. Classic Evolution: 8D, S17, DAS, hole card with ace-only
  peek, HE 0.72%. Infinite: NO DAS, Charlie, HE 0.45/0.49/0.53% (sources
  disagree). Free Bet: push-22, HE 1.04% (6D H17 land rules), Evolution RTP
  98.45%. Power: 352-card shoe (no 9s/ten-spots; J/Q/K remain), RTP 98.80%.
- Speed Blackjack: betting/decision windows cut by ~7 s vs classic (~6–10 s to
  act); timeout auto-stands 12+ / auto-hits ≤11; same 8-deck shoe, "no impact on
  the maths" (Evolution).
- Hotkeys: `RegisterHotKey` + message pump must share a thread; MOD_NOREPEAT;
  **never default to Ctrl+Alt combos** (AltGr = Ctrl+Alt on EU layouts).
- Windows.Media.Ocr exposes NO per-word confidence (legacy UWP API) — health
  signals must be parse-state heuristics.
- Toasts: `windows-toasts` needs only `winrt-Windows.UI.Notifications` +
  `winrt-Windows.Data.Xml.Dom` beyond the installed pywinrt 3.2.1; cp314 wheels
  verified; `ToastAudio(silent=True)`; works unpackaged via
  `create_toast_notifier_with_id`.
- pyarrow 24.0.0 ships cp314 win_amd64 wheels (typed Parquet export);
  `sqlite3.Connection.backup()` is stdlib and online-safe; venv SQLite 3.50.4
  supports window functions (LAG).
- EV realization reported by experienced counters spans ~2/3 of theoretical to
  above 100% (bj21) — the spread is the argument for measuring your own.
