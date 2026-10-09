# Autonomy / Browser-Control Assessment

Researched 2026-06-10. Question: can the app stop being a read-only advisor and
**act on the table itself** — place bets and play Hit / Stand / Double / Split
in the browser, with safeguards?

Scope note: you asked me to skip the legal/ToS research, so this is a **technical
feasibility + design** document. One risk line is kept (it's the dominant input to
the recommendation, not a lecture); everything else is mechanics.

---

## TL;DR / Recommendation

**Build the executor in this order, and stop after stage 2 unless you have a very
good reason:**

1. **Dry-run / "ghost" mode** — the app computes the action *and the exact click it
   would make*, draws it on the HUD, logs it. Clicks nothing. (Low effort, pure upside:
   it's also the only honest way to measure whether your detection is good enough to
   ever be trusted with money.)
2. **One-key assisted execute** — ghost mode + a single confirm key/click per action.
   The human stays the actor on every irreversible money move; the app removes the
   *aiming and remembering*, not the *deciding to commit*.
3. **Full unattended autonomy** — **advised against.** Not primarily on principle:
   on engineering risk. See below.

**Two technical conclusions that drive everything:**

- **Don't use Selenium/Playwright DOM automation.** An Evolution live table is a
  **video stream + a `<canvas>`/WebGL UI inside a cross-origin iframe**. The cards
  are *pixels* (that's the whole reason this app needs a YOLO model), and the buttons
  are often canvas-drawn with no DOM node. There is almost nothing meaningful to
  `querySelector`. DOM automation buys you nothing for *reading* and is brittle for
  *clicking*.
- **The hard 80% is not clicking — it's phase/turn detection**, which the app does not
  have yet. `get_snapshot()` already tells you *what to do* (`optimal_action`); it has
  no idea *whether it's legal to do it right now*. That gap is where the bugs that cost
  real money live.

**The one risk line:** automated play is a flat ToS violation on every real-money
operator, and the standard enforcement is *frozen account + confiscated balance and
winnings*, not a warning. So the failure mode of a bug here isn't "lose a hand" — it's
"lose the bankroll, twice over." That asymmetry is the reason the recommendation stops
at stage 2.

---

## 1. What "the browser" actually is here

This is **live-dealer** blackjack (Evolution), not an RNG/software game. That changes
the entire approach:

| Layer | Reality | Consequence for automation |
|---|---|---|
| Cards | Pixels in a **live video stream** of a physical table | Must be read by vision (YOLO) — already solved. No DOM, no API. |
| Table UI (chips, bet spots, Hit/Stand/Double/Split, timer) | HTML/CSS **+ `<canvas>`/WebGL** overlay; newer clients are heavily canvas | Buttons may have **no DOM node**. Locate them by **vision**, click them by **coordinate**. |
| Embedding | Game served from an Evolution origin inside a **cross-origin `<iframe>`** on the operator page | Same-origin policy blocks DOM reach from the page; even via CDP the canvas isn't queryable. |
| Player actions | Sent to Evolution over a **WebSocket**; buttons are just triggers | You cannot cleanly forge the WS messages (signed/sequenced session) — you drive the *UI*, like a human. |
| Timing | Fixed **betting window (~12–15 s)** and **decision timer (~10–13 s)**; miss it → auto-skip/auto-stand | The act-loop has *seconds*, and they're hard deadlines. |

**Takeaway:** the only realistic interface is the same one a human uses — *look at the
screen, move a pointer, click a pixel, look again to confirm*. The app is already half
of that loop (the looking). Autonomy is building the other half (the clicking) **plus**
the part neither half has yet (knowing what phase the game is in).

---

## 2. Why OS/CDP input, not DOM automation

**Selenium / Playwright (DOM-driving) — wrong fit here:**
- Cross-origin iframe → no DOM reach from the operator page.
- The game surface is `<canvas>`/WebGL + `<video>` — *cards are rendered pixels*, so you
  still need vision regardless. Selenium reads nothing useful.
- Action buttons are frequently canvas-drawn (no node) or DOM with obfuscated, dynamic
  attributes → selectors are fragile-to-impossible.
- `navigator.webdriver` and the CDP automation surface are exactly what operator anti-bot
  fingerprints first.

Net: you converge on **vision for reading + coordinate clicks for acting** no matter what.

**Two ways to deliver a coordinate click:**

| Method | How | Pros | Cons |
|---|---|---|---|
| **OS-global input** | `pydirectinput` / `ctypes` `SendInput` move+click at a screen pixel | Hardware-real events; canvas accepts them unconditionally | **Seizes your physical mouse** — you can't use the PC while it runs; window must stay put/foreground |
| **CDP-injected input** (recommended) | Launch Chrome with `--remote-debugging-port`, attach with Playwright `connect_over_cdp`, send `Input.dispatchMouseEvent` at iframe-relative coords | Targets the tab even unfocused; **frees your physical cursor**; survives the window not being on top | Synthetic events; some canvas handlers can flag/ignore non-trusted input; more detectable |

Recommendation: **CDP-injected clicks** as the executor's output path (you keep using
your machine, and it targets the right tab), with OS-input as a fallback if a given
canvas ignores synthetic events. **All reading stays vision-based** (`mss` capture →
YOLO/`cv2`), exactly as today.

---

## 3. Architecture — where it plugs in

The app is already a clean producer/consumer: `DetectionEngine` publishes an immutable
`get_snapshot()`; the Tk GUI is just a consumer. **The executor is a second consumer of
that same snapshot** — it never touches engine internals.

```
                 ┌──────────────────────────────────────┐
 screen (mss) ─► │ DetectionEngine (worker thread)        │
                 │  capture → YOLO → state → snapshot      │
                 └───────────────┬──────────────────────────┘
                         get_snapshot()  (immutable)
                 ┌───────────────┴───────────────┐
                 ▼                                ▼
        ┌─────────────────┐            ┌────────────────────────┐
        │ Tk GUI (today)  │            │  Executor (NEW)         │
        │ renders advice  │            │  decide → guard → click │
        └─────────────────┘            └───────────┬─────────────┘
                                                    │ CDP / OS input
                                                    ▼
                                            Chrome tab (casino)
```

### 3.1 What the snapshot already gives the executor (free)

From `publish_snapshot()` (`lib/logic/engine.py:1313`), per consume:

- `seats[i].optimal_action` — the exact-EV action `S/H/D/P/R` (already computed off the
  Tk/engine threads, so it's *ready* before it's your turn).
- `seats[i].mine` — only act on seats the user owns (`my_seats`).
- `seats[i].can_split`, `seats[i].cards`, `seats[i].total`.
- `bet` / `bet_placed` / `edge_exact` — the bet-ramp size to wager.
- `insurance` — take/decline + even-money.
- `ocr` — `{balance, bet, result}` (via `lib/logic/ocr.py`, `winocr`), for limit checks
  and post-action verification.
- `dealer.locked`, `count` — context.

### 3.2 What it does NOT have — and must gain (the real work)

**`get_snapshot()` has no game *phase*.** It knows cards and counts, not "is betting
open" or "is it my turn." An executor that acts without that is a money-loss generator.
You need a new **`PhaseDetector`** producing a small state machine:

```
IDLE → BETTING_OPEN → DEALING → MY_TURN → WAITING → SETTLE → IDLE
```

Detection signals (all vision/OCR, fits the existing pipeline):

| Transition | Signal | How to detect |
|---|---|---|
| → BETTING_OPEN | "Place your bets" + countdown appears | OCR the timer digits (`winocr` already in repo); template-match the chip tray |
| bet registered | my chip sits on my bet spot | `cv2.matchTemplate` chip on the spot; cross-check `ocr.bet` |
| → MY_TURN | action buttons appear **and are enabled**, active-seat highlight on my seat | template-match buttons; sample button color (grey=disabled); detect the seat highlight ring/glow |
| action taken | buttons disappear / my card count changes | re-capture; compare seat card count to pre-click |
| → SETTLE | result banner | `ocr.result` (already parsed to win/lose/push/blackjack) |

This `PhaseDetector` + a per-control **template library** (saved crops of the Deal/Hit/
Stand/Double/Split/insurance buttons and chip denominations, per resolution) is the bulk
of the build. Everything downstream is a few lines.

### 3.3 The executor loop (pseudocode)

```python
class Executor:
    def step(self, snap, phase):
        if not self.armed:          return            # human-controlled toggle
        if not self.guards_ok(snap, phase): return    # see §5 — abstain on any doubt

        if phase == BETTING_OPEN and not self.bet_done:
            size = clamp(snap["bet_placed"], self.min_bet, self.max_bet)
            self.place_chips(size)                     # template-locate chips + spot
            self.verify(lambda s: chip_on_spot(s) and ocr_bet_matches(s, size))

        elif phase == MY_TURN:
            seat = my_active_seat(snap, phase)         # WHICH seat is lit — critical
            action = snap["seats"][seat]["optimal_action"]   # precomputed, instant
            btn = self.locate_button(action)           # matchTemplate, enabled-state
            if btn is None: return                     # abstain, never guess a pixel
            self.click(btn)                            # CDP Input.dispatchMouseEvent
            self.verify(lambda s: buttons_gone(s) or seat_cards_grew(s, seat))
```

---

## 4. The genuinely hard problems (independent of any policy)

1. **Turn/seat attribution is the crux.** The active-player highlight moves around the
   table; click Hit when the *neighbour's* seat is lit and you've played their hand. This
   is the single most dangerous bug and the app has zero code for it today.
2. **Hard real-time deadlines.** Betting ~12–15 s, decisions ~10–13 s. Capture +
   inference + (worst-case) a 2 s EV compute can blow a decision window. Mitigation: the
   snapshot *already* precomputes `optimal_action`, so the click is instant — but you must
   guarantee the action is ready *before* `MY_TURN`, and have a safe default (Stand) if it
   isn't.
3. **Coordinate fragility.** Hardcoded pixels break when the window moves/resizes, the
   client re-lays-out, an ad/“are you still there?”/disconnect modal pops, or a chip tray
   slides. A blind click into a shifted layout clicks *something else* (raises the bet,
   dismisses a warning, plays the wrong seat). Template-matching + "abstain if not found
   at expected confidence" is mandatory, not optional.
4. **Irreversibility.** Every click is real money with no undo. A false-positive
   `MY_TURN` that doubles a stiff, or a misread card, is an instant loss — and a bot will
   repeat the mistake calmly until a limit trips.
5. **State desync.** Network lag, a missed/misread card, a hand the bot didn't expect →
   internal state drifts from the table. Acting on a desynced state is how you get the
   big losses. Rule: **on any mismatch between expected and observed, disarm and alert —
   never retry blindly** (blind retry = double-bet / double-click).

The advisory app **degrades gracefully** (slightly-wrong advice, human catches it). The
same detection error in an autonomous executor **fails catastrophically** (acts wrongly,
with money, repeatedly). That inversion is the core argument.

---

## 5. Safeguards (the part you asked about)

Treat these as **non-negotiable preconditions**, gated in `guards_ok()` before *any*
click. Abstaining is always safe (worst case the timer auto-stands a hand); guessing is
not.

**Arming & abort**
- Explicit **ARM toggle** (off by default, re-armed every session) — nothing clicks unless armed.
- **Kill switch:** global hotkey + `pyautogui.FAILSAFE` (slam mouse to a screen corner → hard abort). Dead-man behaviour: any unexpected modal → auto-disarm.

**Per-click preconditions (all must hold)**
- **Detection-confidence floor** — act only when card/phase confidence is high; else abstain.
- **Phase certainty** — `MY_TURN`/`BETTING_OPEN` confirmed by *N consecutive frames* **and** the expected control found by template match at the expected confidence.
- **Seat certainty** — the active highlight is unambiguously on *my* seat (`mine == True`); ambiguous → abstain.
- **Geometry sanity** — never click a coordinate outside a known, calibrated control region.

**Money limits (auto-disarm on breach)**
- Max bet per round; max rounds/bets per shoe; **stop-loss** (session P&L floor) and **stop-win**.
- **Balance cross-check:** if `ocr.balance` disagrees with the expected balance beyond a tolerance, disarm — the bot's model of the world is wrong.

**Post-action verification**
- After every click, re-read the screen and confirm the effect (chip on spot / buttons gone / card count grew). **If unconfirmed: disarm and alert — do not re-click.**

**Audit**
- Log every decision + the exact click + a screenshot crop + confidence into the existing
  `SessionStore` (`lib/logic/session_store.py`) for after-the-fact review.

> Note these safeguards protect you from *your own bugs and bankroll* — they do nothing
> about the ToS/detection risk, which is unaffected by how carefully you click.

---

## 6. Build plan & effort

| Stage | Deliverable | New modules | Effort |
|---|---|---|---|
| 0 | **Template library + `PhaseDetector`** (timer OCR, button/highlight detection, state machine). *Most of the work.* | `lib/logic/phase.py`, `assets/controls/*` | ~4–6 d |
| 1 | **Dry-run executor** — computes & draws the click it *would* make; clicks nothing; logs match-rate vs. what you'd do | `lib/logic/executor.py` (ghost mode), HUD overlay | ~2–3 d |
| 2 | **One-key assisted execute** — CDP click path + all §5 guards, fired only on human confirm | `executor.py` (CDP via Playwright `connect_over_cdp`), `lib/interfaces/` arm toggle | ~3–5 d |
| 3 | Full unattended autonomy | (auto-fire the loop) | **not recommended** |

Suggested libraries (Windows): `playwright` (CDP `connect_over_cdp` + `Input.dispatchMouseEvent`),
`opencv` (already in) for `matchTemplate`, `winocr` (already in) for timer/balance,
`pydirectinput`/`ctypes SendInput` + `pyautogui` (FAILSAFE) for the OS-input fallback.

**Gate between stage 1 and 2:** run dry-run for a long, honest sample and measure how
often the *would-click* matches a correct human action across betting, every decision
type, splits, insurance, and the awkward states (disconnects, modals, mid-shoe shuffle).
If that match-rate isn't near-perfect, stage 2 is just an automated way to lose money —
do not arm it.

---

## 7. Recommendation

Build **stage 0 → 1 → 2**. They reuse everything you've already built (the snapshot,
the EV engine, OCR, the session store), they're genuinely useful (ghost mode is also a
detection-quality test bench; one-key assist removes the tedium without removing the
human), and they keep a person on every irreversible money action.

**Stop before stage 3.** Full unattended autonomy multiplies your downside on both axes:
a single detection/phase/seat bug now plays *real money wrong, repeatedly*, and a fully
automated client is the cleanest possible signal for the operator-side enforcement that
freezes the account and confiscates the balance. The advisory tool is defensible as a
human's heads-up display; the moment software places the bet, it's an automated betting
agent, and the asymmetry (one bad night ≠ one bad hand, but the whole bankroll) makes the
expected value of "unattended" negative even before anyone catches it.

If the goal behind "autonomy" is really *closing the loop* — proving the EV engine + bet
ramp + counting actually produce the modelled edge — the zero-risk way to get that is to
point an auto-player at the **built-in trainer/replay simulator** (`lib/logic/trainer.py`)
and let it run millions of hands against your own model. That's the same "autonomous
agent" engineering project, validates the whole stack end-to-end, and costs nothing if a
bug fires.
