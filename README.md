# Blackjack AI

Watches a live blackjack table on a chosen monitor, detects the dealt cards
with a YOLO model (local weights or the Roboflow hosted API), keeps a Hi-Lo
card count across the shoe, and shows basic-strategy advice per seat in a
desktop UI.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

1. The primary monitor is selected automatically — change it in the dropdown
   if the casino stream is elsewhere.
2. Press **Start Detection**. Models initialize in the background (the first
   start takes a few seconds when using the hosted API).
3. Cards appear on the felt as they are dealt; advice (Hit / Stand / Double /
   Split / Surrender) shows under every active seat once the dealer's
   up-card is known.
4. Misread card? Click it and pick the right one. Click the dealer card to
   correct the up-card. The **+** next to a seat adds a hit card manually.

## What the panels show

* **Cards Seen** — per-rank totals for the current shoe (J/Q/K are grouped
  with 10). The +/- buttons adjust both the totals and the running count.
* **Game Info** — round number, Hi-Lo running count, true count (running ÷
  decks remaining), decks remaining, and a bet-size hint from the true count.
* **New Round** clears the table but keeps the shoe count. Rounds also
  auto-advance when the table is cleared. **New Shoe** resets all counts —
  use it after the shuffle (the app reminds you when it spots the cutting
  card).
* **Preview Regions** overlays the seat polygons, dealer area, and the raw
  model detections on a live screenshot — useful to verify the stream lines
  up with the configured regions.

## Configuration

Everything lives in `lib/common/constants.py`: capture regions, model IDs,
confidence thresholds, pacing, theme. The Roboflow API key can be overridden
with the `ROBOFLOW_API_KEY` environment variable.

For lower latency, place local YOLO weights at `models/player_cards.pt` (and
optionally `models/dealer_cards.pt`) and install `ultralytics` — the app
prefers local weights automatically and falls back to the hosted API.

## Project layout

```
main.py                 entry point
assets/                 card images + basic-strategy table (strategy.csv)
lib/
  common/               constants (paths, settings, theme), model class maps
  logic/                engine, models, counting, strategy, capture — no Tk here
  interfaces/           Tk UI: main window, table canvas, card picker, logs
models/                 optional local YOLO weights (gitignored)
output/, logs/          runtime artifacts (gitignored)
archive/                everything not needed to run the app:
                        legacy api/ scripts, ML training datasets, design docs,
                        the old Tk GUI
```

## Architecture notes

The detection engine (`lib/logic/engine.py`) runs on a worker thread and
never touches Tkinter. Each cycle it captures the monitor in memory, skips
inference when the frame hasn't changed, otherwise runs the player and
dealer models in parallel, updates the game state, and publishes an
immutable snapshot. The GUI polls that snapshot from the Tk event loop and
only updates widgets whose content actually changed.

The dealer call stops for the rest of the round once the up-card is
confirmed, and hosted-API uploads are downscaled to 1280px — together with
the frame-skip this keeps API traffic to a small fraction of the old
implementation.
