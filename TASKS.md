# Development Tasks

## Done (v3.0 overhaul)

- [x] Move everything unrelated to the app into `archive/`; promote the app from `NEW/` to the repo root
- [x] Fix soft hands never being detected (broken `is_soft_hand` math)
- [x] Fix pair lookups — A,A and 10/J/Q/K pairs now resolve (was "?"), incl. the `10,1` typo in the strategy CSV
- [x] Fix dealer "10" being read as "1" (first-character bug) and advice computed against the wrong up-card
- [x] Fix round reset — detection no longer goes dead after the first round; rounds auto-advance when the table clears
- [x] Fix the true count (was always 0) and track the shoe across rounds; add decks-remaining + bet hint
- [x] Fix Hi-Lo counting (duplicate ranks were skipped, set was polluted, dealer card was never counted)
- [x] Fix Roboflow center-vs-corner coordinates that mis-assigned cards to neighboring seats
- [x] All Tk updates on the main thread (was crashing/freezing from the worker thread); queue consumer actually wired
- [x] Start/Stop toggle with guard (double Start used to spawn duplicate detection loops)
- [x] Performance: in-memory pipeline (no JPEG disk round-trips), parallel dealer+player inference, dealer call
      stops once locked, static-frame skip, downscaled API uploads, reused mss instance, no matplotlib
- [x] Live card counters (were frozen until manual refresh); manual ± now also adjusts the running count
- [x] 3rd+ card (hit) detection with confirmation; phantom-pair suppression; manual "+" per seat
- [x] Card picker sized to fit, titled per seat/slot, with remove option
- [x] Region preview actually works (monitor wiring was broken) and renders above the table
- [x] Logging window: thread-safe, trimmed display, debounced search, complete filters
- [x] Removed: dead legacy GUI, wizardofodds "Optimal" fetch (broken by design), eval()-based cache,
      hardcoded relative asset paths, committed IDE junk
- [x] requirements.txt + README; assets moved into `assets/`

## Open

- [ ] Split-hand support (after a split a seat holds two hands; needs per-hand card grouping in the seat region)
- [ ] OCR for balance and current bet
- [ ] Interactive region calibration (drag the seat polygons over a live screenshot; persist per resolution)
- [ ] Local YOLO weights for player + dealer models (train/export; drop into `models/` — see README)
- [ ] Fine-tune model confidence/overlap thresholds against real footage
