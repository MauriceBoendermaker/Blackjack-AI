"""Persisted runtime settings: table rules, game config, side bets, app tab.

Every EV the app shows is conditional on the table rules, so they must be
editable per table instead of hardcoded. The profile is a small JSON file in
output/settings.json, loaded at startup (main.py) and written by the settings
dialog. apply() mutates the live constants in place — every consumer
(ev_engine.current_rules, deviations, sidebets.evaluate_all) reads them at
call time, so changes take effect on the next snapshot after the engine's
caches are invalidated (DetectionEngine.refresh_settings)."""

import json
import os
import threading

from . import constants

SETTINGS_PATH = constants.OUTPUT_DIR / "settings.json"

# App-level tunables from the settings dialog's "App" tab: module attribute
# name -> (coerce, lo, hi). Applied via setattr — the engine/worker/GUI read
# these attributes at call time, so a change takes effect on the next cycle.
_APP_ATTRS = {
    "EMPTY_FRAMES_FOR_RESET": (int, 3, 10),
    "CUTTING_CARD_CONFIRM_FRAMES": (int, 1, 5),
    "EV_ADVICE_TIMEOUT_S": (float, 1.0, 10.0),
    "IDLE_REFRESH_GAP_S": (float, 10.0, 600.0),
    "SNAPSHOT_POLL_MS": (int, 60, 500),
    # Bool as 0/1: suit-aware dealer detection via the 52-class card model.
    "DEALER_USE_PLAYER_MODEL": (int, 0, 1),
}
# OCR keys editable from the dialog (interval_s stays code-configured).
_OCR_TOGGLES = ("enabled", "sync_bankroll", "sync_bet")


def snapshot() -> dict:
    """The current editable settings as a plain JSON-serializable dict."""
    return {
        "rules": dict(constants.RULES),
        "deck_count": constants.DECK_COUNT,
        "base_bet": constants.BASE_BET,
        "betting": dict(constants.BETTING),
        "side_bets": {key: {"enabled": bool(cfg.get("enabled")),
                            "stake": float(cfg.get("stake") or 0.0)}
                      for key, cfg in constants.SIDE_BETS.items()},
        "ui": dict(constants.UI),
        "app": {name: getattr(constants, name) for name in _APP_ATTRS},
        "ocr": {key: constants.OCR[key] for key in _OCR_TOGGLES},
        "phase": {"enabled": constants.PHASE["enabled"]},
        # The API key persists in plaintext like every other local setting;
        # ANTHROPIC_API_KEY overrides it without touching the file.
        "vision": {key: constants.VISION[key]
                   for key in ("enabled", "api_key", "model", "triage")},
    }


def apply(data: dict):
    """Apply a settings dict onto the live constants (unknown keys ignored)."""
    rules = data.get("rules", {})
    for key in constants.RULES:
        if key in rules:
            constants.RULES[key] = rules[key]
    if "deck_count" in data:
        constants.DECK_COUNT = max(1, min(8, int(data["deck_count"])))
    if "base_bet" in data:
        constants.BASE_BET = max(1, float(data["base_bet"]))
    betting = data.get("betting", {})
    for key in constants.BETTING:
        if key in betting:
            try:
                constants.BETTING[key] = float(betting[key])
            except (TypeError, ValueError):
                pass
    for key, cfg in data.get("side_bets", {}).items():
        if key not in constants.SIDE_BETS:
            continue
        if "enabled" in cfg:
            constants.SIDE_BETS[key]["enabled"] = bool(cfg["enabled"])
        if "stake" in cfg:
            try:
                constants.SIDE_BETS[key]["stake"] = max(0.0, float(cfg["stake"]))
            except (TypeError, ValueError):
                pass
    ui = data.get("ui", {})
    if "scale" in ui:
        try:
            pct = int(ui["scale"])
        except (TypeError, ValueError):
            pct = constants.UI["scale"]
        # 0 = auto (per-monitor DPI); manual overrides clamp to the scaling
        # module's 75-300% factor bounds.
        constants.UI["scale"] = 0 if pct <= 0 else max(75, min(300, pct))
    app = data.get("app", {})
    for name, (coerce, lo, hi) in _APP_ATTRS.items():
        if name in app:
            try:
                setattr(constants, name, max(lo, min(hi, coerce(app[name]))))
            except (TypeError, ValueError):
                pass
    ocr = data.get("ocr", {})
    for key in _OCR_TOGGLES:
        if key in ocr:
            constants.OCR[key] = int(bool(ocr[key]))
    phase_cfg = data.get("phase", {})
    if "enabled" in phase_cfg:
        constants.PHASE["enabled"] = int(bool(phase_cfg["enabled"]))
    vision = data.get("vision", {})
    for key in ("enabled", "triage"):
        if key in vision:
            constants.VISION[key] = int(bool(vision[key]))
    if "api_key" in vision:
        constants.VISION["api_key"] = str(vision["api_key"] or "")
    if "model" in vision and str(vision["model"] or "").strip():
        constants.VISION["model"] = str(vision["model"]).strip()


# Serializes concurrent writers (the engine io thread and the Tk thread):
# without it, two saves sharing the one tmp path collide and os.replace
# raises PermissionError on Windows.
_SAVE_LOCK = threading.Lock()


def save(data: dict | None = None):
    if data is None:
        data = snapshot()
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace: a crash mid-write must never truncate the profile.
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    with _SAVE_LOCK:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, SETTINGS_PATH)


def load_and_apply() -> bool:
    """Load the saved profile if one exists. Returns True when applied."""
    if not SETTINGS_PATH.exists():
        return False
    try:
        apply(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        return True
    except (OSError, ValueError) as e:
        print(f"Settings profile unreadable ({e}); using defaults.")
        return False
