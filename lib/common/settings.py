"""Persisted runtime settings: table rules, game config, side-bet paytables.

Every EV the app shows is conditional on the table rules, so they must be
editable per table instead of hardcoded. The profile is a small JSON file in
output/settings.json, loaded at startup (main.py) and written by the settings
dialog. apply() mutates the live constants in place — every consumer
(ev_engine.current_rules, deviations, sidebets.evaluate_all) reads them at
call time, so changes take effect on the next snapshot after the engine's
caches are invalidated (DetectionEngine.refresh_settings)."""

import json

from . import constants

SETTINGS_PATH = constants.OUTPUT_DIR / "settings.json"


def snapshot() -> dict:
    """The current editable settings as a plain JSON-serializable dict."""
    return {
        "rules": dict(constants.RULES),
        "deck_count": constants.DECK_COUNT,
        "base_bet": constants.BASE_BET,
        "side_bets": {key: {"enabled": bool(cfg.get("enabled"))}
                      for key, cfg in constants.SIDE_BETS.items()},
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
    for key, cfg in data.get("side_bets", {}).items():
        if key in constants.SIDE_BETS and "enabled" in cfg:
            constants.SIDE_BETS[key]["enabled"] = bool(cfg["enabled"])


def save(data: dict | None = None):
    if data is None:
        data = snapshot()
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


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
