"""Named region-calibration profiles — one per casino table.

A profile bundles BOTH calibration kinds per capture resolution: the
seat/dealer regions (Calibrate Regions editor) and the OCR rects
(balance/bet/result) — a different table moves both, so switching tables
swaps everything at once. monitor_utils and ocr route their load/save/
delete through the ACTIVE profile, so the engine and the editors need no
knowledge of profiles.

Store: output/region_profiles.json
{
  "active": "Default",
  "profiles": {
    "Default": {
      "saved": "2026-06-11T16:40:00",
      "regions": {"2560x1440": {"players": [...], "dealer": [l,t,r,b]}},
      "ocr":     {"2560x1440": {"balance": [l,t,r,b], ...}}
    }
  }
}

Legacy per-resolution files (regions_{w}x{h}.json, ocr_regions_{w}x{h}.json)
are folded into "Default" the first time the store is touched and renamed
to *.migrated so they are never re-imported. A plain read on a fresh
install creates no files. Saving into a profile stamps `saved` (shown in
the GUI's profile list) and makes it the active one.
"""

import json
import re
import threading
from datetime import datetime

from ..common import constants

DEFAULT_NAME = "Default"

# Reads come from the engine worker (set_monitor) while editors write from
# the Tk thread; the RLock serializes the read-modify-write cycles.
_LOCK = threading.RLock()
_LEGACY = (("regions", re.compile(r"^regions_(\d+)x(\d+)\.json$")),
           ("ocr", re.compile(r"^ocr_regions_(\d+)x(\d+)\.json$")))


def store_path():
    return constants.OUTPUT_DIR / "region_profiles.json"


def _res_key(resolution):
    w, h = resolution
    return f"{w}x{h}"


def _new_profile():
    return {"saved": None, "regions": {}, "ocr": {}}


def _empty():
    return {"active": DEFAULT_NAME, "profiles": {DEFAULT_NAME: _new_profile()}}


def _load():
    """The whole store; folds legacy per-resolution files in on first touch."""
    with _LOCK:
        path = store_path()
        if not path.exists():
            return _migrate_legacy()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            profiles = data.get("profiles")
            if not isinstance(profiles, dict) or not profiles:
                raise ValueError("no profiles")
            for prof in profiles.values():
                prof.setdefault("saved", None)
                for kind in ("regions", "ocr"):
                    # setdefault alone lets an existing null/list through,
                    # which would crash every _get/_set downstream.
                    if not isinstance(prof.get(kind), dict):
                        prof[kind] = {}
            if data.get("active") not in profiles:
                data["active"] = sorted(profiles)[0]
            return data
        except OSError as e:
            # Transient read failure (AV/sync lock) on a possibly VALID
            # file: serve defaults for this read but block writes — a save
            # now would clobber recoverable data.
            print(f"Region-profile store unreadable ({e}); "
                  "read-only defaults until it recovers.")
            data = _empty()
            data["_locked"] = True
            return data
        except (ValueError, TypeError, AttributeError) as e:
            # Parse/shape corruption: quarantine the bytes first — the next
            # save would otherwise atomically replace every table's
            # calibration with an empty store.
            print(f"Region-profile store corrupt ({e}); "
                  f"moved to {path.name}.corrupt, starting fresh.")
            try:
                path.replace(path.with_name(path.name + ".corrupt"))
            except OSError:
                pass
            return _empty()


def _migrate_legacy():
    """Pre-profile files become the 'Default' profile, stamped with their
    newest mtime. The store is only written when something was actually
    migrated — a read on a fresh install must not create files."""
    data = _empty()
    default = data["profiles"][DEFAULT_NAME]
    migrated, newest = [], None
    try:
        entries = list(constants.OUTPUT_DIR.iterdir())
    except OSError:
        return data
    for entry in entries:
        for kind, pattern in _LEGACY:
            m = pattern.match(entry.name)
            if not m:
                continue
            try:
                payload = json.loads(entry.read_text(encoding="utf-8"))
                mtime = entry.stat().st_mtime
            except (OSError, ValueError):
                break  # unreadable legacy file: leave it alone
            default[kind][f"{m.group(1)}x{m.group(2)}"] = payload
            newest = mtime if newest is None else max(newest, mtime)
            migrated.append(entry)
            break
    if migrated:
        default["saved"] = datetime.fromtimestamp(newest).isoformat(
            timespec="seconds")
        try:
            _write(data)
        except OSError as e:
            # Disk not writable right now: serve the merged data in memory
            # and leave the legacy files untouched, so the next launch
            # retries the migration cleanly.
            print(f"Region-profile migration deferred ({e}).")
            return data
        for entry in migrated:
            try:
                entry.rename(entry.with_name(entry.name + ".migrated"))
            except OSError:
                pass  # payload already lives in the store; the lingering
                # legacy file is simply never re-imported
    return data


def _write(data):
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------- public

def list_profiles():
    """[{'name', 'saved'}] — Default first, the rest alphabetical."""
    data = _load()
    items = [{"name": name, "saved": prof.get("saved")}
             for name, prof in data["profiles"].items()]
    items.sort(key=lambda p: (p["name"] != DEFAULT_NAME, p["name"].lower()))
    return items


def active_name() -> str:
    return _load()["active"]


def set_active(name) -> bool:
    """Switch the active profile; False when the name doesn't exist."""
    with _LOCK:
        data = _load()
        if name not in data["profiles"]:
            return False
        if data["active"] != name:
            data["active"] = name
            _write(data)
        return True


def _get(kind, resolution):
    data = _load()
    return data["profiles"][data["active"]][kind].get(_res_key(resolution))


def _set(kind, resolution, payload, profile):
    with _LOCK:
        data = _load()
        if data.pop("_locked", False):
            print("Region-profile store is locked — calibration not saved; "
                  "try again in a moment.")
            return
        name = (profile or data["active"]).strip() or data["active"]
        prof = data["profiles"].setdefault(name, _new_profile())
        prof[kind][_res_key(resolution)] = payload
        prof["saved"] = datetime.now().isoformat(timespec="seconds")
        data["active"] = name  # saving into a profile selects it
        _write(data)


def _delete(kind, resolution):
    with _LOCK:
        data = _load()
        if data.pop("_locked", False):
            return
        prof = data["profiles"][data["active"]]
        if prof[kind].pop(_res_key(resolution), None) is not None:
            _write(data)


def get_regions(resolution):
    return _get("regions", resolution)


def set_regions(resolution, payload, profile=None):
    _set("regions", resolution, payload, profile)


def delete_regions(resolution):
    _delete("regions", resolution)


def get_ocr(resolution):
    return _get("ocr", resolution)


def set_ocr(resolution, payload, profile=None):
    _set("ocr", resolution, payload, profile)


def delete_ocr(resolution):
    _delete("ocr", resolution)
