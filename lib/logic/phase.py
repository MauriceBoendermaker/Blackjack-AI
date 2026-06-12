"""Game-phase & turn detection (V4 Feature 1) — the autonomy keystone.

A small state machine that gives the snapshot the one thing it lacked:
WHAT IS HAPPENING RIGHT NOW. The engine knows what to do (optimal_action,
bet size, insurance); this module knows whether it is legal to do it:

    IDLE -> BETTING_OPEN -> DEALING -> MY_TURN -> WAITING -> SETTLE -> IDLE

Signals (all local vision/OCR — the per-frame hot loop never touches a
network API by design):
  * countdown digits + "PLACE YOUR BETS" banner via winocr (lib/logic/ocr.py
    reads them on the engine's io thread; freshness-gated here)
  * action buttons via cv2.matchTemplate against templates captured with the
    Capture Controls editor, with enabled-state color sampling (a greyed
    button matches the shape but not the color it was captured with)
  * bet-spot change detection (the spot template is captured EMPTY; a chip
    on it moves the patch away from the calibration)
  * engine context: activity, seat card counts, owned seats

Templates are profile-keyed: assets/controls/<profile-slug>/<WxH>/<key>.png
with rects in the active table profile (region_profiles "controls" kind).

Pure logic + cv2; no Tkinter, no threads of its own. The DetectionEngine
calls update() once per worker cycle (cheap: a few small template matches)
and publishes the returned state into the snapshot for the GUI/HUD and the
executor (V4 Feature 2) to consume.
"""

import hashlib
import re
import time
from collections import deque

import numpy as np

from ..common import constants
from . import region_profiles

# Phase names (snapshot["phase"]["phase"]).
IDLE = "idle"
BETTING_OPEN = "betting_open"
DEALING = "dealing"
MY_TURN = "my_turn"
WAITING = "waiting"
SETTLE = "settle"
UNKNOWN = "unknown"

PHASE_LABELS = {
    IDLE: "Idle",
    BETTING_OPEN: "Bets open",
    DEALING: "Dealing",
    MY_TURN: "YOUR TURN",
    WAITING: "Waiting",
    SETTLE: "Settling",
    UNKNOWN: "Unknown",
}

# Controls the capture editor can calibrate. The decision buttons drive
# MY_TURN; bet_spot (captured EMPTY) drives chip-on-spot detection.
BUTTON_KEYS = ("hit", "stand", "double", "split")
CONTROL_KEYS = BUTTON_KEYS + ("bet_spot",)


def empty_state() -> dict:
    return {"phase": IDLE, "since": time.time(), "confidence": 0.0,
            "frames": 0, "timer_s": None, "status": None, "buttons": {},
            "calibrated": False, "my_seat": None, "seat_confident": False,
            "bet_spot": None, "triage": None}


# ------------------------------------------------------- template library

def profile_slug(name: str) -> str:
    """Profile name -> filesystem-safe folder name. A short hash of the
    EXACT name is appended so distinct profiles ('Table A' vs 'table a')
    can never share a template folder and overwrite each other's PNGs."""
    base = re.sub(r"[^A-Za-z0-9]+", "-", str(name)).strip("-").lower()
    digest = hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:6]
    return f"{base or 'profile'}-{digest}"


def template_dir(profile_name, resolution):
    w, h = resolution
    return constants.CONTROLS_DIR / profile_slug(profile_name) / f"{w}x{h}"


def save_controls(resolution, captures: dict, profile=None):
    """Persist captured controls: rects into the table profile, template
    crops as PNGs under assets/controls/. `captures` maps control key ->
    {"rect": [l,t,r,b], "crop": BGR ndarray}.

    MERGE semantics: only the submitted keys are (re)written; a control
    calibrated earlier but not captured this time keeps its rect and its
    PNG untouched. The enabled buttons and the empty bet spot cannot
    coexist on one screenshot, so calibration necessarily spans several
    capture sessions — a save must never silently re-crop or drop what an
    earlier session captured.
    """
    import cv2
    # Same normalization rule as region_profiles._set, so the rects and
    # the PNG folder can never land under different profiles.
    name = (profile or "").strip() or region_profiles.active_name()
    folder = template_dir(name, resolution)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        existing = region_profiles.get_controls(resolution, profile=name) or {}
    except (ValueError, TypeError, AttributeError):
        existing = {}
    payload = {key: item for key, item in existing.items()
               if isinstance(item, dict) and item.get("rect")}
    for key, item in captures.items():
        ok, buf = cv2.imencode(".png", item["crop"])
        if not ok:
            raise ValueError(f"could not encode template for '{key}'")
        filename = f"{key}.png"
        (folder / filename).write_bytes(buf.tobytes())
        payload[key] = {"rect": [int(v) for v in item["rect"]],
                        "template": filename}
    region_profiles.set_controls(resolution, payload, profile=name)


def delete_controls(resolution):
    region_profiles.delete_controls(resolution)


def load_controls(resolution) -> dict:
    """{key: {"rect": (l,t,r,b), "template": ndarray|None}} for the ACTIVE
    profile at this resolution; {} when nothing is calibrated. A rect whose
    template file is missing/corrupt still loads (rect-only change
    detection works without it for bet_spot)."""
    import cv2
    try:
        data = region_profiles.get_controls(resolution)
    except (ValueError, TypeError, AttributeError):
        return {}
    if not data:
        return {}
    folder = template_dir(region_profiles.active_name(), resolution)
    out = {}
    for key, item in data.items():
        try:
            rect = tuple(int(v) for v in item["rect"])
            if len(rect) != 4 or rect[2] <= rect[0] or rect[3] <= rect[1]:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        template = None
        path = folder / str(item.get("template") or f"{key}.png")
        try:
            raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            template = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except (OSError, ValueError):
            template = None
        out[key] = {"rect": rect, "template": template}
    return out


# ------------------------------------------------------ template matching

def locate_control(frame_bgr, rect, template_bgr, pad_frac=None):
    """Find a captured control near its calibrated rect.

    Searches the rect inflated by `pad_frac` (tolerates small client
    re-layouts), grayscale TM_CCOEFF_NORMED for the shape, then a mean
    per-channel color distance on the matched patch for the enabled state
    (templates are captured while the button is ENABLED, so a greyed
    button keeps the shape but moves the color).

    Returns {"score", "center" (x, y), "color_dist"} or None when the
    template/search geometry is unusable.
    """
    import cv2
    if template_bgr is None or frame_bgr is None:
        return None
    if pad_frac is None:
        pad_frac = float(constants.PHASE.get("search_pad", 0.35))
    fh, fw = frame_bgr.shape[:2]
    left, top, right, bottom = rect
    pad_x = int((right - left) * pad_frac)
    pad_y = int((bottom - top) * pad_frac)
    left, top = max(0, left - pad_x), max(0, top - pad_y)
    right, bottom = min(fw, right + pad_x), min(fh, bottom + pad_y)
    search = frame_bgr[top:bottom, left:right]
    th, tw = template_bgr.shape[:2]
    if th < 4 or tw < 4 or search.shape[0] < th or search.shape[1] < tw:
        return None
    result = cv2.matchTemplate(
        cv2.cvtColor(search, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY),
        cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(result)
    x, y = loc
    patch = search[y:y + th, x:x + tw]
    color_dist = float(np.mean(np.abs(
        patch.astype(np.int16) - template_bgr.astype(np.int16))))
    return {"score": float(score),
            "center": (left + x + tw // 2, top + y + th // 2),
            "color_dist": color_dist}


def patch_changed(frame_bgr, rect, template_bgr) -> float | None:
    """Mean per-channel color distance of the EXACT calibrated rect vs its
    captured template — chip-on-spot detection (template captured empty).
    None when geometry is unusable."""
    if template_bgr is None or frame_bgr is None:
        return None
    fh, fw = frame_bgr.shape[:2]
    left, top, right, bottom = rect
    left, top = max(0, left), max(0, top)
    right, bottom = min(fw, right), min(fh, bottom)
    patch = frame_bgr[top:bottom, left:right]
    th, tw = template_bgr.shape[:2]
    if patch.shape[0] != th or patch.shape[1] != tw:
        if patch.size == 0 or th < 1 or tw < 1:
            return None
        import cv2
        patch = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(
        patch.astype(np.int16) - template_bgr.astype(np.int16))))


# ------------------------------------------------------------- detector

def _default_log(message, level=None):
    print(message)


class PhaseDetector:
    """Consumes one frame + engine context per cycle, produces the phase
    state dict. Owned and driven by the DetectionEngine worker thread;
    configure() swaps the control set atomically (a dict replace), so a
    reload from the Tk thread cannot tear a running update()."""

    def __init__(self, log=_default_log):
        self.log = log
        self._controls = {}
        self._resolution = None
        self._current = IDLE
        self._since = time.time()
        self._candidate = None
        self._streak = 0
        self._frames_current = 0
        self._timer_history = deque(maxlen=4)  # (monotonic ts, seconds)
        self._last_buttons = {}

    def configure(self, resolution):
        """(Re)load the active profile's control templates for this capture
        resolution. Called from set_monitor and after a capture-editor save."""
        controls = load_controls(resolution)
        self._controls = controls
        self._resolution = tuple(resolution)
        buttons = [k for k in BUTTON_KEYS
                   if controls.get(k, {}).get("template") is not None]
        if buttons:
            self.log(f"Phase detection: {len(buttons)} control template(s) "
                     f"loaded ({', '.join(buttons)}).")
        elif constants.PHASE.get("enabled"):
            self.log("Phase detection: no control templates for this "
                     "resolution — MY_TURN detection off (use Capture "
                     "Controls to calibrate).", level="WARNING")

    def set_controls(self, controls: dict):
        """Adopt an externally prepared control set (the anchor remap,
        V3 E3, hands in rects transformed and templates rescaled to the
        live screen). Atomic dict swap — same contract as configure()."""
        self._controls = dict(controls)
        buttons = [k for k in BUTTON_KEYS
                   if controls.get(k, {}).get("template") is not None]
        if buttons:
            self.log(f"Phase detection: {len(buttons)} control template(s) "
                     f"mapped via anchors ({', '.join(buttons)}).")

    @property
    def calibrated(self) -> bool:
        """True when at least Hit AND Stand templates exist — the minimum
        for trustworthy MY_TURN detection."""
        return all(self._controls.get(k, {}).get("template") is not None
                   for k in ("hit", "stand"))

    def reset(self):
        self._current = IDLE
        self._since = time.time()
        self._candidate, self._streak, self._frames_current = None, 0, 0
        self._timer_history.clear()

    # ------------------------------------------------------------ update

    def update(self, frame_bgr, signals: dict) -> dict:
        """One detection step. `signals` comes from the engine:
        {"activity": str, "cards_on_table": int, "undecided_mine": [idx],
         "my_seats": [idx], "ocr": {"status","result","timer","bet","ts"}}.
        Returns the new phase state dict (also retained internally)."""
        now = time.monotonic()
        cfg = constants.PHASE
        buttons = self._match_buttons(frame_bgr)
        self._last_buttons = buttons

        ocr_vals = signals.get("ocr") or {}
        fresh = (time.time() - float(ocr_vals.get("ts") or 0)
                 <= float(cfg.get("ocr_fresh_s", 3.0)))
        status = ocr_vals.get("status") if fresh else None
        result = ocr_vals.get("result") if fresh else None
        timer_s = ocr_vals.get("timer") if fresh else None
        stray_text = bool(ocr_vals.get("stray_text")) if fresh else False
        if timer_s is not None:
            if not self._timer_history or self._timer_history[-1][1] != timer_s:
                self._timer_history.append((now, timer_s))

        candidate = self._candidate_phase(buttons, status, result, timer_s,
                                          stray_text, signals, now)
        # Confirmation applies on the way INTO the action phases AND on the
        # way OUT of MY_TURN: a one-frame hover highlight or animation
        # sweeping a button must not flap the turn state (a flap would
        # re-fire alerts and queue bogus discipline judgments).
        required = (int(cfg.get("confirm_frames", 2))
                    if (candidate in (MY_TURN, BETTING_OPEN)
                        or self._current == MY_TURN) else 1)
        if candidate == self._current:
            self._candidate, self._streak = None, 0
            self._frames_current += 1
        else:
            if candidate == self._candidate:
                self._streak += 1
            else:
                self._candidate, self._streak = candidate, 1
            if self._streak >= required:
                self._current = candidate
                self._since = time.time()
                self._frames_current = self._streak
                self._candidate, self._streak = None, 0

        seat, seat_confident = self._attribute_seat(signals)
        return {
            "phase": self._current,
            "since": self._since,
            "confidence": self._confidence(buttons),
            "frames": self._frames_current,
            "timer_s": timer_s,
            "status": status,
            "buttons": buttons,
            "calibrated": self.calibrated,
            "my_seat": seat,
            "seat_confident": seat_confident,
            "bet_spot": self._bet_spot(frame_bgr),
            "triage": None,  # filled by the engine's vision-assist hook
        }

    # ----------------------------------------------------------- internals

    def _match_buttons(self, frame_bgr) -> dict:
        cfg = constants.PHASE
        threshold = float(cfg.get("match_threshold", 0.70))
        max_dist = float(cfg.get("enabled_color_dist", 48.0))
        out = {}
        for key in BUTTON_KEYS:
            control = self._controls.get(key)
            if not control or control.get("template") is None:
                continue
            hit = locate_control(frame_bgr, control["rect"], control["template"])
            if hit is None:
                continue
            found = hit["score"] >= threshold
            out[key] = {"found": found,
                        "enabled": found and hit["color_dist"] <= max_dist,
                        "score": round(hit["score"], 3),
                        "center": [int(hit["center"][0]), int(hit["center"][1])]}
        return out

    def _candidate_phase(self, buttons, status, result, timer_s, stray_text,
                         signals, now):
        cards_on_table = int(signals.get("cards_on_table") or 0)
        activity = signals.get("activity") or "waiting"
        hit, stand = buttons.get("hit"), buttons.get("stand")

        if hit and stand and hit["enabled"] and stand["enabled"]:
            return MY_TURN
        if status == "betting":
            return BETTING_OPEN
        if (timer_s is not None and cards_on_table == 0
                and self._timer_ticking(now)):
            # A live countdown with an empty table is the betting window even
            # when the banner OCR missed the phrase. (Deliberately NOT
            # extended to mid-round states: another player's decision timer
            # ticks exactly the same way.)
            return BETTING_OPEN
        if result is not None and cards_on_table > 0:
            return SETTLE
        if activity == "dealing":
            return DEALING
        if hit and stand and hit["found"] != stand["found"]:
            # Exactly one of a calibrated button pair visible: something is
            # occluding the controls (modal / disconnect overlay).
            return UNKNOWN
        if stray_text:
            # Substantial banner-area text that matches no game vocabulary —
            # the signature of a centered modal / disconnect / inactivity
            # dialog covering the table (which also hides BOTH buttons, so
            # the pair-mismatch rule above can't see it).
            return UNKNOWN
        if cards_on_table > 0:
            return WAITING
        return IDLE

    def _timer_ticking(self, now) -> bool:
        """True when the countdown moved downward across recent reads."""
        recent = [(ts, s) for ts, s in self._timer_history if now - ts < 10.0]
        return len(recent) >= 2 and recent[-1][1] < recent[0][1]

    def _attribute_seat(self, signals):
        """WHICH owned seat the decision belongs to — the crux per
        AUTONOMY_PLAN §4.1. Heuristic: the lowest-index owned seat that
        still has an undecidable hand (<21, not bust). Confident only when
        that candidate is unique; the executor must abstain otherwise."""
        undecided = list(signals.get("undecided_mine") or [])
        if len(undecided) == 1:
            return undecided[0], True
        if undecided:
            return undecided[0], False
        mine = sorted(signals.get("my_seats") or [])
        return (mine[0], False) if mine else (None, False)

    def _bet_spot(self, frame_bgr):
        """{"center": [x, y], "changed": bool} for the calibrated bet spot
        (template captured EMPTY, so `changed` ~= a chip sits on it), or
        None when uncalibrated."""
        control = self._controls.get("bet_spot")
        if not control or control.get("template") is None:
            return None
        dist = patch_changed(frame_bgr, control["rect"], control["template"])
        if dist is None:
            return None
        left, top, right, bottom = control["rect"]
        return {"center": [(left + right) // 2, (top + bottom) // 2],
                "changed": dist > float(
                    constants.PHASE.get("enabled_color_dist", 35.0))}

    def _confidence(self, buttons) -> float:
        """0-1: how much the executor may trust the current phase. MY_TURN
        scales with the weakest button match; OCR-driven phases ride the
        confirm streak alone."""
        base = min(1.0, max(self._frames_current, 1)
                   / max(1, int(constants.PHASE.get("confirm_frames", 2))))
        if self._current == MY_TURN and buttons:
            scores = [b["score"] for b in buttons.values() if b["found"]]
            if scores:
                return round(base * min(1.0, min(scores) / 0.9), 3)
        return round(base, 3)
