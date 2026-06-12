"""Anchor-based, resolution-independent calibration (V3 E3).

Profiles key every calibrated region set by exact capture resolution
("regions{WxH}"), so a new resolution, DPI change, or moved casino window
used to mean re-drawing every polygon by hand — and silently-wrong
geometry until then (OCR reading garbage, cards routed to wrong seats).

Anchors fix the class: 2-3 small template crops of STABLE casino-UI
elements (logo, chip tray, menu button) captured once per table profile.
On startup or demand, cv2.matchTemplate finds them on the live frame
across a small scale pyramid, a least-squares fit solves the uniform
scale + offset transform, and the ONE calibrated region set (players,
dealer, OCR rects, control templates) is mapped to whatever the screen
actually shows. A throttled drift detector re-checks the anchors near
their solved positions and warns instead of silently feeding cards to the
wrong seats.

Fail-disabled by design: no fit (too few anchors matched, residual too
high) means the engine KEEPS its current geometry and says so — a guessed
transform moving real executor clicks is the failure mode this must never
have.

Storage: a 4th "anchors" kind in the region-profile store
({key: {"rect": [l,t,r,b], "template": "anchor-<key>.png"}}) with PNG
crops beside the control templates (assets/controls/<slug>/<WxH>/).
All coordinates are monitor-local physical pixels, frames BGR — the
capture pipeline's contract.
"""

import numpy as np

from ..common import constants
from . import region_profiles
from .phase import template_dir

#: Capture-UI anchor slots: key -> human label.
ANCHOR_KEYS = (("anchor_a", "Anchor A (e.g. table logo)"),
               ("anchor_b", "Anchor B (e.g. chip tray)"),
               ("anchor_c", "Anchor C (e.g. menu button)"))

#: Relative scale steps tried around the resolution-ratio guess.
_SCALE_STEPS = (0.90, 0.95, 1.00, 1.05, 1.10)


# ---------------------------------------------------------------- storage

def save_anchors(resolution, captures: dict, profile=None):
    """Persist captured anchors (merge semantics, like phase.save_controls):
    rects into the profile's "anchors" kind, crops as PNGs."""
    import cv2
    name = (profile or "").strip() or region_profiles.active_name()
    folder = template_dir(name, resolution)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        existing = region_profiles.get_anchors(resolution, profile=name) or {}
    except (ValueError, TypeError, AttributeError):
        existing = {}
    payload = {key: item for key, item in existing.items()
               if isinstance(item, dict) and item.get("rect")}
    for key, item in captures.items():
        ok, buf = cv2.imencode(".png", item["crop"])
        if not ok:
            raise ValueError(f"could not encode anchor template '{key}'")
        filename = f"anchor-{key}.png"
        (folder / filename).write_bytes(buf.tobytes())
        payload[key] = {"rect": [int(v) for v in item["rect"]],
                        "template": filename}
    region_profiles.set_anchors(resolution, payload, profile=name)


def load_anchors(resolution, profile=None) -> dict:
    """{key: {"rect": (l,t,r,b), "template": ndarray}} — anchors whose
    template PNG is missing are dropped (a rect alone can't be matched)."""
    import cv2
    try:
        data = region_profiles.get_anchors(resolution, profile=profile)
    except (ValueError, TypeError, AttributeError):
        return {}
    if not data:
        return {}
    name = profile or region_profiles.active_name()
    folder = template_dir(name, resolution)
    out = {}
    for key, item in data.items():
        try:
            rect = tuple(int(v) for v in item["rect"])
            if len(rect) != 4 or rect[2] <= rect[0] or rect[3] <= rect[1]:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        path = folder / str(item.get("template") or f"anchor-{key}.png")
        try:
            raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            template = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except (OSError, ValueError):
            template = None
        if template is not None:
            out[key] = {"rect": rect, "template": template}
    return out


def calibrated_resolution(live_resolution, profile=None):
    """The resolution whose calibration the anchors can map: the live one
    when it has anchors, else the profile's single anchor-bearing
    resolution. None when there is nothing to anchor to."""
    try:
        if region_profiles.get_anchors(live_resolution, profile=profile):
            return tuple(live_resolution)
        keys = region_profiles.anchor_resolutions(profile=profile)
    except (ValueError, TypeError, AttributeError):
        return None
    if len(keys) == 1:
        w, h = keys[0].split("x")
        return (int(w), int(h))
    return None


# ------------------------------------------------------------------ solve

def _match_full(frame_gray, template_bgr, scale):
    """Best (score, center) of the template resized by `scale` over the
    whole frame; None when geometry is unusable."""
    import cv2
    th = max(4, int(round(template_bgr.shape[0] * scale)))
    tw = max(4, int(round(template_bgr.shape[1] * scale)))
    if th >= frame_gray.shape[0] or tw >= frame_gray.shape[1]:
        return None
    template = cv2.resize(template_bgr, (tw, th),
                          interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(frame_gray,
                               cv2.cvtColor(template, cv2.COLOR_BGR2GRAY),
                               cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(result)
    return float(score), (loc[0] + tw / 2.0, loc[1] + th / 2.0)


def solve(frame_bgr, anchor_set: dict, calib_resolution) -> dict | None:
    """Fit the uniform scale+offset mapping calibrated coords to the live
    frame from the matched anchor positions.

    Returns {"scale", "dx", "dy", "score" (worst kept anchor), "residual"
    (max px), "matched", "centers" {key: live center}} or None when fewer
    than ANCHORS["min_anchors"] match / the fit is unsound — the caller
    must then keep its current geometry (fail-disabled)."""
    import cv2
    cfg = constants.ANCHORS
    if frame_bgr is None or not anchor_set:
        return None
    fh, fw = frame_bgr.shape[:2]
    cw, ch = calib_resolution
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    guesses = {round(g * step, 4)
               for g in (fw / cw, fh / ch, 1.0)
               for step in _SCALE_STEPS if 0.3 <= g * step <= 3.0}
    matched = {}
    for key, item in anchor_set.items():
        best = None
        for scale in sorted(guesses):
            hit = _match_full(frame_gray, item["template"], scale)
            if hit and (best is None or hit[0] > best[0]):
                best = (hit[0], hit[1], scale)
        if best and best[0] >= float(cfg["match_threshold"]):
            left, top, right, bottom = item["rect"]
            matched[key] = {"calib": ((left + right) / 2.0,
                                      (top + bottom) / 2.0),
                            "live": best[1], "score": best[0],
                            "tpl_scale": best[2]}
    if len(matched) < int(cfg["min_anchors"]):
        return None

    calib = np.array([m["calib"] for m in matched.values()])
    live = np.array([m["live"] for m in matched.values()])
    c_mean, l_mean = calib.mean(axis=0), live.mean(axis=0)
    dc, dl = calib - c_mean, live - l_mean
    denom = float((dc * dc).sum())
    if denom > 1e-6:
        scale = float((dc * dl).sum()) / denom
    else:
        # Anchors coincide (degenerate geometry): trust the template scale.
        scale = float(np.mean([m["tpl_scale"] for m in matched.values()]))
    if not 0.3 <= scale <= 3.0:
        return None
    dx, dy = (l_mean - scale * c_mean).tolist()
    residual = float(np.abs(live - (scale * calib + (dx, dy))).max())
    if residual > float(cfg["max_residual_px"]):
        return None
    return {"scale": scale, "dx": dx, "dy": dy,
            "score": min(m["score"] for m in matched.values()),
            "residual": residual, "matched": len(matched),
            "centers": {k: m["live"] for k, m in matched.items()}}


# -------------------------------------------------------------- transform

def transform_point(point, fit):
    return (point[0] * fit["scale"] + fit["dx"],
            point[1] * fit["scale"] + fit["dy"])


def transform_rect(rect, fit) -> list:
    left, top = transform_point(rect[:2], fit)
    right, bottom = transform_point(rect[2:], fit)
    return [int(round(left)), int(round(top)),
            int(round(right)), int(round(bottom))]


def transform_regions(payload, fit) -> dict | None:
    """Map a load_custom_regions() payload ({'players', 'dealer'})."""
    if not payload:
        return None
    players = [[list(transform_point(p, fit)) for p in poly]
               for poly in payload["players"]]
    return {"players": players,
            "dealer": transform_rect(payload["dealer"], fit)}


def transform_ocr(payload, fit) -> dict | None:
    if not payload:
        return None
    return {key: transform_rect(rect, fit) for key, rect in payload.items()}


def transform_controls(controls: dict, fit) -> dict:
    """Map phase.load_controls() output: rects transformed, templates
    RESIZED by the solved scale (matchTemplate has no scale invariance —
    an unscaled template would silently never match again)."""
    import cv2
    out = {}
    for key, item in controls.items():
        template = item.get("template")
        if template is not None and abs(fit["scale"] - 1.0) > 1e-3:
            th = max(4, int(round(template.shape[0] * fit["scale"])))
            tw = max(4, int(round(template.shape[1] * fit["scale"])))
            template = cv2.resize(template, (tw, th),
                                  interpolation=cv2.INTER_AREA)
        out[key] = {"rect": tuple(transform_rect(item["rect"], fit)),
                    "template": template}
    return out


# ------------------------------------------------------------------ drift

def check_drift(frame_bgr, anchor_set: dict, fit) -> dict:
    """Re-locate each anchor NEAR its solved position (cheap local search,
    not the full-frame solve). Returns {"ok", "worst_score", "max_shift"}
    — ok=False when an anchor vanished or moved beyond the tolerance."""
    import cv2
    cfg = constants.ANCHORS
    worst, max_shift = 1.0, 0.0
    fh, fw = frame_bgr.shape[:2]
    for item in anchor_set.values():
        template = item["template"]
        th = max(4, int(round(template.shape[0] * fit["scale"])))
        tw = max(4, int(round(template.shape[1] * fit["scale"])))
        template = cv2.resize(template, (tw, th),
                              interpolation=cv2.INTER_AREA)
        left, top, right, bottom = transform_rect(item["rect"], fit)
        pad_x, pad_y = (right - left), (bottom - top)
        left, top = max(0, left - pad_x), max(0, top - pad_y)
        right, bottom = min(fw, right + pad_x), min(fh, bottom + pad_y)
        search = frame_bgr[top:bottom, left:right]
        if search.shape[0] < th or search.shape[1] < tw:
            return {"ok": False, "worst_score": 0.0, "max_shift": None}
        result = cv2.matchTemplate(
            cv2.cvtColor(search, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(template, cv2.COLOR_BGR2GRAY),
            cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        expected = transform_point(
            (((item["rect"][0] + item["rect"][2]) / 2.0),
             ((item["rect"][1] + item["rect"][3]) / 2.0)), fit)
        center = (left + loc[0] + tw / 2.0, top + loc[1] + th / 2.0)
        shift = float(np.hypot(center[0] - expected[0],
                               center[1] - expected[1]))
        worst = min(worst, float(score))
        max_shift = max(max_shift, shift)
    ok = (worst >= float(cfg["match_threshold"])
          and max_shift <= float(cfg["drift_shift_px"]))
    return {"ok": ok, "worst_score": worst, "max_shift": max_shift}
