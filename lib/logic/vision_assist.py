"""Claude vision assist (V4) — optional API helpers for the two cold paths.

Hybrid by explicit design decision: the per-frame hot loop (phase detection,
card vision, OCR) stays local — an API round-trip is 1-4 s against ~12 s
windows and ~$0.005-0.013 per frame, and its coordinates are documented as
approximate. The API earns its keep where latency and approximation don't
hurt:

  * bootstrap(frame): one screenshot -> suggested OCR rects (balance / bet /
    result / timer) and control rects (hit / stand / double / split /
    bet_spot) that PRE-FILL the region editors. A human fine-tunes and
    saves, so approximate boxes are fine — this turns calibrating a new
    table/resolution from the most painful manual step into a 2-minute job.
  * triage(frame): a persistently UNKNOWN screen (modal, disconnect,
    shuffle, lobby) gets a one-line label for the HUD/log. The executor is
    already abstaining when this fires, so 2-4 s of latency costs nothing.

Plain REST via `requests` (already a dependency — no SDK install). Disabled
without an API key: constants.VISION["api_key"] or ANTHROPIC_API_KEY. Image
sizing follows the documented resize rule so returned pixel coordinates map
back onto the native frame exactly.
"""

import base64
import json
import math
import os
import re

from ..common import constants

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

OCR_KEYS = ("balance", "bet", "result", "timer")
CONTROL_KEYS = ("hit", "stand", "double", "split", "bet_spot")

# Models with high-resolution vision (2576 px long edge / 4784 visual
# tokens); everything else gets the classic 1568/1568 limits.
_HIRES_MARKERS = ("opus-4-7", "opus-4-8", "fable", "mythos")


def api_key() -> str:
    return (os.environ.get("ANTHROPIC_API_KEY")
            or str(constants.VISION.get("api_key") or "")).strip()


def available() -> bool:
    """Vision assist is opt-in AND needs a key — without either, every
    caller silently skips (the app stays a pure local advisor)."""
    return bool(constants.VISION.get("enabled")) and bool(api_key())


# ---------------------------------------------------------- image sizing

def _limits(model: str) -> tuple[int, int]:
    model = (model or "").lower()
    if any(marker in model for marker in _HIRES_MARKERS):
        return 2576, 4784
    return 1568, 1568


def count_image_tokens(width: int, height: int) -> int:
    """Visual tokens: one per 28x28 patch (documented cost rule)."""
    return math.ceil(width / 28) * math.ceil(height / 28)


def resized_size(width, height, max_edge=1568, max_tokens=1568):
    """The exact size Claude resizes an image to (reference implementation
    from the vision docs) — uploading at this size makes the returned pixel
    coordinates map 1:1, no rescale guesswork."""

    def fits(w, h):
        return (math.ceil(w / 28) * 28 <= max_edge
                and math.ceil(h / 28) * 28 <= max_edge
                and count_image_tokens(w, h) <= max_tokens)

    if fits(width, height):
        return width, height
    if height > width:
        rh, rw = resized_size(height, width, max_edge, max_tokens)
        return rw, rh
    aspect = width / height
    lo, hi = 1, width  # lo always fits; hi never fits
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect), 1)):
            lo = mid
        else:
            hi = mid
    return lo, max(round(lo / aspect), 1)


def _encode_frame(frame_bgr, max_edge, max_tokens):
    """Resize a BGR frame to Claude's native size and JPEG-encode it.
    Returns (base64_str, scale_x, scale_y) where scale maps API pixel
    coords BACK to native frame coords."""
    import cv2
    h, w = frame_bgr.shape[:2]
    rw, rh = resized_size(w, h, max_edge, max_tokens)
    img = frame_bgr if (rw, rh) == (w, h) else cv2.resize(
        frame_bgr, (rw, rh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("could not JPEG-encode the frame")
    return (base64.standard_b64encode(buf.tobytes()).decode("ascii"),
            w / rw, h / rh)


# ----------------------------------------------------------- API plumbing

def _post(image_b64, prompt, max_tokens=1024, timeout=45) -> str:
    """One image-then-text message -> the response text. Raises
    RuntimeError with the API's message on any failure."""
    import requests
    payload = {
        "model": str(constants.VISION.get("model") or "claude-opus-4-8"),
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg",
                            "data": image_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    resp = requests.post(
        API_URL, json=payload, timeout=timeout,
        headers={"x-api-key": api_key(), "anthropic-version": API_VERSION,
                 "content-type": "application/json"})
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text)
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"Anthropic API {resp.status_code}: {detail[:300]}")
    data = resp.json()
    return "".join(block.get("text", "") for block in data.get("content", [])
                   if block.get("type") == "text")


def _json_block(text):
    """The first JSON object in a model reply (tolerates ```json fences)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"no JSON in vision reply: {text[:200]!r}")
    return json.loads(m.group(0))


# ------------------------------------------------------------- bootstrap

_BOOTSTRAP_PROMPT = f"""This is a screenshot of an online live-dealer \
blackjack table (Evolution-style UI). Locate these elements and return their \
bounding boxes as [x1, y1, x2, y2] in pixel coordinates of THIS image \
(origin top-left). Use null for anything not visible right now.

Text areas to read later with OCR (box the text tightly with a few px of margin):
- "balance": the player's account balance amount
- "bet": the current total bet amount
- "result": where the WIN / LOSE / PUSH / "PLACE YOUR BETS" banner appears
- "timer": the betting/decision countdown number

Clickable controls (box the whole button/spot):
- "hit", "stand", "double", "split": the action buttons
- "bet_spot": the main bet circle/spot where chips are placed

Reply with ONLY a JSON object, no prose:
{{"balance": [x1,y1,x2,y2] or null, "bet": ..., "result": ..., "timer": ...,
 "hit": ..., "stand": ..., "double": ..., "split": ..., "bet_spot": ...}}"""


def bootstrap(frame_bgr) -> dict:
    """One screenshot -> {"ocr": {key: [l,t,r,b]}, "controls": {...}} in
    NATIVE frame pixels, only the keys the model could place. Blocking
    network call — run it off the Tk thread."""
    max_edge, max_tokens = _limits(constants.VISION.get("model"))
    image_b64, sx, sy = _encode_frame(frame_bgr, max_edge, max_tokens)
    reply = _json_block(_post(image_b64, _BOOTSTRAP_PROMPT, max_tokens=1024))
    h, w = frame_bgr.shape[:2]
    out = {"ocr": {}, "controls": {}}
    for key, rect in reply.items():
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            continue
        try:
            l, t, r, b = (float(v) for v in rect)
        except (TypeError, ValueError):
            continue
        rect = [max(0, min(w, int(round(min(l, r) * sx)))),
                max(0, min(h, int(round(min(t, b) * sy)))),
                max(0, min(w, int(round(max(l, r) * sx)))),
                max(0, min(h, int(round(max(t, b) * sy))))]
        if rect[2] - rect[0] < 4 or rect[3] - rect[1] < 4:
            continue
        if key in OCR_KEYS:
            out["ocr"][key] = rect
        elif key in CONTROL_KEYS:
            out["controls"][key] = rect
    return out


# --------------------------------------------------------------- triage

_TRIAGE_PROMPT = """This is a screenshot from a machine watching an online \
live-dealer blackjack table. The local detector cannot identify the current \
screen state. Classify it.

Reply with ONLY a JSON object:
{"label": one of ["betting_open", "dealing", "decision", "result",
 "modal_dialog", "disconnected", "shuffle", "lobby", "not_blackjack",
 "other"],
 "detail": "<one short sentence, e.g. 'an inactivity dialog with an OK
 button is covering the table'>"}"""


def triage(frame_bgr) -> str | None:
    """A short human-readable label for an unidentified screen, e.g.
    'modal_dialog — an inactivity dialog is covering the table'. Blocking
    network call — callers run it on the io pool. Frames are downscaled to
    the cheap (low-res) tier; a coarse label needs no extra fidelity."""
    image_b64, _, _ = _encode_frame(frame_bgr, 1568, 1568)
    reply = _json_block(_post(image_b64, _TRIAGE_PROMPT, max_tokens=300,
                              timeout=20))
    label = str(reply.get("label") or "").strip()
    detail = str(reply.get("detail") or "").strip()
    if not label and not detail:
        return None
    return f"{label} — {detail}" if detail else label
