"""Casino-UI OCR (V2 Feature 5): balance, current bet, result banner.

Engine choice per research: Windows.Media.Ocr via `winocr` — the OS-built-in
engine, no model download, no GPU, faster and more accurate on rendered
screen text than Tesseract. It has no character whitelist, so numeric fields
are post-filtered with a regex (the € glyph commonly OCRs as '?') and the
result banner snaps to its tiny fixed vocabulary.

Regions are per-resolution rectangles in output/ocr_regions_{w}x{h}.json,
drawn with the OCR-region editor. Reads run on the engine's advice thread at
~1 Hz; results sync the bankroll ("ground truth beats bookkeeping"), the
bet-placed field, and cross-check the result banner against settlement.
"""

import json
import re

from ..common import constants

try:
    import winocr  # Windows-only; optional at runtime
    OCR_AVAILABLE = True
except Exception:  # pragma: no cover - import availability depends on the OS
    winocr = None
    OCR_AVAILABLE = False

REGION_KEYS = ("balance", "bet", "result")

_AMOUNT_RE = re.compile(r"\d[\d.,]*")
_RESULT_VOCAB = {
    "win": "win", "won": "win", "wins": "win",
    "lose": "lose", "loss": "lose", "lost": "lose", "bust": "lose",
    "push": "push", "tie": "push",
    "blackjack": "blackjack", "bj": "blackjack",
}


# ------------------------------------------------------------- pure parsing

def parse_amount(text) -> float | None:
    """'? 1,234.56' -> 1234.56; handles EU '1.234,56' too. None when no number."""
    if not text:
        return None
    m = _AMOUNT_RE.search(text)
    if not m:
        return None
    raw = m.group(0).strip(".,")
    if "," in raw and "." in raw:
        # The later separator is the decimal point.
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        head, _, tail = raw.rpartition(",")
        raw = head.replace(",", "") + "." + tail if len(tail) == 2 else raw.replace(",", "")
    elif raw.count(".") > 1 or ("." in raw and len(raw.rpartition(".")[2]) == 3):
        raw = raw.replace(".", "")  # thousands-only dots
    try:
        return float(raw)
    except ValueError:
        return None


def snap_result(text) -> str | None:
    """Banner text -> 'win' | 'lose' | 'push' | 'blackjack' | None."""
    if not text:
        return None
    for word in re.findall(r"[A-Za-z]+", text.lower()):
        if word in _RESULT_VOCAB:
            return _RESULT_VOCAB[word]
    return None


# --------------------------------------------------------- region profiles

def regions_path(resolution):
    w, h = resolution
    return constants.OUTPUT_DIR / f"ocr_regions_{w}x{h}.json"


def load_regions(resolution):
    """{'balance': [l,t,r,b], ...} (keys optional) or None when unset."""
    path = regions_path(resolution)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = {}
        for key in REGION_KEYS:
            rect = data.get(key)
            if rect and len(rect) == 4 and rect[2] > rect[0] and rect[3] > rect[1]:
                out[key] = [int(v) for v in rect]
        return out or None
    except (OSError, ValueError, TypeError) as e:
        print(f"OCR regions unreadable ({e}); OCR disabled.")
        return None


def save_regions(resolution, regions: dict):
    path = regions_path(resolution)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(regions, indent=1), encoding="utf-8")


def delete_regions(resolution):
    path = regions_path(resolution)
    if path.exists():
        path.unlink()


# ----------------------------------------------------------------- reading

def read_regions(frame_bgr, regions: dict) -> dict:
    """OCR each configured crop of a BGR frame -> raw text per region key.
    Crops are upscaled 2x — Windows OCR likes capital heights >= ~30 px."""
    if not OCR_AVAILABLE:
        return {}
    import cv2
    from PIL import Image
    out = {}
    h, w = frame_bgr.shape[:2]
    for key, (left, top, right, bottom) in regions.items():
        left, top = max(0, left), max(0, top)
        right, bottom = min(w, right), min(h, bottom)
        if right - left < 4 or bottom - top < 4:
            continue
        crop = frame_bgr[top:bottom, left:right]
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        try:
            result = winocr.recognize_pil_sync(pil)
            out[key] = (result.get("text", "") if isinstance(result, dict)
                        else getattr(result, "text", ""))
        except Exception:
            out[key] = ""
    return out


def interpret(texts: dict) -> dict:
    """Raw OCR texts -> {'balance': float|None, 'bet': float|None,
    'result': str|None}."""
    return {
        "balance": parse_amount(texts.get("balance")),
        "bet": parse_amount(texts.get("bet")),
        "result": snap_result(texts.get("result")),
    }
