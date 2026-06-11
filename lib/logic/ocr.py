"""Casino-UI OCR (V2 Feature 5): balance, current bet, result banner.

Engine choice per research: Windows.Media.Ocr via `winocr` — the OS-built-in
engine, no model download, no GPU, faster and more accurate on rendered
screen text than Tesseract. It has no character whitelist, so numeric fields
are post-filtered with a regex (the € glyph commonly OCRs as '?') and the
result banner snaps to its tiny fixed vocabulary.

Regions are per-resolution rectangles stored in the ACTIVE table profile
(lib/logic/region_profiles.py), drawn with the OCR-region editor. Reads run
on the engine's I/O thread at ~1 Hz; results sync the bankroll ("ground
truth beats bookkeeping"), the bet-placed field, and cross-check the result
banner against settlement.
"""

import re
import unicodedata

from . import region_profiles

try:
    import winocr  # Windows-only; optional at runtime
    OCR_AVAILABLE = True
except Exception:  # pragma: no cover - import availability depends on the OS
    winocr = None
    OCR_AVAILABLE = False

REGION_KEYS = ("balance", "bet", "result", "timer")

_AMOUNT_RE = re.compile(r"\d[\d.,]*")
_RESULT_VOCAB = {
    "win": "win", "won": "win", "wins": "win",
    "lose": "lose", "loss": "lose", "lost": "lose", "bust": "lose",
    "push": "push", "tie": "push",
    "blackjack": "blackjack", "bj": "blackjack",
}

def _normalize(text) -> str:
    """Accent-stripped lowercase word soup — both the OCR text and the
    phrase vocabulary go through this, so 'PLAATS UW  INZET!' and localized
    accents ('más', 'einsätze') match regardless of how winocr renders
    them."""
    if not text:
        return ""
    ascii_text = unicodedata.normalize("NFKD", str(text)).encode(
        "ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-zA-Z]+", ascii_text.lower()))


# Phase-status phrases on the banner (often the same screen area as the
# result banner). Evolution localizes the client, so the common locales are
# included; everything is matched through _normalize. Longest-first so a
# longer phrase can never be shadowed by a shorter one.
_STATUS_PHRASES = sorted(
    ([(_normalize(p), s) for p, s in [
        ("place your bets", "betting"), ("place bets", "betting"),
        ("bets open", "betting"),
        ("plaats uw inzet", "betting"), ("plaats je inzet", "betting"),
        ("platzieren sie ihre einsätze", "betting"),
        ("placez vos mises", "betting"),
        ("hagan sus apuestas", "betting"), ("haga sus apuestas", "betting"),
        ("fate il vostro gioco", "betting"),
        ("no more bets", "closed"), ("bets closed", "closed"),
        ("betting closed", "closed"),
        ("geen inzetten meer", "closed"),
        ("keine einsätze mehr", "closed"),
        ("keine weiteren einsätze", "closed"),
        ("rien ne va plus", "closed"), ("les jeux sont faits", "closed"),
        ("no más apuestas", "closed"),
        ("make your decision", "decision"), ("decide", "decision"),
    ]]),
    key=lambda item: -len(item[0]))
_TIMER_RE = re.compile(r"(?:(\d+)\s*:\s*)?(\d{1,3})\s*s?", re.ASCII)


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


def snap_status(text) -> str | None:
    """Banner text -> 'betting' | 'closed' | 'decision' | None. Normalizes
    whitespace and accents so 'PLACE  YOUR\\nBETS' and localized banners
    still match."""
    joined = _normalize(text)
    if not joined:
        return None
    for phrase, status in _STATUS_PHRASES:
        if phrase in joined:
            return status
    return None


def parse_timer(text) -> int | None:
    """Countdown text -> whole seconds ('9' -> 9, '0:12' -> 12, '8s' -> 8).
    None when no digits; values above 120 are misreads (no live-table
    countdown runs minutes)."""
    if not text:
        return None
    m = _TIMER_RE.search(text)
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) else 0
    seconds = minutes * 60 + int(m.group(2))
    return seconds if 0 <= seconds <= 120 else None


# --------------------------------------------------------- region profiles

def load_regions(resolution):
    """{'balance': [l,t,r,b], ...} (keys optional) or None when unset —
    the ACTIVE table profile's OCR rects for this resolution."""
    try:
        data = region_profiles.get_ocr(resolution)
        if data is None:
            return None
        out = {}
        for key in REGION_KEYS:
            rect = data.get(key)
            if rect and len(rect) == 4 and rect[2] > rect[0] and rect[3] > rect[1]:
                out[key] = [int(v) for v in rect]
        return out or None
    except (ValueError, TypeError, AttributeError) as e:
        print(f"OCR regions unreadable ({e}); OCR disabled.")
        return None


def save_regions(resolution, regions: dict):
    """Save into the active table profile (stamps its date)."""
    region_profiles.set_ocr(resolution, regions)


def delete_regions(resolution):
    region_profiles.delete_ocr(resolution)


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
    'result': str|None, 'status': str|None, 'timer': int|None,
    'stray_text': bool}.

    `status` (betting open/closed) reads the result-banner area — Evolution
    shows "PLACE YOUR BETS" where results appear — with the dedicated timer
    region as a fallback for tables that print it next to the countdown.
    `stray_text` flags substantial banner-area text that matches NO known
    vocabulary — the signature of a modal/disconnect dialog covering the
    table (the phase detector treats it as an UNKNOWN-screen signal)."""
    status = snap_status(texts.get("result")) or snap_status(texts.get("timer"))
    result = snap_result(texts.get("result"))
    words = [w for w in _normalize(texts.get("result")).split() if len(w) >= 3]
    return {
        "balance": parse_amount(texts.get("balance")),
        "bet": parse_amount(texts.get("bet")),
        "result": result,
        "status": status,
        "timer": parse_timer(texts.get("timer")),
        "stray_text": bool(len(words) >= 2 and status is None
                           and result is None),
    }
