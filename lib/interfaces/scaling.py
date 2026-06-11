"""Per-monitor DPI scaling: live named fonts, pixel helpers, a rescale watch.

init() resolves the window's DPI once and replaces the constants.FONT_*
tuples with named tkfont.Font objects sized in physical pixels — every
widget created with `font=constants.FONT_*` is then linked to the named
font, so reconfiguring it on a DPI change restyles the whole UI in place.
watch() detects the window moving between monitors of different DPI.

resolve_scale/px/size are pure; importing this module never needs a display
(Tk objects are only created inside init). Everything runs on the Tk thread.
"""

import ctypes
import tkinter as tk
import tkinter.font as tkfont

from ..common import constants

# Scale-factor bounds: below 0.75 text turns unreadable, above 3.0 nothing
# fits even on the largest monitor.
_MIN_SCALE, _MAX_SCALE = 0.75, 3.0
# Relative scale changes smaller than this are noise (DPI reads jitter while
# the window straddles two monitors mid-drag).
_CHANGE_THRESHOLD = 0.02
_WATCH_DEBOUNCE_MS = 250

# 96-dpi baseline tuples, captured before init() replaces the constants
# attributes with live Font objects.
_BASELINES = {name: getattr(constants, name) for name in (
    "FONT_TITLE", "FONT_SECTION", "FONT_BODY", "FONT_BODY_BOLD",
    "FONT_SMALL", "FONT_BIG_VALUE")}

_scale = 1.0
_fonts = {}       # constants attr name -> tkfont.Font, created by init()
_callbacks = []   # fired after every rescale (fonts already updated)
_resizing = False  # re-entrancy guard: our own geometry() fires <Configure>


def resolve_scale(dpi, override_pct):
    """Scale factor from DPI, or from a manual override percent.

    Pure (no Tk). A falsy override means auto: dpi / 96. Either way the
    result is clamped to [0.75, 3.0]."""
    if override_pct:
        factor = float(override_pct) / 100.0
    else:
        factor = float(dpi) / 96.0
    return min(max(factor, _MIN_SCALE), _MAX_SCALE)


def scale():
    """Current scale factor (1.0 = 96 dpi)."""
    return _scale


def px(n):
    """A 96-dpi pixel measure scaled to the current factor."""
    return round(n * _scale)


def size(wh_tuple):
    """Scaled copy of a (width, height) tuple — card render sizes."""
    return (round(wh_tuple[0] * _scale), round(wh_tuple[1] * _scale))


def on_change(callback):
    """Register a no-arg callback fired after every rescale."""
    _callbacks.append(callback)


def init(root):
    """Resolve the root window's DPI and install the live pixel-sized fonts.

    Must run right after the Tk root exists and before any other widget so
    every later `font=constants.FONT_*` picks up the named Font objects.
    Returns the resolved scale factor."""
    global _scale
    _scale = resolve_scale(_window_dpi(root), constants.UI["scale"])
    _configure_tk_scaling(root)
    for name, base in _BASELINES.items():
        font = _make_font(root, name, base)
        _fonts[name] = font
        setattr(constants, name, font)
    return _scale


def watch(root):
    """Rescale live when the window lands on a monitor with a different DPI.

    <Configure> fires constantly during drags, so the DPI re-read sits
    behind a 250 ms after()-debounce; only a >=2% scale change does work."""
    state = {"job": None}

    def check():
        state["job"] = None
        new_scale = resolve_scale(_window_dpi(root), constants.UI["scale"])
        if abs(new_scale - _scale) / _scale < _CHANGE_THRESHOLD:
            return
        ratio = new_scale / _scale
        _apply_scale(root, new_scale)
        _resize_window(root, ratio)
        _fire_callbacks()

    def on_configure(event):
        if event.widget is not root or _resizing:
            return
        if state["job"] is not None:
            root.after_cancel(state["job"])
        state["job"] = root.after(_WATCH_DEBOUNCE_MS, check)

    root.bind("<Configure>", on_configure, add="+")


def apply_override(root, pct):
    """Settings-dialog hook: 0 = auto (per-monitor), else a fixed percent."""
    constants.UI["scale"] = pct
    old = _scale
    _apply_scale(root, resolve_scale(_window_dpi(root), pct))
    _resize_window(root, _scale / old)
    _fire_callbacks()


# ------------------------------------------------------------------ internals

def _window_dpi(root):
    """Per-monitor DPI of the window via Win32; 96 on any failure."""
    try:
        return int(ctypes.windll.user32.GetDpiForWindow(root.winfo_id())) or 96
    except Exception:
        return 96


def _font_px(base_pt):
    """Baseline point size -> physical pixels (negative = pixels to Tk)."""
    return -round(base_pt * 4.0 / 3.0 * _scale)


def _configure_tk_scaling(root):
    # Fonts are sized in pixels directly, so Tk's points-per-pixel factor is
    # set to match the same scale — point-based measures stay consistent
    # without double-scaling the fonts.
    root.tk.call("tk", "scaling", _scale * 96.0 / 72.0)


def _make_font(root, name, base):
    """Create (or re-attach and reconfigure) the named font for one role."""
    weight = "bold" if "bold" in base[2:] else "normal"
    try:
        return tkfont.Font(root=root, name=f"BJ_{name}", exists=False,
                           family=base[0], size=_font_px(base[1]), weight=weight)
    except tk.TclError:
        # Same interpreter, second init: the named font survives — reuse it.
        font = tkfont.Font(root=root, name=f"BJ_{name}", exists=True)
        font.configure(family=base[0], size=_font_px(base[1]), weight=weight)
        return font


def _apply_scale(root, new_scale):
    """Update tk scaling and reconfigure the named fonts in place — every
    live widget holding a FONT_* object updates instantly."""
    global _scale
    _scale = new_scale
    _configure_tk_scaling(root)
    for name, base in _BASELINES.items():
        font = _fonts.get(name)
        if font is not None:
            font.configure(size=_font_px(base[1]))


def _resize_window(root, ratio):
    """Resize the toplevel proportionally so the layout keeps its density."""
    global _resizing
    if _resizing or abs(ratio - 1.0) < 0.001:
        return
    _resizing = True
    width = max(200, round(root.winfo_width() * ratio))
    height = max(150, round(root.winfo_height() * ratio))
    root.geometry(f"{width}x{height}")
    # Released on a timer so the <Configure> burst from our own resize has
    # drained before the watcher may schedule another DPI check.
    root.after(_WATCH_DEBOUNCE_MS + 50, _end_resize)


def _end_resize():
    global _resizing
    _resizing = False


def _fire_callbacks():
    for callback in list(_callbacks):
        try:
            callback()
        except Exception as e:
            # One listener failing must not block the others' re-render.
            print(f"scaling on_change callback failed: {e}")
