"""Repro for claimed defect: UI scale 200% override grows window past screen.

Mirrors main.py exactly: per-monitor DPI awareness BEFORE tkinter, then the
real lib.interfaces.scaling module, the real _initial_geometry math from
ModernBlackjackGUI, and the real production call from settings_dialog.py
line 262: scaling.apply_override(root, 200).
"""
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(2)  # same as main.py

import sys
sys.path.insert(0, r"C:\Users\mauri\Documents\python\Blackjack-AI")

import tkinter as tk
import screeninfo
from lib.common import constants
from lib.interfaces import scaling

root = tk.Tk()
root.geometry("+100+100")  # ensure on primary monitor (2560x1600 @ 144dpi)
root.update_idletasks()

s = scaling.init(root)
print("auto scale after init:", s)

# --- ModernBlackjackGUI._initial_geometry, verbatim math ---
monitors = screeninfo.get_monitors()
mon = next((m for m in monitors if getattr(m, "is_primary", False)), monitors[0])
max_w, max_h = int(mon.width * 0.92), int(mon.height * 0.92)
width, height = min(scaling.px(1500), max_w), min(scaling.px(950), max_h)
print("initial geometry: %dx%d (monitor %dx%d, 92%% cap %dx%d)" % (
    width, height, mon.width, mon.height, max_w, max_h))

root.geometry(f"{width}x{height}")
root.minsize(min(scaling.px(1150), width), min(scaling.px(760), height))
root.update_idletasks()
root.update()
print("before override: winfo=%dx%d  state=%s" % (
    root.winfo_width(), root.winfo_height(), root.state()))

# --- _apply_chrome_scale stand-in: re-applies minsize only (as in the app) ---
def chrome():
    w, h = min(scaling.px(1500), max_w), min(scaling.px(950), max_h)
    root.minsize(min(scaling.px(1150), w), min(scaling.px(760), h))
scaling.on_change(chrome)

# --- the exact production call: settings_dialog.py _save -> line 262 ---
scaling.apply_override(root, 200)
root.update_idletasks()
root.update()

vw, vh = root.winfo_width(), root.winfo_height()
x, y = root.winfo_x(), root.winfo_y()
print("after apply_override(200): winfo=%dx%d at +%d+%d  geom=%s" % (
    vw, vh, x, y, root.geometry()))
print("primary monitor: %dx%d" % (mon.width, mon.height))
right_overflow = (x + vw) - mon.width
bottom_overflow = (y + vh) - mon.height
print("VERDICT: %s  (right edge %+d px, bottom edge %+d px vs monitor)" % (
    "OVERFLOWS primary monitor" if (right_overflow > 0 or bottom_overflow > 0)
    else "fits", right_overflow, bottom_overflow))

# Also sanity-check the reverse path heals it (claim says shrink direction works)
scaling.apply_override(root, 150)
root.update_idletasks()
root.update()
print("after apply_override(150) again: winfo=%dx%d" % (
    root.winfo_width(), root.winfo_height()))
root.destroy()
