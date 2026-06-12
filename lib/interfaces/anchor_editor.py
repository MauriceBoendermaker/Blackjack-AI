"""Anchor capture editor (V3 E3).

Drag a box over 2-3 STABLE casino-UI elements — the table logo, the chip
tray, the menu button — on a live screenshot. Save crops them as anchor
templates beside the control templates; from then on the engine matches
them at startup (and on Re-anchor) to solve the scale+offset transform
that maps this profile's calibrated regions to whatever the screen
actually shows — new resolution, moved window, different DPI.

Pick elements that NEVER move or change appearance during play (no cards,
no chips, no buttons that grey out), spread far apart — the farther
apart, the more accurate the solved scale.
"""

import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

from ..common import constants
from ..logic import anchors, region_profiles

C = constants.COLORS
HANDLE = 5
COLORS = {"anchor_a": "#2fbf71", "anchor_b": "#4dabf7",
          "anchor_c": "#ffa94d"}
LABELS = dict(anchors.ANCHOR_KEYS)


class AnchorEditor(tk.Toplevel):
    def __init__(self, parent, capture, on_save=None):
        super().__init__(parent)
        self.title("Capture Anchors — pick 2-3 stable UI elements "
                   "(logo, chip tray, menu)")
        self.configure(bg=C["bg_primary"])
        self.on_save = on_save
        self.resolution = capture.resolution

        self._frame_bgr = capture.grab_bgr()
        img = Image.fromarray(cv2.cvtColor(self._frame_bgr,
                                           cv2.COLOR_BGR2RGB))
        max_w = min(1380, self.winfo_screenwidth() - 80)
        max_h = self.winfo_screenheight() - 220
        self.scale = min(max_w / img.width, max_h / img.height, 1.0)
        view = img.resize((int(img.width * self.scale),
                           int(img.height * self.scale)), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(view)

        self.canvas = tk.Canvas(self, width=view.width, height=view.height,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=10, pady=(10, 6))
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")

        saved = {}
        try:
            saved = region_profiles.get_anchors(self.resolution) or {}
        except (ValueError, TypeError, AttributeError):
            pass
        self.rects = {}
        self.include = {}
        for i, (key, _label) in enumerate(anchors.ANCHOR_KEYS):
            item = saved.get(key) or {}
            rect = item.get("rect")
            if rect and len(rect) == 4:
                left, top, right, bottom = rect
            else:
                left, top = 40, 40 + i * 70
                right, bottom = left + 120, top + 46
            s = self.scale
            self.rects[key] = [[left * s, top * s], [right * s, bottom * s]]
            self.include[key] = tk.BooleanVar(value=False)

        self._drag = None
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>",
                         lambda e: setattr(self, "_drag", None))

        checks = tk.Frame(self, bg=C["bg_primary"])
        checks.pack(fill=tk.X, padx=10)
        tk.Label(checks, text="Capture now:", bg=C["bg_primary"],
                 fg=C["text_secondary"], font=constants.FONT_SMALL
                 ).pack(side=tk.LEFT)
        for key, label in anchors.ANCHOR_KEYS:
            tk.Checkbutton(checks, text=label, variable=self.include[key],
                           bg=C["bg_primary"], fg=COLORS[key],
                           font=constants.FONT_SMALL,
                           selectcolor=C["bg_primary"],
                           activebackground=C["bg_primary"]
                           ).pack(side=tk.LEFT, padx=4)

        bar = tk.Frame(self, bg=C["bg_primary"])
        bar.pack(fill=tk.X, padx=10, pady=(2, 10))
        tk.Label(bar, text=("Anchors must never move or change during play; "
                            "spread them apart for an accurate fit."),
                 bg=C["bg_primary"], fg=C["text_secondary"],
                 font=constants.FONT_SMALL).pack(side=tk.LEFT)
        for text, cmd, bg in (("Save", self._save, C["success"]),
                              ("Clear anchors", self._clear, C["warning"]),
                              ("Cancel", self.destroy, C["text_secondary"])):
            tk.Button(bar, text=text, command=cmd, bg=bg,
                      fg="white" if bg != C["warning"] else C["text_primary"],
                      relief="flat", padx=14, pady=6,
                      font=constants.FONT_BODY,
                      cursor="hand2").pack(side=tk.RIGHT, padx=4)
        self.profile_var = tk.StringVar(value=region_profiles.active_name())
        entry = tk.Entry(bar, textvariable=self.profile_var, width=16,
                         font=constants.FONT_BODY)
        entry.pack(side=tk.RIGHT, padx=(12, 2))
        tk.Label(bar, text="Save as profile:", bg=C["bg_primary"],
                 fg=C["text_secondary"], font=constants.FONT_BODY
                 ).pack(side=tk.RIGHT)

        self._redraw()
        self.grab_set()

    # ------------------------------------------------------------ drawing

    def _redraw(self):
        self.canvas.delete("overlay")
        for key, ((x1, y1), (x2, y2)) in self.rects.items():
            color = COLORS[key]
            dimmed = not self.include[key].get()
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=color, width=1 if dimmed else 2,
                dash=(3, 3) if dimmed else None, tags="overlay")
            self.canvas.create_text(min(x1, x2) + 4, min(y1, y2) - 10,
                                    text=LABELS[key], fill=color, anchor="w",
                                    font=constants.FONT_BODY_BOLD,
                                    tags="overlay")
            for x, y in ((x1, y1), (x2, y2)):
                self.canvas.create_oval(x - HANDLE, y - HANDLE, x + HANDLE,
                                        y + HANDLE, fill=color,
                                        outline="white", tags="overlay")

    def _press(self, event):
        best, best_d = None, (HANDLE * 3) ** 2
        for key, corners in self.rects.items():
            for i, (x, y) in enumerate(corners):
                d = (x - event.x) ** 2 + (y - event.y) ** 2
                if d < best_d:
                    best, best_d = (key, i), d
        self._drag = best

    def _move(self, event):
        if self._drag is None:
            return
        key, i = self._drag
        self.rects[key][i] = [min(max(event.x, 0), self.canvas.winfo_width()),
                              min(max(event.y, 0),
                                  self.canvas.winfo_height())]
        self.include[key].set(True)
        self._redraw()

    # --------------------------------------------------------------- save

    def _save(self):
        captures = {}
        h, w = self._frame_bgr.shape[:2]
        for key, ((x1, y1), (x2, y2)) in self.rects.items():
            if not self.include[key].get():
                continue
            rect = [max(0, int(min(x1, x2) / self.scale)),
                    max(0, int(min(y1, y2) / self.scale)),
                    min(w, int(max(x1, x2) / self.scale)),
                    min(h, int(max(y1, y2) / self.scale))]
            if rect[2] - rect[0] < 12 or rect[3] - rect[1] < 12:
                messagebox.showerror(
                    "Capture Anchors",
                    f"The {LABELS[key]} box is too small to match reliably — "
                    "drag it over a distinctive element or uncheck it.",
                    parent=self)
                return
            crop = self._frame_bgr[rect[1]:rect[3], rect[0]:rect[2]].copy()
            captures[key] = {"rect": rect, "crop": crop}
        if not captures:
            messagebox.showerror(
                "Capture Anchors",
                "Nothing is checked — check the anchor(s) to capture from "
                "this screenshot.", parent=self)
            return
        target = self.profile_var.get().strip() or \
            region_profiles.active_name()
        try:
            existing = region_profiles.get_anchors(
                self.resolution, profile=target) or {}
        except (ValueError, TypeError, AttributeError):
            existing = {}
        if len(set(existing) | set(captures)) < \
                int(constants.ANCHORS["min_anchors"]):
            messagebox.showerror(
                "Capture Anchors",
                f"At least {constants.ANCHORS['min_anchors']} anchors are "
                "needed for a trustworthy fit — capture another one.",
                parent=self)
            return
        try:
            anchors.save_anchors(self.resolution, captures,
                                 profile=self.profile_var.get())
        except (OSError, ValueError) as e:
            messagebox.showerror("Capture Anchors",
                                 f"Saving anchors failed: {e}", parent=self)
            return
        if self.on_save:
            self.on_save()
        self.destroy()

    def _clear(self):
        active = region_profiles.active_name()
        if not messagebox.askyesno(
                "Clear anchors",
                f"Remove the anchors of profile \"{active}\" for this "
                "resolution? Resolution-independent mapping switches off "
                "until you re-capture.", parent=self):
            return
        region_profiles.delete_anchors(self.resolution)
        if self.on_save:
            self.on_save()
        self.destroy()
