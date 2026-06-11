"""Control-template capture editor (V4 Feature 1).

Drag a rectangle over each casino control on a live screenshot — Hit /
Stand / Double / Split buttons and the (empty) bet spot — and Save crops
them as match templates into assets/controls/<profile-slug>/<WxH>/ with the
rects in the active table profile. The phase detector template-matches
these every cycle to know when the action buttons are live.

Capture rules that matter (shown in the editor):
  * capture the buttons while they are VISIBLE AND ENABLED — the detector
    decides enabled/disabled by color distance from the capture;
  * capture the BET SPOT while it is EMPTY — chip-on-spot detection is
    "the spot no longer looks like its empty calibration".

Those two states never coexist on one screenshot, so calibration spans
several capture sessions. The checkbox therefore means "capture this
control from THIS screenshot": only checked keys are (re)cropped and
saved; unchecked keys keep their previously saved rect and template
untouched (phase.save_controls merges). The optional "✨ Suggest" button
asks the Claude vision assist to pre-fill the boxes from the screenshot.
"""

import threading
import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

from ..common import constants
from ..logic import phase, region_profiles, vision_assist

C = constants.COLORS
HANDLE = 5
COLORS = {"hit": "#2fbf71", "stand": "#4dabf7", "double": "#ffa94d",
          "split": "#c084fc", "bet_spot": "#ff7eb6"}
LABELS = {"hit": "Hit", "stand": "Stand", "double": "Double",
          "split": "Split", "bet_spot": "Bet spot (empty)"}


class ControlCaptureEditor(tk.Toplevel):
    def __init__(self, parent, capture, on_save=None):
        super().__init__(parent)
        self.title("Capture Controls — open this while the action buttons "
                    "are visible and enabled")
        self.configure(bg=C["bg_primary"])
        self.on_save = on_save
        self.resolution = capture.resolution

        self._frame_bgr = capture.grab_bgr()
        img = Image.fromarray(cv2.cvtColor(self._frame_bgr, cv2.COLOR_BGR2RGB))
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
            saved = region_profiles.get_controls(self.resolution) or {}
        except (ValueError, TypeError, AttributeError):
            pass
        self.rects = {}
        self.include = {}
        for i, key in enumerate(phase.CONTROL_KEYS):
            item = saved.get(key) or {}
            rect = item.get("rect")
            if rect and len(rect) == 4:
                l, t, r, b = rect
            else:  # stack defaults top-left so they're visible and grabbable
                l, t = 40, 40 + i * 70
                r, b = l + 150, t + 46
            s = self.scale
            self.rects[key] = [[l * s, t * s], [r * s, b * s]]
            # Unchecked by default: saving only captures CHECKED keys from
            # this screenshot; everything else keeps its earlier
            # calibration (a silent re-crop from a frame where the buttons
            # are greyed or a chip sits on the spot would corrupt the
            # templates). Dragging a box checks it automatically.
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
        for key in phase.CONTROL_KEYS:
            tk.Checkbutton(checks, text=LABELS[key], variable=self.include[key],
                           bg=C["bg_primary"], fg=COLORS[key],
                           font=constants.FONT_SMALL, selectcolor=C["bg_primary"],
                           activebackground=C["bg_primary"]
                           ).pack(side=tk.LEFT, padx=4)

        bar = tk.Frame(self, bg=C["bg_primary"])
        bar.pack(fill=tk.X, padx=10, pady=(2, 10))
        self.status_var = tk.StringVar(
            value="Checked = captured from THIS screenshot (buttons need to "
                  "be enabled, the spot empty); unchecked keep their saved "
                  "calibration.")
        tk.Label(bar, textvariable=self.status_var, bg=C["bg_primary"],
                 fg=C["text_secondary"], font=constants.FONT_SMALL
                 ).pack(side=tk.LEFT)
        for text, cmd, bg in (("Save", self._save, C["success"]),
                              ("Clear calibration", self._clear, C["warning"]),
                              ("Cancel", self.destroy, C["text_secondary"])):
            tk.Button(bar, text=text, command=cmd, bg=bg,
                      fg="white" if bg != C["warning"] else C["text_primary"],
                      relief="flat", padx=14, pady=6, font=constants.FONT_BODY,
                      cursor="hand2").pack(side=tk.RIGHT, padx=4)
        # Save target: a table profile by name (same pattern as the region
        # editor) — prefilled with the active profile.
        self.profile_var = tk.StringVar(value=region_profiles.active_name())
        entry = tk.Entry(bar, textvariable=self.profile_var, width=16,
                         font=constants.FONT_BODY)
        entry.pack(side=tk.RIGHT, padx=(12, 2))
        tk.Label(bar, text="Save as profile:", bg=C["bg_primary"],
                 fg=C["text_secondary"], font=constants.FONT_BODY
                 ).pack(side=tk.RIGHT)
        if vision_assist.available():
            self._suggest_btn = tk.Button(
                bar, text="✨ Suggest", command=self._suggest,
                bg=C["accent"], fg="white", relief="flat", padx=14, pady=6,
                font=constants.FONT_BODY, cursor="hand2")
            self._suggest_btn.pack(side=tk.RIGHT, padx=(0, 12))

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
                              min(max(event.y, 0), self.canvas.winfo_height())]
        self.include[key].set(True)  # touching a box means you want it saved
        self._redraw()

    # --------------------------------------------------------- AI suggest

    def _suggest(self):
        """Ask the Claude vision assist to place the boxes; results pre-fill
        the editor for human fine-tuning (never saved unreviewed)."""
        self._suggest_btn.config(state="disabled")
        self.status_var.set("Asking Claude to locate the controls…")
        frame = self._frame_bgr

        def work():
            try:
                suggestion = vision_assist.bootstrap(frame)
                self.after(0, lambda: self._apply_suggestion(suggestion))
            except Exception as e:
                self.after(0, lambda e=e: self._suggest_failed(e))

        threading.Thread(target=work, daemon=True).start()

    def _apply_suggestion(self, suggestion):
        if not self.winfo_exists():
            return
        self._suggest_btn.config(state="normal")
        boxes = suggestion.get("controls", {})
        for key, rect in boxes.items():
            if key not in self.rects:
                continue
            l, t, r, b = rect
            s = self.scale
            self.rects[key] = [[l * s, t * s], [r * s, b * s]]
            self.include[key].set(True)
        self.status_var.set(
            f"Claude placed {len(boxes)} control(s) — adjust, then Save."
            if boxes else "Claude found no controls on this screenshot — "
                          "are the buttons visible?")
        self._redraw()

    def _suggest_failed(self, error):
        if not self.winfo_exists():
            return
        self._suggest_btn.config(state="normal")
        self.status_var.set(f"Suggest failed: {error}")

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
            if rect[2] - rect[0] < 8 or rect[3] - rect[1] < 8:
                messagebox.showerror(
                    "Capture Controls",
                    f"The {LABELS[key]} box is too small to be a template — "
                    "drag it over the control or uncheck it.", parent=self)
                return
            crop = self._frame_bgr[rect[1]:rect[3], rect[0]:rect[2]].copy()
            captures[key] = {"rect": rect, "crop": crop}
        if not captures:
            messagebox.showerror(
                "Capture Controls",
                "Nothing is checked — check the control(s) to capture from "
                "this screenshot.", parent=self)
            return
        # The save MERGES with the target profile's earlier captures; warn
        # only when the merged result still can't detect the turn.
        target = self.profile_var.get().strip() or \
            region_profiles.active_name()
        try:
            existing = region_profiles.get_controls(
                self.resolution, profile=target) or {}
        except (ValueError, TypeError, AttributeError):
            existing = {}
        merged = set(existing) | set(captures)
        if not {"hit", "stand"} <= merged:
            if not messagebox.askyesno(
                    "Capture Controls",
                    "Hit AND Stand are both needed for turn detection and "
                    "this profile still won't have both — save anyway?",
                    parent=self):
                return
        try:
            phase.save_controls(self.resolution, captures,
                                profile=self.profile_var.get())
        except (OSError, ValueError) as e:
            messagebox.showerror("Capture Controls",
                                 f"Saving templates failed: {e}", parent=self)
            return
        if self.on_save:
            self.on_save()
        self.destroy()

    def _clear(self):
        active = region_profiles.active_name()
        if not messagebox.askyesno(
                "Clear calibration",
                f"Remove the captured controls of profile \"{active}\" for "
                "this resolution? Turn detection switches off until you "
                "re-capture.", parent=self):
            return
        phase.delete_controls(self.resolution)
        if self.on_save:
            self.on_save()
        self.destroy()
