"""Interactive region calibration (Feature 10).

A live screenshot of the selected monitor with the seat polygons and dealer
rectangle drawn on top; every vertex is a draggable handle. Saving writes a
per-resolution profile (output/regions_{w}x{h}.json) that monitor_utils picks
up over the shipped defaults, then the engine reloads its regions.
"""

import tkinter as tk

import cv2
from PIL import Image, ImageTk

from ..common import constants
from ..logic import monitor_utils

C = constants.COLORS
HANDLE = 5  # handle radius in canvas px


class RegionEditor(tk.Toplevel):
    def __init__(self, parent, capture, on_save=None):
        super().__init__(parent)
        self.title("Calibrate Regions — drag the corner handles")
        self.configure(bg=C["bg_primary"])
        self.on_save = on_save
        self.capture = capture
        self.resolution = capture.resolution

        frame_bgr = capture.grab_bgr()
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        # Fit the screenshot into a workable window, remember the scale to
        # convert canvas coords back to native pixels on save.
        max_w, max_h = min(1380, self.winfo_screenwidth() - 80), self.winfo_screenheight() - 180
        self.scale = min(max_w / img.width, max_h / img.height, 1.0)
        view = img.resize((int(img.width * self.scale), int(img.height * self.scale)),
                          Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(view)

        self.canvas = tk.Canvas(self, width=view.width, height=view.height,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=10, pady=(10, 6))
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")

        custom = monitor_utils.load_custom_regions(self.resolution)
        if custom is not None:
            players = custom["players"]
            dealer = custom["dealer"]
        else:
            players = monitor_utils.default_player_regions(self.resolution)
            dealer = monitor_utils.default_dealer_rect(self.resolution)

        # Drop the duplicated closing vertex for editing; re-added on save.
        self.polys = []
        for poly in players:
            pts = [list(p) for p in poly]
            if len(pts) > 1 and pts[0] == pts[-1]:
                pts = pts[:-1]
            self.polys.append([[x * self.scale, y * self.scale] for x, y in pts])
        left, top, right, bottom = dealer
        self.dealer = [[left * self.scale, top * self.scale],
                       [right * self.scale, bottom * self.scale]]

        self._drag = None  # ("poly", i, j) or ("dealer", i)
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))

        bar = tk.Frame(self, bg=C["bg_primary"])
        bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(bar, text="Orange: seat regions · Red: dealer area",
                 bg=C["bg_primary"], fg=C["text_secondary"],
                 font=constants.FONT_SMALL).pack(side=tk.LEFT)
        for text, cmd, bg in (("Save", self._save, C["success"]),
                              ("Reset to defaults", self._reset, C["warning"]),
                              ("Cancel", self.destroy, C["text_secondary"])):
            tk.Button(bar, text=text, command=cmd, bg=bg,
                      fg="white" if bg != C["warning"] else C["text_primary"],
                      relief="flat", padx=14, pady=6, font=constants.FONT_BODY,
                      cursor="hand2").pack(side=tk.RIGHT, padx=4)

        self._redraw()
        self.grab_set()

    # ------------------------------------------------------------- drawing

    def _redraw(self):
        self.canvas.delete("overlay")
        for i, pts in enumerate(self.polys):
            flat = [c for p in pts for c in p]
            self.canvas.create_polygon(*flat, outline="#ff9f1a", fill="", width=2,
                                       tags="overlay")
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            self.canvas.create_text(cx, cy, text=f"P{i + 1}", fill="#ff9f1a",
                                    font=constants.FONT_BODY_BOLD, tags="overlay")
            for p in pts:
                self._handle(p, "#ff9f1a")
        (x1, y1), (x2, y2) = self.dealer
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ff4d4d", width=2,
                                     tags="overlay")
        self.canvas.create_text((x1 + x2) / 2, y1 + 14, text="DEALER", fill="#ff4d4d",
                                font=constants.FONT_BODY_BOLD, tags="overlay")
        for p in self.dealer:
            self._handle(p, "#ff4d4d")

    def _handle(self, p, color):
        x, y = p
        self.canvas.create_oval(x - HANDLE, y - HANDLE, x + HANDLE, y + HANDLE,
                                fill=color, outline="white", tags="overlay")

    # ------------------------------------------------------------- dragging

    def _press(self, event):
        best, best_d = None, (HANDLE * 3) ** 2
        for i, pts in enumerate(self.polys):
            for j, (x, y) in enumerate(pts):
                d = (x - event.x) ** 2 + (y - event.y) ** 2
                if d < best_d:
                    best, best_d = ("poly", i, j), d
        for i, (x, y) in enumerate(self.dealer):
            d = (x - event.x) ** 2 + (y - event.y) ** 2
            if d < best_d:
                best, best_d = ("dealer", i, 0), d
        self._drag = best

    def _move(self, event):
        if self._drag is None:
            return
        x = min(max(event.x, 0), self.canvas.winfo_width())
        y = min(max(event.y, 0), self.canvas.winfo_height())
        kind, i, j = self._drag
        if kind == "poly":
            self.polys[i][j] = [x, y]
        else:
            self.dealer[i] = [x, y]
        self._redraw()

    # --------------------------------------------------------------- save

    def _save(self):
        players = []
        for pts in self.polys:
            native = [[round(x / self.scale, 1), round(y / self.scale, 1)]
                      for x, y in pts]
            native.append(list(native[0]))  # close the polygon (base format)
            players.append(native)
        (x1, y1), (x2, y2) = self.dealer
        dealer = [int(min(x1, x2) / self.scale), int(min(y1, y2) / self.scale),
                  int(max(x1, x2) / self.scale), int(max(y1, y2) / self.scale)]
        monitor_utils.save_custom_regions(self.resolution, players, dealer)
        if self.on_save:
            self.on_save()
        self.destroy()

    def _reset(self):
        monitor_utils.delete_custom_regions(self.resolution)
        if self.on_save:
            self.on_save()
        self.destroy()
