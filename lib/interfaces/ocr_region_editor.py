"""OCR-region editor (V2 Feature 5): drag three rectangles over a live
screenshot — balance, current bet, result banner. Saved per resolution to
output/ocr_regions_{w}x{h}.json; the engine reloads them on save."""

import tkinter as tk

import cv2
from PIL import Image, ImageTk

from ..common import constants
from ..logic import ocr

C = constants.COLORS
HANDLE = 5
COLORS = {"balance": "#4dd2ff", "bet": "#ffd24d", "result": "#ff7eb6"}


class OcrRegionEditor(tk.Toplevel):
    def __init__(self, parent, capture, on_save=None):
        super().__init__(parent)
        self.title("OCR Regions — drag the corners over balance / bet / result")
        self.configure(bg=C["bg_primary"])
        self.on_save = on_save
        self.resolution = capture.resolution

        frame_bgr = capture.grab_bgr()
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        max_w = min(1380, self.winfo_screenwidth() - 80)
        max_h = self.winfo_screenheight() - 180
        self.scale = min(max_w / img.width, max_h / img.height, 1.0)
        view = img.resize((int(img.width * self.scale), int(img.height * self.scale)),
                          Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(view)

        self.canvas = tk.Canvas(self, width=view.width, height=view.height,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=10, pady=(10, 6))
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")

        saved = ocr.load_regions(self.resolution) or {}
        self.rects = {}
        for i, key in enumerate(ocr.REGION_KEYS):
            if key in saved:
                l, t, r, b = saved[key]
            else:  # stack defaults top-left so they're visible and grabbable
                l, t = 40, 40 + i * 90
                r, b = l + 260, t + 50
            s = self.scale
            self.rects[key] = [[l * s, t * s], [r * s, b * s]]

        self._drag = None
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))

        bar = tk.Frame(self, bg=C["bg_primary"])
        bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(bar, text="Blue: balance · Yellow: bet · Pink: result banner",
                 bg=C["bg_primary"], fg=C["text_secondary"],
                 font=constants.FONT_SMALL).pack(side=tk.LEFT)
        for text, cmd, bg in (("Save", self._save, C["success"]),
                              ("Disable OCR", self._clear, C["warning"]),
                              ("Cancel", self.destroy, C["text_secondary"])):
            tk.Button(bar, text=text, command=cmd, bg=bg,
                      fg="white" if bg != C["warning"] else C["text_primary"],
                      relief="flat", padx=14, pady=6, font=constants.FONT_BODY,
                      cursor="hand2").pack(side=tk.RIGHT, padx=4)
        self._redraw()
        self.grab_set()

    def _redraw(self):
        self.canvas.delete("overlay")
        for key, ((x1, y1), (x2, y2)) in self.rects.items():
            color = COLORS[key]
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2,
                                         tags="overlay")
            self.canvas.create_text(min(x1, x2) + 4, min(y1, y2) - 10, text=key,
                                    fill=color, anchor="w",
                                    font=constants.FONT_BODY_BOLD, tags="overlay")
            for x, y in ((x1, y1), (x2, y2)):
                self.canvas.create_oval(x - HANDLE, y - HANDLE, x + HANDLE,
                                        y + HANDLE, fill=color, outline="white",
                                        tags="overlay")

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
        self._redraw()

    def _save(self):
        out = {}
        for key, ((x1, y1), (x2, y2)) in self.rects.items():
            out[key] = [int(min(x1, x2) / self.scale), int(min(y1, y2) / self.scale),
                        int(max(x1, x2) / self.scale), int(max(y1, y2) / self.scale)]
        ocr.save_regions(self.resolution, out)
        if self.on_save:
            self.on_save()
        self.destroy()

    def _clear(self):
        ocr.delete_regions(self.resolution)
        if self.on_save:
            self.on_save()
        self.destroy()
