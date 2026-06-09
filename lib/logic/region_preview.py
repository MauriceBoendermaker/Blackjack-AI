"""Region/debug preview: capture the table once, run the player model, and
draw detected cards + seat polygons + the dealer area onto the screenshot.

Pure logic (no Tkinter) — runs on a worker thread; returns a PIL image.
"""

import cv2
import numpy as np
from PIL import Image

from ..common import constants
from ..common.card_mappings import PLAYER_CLASS_MAP
from .monitor_utils import dealer_area_rect


def build_region_preview(engine) -> Image.Image:
    """Uses the engine's capture + models (shares the lazy singletons).
    Runs on a short-lived worker thread — releases its capture handle on exit."""
    try:
        frame = engine.capture.grab_bgr()
    finally:
        engine.capture.close_local()
    preds = engine.provider.players_model().predict(
        frame, constants.PREDICTION_CONFIDENCE_PLAYERS, constants.PREDICTION_OVERLAP_PLAYERS)

    img = frame.copy()

    # Detected cards (green boxes).
    for p in preds:
        name = PLAYER_CLASS_MAP.get(p["class"], p["class"])
        x1 = int(p["cx"] - p["width"] / 2)
        y1 = int(p["cy"] - p["height"] / 2)
        x2 = int(p["cx"] + p["width"] / 2)
        y2 = int(p["cy"] + p["height"] / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"{name} {p['confidence']:.2f}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

    # Seat polygons (blue) with seat numbers.
    if engine.regions:
        for i, region in enumerate(engine.regions):
            pts = region.vertices.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(255, 120, 0), thickness=2)
            cx, cy = region.vertices.mean(axis=0)
            cv2.putText(img, f"P{i + 1}", (int(cx) - 14, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 120, 0), 2)

    # Dealer area (red).
    left, top, right, bottom = dealer_area_rect(engine.capture.resolution)
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 220), 2)
    cv2.putText(img, "DEALER AREA", (left + 8, top + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
