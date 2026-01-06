import os
from pathlib import Path
from typing import List, Dict, Any
from ..common import constants

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class _PredictionResult:
    def __init__(self, preds: List[Dict[str, Any]], src_path: str | None = None):
        self._preds = preds
        self._src_path = src_path

    def json(self):
        return {"predictions": self._preds}

    def save(self, out_path: str):
        if self._src_path is None:
            return
        try:
            from PIL import Image, ImageDraw
            img = Image.open(self._src_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            for p in self._preds:
                x, y, w, h = p["x"], p["y"], p["width"], p["height"]
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, max(0, y1 - 12)), f"{p['class']} {p['confidence']:.2f}", fill="red")
            img.save(out_path)
        except Exception:
            pass


class LocalYoloModel:
    def __init__(self, weights_path: str):
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. pip install ultralytics")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing weights: {weights_path}")
        self.model = YOLO(weights_path)

    def predict(self, file_path: str, confidence: float = 50, overlap: float = 45):
        conf = max(0.0, min(1.0, confidence / 100.0))
        iou = max(0.0, min(1.0, getattr(constants, "PREDICTION_IOU", 50) / 100.0))
        results = self.model.predict(source=file_path, conf=conf, iou=iou, imgsz=640, verbose=False, device=0)
        preds = []
        if not results:
            return _PredictionResult(preds, src_path=file_path)
        r = results[0]
        names = r.names if hasattr(r, "names") else {}
        for b in r.boxes:
            cls_idx = int(b.cls.item())
            cls_name = str(names.get(cls_idx, str(cls_idx)))
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            conf_val = float(b.conf.item())
            preds.append({
                "x": x1,
                "y": y1,
                "width": w,
                "height": h,
                "class": cls_name,
                "confidence": conf_val,
            })
        return _PredictionResult(preds, src_path=file_path)


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

class Utils:
    def __init__(self):
        self._player_model = None
        self._dealer_model = None

    def initialize_player_model(self):
        if self._player_model is None:
            weights = str(MODELS_DIR / "player_cards.pt")
            self._player_model = LocalYoloModel(weights)
        return self._player_model

    def initialize_dealer_model(self):
        if self._dealer_model is None:
            weights = MODELS_DIR / "dealer_cards.pt"
            if not weights.exists():
                weights = MODELS_DIR / "player_cards.pt"
            self._dealer_model = LocalYoloModel(str(weights))
        return self._dealer_model

    def generate_card_image_path(self, card: str):
        card = card.replace(" ", "_")
        return os.path.join(constants.CARD_FOLDER_PATH, f"{card}.png")
