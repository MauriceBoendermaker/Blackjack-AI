import os
from pathlib import Path
from typing import List, Dict, Any
from ..common import constants

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except Exception:
    ROBOFLOW_AVAILABLE = False

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
        # Reduced from 640 to 416 for 30-40% speed improvement with minimal accuracy loss
        results = self.model.predict(source=file_path, conf=conf, iou=iou, imgsz=416, verbose=False, device=0)
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


class RoboflowModel:
    """Model class that uses Roboflow API for inference"""
    def __init__(self, project_id: str, model_version: int):
        if not ROBOFLOW_AVAILABLE:
            raise RuntimeError("roboflow not installed. pip install roboflow")

        self.project_id = project_id
        self.model_version = model_version
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Roboflow model"""
        try:
            rf = Roboflow(api_key=constants.ROBOFLOW_API_KEY)
            project = rf.workspace().project(self.project_id)
            self.model = project.version(self.model_version).model
            print(f"✓ Initialized Roboflow model: {self.project_id} v{self.model_version}")
        except Exception as e:
            print(f"Failed to initialize Roboflow model {self.project_id}: {e}")
            raise

    def predict(self, file_path: str, confidence: float = 50, overlap: float = 45):
        """Run prediction using Roboflow API"""
        if self.model is None:
            return _PredictionResult([], src_path=file_path)

        try:
            # Roboflow API expects confidence as percentage (0-100)
            result = self.model.predict(file_path, confidence=int(confidence), overlap=int(overlap))

            # Convert Roboflow result format to our standard format
            predictions = result.json().get('predictions', [])

            return _PredictionResult(predictions, src_path=file_path)
        except Exception as e:
            print(f"Prediction error: {e}")
            return _PredictionResult([], src_path=file_path)


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

class Utils:
    def __init__(self):
        self._player_model = None
        self._dealer_model = None

    def initialize_player_model(self):
        """Initialize player card detection model (Roboflow API or local weights)"""
        if self._player_model is None:
            # Check if local weights exist
            weights_path = MODELS_DIR / "player_cards.pt"
            if weights_path.exists() and YOLO is not None:
                print("Using local YOLO model for players")
                self._player_model = LocalYoloModel(str(weights_path))
            elif ROBOFLOW_AVAILABLE:
                print("Using Roboflow API for players")
                self._player_model = RoboflowModel(
                    project_id=constants.PROJECT_ID_PLAYERS,
                    model_version=constants.MODEL_VERSION_PLAYERS
                )
            else:
                raise RuntimeError("No model available. Install either 'roboflow' or 'ultralytics' and provide model weights.")
        return self._player_model

    def initialize_dealer_model(self):
        """Initialize dealer card detection model (Roboflow API or local weights)"""
        if self._dealer_model is None:
            # Check if local weights exist
            weights_path = MODELS_DIR / "dealer_cards.pt"
            if not weights_path.exists():
                weights_path = MODELS_DIR / "player_cards.pt"

            if weights_path.exists() and YOLO is not None:
                print("Using local YOLO model for dealer")
                self._dealer_model = LocalYoloModel(str(weights_path))
            elif ROBOFLOW_AVAILABLE:
                print("Using Roboflow API for dealer")
                self._dealer_model = RoboflowModel(
                    project_id=constants.PROJECT_ID_DEALER,
                    model_version=constants.MODEL_VERSION_DEALER
                )
            else:
                raise RuntimeError("No model available. Install either 'roboflow' or 'ultralytics' and provide model weights.")
        return self._dealer_model

    def generate_card_image_path(self, card: str):
        card = card.replace(" ", "_")
        return os.path.join(constants.CARD_FOLDER_PATH, f"{card}.png")
