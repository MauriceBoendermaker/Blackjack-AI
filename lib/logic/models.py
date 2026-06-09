"""Card-detection backends.

Every backend returns predictions in ONE normalized format:
    {"cx": float, "cy": float, "width": float, "height": float,
     "class": str, "confidence": float}
with cx/cy the BOX CENTER in the coordinate space of the input image.
(The Roboflow REST API natively returns centers; ultralytics returns corners —
both are converted here so downstream code never has to care.)
"""

import threading

import cv2
import numpy as np

from ..common import constants


class ModelError(RuntimeError):
    """Raised when inference fails (network, auth, missing backend...)."""


class LocalYoloModel:
    """Local ultralytics YOLO weights (models/*.pt). Used when available."""

    name = "Local YOLO"

    def __init__(self, weights_path):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ModelError("ultralytics is not installed (pip install ultralytics)") from e
        self._model = YOLO(str(weights_path))
        try:
            import torch
            self._device = 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            self._device = "cpu"

    def predict(self, image_bgr: np.ndarray, confidence=50, overlap=45):
        conf = max(0.0, min(1.0, confidence / 100.0))
        iou = max(0.05, min(1.0, overlap / 100.0))
        results = self._model.predict(
            source=image_bgr, conf=conf, iou=iou, imgsz=640,
            verbose=False, device=self._device,
        )
        preds = []
        if not results:
            return preds
        r = results[0]
        names = getattr(r, "names", {})
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            preds.append({
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "width": x2 - x1,
                "height": y2 - y1,
                "class": str(names.get(int(b.cls.item()), int(b.cls.item()))),
                "confidence": float(b.conf.item()),
            })
        return preds


class RoboflowModel:
    """Hosted Roboflow inference. Downscales uploads and rescales results."""

    name = "Roboflow API"

    def __init__(self, project_id, model_version):
        try:
            from roboflow import Roboflow
        except ImportError as e:
            raise ModelError("roboflow is not installed (pip install roboflow)") from e
        try:
            rf = Roboflow(api_key=constants.ROBOFLOW_API_KEY)
            self._model = rf.workspace().project(project_id).version(model_version).model
        except Exception as e:
            raise ModelError(f"Failed to initialize Roboflow model {project_id}: {e}") from e

    def predict(self, image_bgr: np.ndarray, confidence=50, overlap=45):
        h, w = image_bgr.shape[:2]
        scale = 1.0
        upload = image_bgr
        if w > constants.API_UPLOAD_MAX_WIDTH:
            scale = constants.API_UPLOAD_MAX_WIDTH / w
            upload = cv2.resize(
                image_bgr,
                (constants.API_UPLOAD_MAX_WIDTH, max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        try:
            result = self._model.predict(upload, confidence=int(confidence), overlap=int(overlap))
            raw = result.json().get("predictions", [])
        except Exception as e:
            raise ModelError(f"Roboflow inference failed: {e}") from e

        inv = 1.0 / scale
        return [
            {
                # Roboflow x/y are already the box center.
                "cx": p["x"] * inv,
                "cy": p["y"] * inv,
                "width": p["width"] * inv,
                "height": p["height"] * inv,
                "class": str(p["class"]),
                "confidence": float(p.get("confidence", 0.0)),
            }
            for p in raw
        ]


class ModelProvider:
    """Process-wide lazy, thread-safe access to the two detection models.

    Nothing touches the network or loads weights until the first predict —
    the GUI starts instantly and model init happens on the worker thread.
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._players = None
        self._dealer = None

    @classmethod
    def get(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _build(self, weights_name, project_id, model_version):
        weights = constants.MODELS_DIR / weights_name
        if weights.exists():
            try:
                return LocalYoloModel(weights)
            except Exception as e:
                # Corrupt weights, torch/DLL trouble, version mismatch, ... —
                # report it and fall through to the hosted API.
                print(f"Local weights {weights.name} unusable ({e}); using the hosted API.")
        return RoboflowModel(project_id, model_version)

    def players_model(self):
        with self._lock:
            if self._players is None:
                self._players = self._build(
                    "player_cards.pt", constants.PROJECT_ID_PLAYERS, constants.MODEL_VERSION_PLAYERS)
            return self._players

    def dealer_model(self):
        with self._lock:
            if self._dealer is None:
                self._dealer = self._build(
                    "dealer_cards.pt", constants.PROJECT_ID_DEALER, constants.MODEL_VERSION_DEALER)
            return self._dealer

    @property
    def backend_name(self):
        parts = []
        if self._players is not None:
            parts.append(self._players.name)
        if self._dealer is not None and (not parts or self._dealer.name != parts[0]):
            parts.append(self._dealer.name)
        return " + ".join(parts) if parts else "not loaded"
