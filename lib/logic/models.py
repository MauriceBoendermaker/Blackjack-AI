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
    hosted = False

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


def letterbox(image_bgr: np.ndarray, size: int = 640):
    """Resize keeping aspect, pad to size x size (YOLO preprocessing).
    Returns (padded, ratio, (pad_x, pad_y))."""
    h, w = image_bgr.shape[:2]
    ratio = min(size / w, h and size / h or 1)
    new_w, new_h = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) / 2.0, (size - new_h) / 2.0
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, ratio, (left, top)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float):
    """Greedy non-maximum suppression; returns kept indices."""
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou = inter / np.maximum(1e-9, areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return keep


class OnnxModel:
    """YOLO weights exported to ONNX (ultralytics: model.export(format='onnx')),
    run with onnxruntime — up to ~3x faster on CPU than the torch path and no
    multi-GB torch dependency on the inference machine."""

    name = "Local ONNX"
    hosted = False

    def __init__(self, model_path):
        try:
            import onnxruntime
        except ImportError as e:
            raise ModelError("onnxruntime is not installed "
                             "(pip install onnxruntime)") from e
        try:
            self._session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"])
        except Exception as e:
            raise ModelError(f"Failed to load ONNX model {model_path}: {e}") from e
        self._input = self._session.get_inputs()[0]
        shape = self._input.shape  # (1, 3, H, W) — H/W may be symbolic
        self._size = int(shape[2]) if isinstance(shape[2], int) else 640
        # ultralytics embeds the class-name dict in the model metadata.
        self._names = {}
        meta = self._session.get_modelmeta().custom_metadata_map
        if "names" in meta:
            try:
                import ast
                self._names = {int(k): str(v) for k, v in
                               ast.literal_eval(meta["names"]).items()}
            except (ValueError, SyntaxError):
                pass

    def predict(self, image_bgr: np.ndarray, confidence=50, overlap=45):
        conf = max(0.0, min(1.0, confidence / 100.0))
        iou = max(0.05, min(1.0, overlap / 100.0))
        padded, ratio, (pad_x, pad_y) = letterbox(image_bgr, self._size)
        blob = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        try:
            out = self._session.run(None, {self._input.name: blob})[0]
        except Exception as e:
            raise ModelError(f"ONNX inference failed: {e}") from e
        return self._decode(out[0], conf, iou, ratio, pad_x, pad_y)

    def _decode(self, output, conf, iou, ratio, pad_x, pad_y):
        """ultralytics YOLOv8+ raw head: (4 + n_classes, n_anchors) with
        cx/cy/w/h in letterboxed pixels."""
        if output.shape[0] < output.shape[1]:
            preds = output  # (4+nc, n)
        else:
            preds = output.T
        boxes = preds[:4].T              # (n, cx cy w h)
        class_scores = preds[4:].T       # (n, nc)
        cls = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(cls)), cls]
        mask = scores >= conf
        boxes, scores, cls = boxes[mask], scores[mask], cls[mask]
        if not len(boxes):
            return []
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        keep = nms(xyxy, scores, iou)
        results = []
        for i in keep:
            cx = (boxes[i, 0] - pad_x) / ratio
            cy = (boxes[i, 1] - pad_y) / ratio
            results.append({
                "cx": float(cx), "cy": float(cy),
                "width": float(boxes[i, 2] / ratio),
                "height": float(boxes[i, 3] / ratio),
                "class": self._names.get(int(cls[i]), str(int(cls[i]))),
                "confidence": float(scores[i]),
            })
        return results


class RoboflowModel:
    """Hosted Roboflow inference. Downscales uploads and rescales results."""

    name = "Roboflow API"
    hosted = True  # remote session object — does not survive system sleep

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
        # Preference order: local ONNX (lightest/fastest CPU) -> local .pt
        # (ultralytics) -> hosted API.
        onnx_path = (constants.MODELS_DIR / weights_name).with_suffix(".onnx")
        if onnx_path.exists():
            try:
                return OnnxModel(onnx_path)
            except Exception as e:
                print(f"ONNX weights {onnx_path.name} unusable ({e}); trying .pt.")
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

    def has_hosted(self):
        """True if any initialized backend is a hosted (network) session."""
        with self._lock:
            return any(getattr(m, "hosted", False)
                       for m in (self._players, self._dealer))

    def refresh(self, log=None):
        """Rebuild hosted model sessions and re-validate local weight files.

        After a long idle (system sleep/restore) the hosted Roboflow session
        is stale; a local weights file can also be corrupted while the app
        idles. Hosted backends are rebuilt unconditionally; local backends
        are reloaded only after their file passes an integrity check (exists,
        > 1 KB, constructable) — a failing file is reported and the slot
        falls through to the next backend instead of crashing.

        Blocking work (network handshake, weight load) happens OUTSIDE the
        lock; each rebuilt model is swapped in atomically. `log` may accept
        the engine logger's level= keyword.
        """
        if log is None:
            log = lambda message, level=None: print(
                f"{level}: {message}" if level and level != "INFO" else message)
        for attr, weights_name, project_id, model_version in (
                ("_players", "player_cards.pt",
                 constants.PROJECT_ID_PLAYERS, constants.MODEL_VERSION_PLAYERS),
                ("_dealer", "dealer_cards.pt",
                 constants.PROJECT_ID_DEALER, constants.MODEL_VERSION_DEALER)):
            with self._lock:
                model = getattr(self, attr)
            if model is None:
                continue  # never built — the lazy getter handles first use
            fresh = self._refresh_one(model, weights_name, project_id,
                                      model_version, log)
            with self._lock:
                setattr(self, attr, fresh)

    def _refresh_one(self, model, weights_name, project_id, model_version, log):
        if getattr(model, "hosted", False):
            # Rebuild from scratch via _build: re-checks local weights first,
            # same preference order as startup.
            return self._build(weights_name, project_id, model_version)
        path = constants.MODELS_DIR / weights_name
        if isinstance(model, OnnxModel):
            path = path.with_suffix(".onnx")
        try:
            size = path.stat().st_size if path.exists() else -1
            if size <= 1024:  # a real weights file is megabytes, not bytes
                raise ModelError("file is missing" if size < 0
                                 else f"file is only {size} bytes")
            return type(model)(path)
        except Exception as e:
            log(f"Local weights {path.name} failed validation ({e}); "
                "falling back to the next backend.", level="ERROR")
            return self._build(weights_name, project_id, model_version)

    @property
    def backend_name(self):
        parts = []
        if self._players is not None:
            parts.append(self._players.name)
        if self._dealer is not None and (not parts or self._dealer.name != parts[0]):
            parts.append(self._dealer.name)
        return " + ".join(parts) if parts else "not loaded"
