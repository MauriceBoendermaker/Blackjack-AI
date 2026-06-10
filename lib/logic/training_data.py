"""Active-learning training-data capture (V2 Feature 6).

Every detection event worth learning from gets saved as a labeled sample
under output/training_data/:

  * correction — the user fixed a misread via the card picker: GOLD data,
    the model's proven mistakes with the right answer attached.
  * confirmed  — every Nth multi-frame-confirmed lock: pre-annotated
    positives that keep the dataset balanced.
  * lowconf    — pending hit/dealer candidates that flapped (appeared, then
    failed confirmation): exactly the model's blind spots.

Each sample is a JPEG crop centered on the detection plus a YOLO-format
label (.txt, class index from classes.txt) and a .json sidecar with the
human-readable label and provenance — ready for review in Roboflow Annotate
(upload with `roboflow`'s project.single_upload) or local fine-tuning with
ultralytics, then `model.export(format="onnx")` into models/.
"""

import json
import time
from pathlib import Path

import cv2

from ..common import constants

#: Stable class ordering for YOLO label files (52 cards, suit-major like the
#: detector's a1..d13 scheme).
CLASS_NAMES = [f"{rank} of {suit}"
               for suit in ("Hearts", "Diamonds", "Spades", "Clubs")
               for rank in ("Ace", "2", "3", "4", "5", "6", "7", "8", "9",
                            "10", "Jack", "Queen", "King")]
_CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


class TrainingDataCollector:
    def __init__(self, root=None):
        self.root = Path(root) if root else constants.OUTPUT_DIR / "training_data"
        self._capped_logged = False
        self._count_cache = None

    def _file_count(self):
        if self._count_cache is None:
            self._count_cache = (sum(1 for _ in self.root.glob("*.jpg"))
                                 if self.root.exists() else 0)
        return self._count_cache

    def save_sample(self, frame_bgr, cx, cy, label, kind, box=170) -> bool:
        """Crop `box` px around (cx, cy) and store image + YOLO label + meta.
        Rank-only labels (dealer model) are saved with metadata only — they
        still need a suit before they can train the 52-class model."""
        cfg = constants.TRAINING
        if not cfg.get("enabled"):
            return False
        if self._file_count() >= cfg.get("max_files", 5000):
            if not self._capped_logged:
                self._capped_logged = True
                print(f"Training-data cap reached ({cfg.get('max_files')}); "
                      "not saving more samples.")
            return False
        h, w = frame_bgr.shape[:2]
        half = box // 2
        left, top = int(max(0, cx - half)), int(max(0, cy - half))
        right, bottom = int(min(w, cx + half)), int(min(h, cy + half))
        if right - left < 40 or bottom - top < 40:
            return False
        crop = frame_bgr[top:bottom, left:right]

        self.root.mkdir(parents=True, exist_ok=True)
        classes_txt = self.root / "classes.txt"
        if not classes_txt.exists():
            classes_txt.write_text("\n".join(CLASS_NAMES), encoding="utf-8")

        stem = f"{kind}_{int(time.time() * 1000)}_{label.replace(' ', '-')}"
        if not cv2.imwrite(str(self.root / f"{stem}.jpg"), crop):
            return False

        ch, cw = crop.shape[:2]
        class_idx = _CLASS_INDEX.get(label)
        if class_idx is not None:
            # The card fills roughly the center of the crop; reviewers refine.
            rel_cx = (cx - left) / cw
            rel_cy = (cy - top) / ch
            (self.root / f"{stem}.txt").write_text(
                f"{class_idx} {rel_cx:.4f} {rel_cy:.4f} 0.70 0.85\n",
                encoding="utf-8")
        (self.root / f"{stem}.json").write_text(json.dumps({
            "label": label, "kind": kind, "ts": time.time(),
            "frame_xyxy": [left, top, right, bottom],
            "center": [float(cx), float(cy)],
        }), encoding="utf-8")
        self._count_cache = self._file_count() + 1 if self._count_cache is None \
            else self._count_cache + 1
        return True

    def stats(self) -> dict:
        out = {"correction": 0, "confirmed": 0, "lowconf": 0, "total": 0}
        if not self.root.exists():
            return out
        for f in self.root.glob("*.jpg"):
            kind = f.name.split("_", 1)[0]
            if kind in out:
                out[kind] += 1
            out["total"] += 1
        return out


def upload_batch(project_id=None, batch_name=None, root=None) -> int:
    """Push collected samples to Roboflow for review/annotation; returns the
    number uploaded. Requires network + the roboflow package (already a
    dependency). Call manually, e.g.:
        .venv\\Scripts\\python -c "from lib.logic.training_data import upload_batch; upload_batch()"
    """
    from roboflow import Roboflow
    root = Path(root) if root else constants.OUTPUT_DIR / "training_data"
    if not root.exists():
        return 0
    rf = Roboflow(api_key=constants.ROBOFLOW_API_KEY)
    project = rf.workspace().project(project_id or constants.PROJECT_ID_PLAYERS)
    batch = batch_name or f"active-learning-{time.strftime('%Y%m%d')}"
    uploaded = 0
    for img in sorted(root.glob("*.jpg")):
        label = img.with_suffix(".txt")
        try:
            project.single_upload(
                image_path=str(img),
                annotation_path=str(label) if label.exists() else None,
                batch_name=batch)
            uploaded += 1
        except Exception as e:
            print(f"Upload failed for {img.name}: {e}")
    print(f"Uploaded {uploaded} samples to batch '{batch}'.")
    return uploaded
