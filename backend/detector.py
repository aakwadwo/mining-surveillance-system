"""
Galamsey AI Detector — detection engine
=======================================

Separates two clearly distinct things:

    MODEL mode  real YOLOv12 inference on real pixels
    MOCK  mode  synthetic placeholder boxes for building the UI

------------------------------------------------------------------
WHY MOCK MODE EXISTS, AND WHY IT IS TAGGED SO AGGRESSIVELY
------------------------------------------------------------------
You need to build and demo a dashboard before the model is good, and
you need the UI to work on a machine with no camera and no weights.
Mock mode makes that possible.

But the inherited code did this badly. Its _generate_dummy_detections()
returned boxes that were STRUCTURALLY IDENTICAL to real ones:

    {"class_name": "excavator", "confidence": 0.87, "is_illegal": true}

Nothing in the response said "this is fake". A screenshot of that is
indistinguishable from a screenshot of working AI. In a final-year
project that is not a bug, it is an integrity problem.

So here, synthetic output is impossible to mistake for real output:

  * every Detection carries  source = "mock"
  * every API response carries  synthetic = true  and a warning string
  * mock class names are prefixed  "MOCK "
  * rendered frames get a red "MOCK DATA - NOT AI OUTPUT" banner
  * the log says so at startup, on every mode resolution

If you ever find yourself tempted to remove those tags to make a demo
look better, that is exactly the moment not to.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import config
import taxonomy

logger = logging.getLogger(__name__)

MOCK_PREFIX = "MOCK "
SYNTHETIC_WARNING = (
    "These detections are SYNTHETIC placeholder data, not AI output. "
    "The model is not running. Do not present them as results."
)


# ---------------------------------------------------------------
# Detection record
# ---------------------------------------------------------------
@dataclass
class Detection:
    """One detected object."""

    bbox: list[int]              # [x1, y1, x2, y2] in pixels
    confidence: float
    class_id: int
    class_name: str              # dataset/wire spelling
    display_name: str            # human-facing spelling
    source: str                  # "model" | "mock"  <- never omit this
    mining_activity_weight: float
    timestamp: str

    @property
    def is_synthetic(self) -> bool:
        return self.source == "mock"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_synthetic"] = self.is_synthetic
        return d


# ---------------------------------------------------------------
# Loading Ultralytics
# ---------------------------------------------------------------
def _import_yolo(implementation: str):
    """
    Import the YOLO class from the chosen Ultralytics implementation.

    This is the ONLY place sys.path is touched, and it logs exactly what
    it did. The inherited notebook silently picked up the vendored fork
    because it happened to run from inside yolov12/ — the kind of
    accident that makes "it works on my machine" bugs. Here the choice
    is explicit, configurable, and reported.
    """
    if implementation == "vendored":
        vendored = str(config.YOLOV12_DIR)
        if vendored not in sys.path:
            sys.path.insert(0, vendored)

    import ultralytics
    from ultralytics import YOLO

    where = Path(ultralytics.__file__).parent
    logger.info(
        "Ultralytics %s loaded from %s (implementation=%s)",
        ultralytics.__version__, where, implementation,
    )
    return YOLO, ultralytics.__version__, str(where)


# ---------------------------------------------------------------
# Detector
# ---------------------------------------------------------------
class Detector:
    """Owns the model and produces detections in whichever mode applies."""

    def __init__(self) -> None:
        self.requested_mode: str = config.DETECTION_MODE
        self.mode: str = "mock"          # resolved below
        self.model = None
        self.model_error: str | None = None
        self.ultralytics_version: str | None = None
        self.ultralytics_path: str | None = None
        self.device: str = self._resolve_device()
        self.taxonomy_problems: list[str] = []

        self.total_detections = 0
        self.detections_by_class: dict[int, int] = {c: 0 for c in range(taxonomy.NUM_CLASSES)}
        self.session_start = datetime.now(timezone.utc)
        self.last_inference_ms: float | None = None

        self._resolve_mode()

    # -- setup ---------------------------------------------------
    @staticmethod
    def _resolve_device() -> str:
        if config.DEVICE != "auto":
            return config.DEVICE
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # torch missing or broken
            pass
        return "cpu"

    def _resolve_mode(self) -> None:
        """Decide between model and mock, and say why."""
        requested = self.requested_mode

        if requested == "mock":
            self.mode = "mock"
            logger.warning("DETECTION MODE = MOCK (explicitly requested). %s", SYNTHETIC_WARNING)
            return

        loaded = self._load_model()

        if loaded:
            self.mode = "model"
            logger.info("DETECTION MODE = MODEL  device=%s  weights=%s",
                        self.device, config.MODEL_PATH.name)
            return

        if requested == "model":
            # Explicitly asked for the real thing; do not silently fake it.
            logger.error("DETECTION MODE = MODEL was requested but the model "
                         "could not be loaded: %s", self.model_error)
            self.mode = "unavailable"
            return

        self.mode = "mock"
        logger.warning(
            "DETECTION MODE = MOCK (auto fallback; model unavailable: %s). %s",
            self.model_error, SYNTHETIC_WARNING,
        )

    def _load_model(self) -> bool:
        path = config.MODEL_PATH
        if not path.exists():
            self.model_error = f"weights not found: {path}"
            return False

        try:
            YOLO, version, where = _import_yolo(config.YOLO_IMPLEMENTATION)
            self.ultralytics_version = version
            self.ultralytics_path = where

            model = YOLO(str(path))

            # Guard against the silent-mislabelling failure mode: if the
            # weights were trained on a different taxonomy, every label
            # this system prints would be wrong while nothing errors.
            problems = taxonomy.verify_against_model(dict(model.names))
            if problems:
                self.taxonomy_problems = problems
                self.model_error = (
                    "model class names do not match backend/taxonomy.py: "
                    + "; ".join(problems)
                )
                logger.error("REFUSING TO USE MODEL. %s", self.model_error)
                return False

            # Warm up once so the first real request is not slow, and so
            # an incompatible implementation fails now rather than later.
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            model.predict(dummy, device=self.device, verbose=False)

            self.model = model
            return True

        except Exception as exc:  # noqa: BLE001 - we report everything
            self.model_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Model load failed")
            return False

    # -- inference -----------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.mode == "model" and self.model is not None:
            return self._detect_real(frame)
        if self.mode == "unavailable":
            return []
        return self._detect_mock(frame)

    def _detect_real(self, frame: np.ndarray) -> list[Detection]:
        started = time.perf_counter()
        results = self.model.predict(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=config.INFERENCE_IMGSZ,
            device=self.device,
            verbose=False,
        )
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0

        now = datetime.now(timezone.utc).isoformat()
        out: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                cid = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())

                if not taxonomy.is_valid_class_id(cid):
                    logger.warning("model emitted out-of-range class id %s; skipping", cid)
                    continue

                out.append(
                    Detection(
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        confidence=round(conf, 4),
                        class_id=cid,
                        class_name=taxonomy.dataset_name(cid),
                        display_name=taxonomy.display_name(cid),
                        source="model",
                        mining_activity_weight=taxonomy.mining_activity_weight(cid),
                        timestamp=now,
                    )
                )

        self._record(out)
        return out

    def _detect_mock(self, frame: np.ndarray) -> list[Detection]:
        """
        Fixed, obviously-synthetic boxes. Deterministic on purpose: random
        mock data looks like a flickering model and invites false
        confidence. Two boxes, always the same place, always tagged.
        """
        h, w = frame.shape[:2]
        now = datetime.now(timezone.utc).isoformat()

        spec = [
            (0, 0.20, 0.30, 0.45, 0.65, 0.87),   # ExcavationMachinery
            (2, 0.55, 0.40, 0.72, 0.78, 0.92),   # Person
        ]
        out = [
            Detection(
                bbox=[int(w * x1), int(h * y1), int(w * x2), int(h * y2)],
                confidence=conf,
                class_id=cid,
                class_name=MOCK_PREFIX + taxonomy.dataset_name(cid),
                display_name=MOCK_PREFIX + taxonomy.display_name(cid),
                source="mock",
                mining_activity_weight=taxonomy.mining_activity_weight(cid),
                timestamp=now,
            )
            for cid, x1, y1, x2, y2, conf in spec
        ]
        self._record(out)
        return out

    def _record(self, detections: list[Detection]) -> None:
        for d in detections:
            self.total_detections += 1
            self.detections_by_class[d.class_id] = self.detections_by_class.get(d.class_id, 0) + 1

    # -- rendering -----------------------------------------------
    def draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw boxes. Mock frames get an unmissable banner."""
        import cv2

        synthetic = self.mode != "model"

        for d in detections:
            x1, y1, x2, y2 = d.bbox
            # Colour by evidence strength, not by a fabricated legality call.
            weight = d.mining_activity_weight
            colour = (0, 0, 255) if weight >= 0.8 else (0, 165, 255) if weight >= 0.4 else (0, 255, 0)
            if synthetic:
                colour = (255, 0, 255)  # magenta: never used for real output

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            label = f"{d.display_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(frame, label, (x1 + 2, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if synthetic:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.putText(frame, "MOCK DATA - NOT AI OUTPUT", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        return frame

    # -- reporting -----------------------------------------------
    def status(self) -> dict[str, Any]:
        uptime = (datetime.now(timezone.utc) - self.session_start).total_seconds()
        return {
            "mode": self.mode,
            "requested_mode": self.requested_mode,
            "synthetic": self.mode != "model",
            "warning": SYNTHETIC_WARNING if self.mode != "model" else None,
            "model_loaded": self.model is not None,
            "model_path": str(config.MODEL_PATH),
            "model_error": self.model_error,
            "taxonomy_problems": self.taxonomy_problems,
            "device": self.device,
            "ultralytics_version": self.ultralytics_version,
            "ultralytics_path": self.ultralytics_path,
            "yolo_implementation": config.YOLO_IMPLEMENTATION,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "last_inference_ms": self.last_inference_ms,
            "uptime_seconds": round(uptime, 1),
        }

    def statistics(self) -> dict[str, Any]:
        by_class = {
            taxonomy.dataset_name(cid): count
            for cid, count in sorted(self.detections_by_class.items())
        }
        uptime = (datetime.now(timezone.utc) - self.session_start).total_seconds()
        return {
            "synthetic": self.mode != "model",
            "warning": SYNTHETIC_WARNING if self.mode != "model" else None,
            "mode": self.mode,
            "total_detections": self.total_detections,
            "detections_by_class": by_class,
            "session_started": self.session_start.isoformat(),
            "uptime_seconds": round(uptime, 1),
            "detections_per_minute": round(self.total_detections / max(uptime / 60, 1e-6), 2),
        }
