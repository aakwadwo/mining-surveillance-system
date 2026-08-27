#!/usr/bin/env python3
"""
Galamsey AI Detector — backend HTTP API
=======================================

Start it with:

    ./.venv/bin/python backend/app.py

Then:  http://127.0.0.1:5004/health

------------------------------------------------------------------
WHAT THIS FILE IS RESPONSIBLE FOR
------------------------------------------------------------------
HTTP only. It owns routes, request parsing and responses.

It does NOT own:
    paths / ports / modes  -> config.py
    class names            -> taxonomy.py
    model loading, inference, drawing -> detector.py

That split is deliberate. The inherited version did all four in one
448-line file, which is why the class list, the port and the model path
each ended up defined in two or three places that disagreed.

------------------------------------------------------------------
A NOTE ON REST
------------------------------------------------------------------
A REST API is just: a URL names a THING, and the HTTP verb says what
you want to do with it.

    GET  /statistics   -> "give me the statistics"      (read, safe)
    POST /api/detections -> "here is a new detection"   (write, changes state)

GET must never change anything, so it is safe to retry, cache or
refresh. POST may change things, which is why browsers warn before
re-submitting one. Responses are JSON, and the HTTP status code carries
the outcome: 200 ok, 400 your request was malformed, 404 no such thing,
503 the server cannot do this right now.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# backend/ must be importable whether launched from the project root or
# from inside backend/ (see config.py for why this matters).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from flask import Flask, Response, jsonify, request  # noqa: E402
from flask_cors import CORS  # noqa: E402

import config  # noqa: E402
import taxonomy  # noqa: E402
from detector import Detector  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("galamsey")


# ---------------------------------------------------------------
# Application state
# ---------------------------------------------------------------
config.ensure_directories()

_config_problems = config.validate()
if _config_problems:
    for problem in _config_problems:
        logger.error("CONFIG: %s", problem)

detector = Detector()

# Recent detections, newest last. A bounded deque so a long-running
# server cannot exhaust memory.
detection_history: deque[dict[str, Any]] = deque(maxlen=config.DETECTION_HISTORY_LIMIT)
_history_lock = threading.Lock()


# ---------------------------------------------------------------
# Camera — optional, lazy, never required
# ---------------------------------------------------------------
class CameraManager:
    """
    Opens the webcam only when something actually asks for video.

    The inherited code grabbed the camera at import time, so on a
    machine with no camera (or with the permission denied) the server
    printed errors before it had served a single request, and /health
    was unreachable. A camera is a peripheral, not a prerequisite.
    """

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self.last_error: str | None = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> bool:
        if not config.CAMERA_ENABLED:
            self.last_error = "camera disabled by configuration (GALAMSEY_CAMERA=0)"
            return False
        with self._lock:
            if self.is_open:
                return True
            for index in (config.CAMERA_INDEX, 0, 1, 2):
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
                        self._capture = cap
                        self.last_error = None
                        logger.info("camera opened on index %s", index)
                        return True
                cap.release()
            self.last_error = (
                "no usable camera found. On macOS check "
                "System Settings > Privacy & Security > Camera."
            )
            logger.warning(self.last_error)
            return False

    def read(self) -> np.ndarray | None:
        if not self.is_open:
            return None
        ok, frame = self._capture.read()
        return frame if ok else None

    def release(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": config.CAMERA_ENABLED,
            "open": self.is_open,
            "index": config.CAMERA_INDEX,
            "last_error": self.last_error,
        }


camera = CameraManager()


# ---------------------------------------------------------------
# Flask
# ---------------------------------------------------------------
app = Flask(__name__)

# CORS = Cross-Origin Resource Sharing. A browser refuses to let a page
# served from origin A read a response from origin B unless B says it is
# allowed. Our frontend is http://localhost:5501 and this API is
# http://127.0.0.1:5004 — different ports mean different origins, so the
# browser needs that permission. The inherited code used CORS(app),
# allowing EVERY site on the internet to call this API from a visitor's
# browser. We allow only the frontend's own origins.
CORS(app, origins=config.CORS_ORIGINS)


def _record(detections: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "synthetic": detector.mode != "model",
        "detections": detections,
        "metadata": meta or {},
    }
    with _history_lock:
        detection_history.append(entry)


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.get("/")
def index():
    """API index. Lists what this server can do."""
    return jsonify(
        {
            "service": "Galamsey AI Detector backend",
            "version": "2.0.0",
            "status": "running",
            "detection_mode": detector.mode,
            "synthetic": detector.mode != "model",
            "endpoints": {
                "GET /health": "server, model and camera status",
                "GET /statistics": "detection counts for this session",
                "GET /detections": "recent detection history",
                "POST /api/detections": "submit a detection record",
                "GET /client-config": "browser-safe config (Maps key)",
                "GET /config": "current configuration",
                "POST /config": "update the confidence threshold",
                "GET /video_feed": "MJPEG stream with detections (needs a camera)",
                "POST /detect": "run detection on one uploaded image",
            },
        }
    )


@app.get("/client-config")
def client_config():
    """
    Browser-safe configuration, including the Google Maps key.

    The key lives in the git-ignored .env and is read from the
    environment - it is never present in a committed file. Serving it
    here does not make it secret (a browser key never is), but it does
    keep it out of version control and puts every credential in ONE
    place instead of scattered through HTML.
    """
    return jsonify(config.client_config())


@app.get("/health")
def health():
    """
    Health check. The first thing to call when something seems wrong.

    Returns 200 when the server can do its job, 503 when it cannot.
    A monitoring system reads the status code; a human reads the body.
    """
    det = detector.status()
    healthy = detector.mode in ("model", "mock")

    body = {
        "status": "healthy" if healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detection": det,
        "camera": camera.status(),
        "taxonomy": {
            "num_classes": taxonomy.NUM_CLASSES,
            "classes": {
                cid: {
                    "dataset_name": taxonomy.dataset_name(cid),
                    "display_name": taxonomy.display_name(cid),
                }
                for cid in range(taxonomy.NUM_CLASSES)
            },
        },
        "config_problems": _config_problems,
        "history_size": len(detection_history),
    }
    return jsonify(body), (200 if healthy else 503)


@app.get("/statistics")
def statistics():
    return jsonify(detector.statistics())


@app.get("/detections")
def get_detections():
    """
    Recent detections, newest last.

    ?limit=N caps how many are returned.
    """
    try:
        limit = int(request.args.get("limit", config.DETECTION_HISTORY_LIMIT))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    with _history_lock:
        items = list(detection_history)[-max(limit, 0):]

    return jsonify(
        {
            "count": len(items),
            "synthetic": detector.mode != "model",
            "warning": None if detector.mode == "model" else
                       "History may contain synthetic (mock) detections; check each entry.",
            "detections": items,
        }
    )


@app.post("/api/detections")
def post_detections():
    """
    Accept a detection record from the frontend.

    The frontend was written against frontend/DETECTION_API.md, which
    specifies POST /api/detections. That endpoint never existed, so every
    submission failed and piled up in the browser's retry queue. This
    implements the contract the frontend already speaks.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"success": False, "error": "body must be a JSON object"}), 400

    detections = payload.get("detections")
    if not isinstance(detections, list):
        return jsonify({"success": False, "error": "'detections' must be a list"}), 400

    location = payload.get("location") or {}
    if location and not all(isinstance(location.get(k), (int, float)) for k in ("lat", "lng")):
        return jsonify({"success": False, "error": "location.lat and location.lng must be numbers"}), 400

    _record(detections, {"location": location, "source": "frontend",
                         **(payload.get("metadata") or {})})

    return jsonify(
        {
            "success": True,
            "message": "detection recorded",
            "received": len(detections),
            "history_size": len(detection_history),
        }
    ), 201


@app.post("/detect")
def detect_image():
    """
    Run the detector on a single uploaded image.

    Why this endpoint exists: without it, the ONLY way to exercise the
    model is the webcam, so the detection pipeline cannot be tested on a
    machine with no camera, in CI, or from a script. It is also the
    simplest way to demonstrate the system in a viva.

        curl -F image=@photo.jpg http://127.0.0.1:5004/detect
    """
    if "image" not in request.files:
        return jsonify({"error": "no file part named 'image'"}), 400

    upload = request.files["image"]
    raw = np.frombuffer(upload.read(), dtype=np.uint8)
    if raw.size == 0:
        return jsonify({"error": "empty upload"}), 400

    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    results = [d.to_dict() for d in detector.detect(frame)]
    _record(results, {"source": "upload", "filename": upload.filename})

    return jsonify(
        {
            "synthetic": detector.mode != "model",
            "warning": None if detector.mode == "model" else
                       "SYNTHETIC placeholder data - the model is not running",
            "mode": detector.mode,
            "image": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
            "inference_ms": detector.last_inference_ms,
            "count": len(results),
            "detections": results,
        }
    )


@app.route("/config", methods=["GET", "POST"])
def configuration():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        threshold = payload.get("threshold", payload.get("confidence_threshold"))
        if threshold is None:
            return jsonify({"error": "expected 'threshold'"}), 400
        try:
            value = float(threshold)
        except (TypeError, ValueError):
            return jsonify({"error": "threshold must be a number"}), 400
        if not 0.0 < value <= 1.0:
            return jsonify({"error": "threshold must be in (0, 1]"}), 400

        config.CONFIDENCE_THRESHOLD = value
        logger.info("confidence threshold set to %.2f", value)
        return jsonify({"success": True, "confidence_threshold": value})

    return jsonify({**config.as_dict(), "detection": detector.status()})


def _frames():
    """Generator yielding MJPEG parts."""
    while True:
        frame = camera.read()
        if frame is None:
            break
        detections = detector.detect(frame)
        frame = detector.draw(frame, detections)
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.get("/video_feed")
def video_feed():
    """
    Live MJPEG stream with detections drawn on.

    Returns 503 (not 500) when there is no camera: the server is fine,
    it just cannot provide this particular resource right now.
    """
    if not camera.open():
        return jsonify({"error": "camera unavailable", "detail": camera.last_error}), 503
    return Response(_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "no such endpoint", "hint": "GET / lists all endpoints"}), 404


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
def main() -> int:
    status = detector.status()
    logger.info("=" * 62)
    logger.info("Galamsey AI Detector backend")
    logger.info("  detection mode : %s%s", status["mode"],
                "   *** SYNTHETIC DATA ***" if status["synthetic"] else "")
    logger.info("  model          : %s", config.MODEL_PATH.name if status["model_loaded"] else "not loaded")
    logger.info("  device         : %s", status["device"])
    logger.info("  ultralytics    : %s (%s)", status["ultralytics_version"], config.YOLO_IMPLEMENTATION)
    logger.info("  listening on   : http://%s:%s", config.HOST, config.PORT)
    logger.info("  CORS allows    : %s", ", ".join(config.CORS_ORIGINS))
    logger.info("=" * 62)

    try:
        # debug=False by default: the Werkzeug debugger is a remote code
        # execution console, and its reloader would construct the
        # Detector twice (loading the model twice and fighting over the
        # camera). Set GALAMSEY_DEBUG=1 only on a trusted machine.
        app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        camera.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
