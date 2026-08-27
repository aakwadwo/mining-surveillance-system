"""
Galamsey AI Detector — configuration
====================================

THE SINGLE SOURCE OF TRUTH FOR PATHS, PORTS AND MODES.

Nothing else may hardcode a path, a port, or a detection mode.

------------------------------------------------------------------
WHY EVERY PATH IS ANCHORED TO __file__
------------------------------------------------------------------
The inherited code used relative paths:

    model_path   = "yolov12"
    dataset_path = "illegal mining activities"

A relative path is resolved against the CURRENT WORKING DIRECTORY —
whatever folder your shell happened to be in. Run `python app.py` from
backend/ and "yolov12" means "backend/yolov12", which does not exist.
The model was therefore never found, and the server silently fell back
to fake detections. Nothing errored; it just quietly did the wrong thing.

    Path("yolov12")                  -> depends on the shell. Unpredictable.
    Path(__file__).resolve().parent  -> depends on the FILE. Deterministic.

__file__ is the location of this source file. Anchoring to it means the
server behaves identically whether it is started from the project root,
from backend/, from an IDE, or by a service manager.

------------------------------------------------------------------
OVERRIDING
------------------------------------------------------------------
Every value below can be overridden with an environment variable, so
deployment never requires editing source:

    GALAMSEY_PORT=5005 ./.venv/bin/python backend/app.py
    GALAMSEY_DETECTION_MODE=mock ./.venv/bin/python backend/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Load secrets from the git-ignored .env at the project root, so that
# neither the backend nor the frontend ever carries a credential in a
# committed file. See .env.example for the template.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # python-dotenv is optional; env vars still work
    pass

# ---------------------------------------------------------------
# Paths — all derived from this file's own location
# ---------------------------------------------------------------
BACKEND_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BACKEND_DIR.parent

FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"
DATASET_DIR: Path = PROJECT_ROOT / "illegal mining activities"
DATASET_YAML: Path = DATASET_DIR / "data.yaml"
YOLOV12_DIR: Path = PROJECT_ROOT / "yolov12"        # vendored upstream fork

MODELS_DIR: Path = BACKEND_DIR / "models"
DATA_DIR: Path = BACKEND_DIR / "data"               # detection history
LOGS_DIR: Path = BACKEND_DIR / "logs"

# The production model. See backend/models/PROVENANCE.md.
#
# Phase 5: trained with pip Ultralytics on dataset-phase5 (536
# photographs). Test-split mAP50 0.503 vs the Run 6 baseline's 0.200.
# Because it was built by the pip package rather than the vendored fork,
# it needs no sys.path tricks - hence YOLO_IMPLEMENTATION defaults to
# "pip" below.
#
# To run the historical baseline instead:
#   GALAMSEY_MODEL=backend/models/galamsey-run6-baseline.pt \
#   GALAMSEY_YOLO_IMPL=vendored ./.venv/bin/python backend/app.py
DEFAULT_MODEL_PATH: Path = MODELS_DIR / "galamsey-phase5.pt"
MODEL_PATH: Path = Path(os.getenv("GALAMSEY_MODEL", DEFAULT_MODEL_PATH))


# ---------------------------------------------------------------
# Detection mode  —  see backend/detector.py
# ---------------------------------------------------------------
#   "model" : run the real neural network. Fails loudly if unavailable.
#   "mock"  : return clearly-labelled synthetic detections for UI work.
#   "auto"  : use "model" if the weights load, otherwise fall back to
#             "mock" — but say so, loudly, in every response.
#
# Mock output is ALWAYS tagged so it can never be mistaken for real AI.
DETECTION_MODE: str = os.getenv("GALAMSEY_DETECTION_MODE", "auto").lower()
VALID_DETECTION_MODES = ("model", "mock", "auto")

# Which Ultralytics implementation to use.
#   "pip"      : the installed ultralytics package. DEFAULT since Phase 5.
#   "vendored" : yolov12/ in this repo (Ultralytics 8.3.63 fork).
#                Required ONLY for the historical Run 6 / Run 8 weights,
#                whose AAttn block uses split qk/v projections instead of
#                upstream's fused qkv.
# The Phase 5 model was trained with the pip package precisely to remove
# this constraint. See PROVENANCE.md.
YOLO_IMPLEMENTATION: str = os.getenv("GALAMSEY_YOLO_IMPL", "pip").lower()


# ---------------------------------------------------------------
# Inference
# ---------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = float(os.getenv("GALAMSEY_CONF", "0.25"))
IOU_THRESHOLD: float = float(os.getenv("GALAMSEY_IOU", "0.45"))
INFERENCE_IMGSZ: int = int(os.getenv("GALAMSEY_IMGSZ", "640"))

# "mps" = Apple GPU, "cpu", "cuda", or "auto" to pick the best available.
DEVICE: str = os.getenv("GALAMSEY_DEVICE", "auto").lower()


# ---------------------------------------------------------------
# Server
# ---------------------------------------------------------------
# 127.0.0.1 by default, NOT 0.0.0.0. The inherited code bound to
# 0.0.0.0 with debug=True, which exposes the Werkzeug debugger — an
# arbitrary-code-execution console — to every machine on the network.
HOST: str = os.getenv("GALAMSEY_HOST", "127.0.0.1")

# 5004 is the agreed development port: app.py already used it, and
# macOS AirPlay Receiver occupies 5000.
PORT: int = int(os.getenv("GALAMSEY_PORT", "5004"))

DEBUG: bool = os.getenv("GALAMSEY_DEBUG", "0") == "1"

# Origins allowed to call this API from a browser. The frontend is
# served on 5501 (see tools/serve_frontend.sh).
CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.getenv(
        "GALAMSEY_CORS_ORIGINS",
        "http://localhost:5501,http://127.0.0.1:5501",
    ).split(",")
    if o.strip()
]


# ---------------------------------------------------------------
# Camera  —  optional, never a prerequisite
# ---------------------------------------------------------------
# The backend must start and pass health checks with no camera at all.
CAMERA_ENABLED: bool = os.getenv("GALAMSEY_CAMERA", "1") == "1"
CAMERA_INDEX: int = int(os.getenv("GALAMSEY_CAMERA_INDEX", "0"))
CAMERA_WIDTH: int = int(os.getenv("GALAMSEY_CAMERA_WIDTH", "640"))
CAMERA_HEIGHT: int = int(os.getenv("GALAMSEY_CAMERA_HEIGHT", "480"))
CAMERA_FPS: int = int(os.getenv("GALAMSEY_CAMERA_FPS", "15"))

# How many recent detections /detections keeps in memory.
DETECTION_HISTORY_LIMIT: int = int(os.getenv("GALAMSEY_HISTORY", "100"))


# ---------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------
# Read from the environment (populated from .env). NEVER hardcode.
GOOGLE_MAPS_API_KEY: str = os.getenv("GOOGLE_MAPS_API_KEY", "")


def client_config() -> dict:
    """
    The subset of configuration the BROWSER is allowed to receive.

    Be deliberate about what goes in here: everything returned is public
    to anyone who can reach the server. A Google Maps browser key is
    already public by nature (it is embedded in every page that uses it),
    so serving it costs nothing that was not already spent - but it must
    never sit in a committed file, and no other secret may join it.
    """
    return {
        "googleMapsApiKey": GOOGLE_MAPS_API_KEY,
        "hasGoogleMapsKey": bool(GOOGLE_MAPS_API_KEY),
        "backendUrl": f"http://{HOST}:{PORT}",
    }


def ensure_directories() -> None:
    """Create the runtime directories if they are missing."""
    for d in (MODELS_DIR, DATA_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def validate() -> list[str]:
    """Return a list of configuration problems; empty means all good."""
    problems: list[str] = []

    if DETECTION_MODE not in VALID_DETECTION_MODES:
        problems.append(
            f"GALAMSEY_DETECTION_MODE={DETECTION_MODE!r} is not one of "
            f"{VALID_DETECTION_MODES}"
        )
    if YOLO_IMPLEMENTATION not in ("vendored", "pip"):
        problems.append(
            f"GALAMSEY_YOLO_IMPL={YOLO_IMPLEMENTATION!r} must be 'vendored' or 'pip'"
        )
    if not 0.0 < CONFIDENCE_THRESHOLD <= 1.0:
        problems.append(f"GALAMSEY_CONF={CONFIDENCE_THRESHOLD} must be in (0, 1]")
    if not DATASET_YAML.exists():
        problems.append(f"dataset yaml not found: {DATASET_YAML}")
    if YOLO_IMPLEMENTATION == "vendored" and not (YOLOV12_DIR / "ultralytics").is_dir():
        problems.append(f"vendored ultralytics not found in {YOLOV12_DIR}")

    return problems


def as_dict() -> dict:
    """Serialisable snapshot, used by the /config endpoint and tests."""
    return {
        "project_root": str(PROJECT_ROOT),
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "dataset_yaml": str(DATASET_YAML),
        "detection_mode": DETECTION_MODE,
        "yolo_implementation": YOLO_IMPLEMENTATION,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "imgsz": INFERENCE_IMGSZ,
        "device": DEVICE,
        "host": HOST,
        "port": PORT,
        "debug": DEBUG,
        "cors_origins": CORS_ORIGINS,
        "camera_enabled": CAMERA_ENABLED,
        "camera_index": CAMERA_INDEX,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(as_dict(), indent=2))
    issues = validate()
    print("\nvalidate():", "OK" if not issues else issues)
