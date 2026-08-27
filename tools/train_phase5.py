#!/usr/bin/env python3
"""
Phase 5 training run.

    ./.venv/bin/python tools/train_phase5.py                 # full run
    ./.venv/bin/python tools/train_phase5.py --epochs 2      # smoke test
    ./.venv/bin/python tools/train_phase5.py --name my-run

------------------------------------------------------------------
WHY THIS USES pip ULTRALYTICS, NOT THE VENDORED FORK
------------------------------------------------------------------
The Run 6 baseline was trained with the fork in yolov12/ (Ultralytics
8.3.63), whose AAttn block uses split qk/v projections. Upstream
Ultralytics uses a fused qkv, so fork-built weights load but crash at
the first forward pass under pip Ultralytics.

Training here with the PIP package produces weights that:
  * run under the maintained, installed Ultralytics
  * need no sys.path tricks at inference time
  * remove the vendored-fork constraint from the whole project

This script deliberately does NOT put yolov12/ on sys.path, and asserts
that the fork is not shadowing the pip package before it starts.

------------------------------------------------------------------
COMPARABILITY WITH RUN 6
------------------------------------------------------------------
Run 6: yolov12n (COCO-pretrained), 100 epochs, batch 16, imgsz 640,
       optimizer auto, lr0 0.01, seed 0.

Those are kept identical here so the only meaningful variable is THE
DATASET. Changing hyperparameters at the same time as the data would
make it impossible to attribute any difference to either.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import taxonomy  # noqa: E402

DATA_YAML = PROJECT_ROOT / "dataset-phase5" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "phase5"

# --- Hyperparameters: matched to Run 6 so the dataset is the variable ---
BASE_MODEL = "yolo12n.pt"     # pip Ultralytics' own COCO-pretrained nano
EPOCHS = 100
BATCH = 16
IMGSZ = 640
SEED = 0
OPTIMIZER = "auto"
LR0 = 0.01
PATIENCE = 100                # no early stopping, same as Run 6


def resolve_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--name", default="yolo12n-phase5")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    ul_path = Path(ultralytics.__file__).parent
    if "yolov12" in str(ul_path):
        print("ERROR: the vendored fork is shadowing pip ultralytics.")
        print(f"       ultralytics loaded from {ul_path}")
        return 1

    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found. Run tools/recover_dataset.py first.")
        return 1

    device = args.device or resolve_device()

    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "ultralytics_path": str(ul_path),
        "device": device,
        "base_model": BASE_MODEL,
        "architecture": "YOLOv12-nano",
        "data": str(DATA_YAML),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": SEED,
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "patience": PATIENCE,
        "classes": list(taxonomy.DATASET_NAMES),
    }

    print("=" * 70)
    print("PHASE 5 TRAINING")
    print("=" * 70)
    for k, v in env.items():
        print(f"  {k:18s}: {v}")
    print("=" * 70)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(BASE_MODEL)          # downloads COCO weights on first use
    started = time.time()
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=SEED,
        optimizer=OPTIMIZER,
        lr0=LR0,
        patience=PATIENCE,
        device=device,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        plots=True,
        val=True,
        workers=0,                    # avoids macOS fork/semaphore warnings
        deterministic=True,
    )
    duration = time.time() - started

    out = RUNS_DIR / args.name
    env["duration_seconds"] = round(duration, 1)
    env["duration_human"] = f"{duration/3600:.2f} h"
    env["output_dir"] = str(out)
    (out / "phase5_environment.json").write_text(json.dumps(env, indent=2))

    print(f"\nfinished in {duration/60:.1f} min")
    print(f"weights: {out / 'weights' / 'best.pt'}")
    print(f"env    : {out / 'phase5_environment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
