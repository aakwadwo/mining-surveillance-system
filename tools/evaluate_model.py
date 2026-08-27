#!/usr/bin/env python3
"""
Evaluate one model on one dataset split and emit metrics as JSON.

    ./.venv/bin/python tools/evaluate_model.py \
        --model backend/models/galamsey-run6-baseline.pt \
        --data dataset-phase5/data.yaml --split test --impl vendored

------------------------------------------------------------------
WHY THIS IS A SEPARATE, SINGLE-MODEL SCRIPT
------------------------------------------------------------------
Run 6 was built by the vendored fork (Ultralytics 8.3.63, split qk/v
attention) and the Phase 5 model by pip Ultralytics (fused qkv). The two
packages cannot be imported into the same Python process - whichever
lands on sys.path first wins, and the other model then crashes at its
first forward pass.

So each evaluation runs in its OWN process with `--impl` choosing the
implementation, and the comparison is done by a caller that reads the
JSON from both runs. Trying to be clever and do it in one process is
exactly how you end up silently evaluating the wrong model.

------------------------------------------------------------------
A NOTE ON FAIR COMPARISON
------------------------------------------------------------------
Both models MUST be evaluated on the same split of the same dataset,
with the same confidence/IoU settings. Comparing a number from one
model's own training log against another model's fresh evaluation is
not a comparison - the val sets differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--impl", default="pip", choices=["pip", "vendored"])
    ap.add_argument("--conf", type=float, default=0.001)   # standard for mAP
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.impl == "vendored":
        # COMPATIBILITY SHIM, applied in OUR code - the vendored fork's
        # source is never modified.
        #
        # Ultralytics 8.3.63 computes average precision with np.trapz,
        # which NumPy 2.0 removed (renamed to np.trapezoid). Inference
        # does not touch it, which is why prediction works and only
        # val() explodes. Restoring the old name is a faithful alias:
        # np.trapezoid IS the same trapezoidal integration.
        import numpy as _np

        if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
            _np.trapz = _np.trapezoid
        sys.path.insert(0, str(PROJECT_ROOT / "yolov12"))

    import torch
    import ultralytics
    from ultralytics import YOLO

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLO(args.model)

    results = model.val(
        data=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        verbose=False,
        plots=False,
        workers=0,
    )

    box = results.box
    names = dict(model.names)

    per_class = {}
    # results.box.ap_class_index lists which class ids actually had labels
    for i, cid in enumerate(list(box.ap_class_index)):
        cid = int(cid)
        per_class[names.get(cid, str(cid))] = {
            "class_id": cid,
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
            "mAP50": round(float(box.ap50[i]), 4),
            "mAP50-95": round(float(box.ap[i]), 4),
        }

    payload = {
        "model": args.model,
        "model_names": names,
        "num_classes": len(names),
        "data": args.data,
        "split": args.split,
        "implementation": args.impl,
        "ultralytics": ultralytics.__version__,
        "ultralytics_path": str(Path(ultralytics.__file__).parent),
        "device": device,
        "conf": args.conf,
        "iou": args.iou,
        "overall": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50-95": round(float(box.map), 4),
        },
        "per_class": per_class,
        "speed_ms": {k: round(float(v), 2) for k, v in results.speed.items()},
    }

    text = json.dumps(payload, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text)
        print(f"wrote {args.json_out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
