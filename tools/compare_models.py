#!/usr/bin/env python3
"""
Compare the Run 6 baseline against the Phase 5 model, honestly.

    ./.venv/bin/python tools/compare_models.py

Evaluates BOTH models on the SAME split of the SAME dataset, each in its
own subprocess (they need different Ultralytics implementations - see
tools/evaluate_model.py), then prints a side-by-side table.

------------------------------------------------------------------
RULES THIS SCRIPT ENFORCES
------------------------------------------------------------------
1. Same data, same split, same conf/IoU for both models. Quoting a
   number from one model's old training log against another model's
   fresh evaluation is not a comparison.
2. Every class is shown, not just the ones that improved.
3. Regressions are marked as clearly as improvements.
4. mAP is reported alongside per-class recall, because a headline mAP
   can rise while the classes you actually care about get worse.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import taxonomy  # noqa: E402

PY = str(PROJECT_ROOT / ".venv" / "bin" / "python")
EVAL = str(PROJECT_ROOT / "tools" / "evaluate_model.py")
DATA = str(PROJECT_ROOT / "dataset-phase5" / "data.yaml")
SCRATCH = PROJECT_ROOT / "runs" / "phase5" / "_eval"

MODELS = [
    ("Run 6 baseline", "backend/models/galamsey-run6-baseline.pt", "vendored"),
    ("Phase 5",        "runs/phase5/yolo12n-phase5/weights/best.pt", "pip"),
]


def evaluate(label: str, model: str, impl: str, split: str) -> dict | None:
    path = PROJECT_ROOT / model
    if not path.exists():
        print(f"  !! {label}: weights not found at {model}")
        return None
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = SCRATCH / f"{label.replace(' ', '_').lower()}_{split}.json"
    print(f"  evaluating {label} ({impl}) on split={split} ...")
    proc = subprocess.run(
        [PY, EVAL, "--model", str(path), "--data", DATA, "--split", split,
         "--impl", impl, "--json-out", str(out)],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    if proc.returncode != 0:
        print(f"  !! {label} failed:\n{proc.stderr[-1500:]}")
        return None
    return json.loads(out.read_text())


def arrow(new: float, old: float, eps: float = 0.005) -> str:
    if new > old + eps:
        return "UP"
    if new < old - eps:
        return "DOWN"
    return "="


def main() -> int:
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    print("=" * 78)
    print(f"MODEL COMPARISON  -  dataset-phase5, split={split}")
    print("=" * 78)

    results = {}
    for label, model, impl in MODELS:
        r = evaluate(label, model, impl, split)
        if r:
            results[label] = r

    if len(results) < 2:
        print("\nCannot compare: at least one model failed to evaluate.")
        return 1

    base, new = results["Run 6 baseline"], results["Phase 5"]

    print(f"\n{'metric':<26}{'Run 6':>10}{'Phase 5':>11}{'change':>10}{'':>6}")
    print("-" * 78)
    for key in ("precision", "recall", "mAP50", "mAP50-95"):
        b, n = base["overall"][key], new["overall"][key]
        print(f"{key:<26}{b:>10.4f}{n:>11.4f}{n - b:>+10.4f}   {arrow(n, b)}")

    print(f"\n{'per-class RECALL':<26}{'Run 6':>10}{'Phase 5':>11}{'change':>10}")
    print("-" * 78)
    for cid in range(taxonomy.NUM_CLASSES):
        name = taxonomy.dataset_name(cid)
        legacy = {"ProcessingEquipment": "ProcessingEquiment"}.get(name)
        def get(res, field):
            pc = res["per_class"]
            entry = pc.get(name) or (pc.get(legacy) if legacy else None)
            return entry[field] if entry else None
        b, n = get(base, "recall"), get(new, "recall")
        if b is None and n is None:
            print(f"{name:<26}{'--- not present in this split ---':>31}")
            continue
        b = 0.0 if b is None else b
        n = 0.0 if n is None else n
        print(f"{name:<26}{b:>10.4f}{n:>11.4f}{n - b:>+10.4f}   {arrow(n, b)}")

    print(f"\n{'per-class mAP50':<26}{'Run 6':>10}{'Phase 5':>11}{'change':>10}")
    print("-" * 78)
    for cid in range(taxonomy.NUM_CLASSES):
        name = taxonomy.dataset_name(cid)
        legacy = {"ProcessingEquipment": "ProcessingEquiment"}.get(name)
        def get(res):
            pc = res["per_class"]
            entry = pc.get(name) or (pc.get(legacy) if legacy else None)
            return entry["mAP50"] if entry else None
        b, n = get(base), get(new)
        if b is None and n is None:
            continue
        b = 0.0 if b is None else b
        n = 0.0 if n is None else n
        print(f"{name:<26}{b:>10.4f}{n:>11.4f}{n - b:>+10.4f}   {arrow(n, b)}")

    print(f"\n{'inference speed (ms/img)':<26}"
          f"{base['speed_ms'].get('inference', 0):>10.2f}"
          f"{new['speed_ms'].get('inference', 0):>11.2f}")
    print(f"{'ultralytics':<26}{base['ultralytics']:>10}{new['ultralytics']:>11}")
    print(f"{'implementation':<26}{base['implementation']:>10}{new['implementation']:>11}")

    combined = SCRATCH / f"comparison_{split}.json"
    combined.write_text(json.dumps(results, indent=2))
    print(f"\nfull metrics: {combined.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
