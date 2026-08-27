#!/usr/bin/env python3
"""
Dataset validator for the Galamsey AI Detector.

Run it after ANY change to the dataset, and before any training run:

    ./.venv/bin/python tools/validate_dataset.py

Optionally validate a different candidate dataset before adopting it:

    ./.venv/bin/python tools/validate_dataset.py /path/to/other/dataset

Exit code 0 = dataset is sound. Non-zero = do not train on it.

------------------------------------------------------------------
WHAT IT CHECKS, AND WHY EACH CHECK EARNED ITS PLACE
------------------------------------------------------------------
Every check below exists because this project actually hit that
problem, or because we found it in a neighbouring dataset:

  1. data.yaml paths resolve          -> the inherited data.yaml used
                                         '../train/images' and pointed
                                         one directory too high.
  2. images <-> labels pair up        -> a missing .txt silently means
                                         "this image has no objects",
                                         which teaches the model to
                                         detect nothing.
  3. label rows have exactly 5 fields -> a neighbouring dataset mixed
                                         box rows (5 fields) with
                                         polygon rows (11 fields).
  4. class ids within 0..nc-1         -> that same dataset contained
                                         class id 6 in a 6-class set.
  5. names match the taxonomy module  -> three different class lists
                                         existed in this codebase.
  6. coordinates normalised to 0..1   -> YOLO requires fractions, not
                                         pixels. Pixel values look
                                         plausible and train to garbage.
  7. width/height > 0                 -> zero-area boxes produce NaN loss.
  8. images actually decode           -> a truncated JPEG stops training
                                         hours in.
  9. NO IMAGE APPEARS IN TWO SPLITS   -> the single most damaging and
                                         least visible dataset bug. If a
                                         photo is in train and in val,
                                         the model has memorised the
                                         answer and your metrics are
                                         fiction. Compared by pixel
                                         content, not filename, because
                                         Roboflow gives the same photo a
                                         different .rf.<hash> per export.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

# The taxonomy module is the single source of truth for class names.
# It lives in backend/, so make that importable regardless of where
# this script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import taxonomy  # noqa: E402

DEFAULT_DATASET_DIR = PROJECT_ROOT / "illegal mining activities"
# Overridable so the validator can be pointed at any candidate dataset
# (e.g. to check a new export BEFORE adopting it).
DATASET_DIR = DEFAULT_DATASET_DIR
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


class Report:
    """Collects problems, separated by severity."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


def read_data_yaml(path: Path, rep: Report) -> dict:
    """Minimal YAML read. Avoids a hard dependency just for 12 lines."""
    import yaml

    if not path.exists():
        rep.error(f"data.yaml not found at {path}")
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def dhash(path: Path, size: int = 16) -> str | None:
    """
    Perceptual hash: a fingerprint of what the picture LOOKS like.

    MD5 answers "same bytes?". Two exports of the same photo at
    different resolutions have different bytes but are the same
    picture — and for split-leakage that is the question that matters.
    """
    from PIL import Image

    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return None
    px = list(img.tobytes())  # row-major grayscale bytes
    bits = []
    for row in range(size):
        start = row * (size + 1)
        line = px[start : start + size + 1]
        bits.extend("1" if line[c] > line[c + 1] else "0" for c in range(size))
    return f"{int(''.join(bits), 2):064x}"


def validate() -> tuple[Report, dict]:
    rep = Report()
    stats: dict = {"splits": {}, "classes": Counter(), "total_images": 0, "total_objects": 0}

    # ---- 1. data.yaml ------------------------------------------------
    yaml_path = DATASET_DIR / "data.yaml"
    cfg = read_data_yaml(yaml_path, rep)
    if not cfg:
        return rep, stats

    nc = cfg.get("nc")
    names = cfg.get("names") or []

    if nc != taxonomy.NUM_CLASSES:
        rep.error(f"data.yaml nc={nc} but taxonomy expects {taxonomy.NUM_CLASSES}")

    if list(names) != list(taxonomy.DATASET_NAMES):
        rep.error(
            "data.yaml names do not match backend/taxonomy.py\n"
            f"        data.yaml: {list(names)}\n"
            f"        taxonomy : {list(taxonomy.DATASET_NAMES)}"
        )

    # ---- 2. declared split paths resolve -----------------------------
    split_dirs: dict[str, Path] = {}
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        declared = cfg.get(key)
        if not declared:
            rep.error(f"data.yaml has no '{key}:' entry")
            continue
        resolved = (yaml_path.parent / declared).resolve()
        if not resolved.exists():
            rep.error(f"{key}: '{declared}' resolves to {resolved} which does NOT exist")
        else:
            split_dirs[split] = resolved

    # ---- 3..8 per-split checks ---------------------------------------
    hashes: dict[str, dict[str, str]] = defaultdict(dict)

    for split in SPLITS:
        img_dir = split_dirs.get(split)
        if img_dir is None:
            continue
        lbl_dir = img_dir.parent / "labels"
        if not lbl_dir.exists():
            rep.error(f"{split}: labels directory missing at {lbl_dir}")
            continue

        images = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
        )
        labels = sorted(
            p for p in lbl_dir.iterdir()
            if p.suffix == ".txt" and not p.name.startswith(".")
        )

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        for stem in sorted(img_stems - lbl_stems):
            rep.error(f"{split}: image '{stem}' has NO label file")
        for stem in sorted(lbl_stems - img_stems):
            rep.error(f"{split}: label '{stem}.txt' has NO matching image")

        split_classes: Counter = Counter()
        n_objects = 0
        n_empty = 0
        formats: Counter = Counter()

        for lbl in labels:
            try:
                raw = lbl.read_text(encoding="utf-8")
            except Exception as exc:
                rep.error(f"{split}: cannot read {lbl.name}: {exc}")
                continue

            rows = [ln for ln in raw.splitlines() if ln.strip()]
            if not rows:
                n_empty += 1
                continue

            for lineno, row in enumerate(rows, 1):
                parts = row.split()
                formats[len(parts) - 1] += 1

                if len(parts) != 5:
                    rep.error(
                        f"{split}/{lbl.name}:{lineno}: expected 5 fields "
                        f"(class cx cy w h), found {len(parts)}"
                    )
                    continue

                try:
                    cid = int(float(parts[0]))
                    cx, cy, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    rep.error(f"{split}/{lbl.name}:{lineno}: non-numeric value")
                    continue

                if not taxonomy.is_valid_class_id(cid):
                    rep.error(
                        f"{split}/{lbl.name}:{lineno}: class id {cid} outside "
                        f"0..{taxonomy.NUM_CLASSES - 1}"
                    )
                    continue

                for label, value in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
                    if not (-1e-6 <= value <= 1 + 1e-6):
                        rep.error(
                            f"{split}/{lbl.name}:{lineno}: {label}={value} not "
                            f"normalised to 0..1 (are these pixel coordinates?)"
                        )
                if w <= 0 or h <= 0:
                    rep.error(f"{split}/{lbl.name}:{lineno}: zero-area box w={w} h={h}")

                split_classes[cid] += 1
                n_objects += 1

        # image decodability + perceptual hash
        unreadable = 0
        for img in images:
            h = dhash(img)
            if h is None:
                rep.error(f"{split}: image will not decode: {img.name}")
                unreadable += 1
            else:
                hashes[split][img.name] = h

        stats["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": n_objects,
            "empty_labels": n_empty,
            "unreadable": unreadable,
            "classes": split_classes,
            "formats": dict(formats),
        }
        stats["classes"].update(split_classes)
        stats["total_images"] += len(images)
        stats["total_objects"] += n_objects

        if n_empty:
            rep.warn(f"{split}: {n_empty} label file(s) contain no objects (background images)")

    # ---- 9. cross-split leakage --------------------------------------
    seen: dict[str, tuple[str, str]] = {}
    for split in SPLITS:
        for name, h in hashes.get(split, {}).items():
            if h in seen:
                other_split, other_name = seen[h]
                if other_split != split:
                    rep.error(
                        f"SPLIT LEAKAGE: '{name}' ({split}) is the same image as "
                        f"'{other_name}' ({other_split})"
                    )
                else:
                    rep.warn(f"{split}: duplicate image '{name}' == '{other_name}'")
            else:
                seen[h] = (split, name)

    return rep, stats


def print_report(rep: Report, stats: dict) -> None:
    line = "=" * 66
    print(line)
    print("DATASET VALIDATION REPORT")
    print(line)
    print(f"dataset : {DATASET_DIR}")
    print()

    print(f"{'split':<8}{'images':>8}{'labels':>8}{'objects':>9}{'empty':>7}{'formats':>12}")
    print("-" * 66)
    for split in SPLITS:
        s = stats["splits"].get(split)
        if not s:
            print(f"{split:<8}{'--- not validated ---':>44}")
            continue
        print(
            f"{split:<8}{s['images']:>8}{s['labels']:>8}{s['objects']:>9}"
            f"{s['empty_labels']:>7}{str(s['formats']):>12}"
        )
    print("-" * 66)
    print(f"{'TOTAL':<8}{stats['total_images']:>8}{'':>8}{stats['total_objects']:>9}")
    print()

    print("class distribution")
    print("-" * 66)
    total = sum(stats["classes"].values()) or 1
    print(f"{'id':<4}{'dataset name':<22}{'display name':<22}{'count':>7}{'share':>8}")
    for cid in range(taxonomy.NUM_CLASSES):
        n = stats["classes"].get(cid, 0)
        print(
            f"{cid:<4}{taxonomy.dataset_name(cid):<22}{taxonomy.display_name(cid):<22}"
            f"{n:>7}{100 * n / total:>7.1f}%"
        )
    print()

    counts = [stats["classes"].get(c, 0) for c in range(taxonomy.NUM_CLASSES)]
    if min(counts) > 0:
        print(f"class imbalance ratio (max/min): {max(counts) / min(counts):.1f}x")
    print()

    if rep.warnings:
        print(f"WARNINGS ({len(rep.warnings)})")
        print("-" * 66)
        for w in rep.warnings:
            print(f"  ! {w}")
        print()

    if rep.errors:
        print(f"ERRORS ({len(rep.errors)})")
        print("-" * 66)
        for e in rep.errors:
            print(f"  X {e}")
        print()
        print("RESULT: FAILED - do not train on this dataset until fixed.")
    else:
        print("RESULT: PASSED - dataset is structurally sound.")
    print(line)


def main(argv: list[str] | None = None) -> int:
    global DATASET_DIR
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        DATASET_DIR = Path(argv[0]).expanduser().resolve()
    rep, stats = validate()
    print_report(rep, stats)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
