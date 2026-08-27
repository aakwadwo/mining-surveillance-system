#!/usr/bin/env python3
"""
Build the Phase 5 training dataset.

    ./.venv/bin/python tools/recover_dataset.py --report-only   # inspect only
    ./.venv/bin/python tools/recover_dataset.py                 # build it

Combines:
  * Dataset A  — "illegal mining activities/", 87 photographs, the
                 authoritative Phase 4 baseline
  * Recovered  — the 475-image `galamsey-ev0ma` v2 export, remapped from
                 its 7-class taxonomy to the project's 6 classes

into `dataset-phase5/`.

------------------------------------------------------------------
WHAT THIS SCRIPT WILL NOT DO
------------------------------------------------------------------
It never writes to, moves or deletes anything in:
    illegal mining activities/                      (Dataset A)
    ~/Downloads/galamsey.v2i.yolov12/               (pristine source)
    ~/Downloads/yolov12/yolov12/dataset/            (contaminated copy)
It only reads. The output directory is created fresh.

The contaminated Dataset B copy is not read at all: the pristine
`galamsey.v2i.yolov12` export was verified byte-identical to it, so we
work from the clean original. See docs/TAXONOMY_MAPPING.md.

------------------------------------------------------------------
WHY THE SPLIT IS BUILT THE WAY IT IS
------------------------------------------------------------------
Dataset A's existing valid (17) and test (9) photographs stay in valid
and test. That is what makes the Run 6 comparison meaningful: Run 6 was
evaluated on those exact images and has never trained on them, so both
models can be scored on common ground.

Recovered images are then split on top, seeded and deterministic, so
the evaluation sets are large enough to mean something.

Every image is fingerprinted with a perceptual hash BEFORE splitting.
Duplicates are dropped, and no photograph can appear in two splits.
Filenames cannot be trusted for this: Roboflow gives the same photo a
different `.rf.<hash>` in every export.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import taxonomy  # noqa: E402

# ---------------------------------------------------------------
# Inputs and outputs
# ---------------------------------------------------------------
DATASET_A = PROJECT_ROOT / "illegal mining activities"
SOURCE_DIR = Path("/Users/kwadwo/Downloads/galamsey.v2i.yolov12/train")
OUTPUT_DIR = PROJECT_ROOT / "dataset-phase5"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 0
VALID_FRACTION = 0.15
TEST_FRACTION = 0.10

# ---------------------------------------------------------------
# The verified mapping. See docs/TAXONOMY_MAPPING.md for the evidence
# behind every row; do not edit this without updating that document.
# ---------------------------------------------------------------
SOURCE_NAMES = [
    "PERSON", "chanfine machine", "excavator", "galamsey boat",
    "truck", "washing machine", "water pump",
]
CLASS_MAP: dict[int, int] = {
    0: 2,   # PERSON            -> Person
    1: 3,   # chanfine machine  -> ProcessingEquipment (pumping rig)
    2: 0,   # excavator         -> ExcavationMachinery
    3: 3,   # galamsey boat     -> ProcessingEquipment (floating dredge)
    4: 5,   # truck             -> Vehicle
    5: 3,   # washing machine   -> ProcessingEquipment (wash plant)
    6: 3,   # water pump        -> ProcessingEquipment (pumping rig)
}


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def dhash(path: Path, size: int = 16) -> str | None:
    """Fingerprint of what the picture LOOKS like, robust to re-encoding."""
    from PIL import Image

    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return None
    px = list(img.tobytes())
    bits = []
    for row in range(size):
        start = row * (size + 1)
        line = px[start:start + size + 1]
        bits.extend("1" if line[c] > line[c + 1] else "0" for c in range(size))
    return f"{int(''.join(bits), 2):064x}"


def safe_output_name(prefix: str, img: Path, index: int) -> str:
    """
    Build a filesystem-safe output filename.

    Roboflow bakes the original (sometimes very long) headline into the
    filename. One source image here is 200+ characters, and macOS caps a
    single path component at 255 BYTES - adding a prefix overflowed it.

    We keep a readable head of the original name plus the Roboflow
    .rf.<hash>, which is already unique, and fall back to a positional
    index if that is somehow absent. The full original path is recorded
    in MANIFEST.json, so nothing is lost.
    """
    stem, suffix = img.stem, img.suffix.lower()
    if ".rf." in stem:
        head, rf = stem.split(".rf.", 1)
        rf = rf[:32]
    else:
        head, rf = stem, f"{index:05d}"
    head = head[:60].rstrip("._-")
    return f"{prefix}{head}.rf.{rf}{suffix}"


def images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
    )


class Stats:
    def __init__(self) -> None:
        self.rejections: Counter = Counter()
        self.rejected_examples: dict[str, list[str]] = defaultdict(list)

    def reject(self, reason: str, detail: str = "") -> None:
        self.rejections[reason] += 1
        if detail and len(self.rejected_examples[reason]) < 3:
            self.rejected_examples[reason].append(detail)


def convert_label(text: str, stats: Stats, filename: str) -> list[str] | None:
    """
    Convert one source label file to project-taxonomy YOLO boxes.

    Returns the converted lines, or None if the whole file is unusable.
    Individual bad rows are rejected, not repaired.
    """
    out: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        where = f"{filename}:{lineno}"

        try:
            src_cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            stats.reject("non-numeric value", where)
            continue

        if src_cls not in CLASS_MAP:
            stats.reject(f"class id {src_cls} not in source taxonomy (0-6)", where)
            continue

        if len(coords) == 4:                      # already a box
            cx, cy, w, h = coords
            x0, x1 = cx - w / 2, cx + w / 2
            y0, y1 = cy - h / 2, cy + h / 2
        elif len(coords) >= 6 and len(coords) % 2 == 0:   # polygon
            xs, ys = coords[0::2], coords[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        else:
            stats.reject(f"unusable coordinate count ({len(coords)})", where)
            continue

        if any(v < -1e-6 or v > 1 + 1e-6 for v in (x0, x1, y0, y1)):
            stats.reject("coordinates not normalised to 0-1", where)
            continue

        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(1.0, x1), min(1.0, y1)
        w, h = x1 - x0, y1 - y0
        if w <= 1e-6 or h <= 1e-6:
            stats.reject("zero-area box after conversion", where)
            continue

        tgt = CLASS_MAP[src_cls]
        out.append(f"{tgt} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {w:.6f} {h:.6f}")

    return out


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def build(report_only: bool) -> int:
    stats = Stats()
    print("=" * 70)
    print("PHASE 5 DATASET RECOVERY")
    print("=" * 70)

    if not SOURCE_DIR.is_dir():
        print(f"ERROR: pristine source not found at {SOURCE_DIR}")
        return 1

    # ---- 1. Index Dataset A, keeping its split assignment -----------
    print("\n[1] Indexing Dataset A (authoritative baseline)")
    a_items: dict[str, list[tuple[Path, Path, str]]] = {"train": [], "valid": [], "test": []}
    a_hashes: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        for img in images_in(DATASET_A / split / "images"):
            lbl = DATASET_A / split / "labels" / f"{img.stem}.txt"
            if not lbl.exists():
                stats.reject("Dataset A image without label", img.name)
                continue
            h = dhash(img)
            if h is None:
                stats.reject("Dataset A image unreadable", img.name)
                continue
            if h in a_hashes:
                stats.reject("duplicate within Dataset A", img.name)
                continue
            a_hashes[h] = f"A/{split}/{img.name}"
            a_items[split].append((img, lbl, h))
        print(f"    {split:5s}: {len(a_items[split]):>3} images")
    print(f"    unique photographs in Dataset A: {len(a_hashes)}")

    # ---- 2. Read + convert the pristine recovered source -------------
    print(f"\n[2] Reading pristine source: {SOURCE_DIR.parent.name}")
    src_images = images_in(SOURCE_DIR / "images")
    print(f"    source images: {len(src_images)}")

    recovered: list[tuple[Path, list[str], str]] = []
    seen_src: dict[str, str] = {}
    src_class_counts: Counter = Counter()

    for img in src_images:
        lbl = SOURCE_DIR / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            stats.reject("source image without label", img.name)
            continue

        h = dhash(img)
        if h is None:
            stats.reject("source image unreadable", img.name)
            continue
        if h in a_hashes:
            stats.reject("perceptual duplicate of a Dataset A photo", img.name)
            continue
        if h in seen_src:
            stats.reject("duplicate within the source export", img.name)
            continue

        for line in lbl.read_text().splitlines():
            if line.split():
                try:
                    src_class_counts[int(float(line.split()[0]))] += 1
                except ValueError:
                    pass

        lines = convert_label(lbl.read_text(), stats, img.name)
        if not lines:
            stats.reject("no usable annotations after conversion", img.name)
            continue

        seen_src[h] = img.name
        recovered.append((img, lines, h))

    print(f"    accepted    : {len(recovered)}")
    print(f"    rejected    : {len(src_images) - len(recovered)}")

    # ---- 3. Split the recovered images -------------------------------
    print("\n[3] Splitting recovered images (seeded, deterministic)")
    rng = random.Random(SEED)
    shuffled = recovered[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * TEST_FRACTION)
    n_valid = int(n * VALID_FRACTION)
    split_of = {}
    for i, (img, _, h) in enumerate(shuffled):
        split_of[h] = "test" if i < n_test else "valid" if i < n_test + n_valid else "train"
    print(f"    train {sum(1 for v in split_of.values() if v == 'train'):>3}"
          f"  valid {sum(1 for v in split_of.values() if v == 'valid'):>3}"
          f"  test {sum(1 for v in split_of.values() if v == 'test'):>3}")

    # ---- 4. Assemble the final plan ----------------------------------
    plan: dict[str, list] = {"train": [], "valid": [], "test": []}
    for split in ("train", "valid", "test"):
        for img, lbl, h in a_items[split]:
            plan[split].append(("A", img, lbl.read_text().splitlines(), h))
    for img, lines, h in recovered:
        plan[split_of[h]].append(("R", img, lines, h))

    # leakage guard
    seen_all: dict[str, str] = {}
    leaks = []
    for split, items in plan.items():
        for _, img, _, h in items:
            if h in seen_all and seen_all[h] != split:
                leaks.append(f"{img.name}: {seen_all[h]} and {split}")
            seen_all[h] = split
    if leaks:
        print("    LEAKAGE DETECTED:")
        for x in leaks[:10]:
            print("      " + x)
        return 1
    print("    leakage check: PASSED (no photograph appears in two splits)")

    # ---- 5. Report ----------------------------------------------------
    print("\n[4] Source class distribution and mapping")
    print(f"    {'src':<4}{'source name':<20}{'objects':>8}   ->  {'tgt':<4}{'target name'}")
    for c in range(7):
        t = CLASS_MAP[c]
        print(f"    {c:<4}{SOURCE_NAMES[c]:<20}{src_class_counts[c]:>8}   ->  "
              f"{t:<4}{taxonomy.dataset_name(t)}")

    print("\n[5] Final dataset")
    per_split_classes: dict[str, Counter] = {s: Counter() for s in plan}
    per_split_images: dict[str, Counter] = {s: Counter() for s in plan}
    origin = Counter()
    for split, items in plan.items():
        for src, img, lines, _ in items:
            origin[(split, src)] += 1
            present = set()
            for line in lines:
                if line.split():
                    cid = int(line.split()[0])
                    per_split_classes[split][cid] += 1
                    present.add(cid)
            for cid in present:
                per_split_images[split][cid] += 1

    print(f"    {'split':<8}{'images':>8}{'from A':>9}{'recovered':>11}{'objects':>9}")
    total_img = total_obj = 0
    for split in ("train", "valid", "test"):
        ni = len(plan[split]); no = sum(per_split_classes[split].values())
        total_img += ni; total_obj += no
        print(f"    {split:<8}{ni:>8}{origin[(split,'A')]:>9}{origin[(split,'R')]:>11}{no:>9}")
    print(f"    {'TOTAL':<8}{total_img:>8}{'':>9}{'':>11}{total_obj:>9}")

    print(f"\n    {'id':<4}{'class':<24}{'objects':>9}{'images':>8}{'share':>8}")
    grand = Counter()
    grand_img = Counter()
    for split in plan:
        grand.update(per_split_classes[split])
        grand_img.update(per_split_images[split])
    tot = sum(grand.values()) or 1
    for cid in range(taxonomy.NUM_CLASSES):
        print(f"    {cid:<4}{taxonomy.dataset_name(cid):<24}{grand[cid]:>9}"
              f"{grand_img[cid]:>8}{100*grand[cid]/tot:>7.1f}%")
    counts = [grand[c] for c in range(taxonomy.NUM_CLASSES)]
    if min(counts) > 0:
        print(f"    imbalance (max/min objects): {max(counts)/min(counts):.1f}x")

    if stats.rejections:
        print("\n[6] Rejections")
        for reason, count in stats.rejections.most_common():
            print(f"    {count:>5}  {reason}")
            for ex in stats.rejected_examples.get(reason, []):
                print(f"           e.g. {ex[:70]}")

    if report_only:
        print("\n--report-only: nothing was written.")
        return 0

    # ---- 6. Write -----------------------------------------------------
    print(f"\n[7] Writing {OUTPUT_DIR.name}/")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    manifest = []
    for split, items in plan.items():
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        for index, (src, img, lines, h) in enumerate(items):
            prefix = "a_" if src == "A" else "r_"
            name = safe_output_name(prefix, img, index)
            shutil.copy2(img, OUTPUT_DIR / split / "images" / name)
            (OUTPUT_DIR / split / "labels" / f"{Path(name).stem}.txt").write_text(
                "\n".join(l for l in lines if l.strip()) + "\n"
            )
            manifest.append({
                "split": split, "origin": "dataset-a" if src == "A" else "galamsey-ev0ma-v2",
                "output_name": name, "source_path": str(img), "phash": h,
            })

    names = list(taxonomy.DATASET_NAMES)
    (OUTPUT_DIR / "data.yaml").write_text(
        "# Galamsey AI Detector - Phase 5 combined dataset\n"
        "# GENERATED by tools/recover_dataset.py - do not edit by hand.\n"
        "#\n"
        "# Dataset A (87 photographs, authoritative Phase 4 baseline)\n"
        "# + galamsey-ev0ma v2 (475 photographs), remapped from its\n"
        "#   7-class taxonomy per docs/TAXONOMY_MAPPING.md.\n"
        "#\n"
        "# Paths are relative to THIS FILE's directory. No 'path:' key on\n"
        "# purpose: a relative 'path' is resolved against Ultralytics'\n"
        "# global datasets_dir, not against the yaml.\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        f"nc: {taxonomy.NUM_CLASSES}\n"
        f"names: {names}\n"
    )
    (OUTPUT_DIR / "MANIFEST.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/recover_dataset.py",
        "seed": SEED,
        "sources": {
            "dataset_a": str(DATASET_A),
            "recovered": str(SOURCE_DIR.parent),
        },
        "class_map_source_to_target": {str(k): v for k, v in CLASS_MAP.items()},
        "source_names": SOURCE_NAMES,
        "target_names": names,
        "rejections": dict(stats.rejections),
        "counts": {s: len(plan[s]) for s in plan},
        "files": manifest,
    }, indent=2))
    print(f"    wrote {len(manifest)} image/label pairs + data.yaml + MANIFEST.json")
    print("\nNext:  ./.venv/bin/python tools/validate_dataset.py dataset-phase5")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="analyse and report without writing anything")
    raise SystemExit(build(ap.parse_args().report_only))
