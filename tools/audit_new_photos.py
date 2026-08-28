#!/usr/bin/env python3
"""
Audit a folder of NEW photographs before annotating them.

    ./.venv/bin/python tools/audit_new_photos.py ~/Desktop/new-photos

This is step 1 of the model-improvement pipeline. Run it BEFORE spending
hours drawing boxes, because it answers the two questions that decide
whether that work is worth doing:

    1. Are these photographs actually new, or duplicates of ones we
       already have?
    2. Do they contain the classes the model is weak at?

------------------------------------------------------------------
WHY THIS EXISTS
------------------------------------------------------------------
Annotating is the expensive part - a human drawing a box around every
object. Annotating a duplicate is wasted effort AND actively harmful: if
a near-identical photo ends up in train and another copy in test, the
model has effectively seen the answers and the test score becomes
meaningless. That is called DATA LEAKAGE, and it silently inflates your
results.

Duplicates are compared by PERCEPTUAL HASH, not by filename or file
size. The same photograph saved twice - resized, re-compressed, renamed
- has completely different bytes but is still the same picture. Only a
content fingerprint catches that.

------------------------------------------------------------------
WHAT IT DOES NOT DO
------------------------------------------------------------------
It does not annotate, modify, move or delete anything. It reads the new
folder and the existing dataset, and prints a report. Deciding what to
keep is a human judgement - this tool just makes it an informed one.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import taxonomy  # noqa: E402

DEFAULT_EXISTING = PROJECT_ROOT / "dataset-phase5"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Classes the current model is weakest at, and roughly how many more
# images of each would meaningfully help. Derived from the Phase 5
# per-class results and image counts, not from guesswork.
PRIORITY = {
    4: ("Structure", 14, "CRITICAL - only 14 images; effectively unsolved"),
    1: ("MiningTool", 56, "HIGH - 56 images; recall 0.27"),
    3: ("ProcessingEquipment", 118, "MEDIUM - usable but thin"),
    5: ("Vehicle", 125, "MEDIUM"),
    0: ("ExcavationMachinery", 314, "OK"),
    2: ("Person", 143, "SATURATED - more of these will not help"),
}


def dhash(path: Path, size: int = 16) -> str | None:
    """
    A fingerprint of what the picture LOOKS like.

    Shrinks the image to a tiny grid, then records whether each pixel is
    brighter than the one to its right. Two photographs that look the
    same produce nearly the same fingerprint even if one was resized,
    re-saved or renamed.
    """
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


def hamming(a: str, b: str) -> int:
    """How many bits differ between two fingerprints. 0 = identical."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def images_in(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
    )


def existing_fingerprints(dataset: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        folder = dataset / split / "images"
        if not folder.is_dir():
            continue
        for img in images_in(folder):
            h = dhash(img)
            if h:
                out[h] = f"{split}/{img.name}"
    return out


def existing_class_counts(dataset: Path) -> tuple[collections.Counter, collections.Counter]:
    objects: collections.Counter = collections.Counter()
    images: collections.Counter = collections.Counter()
    for split in ("train", "valid", "test"):
        folder = dataset / split / "labels"
        if not folder.is_dir():
            continue
        for lbl in folder.glob("*.txt"):
            present = set()
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if parts:
                    cid = int(float(parts[0]))
                    objects[cid] += 1
                    present.add(cid)
            for cid in present:
                images[cid] += 1
    return objects, images


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new_folder", help="folder containing the new photographs")
    ap.add_argument("--existing", default=str(DEFAULT_EXISTING),
                    help="existing dataset to compare against")
    ap.add_argument("--threshold", type=int, default=8,
                    help="max Hamming distance treated as a duplicate (default 8)")
    args = ap.parse_args()

    new_dir = Path(args.new_folder).expanduser().resolve()
    existing_dir = Path(args.existing).expanduser().resolve()

    if not new_dir.is_dir():
        print(f"ERROR: {new_dir} is not a folder")
        return 1
    if not existing_dir.is_dir():
        print(f"ERROR: existing dataset not found at {existing_dir}")
        return 1

    line = "=" * 70
    print(line)
    print("NEW PHOTOGRAPH AUDIT")
    print(line)
    print(f"new      : {new_dir}")
    print(f"existing : {existing_dir}")
    print()

    # ---- 1. current state of the dataset ---------------------------
    obj_counts, img_counts = existing_class_counts(existing_dir)
    print("[1] What the existing dataset already has")
    print(f"    {'id':<4}{'class':<22}{'objects':>9}{'images':>8}   priority")
    for cid in range(taxonomy.NUM_CLASSES):
        _, _, note = PRIORITY.get(cid, ("", 0, ""))
        print(f"    {cid:<4}{taxonomy.dataset_name(cid):<22}"
              f"{obj_counts[cid]:>9}{img_counts[cid]:>8}   {note}")
    print()

    # ---- 2. index the new folder -----------------------------------
    new_images = images_in(new_dir)
    print(f"[2] Scanning {len(new_images)} new image(s)")
    if not new_images:
        print("    nothing to audit")
        return 0

    unreadable: list[Path] = []
    fingerprints: dict[Path, str] = {}
    for img in new_images:
        h = dhash(img)
        if h is None:
            unreadable.append(img)
        else:
            fingerprints[img] = h
    print(f"    readable   : {len(fingerprints)}")
    print(f"    unreadable : {len(unreadable)}")
    print()

    # ---- 3. duplicates against the existing dataset ----------------
    print("[3] Duplicate check against the existing dataset")
    print("    (perceptual hash - catches the same photo re-saved or renamed)")
    existing = existing_fingerprints(existing_dir)
    dup_existing: list[tuple[Path, str, int]] = []
    for img, h in fingerprints.items():
        best, best_d = None, 999
        for eh, name in existing.items():
            d = hamming(h, eh)
            if d < best_d:
                best, best_d = name, d
        if best_d <= args.threshold:
            dup_existing.append((img, best, best_d))
    print(f"    already in the dataset : {len(dup_existing)}")
    for img, match, d in dup_existing[:10]:
        print(f"      {img.name[:48]:<50} == {match} (distance {d})")
    if len(dup_existing) > 10:
        print(f"      ... and {len(dup_existing) - 10} more")
    print()

    # ---- 4. duplicates within the new folder -----------------------
    print("[4] Duplicate check WITHIN the new folder")
    seen: dict[str, Path] = {}
    dup_internal: list[tuple[Path, Path, int]] = []
    remaining = [p for p in fingerprints if p not in {d[0] for d in dup_existing}]
    for img in remaining:
        h = fingerprints[img]
        hit = None
        for sh, other in seen.items():
            d = hamming(h, sh)
            if d <= args.threshold:
                hit = (other, d)
                break
        if hit:
            dup_internal.append((img, hit[0], hit[1]))
        else:
            seen[h] = img
    print(f"    internal duplicates : {len(dup_internal)}")
    for img, other, d in dup_internal[:5]:
        print(f"      {img.name[:44]:<46} == {other.name[:24]} (distance {d})")
    print()

    # ---- 5. verdict -------------------------------------------------
    genuinely_new = len(seen)
    print(line)
    print("VERDICT")
    print(line)
    print(f"  submitted           : {len(new_images)}")
    print(f"  unreadable          : {len(unreadable)}")
    print(f"  already in dataset  : {len(dup_existing)}")
    print(f"  duplicated in folder: {len(dup_internal)}")
    print(f"  GENUINELY NEW       : {genuinely_new}")
    print()
    if genuinely_new:
        print(f"  Annotating these would take the dataset from "
              f"{sum(img_counts.values()) and len(existing)} to "
              f"{len(existing) + genuinely_new} photographs.")
    print()
    print("  WHAT TO ANNOTATE FIRST")
    print("  The model's weakness follows the data almost exactly, so the")
    print("  value of a photograph depends on WHICH classes it contains:")
    print()
    for cid in (4, 1, 3, 5, 0, 2):
        name, imgs, note = PRIORITY[cid]
        print(f"    {name:<22} {imgs:>4} images now   {note}")
    print()
    print("  A photograph containing a Structure is worth many times one")
    print("  containing only people: Person already has 1,435 objects and")
    print("  adding more will not improve it, while Structure has 38.")
    print()
    print("  NEXT STEPS")
    print("    1. discard the duplicates listed above")
    print("    2. annotate the genuinely new images (Roboflow / LabelImg)")
    print("       using EXACTLY these six classes, in this id order:")
    for cid in range(taxonomy.NUM_CLASSES):
        print(f"         {cid}  {taxonomy.dataset_name(cid)}")
    print("    3. export in YOLO format")
    print("    4. ./.venv/bin/python tools/validate_dataset.py <folder>")
    print("    5. ./.venv/bin/python tools/train_phase5.py --data <folder>/data.yaml \\")
    print("           --name yolo12n-phase6")
    print("    6. ./.venv/bin/python tools/compare_models.py test")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
