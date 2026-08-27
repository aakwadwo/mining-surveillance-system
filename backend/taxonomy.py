"""
Galamsey AI Detector — Detection class taxonomy
===============================================

THE SINGLE SOURCE OF TRUTH FOR DETECTION CLASSES.

Nothing else in this project may hardcode a class name or class id.
Import from here instead.

------------------------------------------------------------------
WHY THIS FILE EXISTS
------------------------------------------------------------------
An object detector does not output words. It outputs an integer:
"slot 2, confidence 0.83". The word "Person" exists only in the
dataset's data.yaml and, frozen at training time, inside the model
checkpoint (best.pt).

If two parts of a system disagree about what slot 2 means, nothing
crashes. The system just prints the wrong label, confidently. That is
the most dangerous failure mode in applied computer vision.

Before this file existed, this project had THREE different class
lists that did not agree with each other or with the dataset:

  data.yaml / trained weights : 6 classes  (authoritative)
  backend/app.py              : 11 invented classes
  frontend/script.js          : 3 invented classes

------------------------------------------------------------------
THE AUTHORITATIVE TAXONOMY
------------------------------------------------------------------
Defined by:  "illegal mining activities/data.yaml"  (nc: 6)
Confirmed embedded in the baseline weights
             runs/train_illegal_mining/yolov12_illegal_mining6/weights/best.pt

------------------------------------------------------------------
THE "ProcessingEquipment" SPELLING  (corrected in Phase 5)
------------------------------------------------------------------
Class 3 was spelled "ProcessingEquiment" — missing the 'p' — in the
original Roboflow project, in Dataset A's data.yaml, and inside the
Run 6 baseline weights.

Phase 4 deliberately preserved the typo, because correcting the name in
one place while the weights still carried the old spelling would
desynchronise data from model. That was the right call at the time.

Phase 5 retrains the model on a rebuilt dataset, which is exactly the
moment the correction can be made SAFELY and SYNCHRONOUSLY across:
    this file, dataset-phase5/data.yaml, "illegal mining activities"/
    data.yaml, frontend/config.js, the tests, and the documentation.

The numeric ids did NOT move. Only the human-readable string changed.

Run 6 still carries the old spelling and is retained as historical
evidence, so LEGACY_NAME_ALIASES below lets verify_against_model()
accept it with a warning instead of rejecting a model that is otherwise
perfectly valid. Aliases are for KNOWN, DOCUMENTED spelling changes
only - never a place to paper over a genuine taxonomy mismatch.
"""

from __future__ import annotations

# ---------------------------------------------------------------
# 1. Canonical taxonomy — ids are FROZEN. Never reorder or renumber.
#    The index in this tuple IS the class id the model emits.
# ---------------------------------------------------------------
DATASET_NAMES: tuple[str, ...] = (
    "ExcavationMachinery",   # 0
    "MiningTool",            # 1
    "Person",                # 2
    "ProcessingEquipment",   # 3  (typo "ProcessingEquiment" fixed in Phase 5)
    "Structure",             # 4
    "Vehicle",               # 5
)

NUM_CLASSES: int = len(DATASET_NAMES)  # 6

# ---------------------------------------------------------------
# 2. Human-facing names. Presentation layer only.
# ---------------------------------------------------------------
DISPLAY_NAMES: dict[int, str] = {
    0: "Excavation Machinery",
    1: "Mining Tool",
    2: "Person",
    3: "Processing Equipment",
    4: "Structure",
    5: "Vehicle",
}

# ---------------------------------------------------------------
# 2b. Known historical spellings, accepted with a warning.
#
#     ONLY for documented renames where the CONCEPT and the ID are
#     unchanged. Never add an entry to silence a real mismatch: two
#     models disagreeing about what class 3 MEANS is precisely the
#     failure this module exists to prevent.
# ---------------------------------------------------------------
LEGACY_NAME_ALIASES: dict[int, set[str]] = {
    3: {"ProcessingEquiment"},   # pre-Phase-5 spelling (Run 6 weights)
}

# ---------------------------------------------------------------
# 3. What each class actually means, for the UI and for the report.
# ---------------------------------------------------------------
DESCRIPTIONS: dict[int, str] = {
    0: "Excavators, bulldozers and other earth-moving machinery.",
    1: "Hand tools and small mining implements: shovels, pans, sluices.",
    2: "People present at the site.",
    3: "Washing plants, sluice boxes, dredges and pumping rigs.",
    4: "Buildings, shelters and site structures.",
    5: "Road vehicles: trucks, cars, motorbikes.",
}

# ---------------------------------------------------------------
# 4. Evidence weighting.
#
#    IMPORTANT — READ THIS BEFORE USING IT.
#
#    None of these six classes proves illegality on its own. A person
#    or a truck is not evidence of a crime; legal mining sites contain
#    excavators too. Whether an activity is ILLEGAL depends on WHERE it
#    happens (inside a licensed concession or a protected reserve),
#    which is geofence information the frontend holds — not something
#    a vision model can determine from pixels.
#
#    So this table does NOT say "illegal". It ranks how strongly each
#    class indicates INDUSTRIAL MINING ACTIVITY. The backend reports
#    evidence; the illegal/legal judgement is made elsewhere, with
#    geofence context, and is ultimately a human decision.
# ---------------------------------------------------------------
MINING_ACTIVITY_WEIGHT: dict[int, float] = {
    0: 1.0,   # heavy machinery — strongest single indicator
    1: 0.8,   # mining tools    — strong
    2: 0.2,   # people          — weak on its own
    3: 1.0,   # processing rigs — strongest single indicator
    4: 0.4,   # structures      — moderate
    5: 0.3,   # vehicles        — weak on its own
}

# ---------------------------------------------------------------
# 5. Legacy vocabularies found in the inherited code.
#    Kept ONLY so old data and logs can be interpreted. Do not extend.
# ---------------------------------------------------------------
LEGACY_BACKEND_CLASSES: tuple[str, ...] = (
    "excavator", "bulldozer", "water_pump", "wash_plant", "truck",
    "conveyor", "drill", "mining_equipment", "illegal_site",
    "legal_site", "suspicious_activity",
)

LEGACY_FRONTEND_CLASSES: tuple[str, ...] = ("excavator", "pump", "washplant")


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def display_name(class_id: int) -> str:
    """Human-readable name for a class id, safe on unknown ids."""
    return DISPLAY_NAMES.get(class_id, f"Unknown({class_id})")


def dataset_name(class_id: int) -> str:
    """Dataset/wire name for a class id, safe on unknown ids."""
    if 0 <= class_id < NUM_CLASSES:
        return DATASET_NAMES[class_id]
    return f"Unknown({class_id})"


def is_valid_class_id(class_id: int) -> bool:
    return isinstance(class_id, int) and 0 <= class_id < NUM_CLASSES


def mining_activity_weight(class_id: int) -> float:
    return MINING_ACTIVITY_WEIGHT.get(class_id, 0.0)


def verify_against_model(model_names: dict[int, str]) -> list[str]:
    """
    Compare a loaded model's embedded class names against this taxonomy.

    Ultralytics exposes model.names as {id: name}. A mismatch means the
    weights were trained on a DIFFERENT taxonomy, and every label the
    system prints would be wrong while nothing errors. Callers must
    treat a non-empty result as a hard failure, not a warning.

    Documented historical spellings (LEGACY_NAME_ALIASES) are accepted,
    so the Run 6 baseline still loads after the Phase 5 rename.

    Returns a list of human-readable problems; empty list == match.
    """
    problems: list[str] = []

    if len(model_names) != NUM_CLASSES:
        problems.append(
            f"class count mismatch: model has {len(model_names)}, "
            f"taxonomy expects {NUM_CLASSES}"
        )

    for cid in range(NUM_CLASSES):
        expected = DATASET_NAMES[cid]
        actual = model_names.get(cid)
        if actual is None:
            problems.append(f"class {cid}: missing from model (expected {expected!r})")
        elif actual != expected:
            if actual in LEGACY_NAME_ALIASES.get(cid, set()):
                # Documented historical spelling. Same id, same concept.
                continue
            problems.append(
                f"class {cid}: model says {actual!r}, taxonomy expects {expected!r}"
            )

    extra = [cid for cid in model_names if cid >= NUM_CLASSES]
    if extra:
        problems.append(
            f"model has extra classes not in this taxonomy: "
            f"{ {cid: model_names[cid] for cid in sorted(extra)} }"
        )

    return problems


def as_report() -> str:
    """Printable summary, used by the health endpoint and the tests."""
    lines = [f"Taxonomy: {NUM_CLASSES} classes (authoritative)"]
    for cid in range(NUM_CLASSES):
        lines.append(
            f"  {cid}  {DATASET_NAMES[cid]:<22} -> {display_name(cid):<22} "
            f"weight={mining_activity_weight(cid):.1f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(as_report())
