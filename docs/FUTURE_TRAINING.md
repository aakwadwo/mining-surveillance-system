# Next model: the improvement pipeline

**Nothing here has been run.** This is the prepared procedure for when
the additional photographs arrive.

The current model (`galamsey-phase5.pt`) stays untouched until a new one
is proven better on the same test set.

---

## The one-line summary

> The model's weakness follows the data almost exactly. **Every class
> whose training data grew in Phase 5 improved; the two that did not
> grow did not.** So the fix is more photographs of the *right* classes —
> not a bigger model, more epochs, or a different architecture.

---

## What photographs are actually worth having

This is the most important page in this document. Not all photographs
help, and some make the dataset *worse*.

| Class | Images now | Priority | Why |
|---|---|---|---|
| **Structure** | **14** | 🔴 **CRITICAL** | Effectively unsolved. Buildings, shelters, huts, sheds, tarpaulin camps. |
| **MiningTool** | **56** | 🟠 **HIGH** | 0.27 recall. Shovels, pans, sluice boxes, pickaxes, buckets. |
| ProcessingEquipment | 118 | 🟡 medium | Wash plants, dredges, pumps, "chanfan" engines. |
| Vehicle | 125 | 🟡 medium | Trucks, pickups, motorbikes. |
| ExcavationMachinery | 314 | 🟢 fine | Already the best-performing class. |
| **Person** | **143** | ⛔ **SATURATED** | 1,435 objects already. **More will not help.** |

### ⚠️ Why more Person photos would make things worse

Adding 200 crowd photographs adds perhaps 1,500 more `Person` objects
and zero `Structure`. The imbalance goes from 37.8× to worse. The model
gets even better at the class it's already good at, and no better at the
ones holding it back.

**A single photograph containing a shed and a shovel is worth more than
fifty photographs of people working.**

### Diversity matters as much as count

Fifty photographs of the same shed from the same angle teach the model
"that shed", not "sheds". Aim to vary:

| Vary this | Why |
|---|---|
| **Lighting** | dawn, midday glare, overcast, dusk — a model trained only at noon fails in the morning |
| **Camera angle** | ground level, elevated, drone-style overhead |
| **Object scale** | close-ups AND objects small in the distance |
| **Background** | forest, river, bare pit, roadside, farmland |
| **Occlusion** | partly hidden behind machinery, vegetation, people |
| **Equipment variety** | different makes, colours, ages, rusted and new |
| **Weather** | dry dust, wet mud, after rain |

Rural and remote settings especially — that is where the system would
actually be used, and it is what the current dataset has least of.

---

## The pipeline

```
ADDITIONAL PHOTOGRAPHS
   ↓  tools/audit_new_photos.py        ← START HERE, before annotating
AUDIT + DEDUPLICATION
   ↓  Roboflow / LabelImg
ANNOTATION
   ↓  tools/validate_dataset.py
LABEL VALIDATION
   ↓
CLASS BALANCE CHECK
   ↓
LEAKAGE CHECK  (perceptual hash)
   ↓
TRAIN / VALID / TEST SPLIT
   ↓  tools/train_phase5.py --data ... --name ...
TRAIN
   ↓  tools/compare_models.py
EVALUATE + PER-CLASS ANALYSIS
   ↓
COMPARE WITH PHASE 5
   ↓
SELECT MODEL  →  INTEGRATE  →  FULL SYSTEM VALIDATION
```

---

## Step by step

### 1. Audit before annotating

```bash
./.venv/bin/python tools/audit_new_photos.py ~/path/to/new-photos
```

Tells you how many are genuinely new versus duplicates of what you
already have, and which classes to prioritise.

**Do this first.** Annotating is the expensive step, and annotating a
duplicate is worse than wasted: a near-identical photo in train and test
means the model has seen the answers, and your score becomes fiction.

### 2. Annotate

Use **exactly** these six classes, in this id order — the ids are the
contract, and changing them silently breaks everything:

```
0  ExcavationMachinery
1  MiningTool
2  Person
3  ProcessingEquipment
4  Structure
5  Vehicle
```

Export in **YOLO format** (one `.txt` per image; `class cx cy w h`,
all normalised 0–1).

### 3. Combine with the existing dataset

Keep the existing 536 photographs. Add the new ones. **Preserve the
current valid/test photographs in valid/test** so the comparison against
Phase 5 stays fair — a model evaluated on a different test set cannot be
compared with one evaluated on this one.

`tools/recover_dataset.py` shows the pattern: perceptual-hash every
image, drop duplicates, split deterministically with a fixed seed, and
verify no photograph appears in two splits.

### 4. Validate

```bash
./.venv/bin/python tools/validate_dataset.py dataset-phase6
```

Must print `RESULT: PASSED`. It checks paths, image↔label pairing,
5-field rows, class ids in range, names matching `taxonomy.py`,
normalised coordinates, non-zero areas, decodable images, and
cross-split leakage.

### 5. Check the balance actually improved

Compare the new per-class image counts against `docs/BASELINE.md`. If
`Structure` is still in single digits, the run will not fix it — go back
and collect more rather than spending 30 minutes training.

### 6. Train — **changing only the dataset**

```bash
./.venv/bin/python tools/train_phase5.py \
    --data dataset-phase6/data.yaml --name yolo12n-phase6
```

Every other setting stays identical (epochs 100, batch 16, imgsz 640,
seed 0, lr0 0.01, YOLOv12-nano). ~33 minutes on an M4 Pro.

**Do not** also switch to `yolo12s`, change the learning rate, or train
for 200 epochs. If you change five things and it improves, you have
learned nothing about which one mattered.

### 7. Evaluate on the same test set

```bash
./.venv/bin/python tools/compare_models.py test
./.venv/bin/python tools/compare_models.py val
```

### 8. Judge it honestly

A higher mAP50 is **not** sufficient. Check:

- Did `Structure` and `MiningTool` improve? Those were the point.
- Did anything **regress**? Report it if so.
- Is the per-class support large enough to trust? `Structure` had 1 test
  image at baseline — if it still does, its number is noise.
- Does the improvement track the data growth per class, as it did in
  Phase 5? That pattern is the evidence, not the headline number.

### 9. Integrate — only if genuinely better

```bash
cp runs/phase5/yolo12n-phase6/weights/best.pt backend/models/galamsey-phase6.pt
# then update DEFAULT_MODEL_PATH in backend/config.py
```

**Never delete `galamsey-phase5.pt`.** It is the baseline the next
comparison is measured against, and the evidence for your write-up.

Update `backend/models/PROVENANCE.md` with the new model's full training
configuration and measured metrics.

### 10. Full system validation

```bash
./.venv/bin/python -m pytest tests/ -q      # must stay at 104+ passing
./tools/run_dev.sh                          # then open localhost:5501
```

Confirm live detection still works end to end with the new weights, and
that `taxonomy.verify_against_model()` reports an exact match.

---

## If the new model is worse

That is a legitimate, reportable result — not a failure to hide.

Likely explanations, in order of probability:

1. The new photographs were mostly `Person`, worsening imbalance.
2. Annotation quality varied between the old and new batches.
3. Duplicates slipped through and caused leakage (check the validator).
4. Not enough new data to move the needle.

Keep `galamsey-phase5.pt` as the production model and say why in the
write-up. A negative result you can explain is worth more than a
positive one you cannot.
