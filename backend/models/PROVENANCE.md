# Model Provenance

Where these weight files came from, what they were trained on, and which
one you are allowed to trust. Written during Phase 4.

**Read this before using any model in this folder.**

---

## Summary

| File | Classes | mAP50 | Reproducible? | Use |
|---|---|---|---|---|
| `galamsey-run6-baseline.pt` | **6** | 0.255 | **yes** | baseline for testing the inference pipeline |
| `archive/galamsey-run8-12class.pt` | 12 | 0.294 | **no** | historical evidence only — **do not deploy** |

Neither model is production quality. Both are weak. They exist so the
system can be built and tested end-to-end before Phase 5 retraining.

---

## `galamsey-run6-baseline.pt` — the baseline

Copied from
`~/Downloads/yolov12/runs/train_illegal_mining/yolov12_illegal_mining6/weights/best.pt`
(byte-identical; the original is untouched).

| | |
|---|---|
| Trained | 2025-11-16 12:03 → 12:35 (2.3 h wall clock) |
| Base model | `yolov12n.pt` (COCO-pretrained nano) |
| Architecture | YOLOv12-nano, 2,520,834 parameters |
| Built by | Ultralytics **8.3.63** — the vendored fork in `yolov12/` |
| Dataset | `~/Downloads/yolov12/yolov12/dataset` in its clean 16 Nov state |
| Epochs | 100 (batch 16, imgsz 640, lr0 0.01, optimizer auto, seed 0) |

**Embedded class names** — these match `backend/taxonomy.py` exactly:

```
0 ExcavationMachinery   1 MiningTool   2 Person
3 ProcessingEquiment    4 Structure    5 Vehicle
```

Note class 3 carries the upstream `ProcessingEqui**ment**` typo. That is
expected and is corrected for display only. See `backend/taxonomy.py`.

**Validation metrics (stored in the checkpoint):**

| Precision | Recall | mAP50 | mAP50-95 | fitness |
|---|---|---|---|---|
| 0.506 | 0.255 | 0.255 | 0.118 | 0.131 |

**Per-class recall** (from the run's confusion matrix):

| Class | Recall |
|---|---|
| ExcavationMachinery | 0.08 |
| MiningTool | 0.39 |
| Person | 0.48 |
| ProcessingEquiment | 0.22 |
| Structure | **0.00** |
| Vehicle | **0.00** |

**Be honest about this model.** It misses 51–100 % of objects depending
on class, and never detects `Structure` or `Vehicle` at all. It is a
proof that the pipeline works, not a working detector.

### Why it is called "reproducible"

Its training dataset still exists, and its class list matches the
project's authoritative taxonomy. The dataset it used is the augmented
(3×) export of the **same 61 source photographs** as the project's
`illegal mining activities/` — verified by perceptual hashing at a median
Hamming distance of 1 bit in 256. So the baseline is traceable to data we
still hold.

---

## `archive/galamsey-run8-12class.pt` — evidence only

Copied from `.../yolov12_illegal_mining8/weights/best.pt`.

| | |
|---|---|
| Trained | 2025-11-26 09:54 → 16:48 (10.1 h) |
| Base model | `yolov12n.pt` |
| Built by | Ultralytics 8.3.63 (vendored fork) |
| Dataset | `/Users/kwadwo/Desktop/yolo/dataset/data.yaml` — **DELETED** |
| Epochs | 100 (best epoch 55) |

**Embedded class names — 12, not 6:**

```
0 ExcavationMachinery   6 chanfine machine
1 MiningTool            7 excavator
2 Person                8 galamsey boat
3 ProcessingEquipment   9 truck
4 Structure            10 washing machine
5 Vehicle              11 water pump
```

**Metrics:** P 0.701 / R 0.238 / mAP50 0.294 / mAP50-95 0.138.

### Why it is NOT used, despite better numbers

1. **Its training dataset no longer exists.** `~/Desktop/yolo/` is gone.
   You cannot state in a dissertation what this model was trained on,
   and you cannot reproduce or extend it.
2. **12 classes for a 6-class project.** Classes 6–11 have completely
   empty rows *and* columns in the confusion matrix — never predicted,
   never present in validation. Six of twelve outputs are dead weight.
3. **The taxonomy is incoherent.** It contains redundant pairs
   (`ExcavationMachinery` vs `excavator`, `Vehicle` vs `truck`,
   `ProcessingEquipment` vs `washing machine`). Two class lists were
   concatenated without reconciliation, so the model was penalised for
   splitting one concept across two labels.
4. It would not match `backend/taxonomy.py`, so
   `taxonomy.verify_against_model()` rejects it by design.

It is kept because it is genuine prior work and useful evidence of what
was tried. **Do not deploy it.**

---

## Running these weights: an important constraint

Both checkpoints were produced by the **vendored fork** in `yolov12/`
(Ultralytics 8.3.63). They **cannot** be run by pip `ultralytics`:

| | `AAttn` internals | `pe` kernel |
|---|---|---|
| Fork 8.3.63 (built these) | `self.qk` + `self.v` (split) | 5 |
| pip 8.3.217 / 8.3.253 | `self.qkv` (fused) | 7 |

pip Ultralytics loads the file successfully — names, `nc` and parameter
count all read back correctly — and then fails at `forward()` with:

```
AttributeError: 'AAttn' object has no attribute 'qkv'. Did you mean: 'qk'?
```

So the backend selects the fork explicitly for inference, and logs which
implementation it used at startup. Verified working under torch 2.13 on
CPU (661 ms) and MPS (56 ms) with identical detections.

**Phase 5 retraining with pip `ultralytics` will remove this constraint**
and produce upstream-compatible weights.

---

## Other files in `archive/`

| File | What it is |
|---|---|
| `run6-args.yaml`, `run8-args.yaml` | exact training configuration used |
| `run6-results.csv`, `run8-results.csv` | per-epoch metrics for all 100 epochs |

These let you reconstruct the training curves without the original run
directories.

---

## Originals

Nothing here was moved. The source run directories remain at:

```
~/Downloads/yolov12/runs/train_illegal_mining/
```

along with confusion matrices, PR curves and validation batch images.
`~/Downloads/yolov12/illegal_mining.zip` holds a verified
byte-identical backup of the clean pre-contamination dataset plus run 6's
weights. **Do not delete any of it.**
