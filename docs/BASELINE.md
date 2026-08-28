# Frozen engineering baseline

Recorded so the **next** model can be compared against this one fairly.
Everything here was measured, not estimated.

---

## The commit

| | |
|---|---|
| Branch | `phase-6-integration` |
| Baseline commit | see `git log -1` on this branch |
| `main` | `48bff44` — the untouched inherited project |
| Other branches | `phase-4-cleanup`, `phase-5-data-recovery` |

---

## The model

| | |
|---|---|
| File | `backend/models/galamsey-phase5.pt` (5,526,554 bytes) |
| Architecture | YOLOv12-nano |
| Parameters | 2,569,218 |
| Base weights | `yolo12n.pt` (COCO-pretrained, pip Ultralytics) |
| Built by | Ultralytics **8.3.253 (pip)** — needs no vendored fork |
| Classes | exactly the six in `backend/taxonomy.py` |

### Training configuration — **keep all of this identical next time**

```
epochs      100        batch      16         imgsz     640
seed        0          optimizer  auto       lr0       0.01
patience    100        deterministic True    workers   0
device      mps        duration   0.55 h on an Apple M4 Pro
```

Reproduce exactly:

```bash
./.venv/bin/python tools/train_phase5.py
```

---

## The dataset

`dataset-phase5/` — **536 unique photographs, 2,722 objects**

| Split | Images |
|---|---|
| train | 399 |
| valid | 84 |
| test | 53 |

| id | class | objects | images |
|---|---|---|---|
| 0 | ExcavationMachinery | 423 | 314 |
| 1 | MiningTool | 391 | 56 |
| 2 | Person | 1,435 | 143 |
| 3 | ProcessingEquipment | 273 | 118 |
| 4 | **Structure** | **38** | **14** |
| 5 | Vehicle | 162 | 125 |

Object imbalance 37.8×. Cross-split leakage verified zero by perceptual
hash; `MANIFEST.json` records the origin and fingerprint of all 536.

---

## Measured metrics — the numbers to beat

Held-out **test** split, conf 0.001, IoU 0.6 (the standard mAP settings):

| Metric | Value |
|---|---|
| Precision | **0.4915** |
| Recall | **0.5584** |
| mAP50 | **0.5031** |
| mAP50-95 | **0.2601** |

Per-class recall (test):

| Class | Recall |
|---|---|
| ExcavationMachinery | 0.730 |
| MiningTool | 0.273 |
| Person | 0.812 |
| ProcessingEquipment | 0.500 |
| Structure | 0.333 |
| Vehicle | 0.703 |

⚠️ `Structure` has **1 test image** and `MiningTool` **5**, so those two
per-class figures are noise-dominated. Do not read a small change in
them as a trend.

Reproduce:

```bash
./.venv/bin/python tools/compare_models.py test
./.venv/bin/python tools/compare_models.py val
```

---

## Measured performance

Apple M4 Pro, MPS:

| | |
|---|---|
| model load + warm-up | 1300 ms (one-off) |
| inference, cold | 284 ms |
| **inference, warm** | **18.7 ms median**, 20.0 ms mean |
| `/api/detect` round trip | ~30–110 ms (HTTP + multipart on top) |
| JPEG encode per frame | 0.5 ms |
| DB insert per event | 0.37 ms |
| DB read, 50 events | 0.56 ms |
| peak memory | 560 MB |
| backend busy at 1500 ms interval | **1.3 %** |

---

## Runtime configuration at baseline

```
GALAMSEY_MODEL             backend/models/galamsey-phase5.pt
GALAMSEY_YOLO_IMPL         pip
GALAMSEY_DEVICE            auto  -> mps
GALAMSEY_CONF              0.25          (live inference threshold)
GALAMSEY_IOU               0.45          (live NMS threshold)
GALAMSEY_IMGSZ             640
GALAMSEY_PORT              5004
GALAMSEY_MAX_UPLOAD        16777216
GALAMSEY_MAX_PIXELS        40000000
GALAMSEY_HISTORY_MAX       5000
DETECTION_INTERVAL_MS      1500          (frontend/config.js)
DETECTION_MAX_WIDTH        960
```

**Evaluation uses different thresholds from live inference on purpose.**
mAP is measured at conf 0.001 so the whole precision/recall curve is
visible; the live dashboard uses 0.25 so the screen isn't covered in
low-confidence guesses. Both are recorded here so the next comparison
uses the same settings.

---

## Tests

**104 passing**, ~8 s, no failures, no skips.

```bash
./.venv/bin/python -m pytest tests/ -q
```

---

## Known limitations at baseline

1. `Structure` effectively unsolved — 14 images.
2. `MiningTool` weak — 0.27 recall, 56 images.
3. Precision 0.49 — about half of what it reports is wrong.
4. Geofence **creation** unavailable — Google removed `DrawingManager`
   in Maps JS v3.65. Not faked.
5. Camera-permission testing cannot be automated.
6. Google Maps key present in git history (commits `48bff44`, `d63a8dd`).
7. Prototype only — no auth, no deployment, single user.

---

## Why the next experiment must change only the dataset

If the next model is better, you need to be able to say **why**. Change
the data *and* the architecture *and* the learning rate at once, and an
improvement tells you nothing about which change caused it.

Phase 5 already demonstrated the value of this discipline: hyper-
parameters were held identical to the previous run, so the 2.5× mAP50
improvement was attributable to the data — and the result was even
cleaner than the headline, because **every class whose data grew
improved, and the two whose data did not grow did not.** That is a
causal finding, not a lucky number.

So for the next run, change **`--data` and `--name` only**:

```bash
./.venv/bin/python tools/train_phase5.py \
    --data dataset-phase6/data.yaml --name yolo12n-phase6
```

Keep constant: architecture, model size, epochs, batch, imgsz, seed,
optimizer, lr0, patience, evaluation thresholds, and the evaluation
procedure.
