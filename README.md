# Galamsey AI Detector

A computer-vision system for detecting illegal small-scale gold mining
(*galamsey*) activity in Ghana from camera or drone imagery, with a web
dashboard that plots detections on a map against geofenced areas.

> **Project status: live end-to-end application.**
> The dashboard now runs real detection on your camera:
> `camera → /api/detect → galamsey-phase5.pt → bounding boxes`, with the
> Detection Report, statistics and alerts all driven by the same real
> inference result, and detection history persisted to SQLite.
>
> Backend, frontend, dataset and inference all work and are tested.
> Phase 5 recovered 449 additional photographs, rebuilt the dataset, and
> retrained: **test mAP50 improved from 0.200 to 0.503 (2.5×)**, with
> every class improving. `Structure` remains unsolved (10 training
> images). See [Current limitations](#current-limitations).

---

## Contents

1. [Architecture](#architecture)
2. [Folder structure](#folder-structure)
3. [Prerequisites](#prerequisites)
4. [Setup](#setup)
5. [Running it](#running-it)
6. [The dataset](#the-dataset)
7. [The AI model](#the-ai-model)
8. [Detection modes: mock vs model](#detection-modes-mock-vs-model)
9. [API reference](#api-reference)
10. [Testing](#testing)
11. [Configuration reference](#configuration-reference)
12. [Current limitations](#current-limitations)
13. [Common errors and fixes](#common-errors-and-fixes)
14. [Assumptions made](#assumptions-made)

---

## Architecture

Three independent pieces. The frontend and backend are **separate
programs** that talk over HTTP.

```
   ┌──────────────────────────────────────────────────────────────┐
   │  BROWSER — http://localhost:5501                             │
   │                                                              │
   │   frontend/index.html + styles.css + script.js               │
   │   frontend/config.js       (committed, NO secrets at all)    │
   │        Maps key fetched from the backend at /client-config   │
   │                                                              │
   │   ┌───────────┐  ┌────────────────┐  ┌───────────────────┐   │
   │   │ Video     │  │ Detection      │  │ Google Map        │   │
   │   │ panel     │  │ Report panel   │  │ + geofences       │   │
   │   └───────────┘  └────────────────┘  └───────────────────┘   │
   └───────┬──────────────────────────────────────┬───────────────┘
           │  HTTP + CORS                         │ Maps JS API
           │                                      ▼
           │                            ┌───────────────────┐
           │                            │ Google Cloud      │
           │                            │ Maps / Geocoding  │
           │                            └───────────────────┘
           ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  BACKEND — http://127.0.0.1:5004   (backend/app.py, Flask)   │
   │                                                              │
   │   app.py       HTTP routes only                              │
   │   config.py    paths, ports, modes   (single source)         │
   │   taxonomy.py  the 6 class names     (single source)         │
   │   detector.py  model loading + inference + drawing           │
   │                        │                                     │
   │                        ▼                                     │
   │   backend/models/galamsey-phase5.pt   (6 classes, pip YOLO)  │
   │   .env  ->  GOOGLE_MAPS_API_KEY (git-ignored)                │
   └───────┬──────────────────────────────────────────────────────┘
           │ imports YOLO from
           ▼
   ┌──────────────────────────┐   ┌──────────────────────────────┐
   │ pip ultralytics 8.3.253  │   │ dataset-phase5/              │
   │ (yolov12/ fork needed    │   │ 536 photographs, 6 classes   │
   │  only for Run 6/8)       │   │ train 399 / valid 84 / test 53│
   └──────────────────────────┘   └──────────────────────────────┘
```

Deeper design notes, including why there are two Ultralytics
implementations, are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Folder structure

```
final year ai/
├── frontend/                    static dashboard — no build step
│   ├── index.html
│   ├── script.js                one GalamseyDetector class (~4000 lines)
│   ├── styles.css
│   ├── config.js                committed config, NO secrets
│   ├── config.local.example.js  template — copy to config.local.js
│   ├── map-popup.html/.js       map in a separate tab
│   └── DETECTION_API.md         original frontend↔device contract
│
├── backend/
│   ├── app.py                   Flask routes
│   ├── config.py                paths, ports, modes
│   ├── taxonomy.py              THE class definitions
│   ├── detector.py              model loading and inference
│   └── models/
│       ├── galamsey-phase5.pt          THE PRODUCTION MODEL
│       ├── galamsey-run6-baseline.pt   historical Phase 4 baseline
│       ├── PROVENANCE.md               where every model came from
│       ├── phase5/                     training + evaluation artefacts
│       └── archive/                    run 8 + its metrics
│
├── dataset-phase5/              THE TRAINING DATASET (generated)
│   ├── data.yaml
│   ├── MANIFEST.json            origin + phash of all 536 files
│   └── train/ valid/ test/
│
├── illegal mining activities/   Dataset A - the Phase 4 baseline,
│   ├── data.yaml                preserved unmodified; 87 photographs
│   └── train/ valid/ test/      all of which are inside dataset-phase5
│
├── yolov12/                     vendored upstream fork — DO NOT EDIT
│
├── tools/
│   ├── recover_dataset.py       rebuild dataset-phase5 from sources
│   ├── train_phase5.py          reproducible training run
│   ├── evaluate_model.py        metrics for one model
│   ├── compare_models.py        Run 6 vs Phase 5, same data
│   ├── validate_dataset.py      structural dataset checks
│   ├── serve_frontend.sh        static server on :5501
│   ├── run_dev.sh               start backend + frontend together
│   ├── check_camera.py          macOS camera diagnostic
│   └── check_imports.py         dependency diagnostic
│
├── tests/test_phase4.py         56 tests
├── .env                         SECRETS - git-ignored
├── .env.example                 template for .env
├── requirements.txt             runtime dependencies
├── requirements-dev.txt         + test tooling
└── .venv/                       git-ignored
```

---

## Prerequisites

| | |
|---|---|
| **Python** | **3.12** (3.11+ works). Verified on 3.12.11, arm64. |
| OS | macOS 26 on Apple Silicon (developed here). Linux/Windows should work; only camera handling is macOS-specific. |
| Disk | ~4 GB for `.venv` (PyTorch is large) |
| Google Maps API key | optional — everything except the map works without it |
| Webcam | optional — the backend runs fine without one |

**Why 3.12:** the inherited `backend/venv` used Python 3.9.6, which
reached end of life in October 2025. PyTorch 2.13, OpenCV 5 and
Ultralytics 8.3 all ship arm64 wheels for 3.12.

---

## Setup

```bash
cd "/Users/kwadwo/Downloads/final year ai"

# 1. Create an isolated environment (a private Python + site-packages
#    for this project only; nothing is installed system-wide).
python3 -m venv .venv

# 2. Install dependencies
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

# 3. Optional: test tooling
./.venv/bin/python -m pip install -r requirements-dev.txt

# 4. Optional: Google Maps key
cp .env.example .env
#   then edit .env and set GOOGLE_MAPS_API_KEY=...
#   The backend reads it and serves it to the browser at /client-config,
#   so no credential ever sits in a frontend file.
```

Verify:

```bash
./.venv/bin/python -c "import torch, cv2, ultralytics, flask; print('ok')"
./.venv/bin/python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
./.venv/bin/python tools/validate_dataset.py dataset-phase5
```

---

## Running it

**Both at once (easiest):**

```bash
./tools/run_dev.sh          # Ctrl+C stops both
```

**Or separately, in two terminals:**

```bash
# terminal 1 — backend
./.venv/bin/python backend/app.py
#   -> http://127.0.0.1:5004

# terminal 2 — frontend
./tools/serve_frontend.sh
#   -> http://localhost:5501
```

Then open **<http://localhost:5501>**.

> ### Do not open `index.html` by double-clicking it
>
> That gives a `file://` URL, which browsers treat as an **insecure
> context**. In an insecure context the browser disables
> `getUserMedia` (webcam), `navigator.geolocation` (map tracking) and
> cross-tab `BroadcastChannel`, and `fetch` to the backend is blocked
> because `file://` has an opaque `null` origin no server can allow.
>
> `http://localhost` **is** a secure context — browsers make that
> exception specifically so local development works.

### Why two ports?

They are two separate programs. A **port** is a numbered door on your
machine that one program listens at. The Python backend listens at 5004;
the static file server listens at 5501. `localhost` (`127.0.0.1`) means
"this computer" — traffic never leaves your machine.

Because the ports differ, the browser considers them **different
origins**, and by default refuses to let a page from one read responses
from the other. **CORS** (Cross-Origin Resource Sharing) is the
mechanism by which the backend says "requests from `localhost:5501` are
allowed". That allow-list is in `backend/config.py`.

---

## The dataset

`dataset-phase5/` — **536 unique photographs, 2,722 objects.** Generated
by `tools/recover_dataset.py` from two Roboflow sources (both CC BY 4.0),
and fully reproducible.

| Split | Images | from Dataset A | recovered | Objects |
|---|---|---|---|---|
| train | 399 | 61 | 338 | 1,983 |
| valid | 84 | 17 | 67 | 487 |
| test | 53 | 9 | 44 | 252 |
| **Total** | **536** | **87** | **449** | **2,722** |

### Where it came from

**Dataset A** (`illegal mining activities/`, 87 photographs) is the
Phase 4 baseline, preserved unmodified. Its valid/test photographs stay
in valid/test so the Run 6 comparison remains fair.

**The recovered 449** came from the Roboflow project
`galamsey-zutez/galamsey-ev0ma` v2 (475 images), whose labels used a
different 7-class taxonomy and polygon format. Establishing that
taxonomy from documented evidence — not from guessing — and remapping it
is written up in [`docs/TAXONOMY_MAPPING.md`](docs/TAXONOMY_MAPPING.md).
26 of the 475 were rejected: 15 perceptual duplicates of Dataset A
photographs, 11 duplicates within the export.

**A historical dataset that was NOT used:** a 657-file folder at
`~/Downloads/yolov12/yolov12/dataset/` mixes two incompatible class
schemas — its class 0 is people while ours is excavators. Training on it
would silently teach the model that excavators are people. It is left
untouched as evidence. Its clean 182-file subset was also excluded: those
are the *same 61 photographs* as Dataset A, exported three times each
with augmentation, so they add file count but no information.

### Leakage

Every photograph is fingerprinted with a perceptual hash before
splitting, so no image can appear in two splits. Filenames cannot be
trusted for this: Roboflow gives the same photo a different
`.rf.<hash>` in every export. Verified by test and by the manifest.

**Six classes.** The number in the label file is what matters; the name
is only a label humans read.

| id | name | objects | images | share |
|---|---|---|---|---|
| 0 | `ExcavationMachinery` | 423 | 314 | 15.5 % |
| 1 | `MiningTool` | 391 | 56 | 14.4 % |
| 2 | `Person` | 1,435 | 143 | 52.7 % |
| 3 | `ProcessingEquipment` | 273 | 118 | 10.0 % |
| 4 | `Structure` | 38 | **14** | 1.4 % |
| 5 | `Vehicle` | 162 | 125 | 6.0 % |

Object-count imbalance improved from 61.5× (Dataset A) to 37.8×.
`Vehicle` grew 10.1×; `MiningTool` and `Structure` did not grow at all,
because the recovered export does not annotate them. **`Structure` has
only 10 training images** and is the dataset's real weak point.

The upstream typo `ProcessingEqui**ment**` was corrected to
`ProcessingEqui**p**ment` in Phase 5, synchronously across the taxonomy
module, both `data.yaml` files, the frontend config, the tests and the
docs. The Run 6 weights still carry the old spelling and are accepted
via a single documented alias in `backend/taxonomy.py`.

### YOLO label format

One `.txt` per image, one line per object:

```
class_id  x_center  y_center  width  height
2         0.4832    0.6104    0.1123 0.2456
```

All four coordinates are **normalised to 0–1** — fractions of the image
width and height, not pixels. This is what lets the same label file stay
correct when the image is resized: YOLO scales every image to 640×640
before training, and fractions survive that unchanged where pixel values
would not. It also keeps all inputs on a similar numeric scale, which
neural networks train on far more stably.

### Validating it

```bash
./.venv/bin/python tools/validate_dataset.py
```

Checks path resolution, image↔label pairing, 5-field rows, class ids in
range, names matching the taxonomy, normalised coordinates, non-zero
areas, decodable images, and **cross-split leakage compared by
perceptual hash** (the same photo in train and test makes your metrics
meaningless, and filenames won't reveal it).

---

## The AI model

**YOLOv12-nano**, fine-tuned from COCO-pretrained `yolo12n.pt`.
2,557,898 parameters, trained with **pip Ultralytics 8.3.253** so it
needs no vendored fork. Full history in
[`backend/models/PROVENANCE.md`](backend/models/PROVENANCE.md).

| | `galamsey-phase5.pt` (production) |
|---|---|
| Classes | 6, exact match to `taxonomy.py` |
| Trained | 100 epochs, batch 16, imgsz 640, seed 0, 0.55 h on MPS |
| Dataset | `dataset-phase5` — 536 photographs |
| Reproducible | yes — `tools/train_phase5.py` |

### Measured on the held-out test split

Both models evaluated on the **same** data with the same settings:

| Metric | Run 6 (Phase 4) | **Phase 5** | change |
|---|---|---|---|
| Precision | 0.357 | **0.492** | +0.134 |
| Recall | 0.268 | **0.558** | +0.290 |
| mAP50 | 0.200 | **0.503** | **2.5×** |
| mAP50-95 | 0.079 | **0.260** | 3.3× |

Per-class recall — every class improved:

| Class | Run 6 | Phase 5 | data growth |
|---|---|---|---|
| ExcavationMachinery | 0.595 | 0.730 | 3.9× |
| MiningTool | 0.182 | 0.273 | none |
| Person | 0.616 | 0.812 | 1.5× |
| ProcessingEquipment | 0.147 | 0.500 | 3.0× |
| Structure | 0.070 | 0.333 | none |
| Vehicle | **0.000** | **0.703** | 10.1× |

**Be honest about the caveats in your write-up.** On the *validation*
split, `MiningTool` (−0.07) and `Structure` (−0.13) went the other way.
Those are precisely the two classes whose data did not grow, and their
support is tiny — `Structure` has 3 validation images and 1 test image,
so its numbers are noise in both directions.

The result that matters is causal, not just numerical: **every class
whose training data grew improved substantially and consistently; the
two that did not grow did not reliably improve.** `Structure` remains
unsolved with 10 training images.

### How inference works

```
image → resize 640×640 → YOLOv12 → [class_id, confidence, x1,y1,x2,y2] …
      → filter by confidence (default 0.25)
      → map class_id to a name via taxonomy.py
      → JSON to the frontend
```

**mAP50** = mean Average Precision at 50 % box overlap, from 0 to 1.
"On average across classes, how well does the model find objects while
avoiding false alarms, counting a box correct if it overlaps the true
box by at least half?"

Reproduce the comparison yourself:

```bash
./.venv/bin/python tools/compare_models.py test
./.venv/bin/python tools/compare_models.py val
```

### A guard you should know about

Before any model is used, `taxonomy.verify_against_model()` compares the
class names **embedded in the weights** against `taxonomy.py`, and
**refuses the model on mismatch**. A detector outputs a number; if the
weights say slot 2 is `ExcavationMachinery` while the app thinks it is
`Person`, nothing errors and every label is silently wrong. That is the
single most dangerous failure mode in applied computer vision, and this
project has already been bitten by it once.

---

## Detection modes: mock vs model

Set with `GALAMSEY_DETECTION_MODE`:

| Mode | Behaviour |
|---|---|
| `auto` *(default)* | use the real model if it loads; otherwise fall back to mock and **say so** |
| `model` | real model only. If it cannot load: mode `unavailable`, **zero** detections, never fakes |
| `mock` | synthetic placeholder boxes, for UI work with no model or camera |

**Mock output is impossible to mistake for real output:**

- every detection carries `source: "mock"` and `is_synthetic: true`
- every response carries `synthetic: true` plus a warning string
- class names are prefixed `MOCK `
- rendered video frames get a red **`MOCK DATA - NOT AI OUTPUT`** banner
- the dashboard shows a mock-mode banner

This matters for academic integrity: the inherited code returned fake
boxes that were structurally identical to real ones, so a screenshot of
placeholder data was indistinguishable from working AI.

```bash
GALAMSEY_DETECTION_MODE=mock ./.venv/bin/python backend/app.py
```

---

## API reference

Base URL `http://127.0.0.1:5004`. All responses are JSON except
`/video_feed`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | index of all endpoints |
| **POST** | **`/api/detect`** | **run the model on one image — THE inference endpoint** |
| GET | `/health` | server, model, camera and taxonomy status |
| GET | `/statistics` | detection counts this session |
| GET | `/detections?limit=N` | **persisted** detection history (survives restart) |
| POST | `/api/detections` | submit a detection record (frontend uses this) |
| POST | `/detect` | deprecated alias of `/api/detect` (Phase 4 shape) |
| GET | `/client-config` | browser-safe config (Maps key from `.env`) |
| GET | `/config` | current configuration |
| POST | `/config` | update the confidence threshold |
| GET | `/video_feed` | MJPEG stream with boxes drawn (needs a camera) |

**REST in one line:** the URL names a *thing*, the verb says what you
want to do with it. `GET` reads and must never change anything (so it is
safe to retry or refresh); `POST` creates or changes something. The
status code carries the outcome: 200 ok, 201 created, 400 your request
was malformed, 404 no such thing, 503 the server is fine but cannot do
this right now.

```bash
curl http://127.0.0.1:5004/health
curl -F image=@dataset-phase5/test/images/*.jpg http://127.0.0.1:5004/api/detect
```

### `POST /api/detect` — the contract

```json
{
  "success": true,
  "detections": [
    {"class_id": 2, "class_name": "Person", "confidence": 0.9527,
     "bbox": {"x1": 104, "y1": 132, "x2": 146, "y2": 243}}
  ],
  "image_width": 639,
  "image_height": 426,
  "inference_time_ms": 108.0,
  "synthetic": false
}
```

`bbox` coordinates are **pixels in the submitted image** — which is why
`image_width`/`image_height` come back with them. A client cannot scale
correctly without knowing what the coordinates are relative to.

| Status | Meaning |
|---|---|
| 200 | inference ran |
| 400 | no image / empty upload / undecodable bytes |
| 413 | upload too large (body > 16 MB, or decoded image > 40 M pixels) |
| 503 | the model is not loaded — **never** substituted with fake data |
| 500 | unexpected failure (logged server-side) |

---

## How live detection works

```
   <video>            the browser renders your camera, 30-60 fps,
      |               hardware accelerated — the backend never touches it
      |  every 1500 ms
      v
   offscreen <canvas>  one frame, downscaled to 960px wide
      |
      v  toBlob -> JPEG
   POST /api/detect
      |
      v
   galamsey-phase5.pt  real YOLOv12 inference, ~40-110 ms on MPS
      |
      v  JSON
   ONE result object drives everything:
      +-- bounding-box overlay   (absolutely-positioned divs ON TOP)
      +-- Detection Report
      +-- confidence / site type / suggested action
      +-- alerts
      +-- persisted event (SQLite)
```

**Why the overlay is a separate layer.** The backend *could* draw boxes
onto the frame and send back a picture. It doesn't: the browser already
renders video smoothly for free, so round-tripping every frame through
Python would make it jerky for no gain. Burned-in boxes also can't be
restyled, hidden or made accessible — and video (smooth) and detections
(~every 1.5 s) update at different rates.

**What happens if the backend dies mid-session.** After 3 consecutive
failures the overlay is cleared and the report is blanked to "Detection
unavailable / Backend unreachable". A frozen picture of the last
successful inference looks exactly like a live one, so showing it would
be the same integrity problem as showing mock data as real. The loop
keeps retrying with a back-off and recovers automatically — verified by
killing the backend mid-session and restarting it.

**Why only one request is ever in flight.** `setInterval(send, 100)`
would post frames faster than ~110 ms inference can drain them; the
queue grows without limit and boxes fall further behind reality. The
loop is `capture → infer → render → WAIT → repeat` with an explicit
in-flight guard, so a slow cycle just delays the next one.

**Coordinate scaling.** Three spaces are involved: the frame we POSTed
(e.g. 960×540), the `<video>` element's box (e.g. 1004×565), and the
picture as actually painted inside it. The video is `object-fit: cover`,
so it's scaled to *fill* and *cropped* — not stretched:

```
scale   = max(boxW/srcW, boxH/srcH)     ← max, not min
offsetX = (boxW - srcW*scale) / 2       ← negative when cropped
screenX = srcX*scale + offsetX
```

Naive scaling ignores the crop and misplaces every box in a way that
looks *almost* right. Boxes are recomputed on window resize.

Tunable in `frontend/config.js`: `DETECTION_INTERVAL_MS` (1500),
`DETECTION_MAX_WIDTH` (960), `DETECTION_JPEG_QUALITY` (0.8).

| Interval | Effect |
|---|---|
| 200 ms | boxes track closely, backend ~55 % busy, laptop heats up |
| **1500 ms** | **backend ~93 % idle, boxes feel current** |
| 5 s | almost free, but boxes visibly lag |

Mining sites do not change in 200 ms.

---

## Detection history

Events are persisted to **SQLite** at `backend/data/detections.db` — part
of the Python standard library, so this adds **no new dependencies** and
no server to run. History survives a backend restart.

**Events, not frames.** Only cycles that actually detected something are
stored, and consecutive identical results within 10 s count as one
ongoing event. Standing in front of the camera for a minute produces one
row, not forty. **Raw frames are never stored** — we keep what the model
concluded, not the pixels, which is a privacy decision as much as a disk
one. Capped at 5000 events (`GALAMSEY_HISTORY_MAX`), oldest trimmed.

---

## Reproducing everything from scratch

```bash
# environment
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-dev.txt

# dataset  (reads the two Roboflow sources, writes dataset-phase5/)
./.venv/bin/python tools/recover_dataset.py --report-only   # inspect
./.venv/bin/python tools/recover_dataset.py                 # build
./.venv/bin/python tools/validate_dataset.py dataset-phase5

# train  (~33 min on an M4 Pro)
./.venv/bin/python tools/train_phase5.py

# evaluate and compare against the Run 6 baseline
./.venv/bin/python tools/compare_models.py test
./.venv/bin/python tools/compare_models.py val

# run it
./tools/run_dev.sh          # backend :5004 + frontend :5501

# test
./.venv/bin/python -m pytest tests/ -v
```

---

## Measured performance

Apple M4 Pro, MPS. No optimisation was needed:

| | |
|---|---|
| model load + warm-up | 1300 ms (one-off at startup) |
| inference, cold | 284 ms |
| **inference, warm** | **18.7 ms median** (20.0 ms mean) |
| JPEG encode per frame | 0.5 ms |
| DB insert per event | 0.37 ms |
| DB read, 50 events | 0.56 ms |
| peak memory | 560 MB |

At the 1500 ms interval the backend is busy **1.3 %** of the time.
(An end-to-end `/api/detect` round trip is ~30–110 ms; the extra over
19 ms is HTTP and multipart overhead, not the model.)

---

## Testing

```bash
./.venv/bin/python -m pytest tests/ -v
```

104 tests, ~8 seconds, no running server required (Flask's test client
exercises the app in-process). Each test guards a bug this project
actually had.

| Group | Covers |
|---|---|
| A Environment | venv in use, Python version, native imports, torch actually computes |
| B Dataset | paths resolve, names match, validator passes **and correctly fails** on a broken dataset |
| C Taxonomy | ids frozen, typo handling, run-8 rejection, frontend↔backend agreement |
| D Backend API | every endpoint, valid and invalid inputs, synthetic flagging |
| E Camera | healthy with no camera; `/video_feed` degrades to 503, never 500 |
| F Frontend | no API key in tracked files, balanced HTML, CSS fixes intact |
| G Phase 5 | dataset validates, no split leakage, manifest provenance, all labels are boxes, model loads without the fork, embedded names match exactly |
| H Detect API | real inference, contract fields, boxes inside the image, confidence is a probability, taxonomy match, all failure modes are 400 not 500, 503 when the model is unavailable, legacy alias unchanged |
| I History | insert, read back, survives reopening the database, empty frames skipped, de-duplication, retention cap, aggregates |
| J Limits | body-size cap, decompression bomb rejected on decoded size, 413-not-500, filename reduction, stale-clearing / in-flight / cover-scaling logic present |

Manual checks:

```bash
./.venv/bin/python tools/check_imports.py     # dependency diagnostic
./.venv/bin/python tools/check_camera.py      # macOS camera permissions
./.venv/bin/python tools/validate_dataset.py  # dataset report
```

---

## Configuration reference

Everything is overridable by environment variable — deployment never
means editing source.

| Variable | Default | Meaning |
|---|---|---|
| `GALAMSEY_DETECTION_MODE` | `auto` | `auto` / `model` / `mock` |
| `GALAMSEY_MODEL` | `backend/models/galamsey-phase5.pt` | weights to load |
| `GALAMSEY_YOLO_IMPL` | `pip` | `pip`, or `vendored` for Run 6/8 |
| `GOOGLE_MAPS_API_KEY` | *(from `.env`)* | served to the browser at `/client-config` |
| `GALAMSEY_DEVICE` | `auto` | `auto` / `mps` / `cpu` / `cuda` |
| `GALAMSEY_CONF` | `0.25` | confidence threshold |
| `GALAMSEY_IOU` | `0.45` | NMS IoU threshold |
| `GALAMSEY_IMGSZ` | `640` | inference image size |
| `GALAMSEY_HOST` | `127.0.0.1` | bind address |
| `GALAMSEY_PORT` | `5004` | backend port |
| `GALAMSEY_DEBUG` | `0` | `1` enables the Flask debugger — **trusted machines only** |
| `GALAMSEY_CORS_ORIGINS` | `http://localhost:5501,http://127.0.0.1:5501` | allowed browser origins |
| `GALAMSEY_CAMERA` | `1` | `0` disables the camera entirely |
| `GALAMSEY_CAMERA_INDEX` | `0` | which camera |
| `GALAMSEY_HISTORY` | `100` | default page size for `/detections` |
| `GALAMSEY_HISTORY_DB` | `backend/data/detections.db` | history database path |
| `GALAMSEY_HISTORY_MAX` | `5000` | max stored events before trimming |
| `GALAMSEY_MAX_UPLOAD` | `16777216` | largest request body (16 MB) |
| `GALAMSEY_MAX_PIXELS` | `40000000` | largest **decoded** image, in pixels |

Frontend settings live in `frontend/config.js`. **Your Maps key goes in
`.env` at the project root** (git-ignored); the backend reads it and
serves it at `GET /client-config`, so no credential is ever stored in a
frontend file. `frontend/config.local.js` remains supported as an
optional fallback for running the frontend without the backend.

### About the Google Maps key

**A browser Maps key cannot be kept secret.** It is delivered to every
visitor's browser; anyone can read it with View Source. Moving it to
`config.local.js` keeps it out of *Git*, which is worth doing, but it is
not encryption.

Real protection is in Google Cloud Console:

1. **Application restrictions → HTTP referrers**: `http://localhost:5501/*`
2. **API restrictions**: Maps JavaScript, Geocoding, Places only
3. **Billing → Budgets & alerts**: a low daily cap

> ⚠️ **Rotate the existing key before pushing this repo anywhere.** It
> is present in the first commit (`48bff44`, the untouched inherited
> baseline), and Git history is permanent — deleting a secret in a later
> commit does not remove it from an earlier one. Bots scrape GitHub for
> `AIza…` patterns within minutes.

---

## Current limitations

**Known and accepted for now:**

1. **`Structure` is unsolved.** 10 training images, 3 validation images,
   1 test image. No amount of retraining fixes that; it needs data.
2. **`MiningTool` is weak** (0.27 recall) — 41 training images, and the
   recovered export does not annotate the class at all.
3. **Class imbalance remains** — 37.8× by object count (improved from
   61.5×), but 22.4× by image coverage, which is *worse* than before
   because the recovered images grew four classes and not the other two.
4. **The model is a nano-scale detector on 399 training images.**
   mAP50 0.503 is a real, usable result for this data scale, not a
   production-grade detector.
5. **Geofence *creation* is genuinely broken, and not faked.** Google
   removed `google.maps.drawing.DrawingManager` in Maps JavaScript API
   v3.65. Saved geofences still render, and clicking the geofence button
   shows an honest explanation rather than silently doing nothing. It has
   **not** been replaced with a stub that pretends to work. Restoring it
   needs a replacement drawing implementation, which is deliberately out
   of scope: the detection pipeline was the priority, and a fake
   geofence would be worse than an absent one because alerts would
   appear to have location context they do not have.
6. **Detection history is in memory only** — it is lost on restart.
7. **No live inference on the webcam stream in the dashboard.** The
   video panel shows the raw webcam; `/video_feed` is the annotated one.
8. **No WebSockets.** `frontend/DETECTION_API.md` describes a WebSocket
   contract; nothing in Phase 4 needs push, and polling `/detections` is
   sufficient. Not built on speculation.
9. **Drone hardware is not integrated.** The Settings dialog still shows
   drone fields; they are inert.
10. **The vendored fork is still required for the two historical
    models** (Run 6, Run 8). The production model does not need it.

**Deferred to Phase 6:**

- **Annotate more `Structure` and `MiningTool`.** These are now the only
  classes holding the model back, and the fix is data, not code.
- **Consider splitting `ProcessingEquipment`.** Four source classes
  (`chanfine machine`, `galamsey boat`, `washing machine`, `water pump`)
  were merged into it. With more data they could be separate classes.
  See `docs/TAXONOMY_MAPPING.md` §4.1.
- **Try a larger backbone** (`yolo12s`) now that the dataset supports it.
- **Persist detection history** to `backend/data/`.
- **Replace or remove geofence drawing** — Google deleted the library.

---

## Common errors and fixes

**`library load disallowed by system policy` / `torch` will not import**
The files are quarantined by macOS (they arrived via WhatsApp/AirDrop).
Rebuild the environment rather than reusing `backend/venv`:
```bash
rm -rf .venv && python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

**`AttributeError: 'AAttn' object has no attribute 'qkv'`**
The weights were built by the vendored fork and you are running them
under pip Ultralytics. Set `GALAMSEY_YOLO_IMPL=vendored` (the default).
See `docs/ARCHITECTURE.md`.

**`AttributeError: module 'numpy' has no attribute 'trapz'`**
You are evaluating a fork-built model (Run 6/8) with NumPy 2. The shim in
`tools/evaluate_model.py` handles it; if you hit it elsewhere, alias
`np.trapz = np.trapezoid` before importing the fork.

**Model refused: "class names do not match"**
Working as designed — the weights use a different taxonomy. Check the
model's own names against `backend/taxonomy.py`. Do not "fix" it by
loosening the check.

**Blank or broken page in the browser**
You almost certainly opened `index.html` from disk. Serve it:
`./tools/serve_frontend.sh`, then <http://localhost:5501>.

**Map area is grey / "Google Maps API Error"**
No `.env` with `GOOGLE_MAPS_API_KEY`, or the backend is not running (the
key is served at `/client-config`), or the key lacks Maps JavaScript API,
or billing is off, or the referrer restriction excludes `localhost:5501`.
Everything except the map still works.

**`Address already in use` on 5004**
```bash
lsof -nP -iTCP:5004 -sTCP:LISTEN     # find it
GALAMSEY_PORT=5005 ./.venv/bin/python backend/app.py   # or move
```
If you try port 5000 on macOS you will collide with AirPlay Receiver.

**CORS error in the console**
The frontend must be on port 5501, or add its origin to
`GALAMSEY_CORS_ORIGINS`.

**Camera not found**
macOS: System Settings → Privacy & Security → Camera, enable your
terminal, then restart it. Diagnose with
`./.venv/bin/python tools/check_camera.py`. The backend runs fine
without a camera.

**`train: ../train/images ... does not exist`**
The Roboflow `../` path bug. Paths must be relative to `data.yaml`'s own
folder. Verify with `tools/validate_dataset.py`.

---

## Assumptions made

Documented so they can be challenged:

1. **`dataset-phase5` (536 photographs) is authoritative.** It combines
   Dataset A with 449 recovered photographs. The contaminated 657-file
   folder is still rejected: 72 % of its training files use a
   conflicting class schema.
2. **The 182-file "clean subset" adds nothing** and was excluded. Verified
   by perceptual hashing to be the *same 61 photographs* as Dataset A,
   exported three times each with augmentation.
3. **The recovered taxonomy mapping is correct.** Established from the
   source export's own `data.yaml`, corroborated by matching export
   timestamps, exact image count, byte-identical files, and visual
   verification of all seven classes. See `docs/TAXONOMY_MAPPING.md`.
4. **Merging four source classes into `ProcessingEquipment` is
   acceptable.** All four are ore-processing or water-handling
   equipment, matching that class's pre-existing definition. Recorded as
   a possible future split.
5. **Run 8 is still not used** despite its dataset being found again in
   Phase 5 — 12 classes for a 6-class project, six of them never
   predicted or validated.
6. **`ProcessingEquipment` is now spelled correctly** everywhere; Run 6's
   old spelling is accepted through one documented alias.
7. **The drone at `192.168.1.100:8080` does not exist.** The frontend
   was written against a device that was never built; it now talks to
   the local backend instead.
8. **Illegality is not decided by the model.** No class proves illegality
   from pixels — legal sites contain excavators too. The backend reports
   *evidence*; the legal/illegal judgement needs geofence context and is
   ultimately a human decision.
9. **A single environment serves backend, tools and training.**
   PyTorch is large; duplicating it would cost gigabytes for no benefit.

---

## Licences

- Dataset: CC BY 4.0 (Roboflow)
- `yolov12/` and Ultralytics: **AGPL-3.0** — note this if you ever
  distribute the project
