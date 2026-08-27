# Galamsey AI Detector

A computer-vision system for detecting illegal small-scale gold mining
(*galamsey*) activity in Ghana from camera or drone imagery, with a web
dashboard that plots detections on a map against geofenced areas.

> **Project status: working end-to-end pipeline, weak model.**
> The backend, frontend, dataset and inference path all work and are
> tested. The detection model itself is an early baseline with
> **mAP50 ≈ 0.26** — it finds some objects and misses many. Improving it
> is Phase 5. See [Current limitations](#current-limitations).

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
   │   frontend/config.js       (committed, no secrets)           │
   │   frontend/config.local.js (git-ignored, holds Maps key)     │
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
   │   backend/models/galamsey-run6-baseline.pt   (6 classes)     │
   └───────┬──────────────────────────────────────────────────────┘
           │ imports YOLO from
           ▼
   ┌──────────────────────────┐   ┌──────────────────────────────┐
   │ yolov12/                 │   │ illegal mining activities/   │
   │ vendored Ultralytics     │   │ 87 images, 6 classes         │
   │ 8.3.63 fork (YOLOv12)    │   │ train 61 / valid 17 / test 9 │
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
│       ├── galamsey-run6-baseline.pt   the baseline (6 classes)
│       ├── PROVENANCE.md               where the weights came from
│       └── archive/                    run 8 + training metrics
│
├── illegal mining activities/   the dataset (authoritative)
│   ├── data.yaml
│   └── train/ valid/ test/      each with images/ and labels/
│
├── yolov12/                     vendored upstream fork — DO NOT EDIT
│
├── tools/
│   ├── validate_dataset.py      structural dataset checks
│   ├── serve_frontend.sh        static server on :5501
│   ├── run_dev.sh               start backend + frontend together
│   ├── check_camera.py          macOS camera diagnostic
│   └── check_imports.py         dependency diagnostic
│
├── tests/test_phase4.py         44 tests
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
cp frontend/config.local.example.js frontend/config.local.js
#   then edit frontend/config.local.js and paste your key
```

Verify:

```bash
./.venv/bin/python -c "import torch, cv2, ultralytics, flask; print('ok')"
./.venv/bin/python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
./.venv/bin/python tools/validate_dataset.py
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

`illegal mining activities/` — a Roboflow export, CC BY 4.0.

| Split | Images | Objects |
|---|---|---|
| train | 61 | 1,190 |
| valid | 17 | 324 |
| test | 9 | 115 |
| **Total** | **87** | **1,629** |

**Six classes.** The number in the label file is what matters; the name
is only a label humans read.

| id | name in `data.yaml` | shown in the UI | share of objects |
|---|---|---|---|
| 0 | `ExcavationMachinery` | Excavation Machinery | 6.7 % |
| 1 | `MiningTool` | Mining Tool | 24.0 % |
| 2 | `Person` | Person | 60.4 % |
| 3 | `ProcessingEquiment` ⚠️ | Processing Equipment | 5.6 % |
| 4 | `Structure` | Structure | 2.3 % |
| 5 | `Vehicle` | Vehicle | 1.0 % |

⚠️ **`ProcessingEquiment` is misspelled in the source data** (missing
`p`). We deliberately do **not** fix it in `data.yaml`, because the same
spelling is baked into the trained weights — changing one without
retraining the other would desynchronise them. It is corrected for
display only, in `backend/taxonomy.py`. Fix it at source at the next
retrain.

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

**YOLOv12-nano**, fine-tuned from COCO-pretrained `yolov12n.pt`.
2,520,834 parameters. Full history in
[`backend/models/PROVENANCE.md`](backend/models/PROVENANCE.md).

| | `galamsey-run6-baseline.pt` |
|---|---|
| Classes | 6, matching `taxonomy.py` exactly |
| Trained | 100 epochs, batch 16, imgsz 640, 2.3 h |
| Precision / Recall | 0.506 / 0.255 |
| mAP50 / mAP50-95 | **0.255** / 0.118 |
| Reproducible | yes |

**Per-class recall:** ExcavationMachinery 0.08, MiningTool 0.39,
Person 0.48, ProcessingEquiment 0.22, Structure 0.00, Vehicle 0.00.

**Be honest about this in your write-up.** The model misses 51–100 % of
objects depending on class and never detects `Structure` or `Vehicle`.
It proves the pipeline works; it is not a deployable detector.

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
| GET | `/health` | server, model, camera and taxonomy status |
| GET | `/statistics` | detection counts this session |
| GET | `/detections?limit=N` | recent detection history |
| POST | `/api/detections` | submit a detection record (frontend uses this) |
| POST | `/detect` | run detection on one uploaded image |
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
curl -F image=@"illegal mining activities/test/images/"*.jpg http://127.0.0.1:5004/detect
```

---

## Testing

```bash
./.venv/bin/python -m pytest tests/ -v
```

44 tests, ~4 seconds, no running server required (Flask's test client
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
| `GALAMSEY_MODEL` | `backend/models/galamsey-run6-baseline.pt` | weights to load |
| `GALAMSEY_YOLO_IMPL` | `vendored` | `vendored` (yolov12/) or `pip` |
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
| `GALAMSEY_HISTORY` | `100` | detections kept in memory |

Frontend settings live in `frontend/config.js`; your Maps key goes in
`frontend/config.local.js` (git-ignored).

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

1. **The model is weak.** mAP50 0.26. `Structure` and `Vehicle` are
   never detected. 61 training images is not enough.
2. **Severe class imbalance** — 61.5× between `Person` (984 objects) and
   `Vehicle` (16).
3. **Geofence *drawing* is unavailable.** Google removed
   `google.maps.drawing.DrawingManager` in Maps JS API v3.65. Saved
   geofences still display; creating new ones needs a replacement
   implementation. The failure is isolated so it no longer breaks the
   map.
4. **The `ProcessingEquiment` typo** persists in the data and weights by
   design (see above).
5. **Detection history is in memory only** — it is lost on restart.
6. **No live inference on the webcam stream in the dashboard.** The
   video panel shows the raw webcam; `/video_feed` is the annotated one.
7. **No WebSockets.** `frontend/DETECTION_API.md` describes a WebSocket
   contract; nothing in Phase 4 needs push, and polling `/detections` is
   sufficient. Not built on speculation.
8. **Drone hardware is not integrated.** The Settings dialog still shows
   drone fields; they are inert.
9. **The vendored-fork constraint** — see `docs/ARCHITECTURE.md`.

**Deliberately deferred to Phase 5:**

- Retrain on more data. A neighbouring folder
  (`~/Downloads/yolov12/yolov12/dataset`) contains **403 photographs not
  present in this dataset**. They are currently unusable because their
  class ids follow a different, conflicting schema — a *label* problem,
  not an *image* problem. Recovering them would multiply the real
  training data by roughly seven and is the single highest-value
  improvement available.

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

**Model refused: "class names do not match"**
Working as designed — the weights use a different taxonomy. Check the
model's own names against `backend/taxonomy.py`. Do not "fix" it by
loosening the check.

**Blank or broken page in the browser**
You almost certainly opened `index.html` from disk. Serve it:
`./tools/serve_frontend.sh`, then <http://localhost:5501>.

**Map area is grey / "Google Maps API Error"**
No `frontend/config.local.js`, or the key lacks Maps JavaScript API, or
billing is off, or the referrer restriction excludes `localhost:5501`.
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

1. **Dataset A (`illegal mining activities/`, 87 images) is
   authoritative.** A larger neighbouring dataset was investigated and
   rejected: 72 % of its training files use a conflicting class schema
   (its class 0 is people, class 2 is excavators — the reverse of ours).
2. **The 182-file "clean subset" of that dataset adds nothing.** It was
   verified by perceptual hashing to be the *same 61 photographs*,
   exported three times each with augmentation. Merging it would inflate
   the file count without adding information and risks split leakage.
3. **Run 6 is the baseline, not run 8**, despite run 8 scoring higher.
   Run 8's training dataset has been deleted, so it is unreproducible,
   and it has 12 classes of which 6 are never used.
4. **`ProcessingEquiment` stays misspelled** in data and weights until a
   retrain corrects both together.
5. **The drone at `192.168.1.100:8080` does not exist.** The frontend
   was written against a device that was never built; it now talks to
   the local backend instead.
6. **Illegality is not decided by the model.** No class proves illegality
   from pixels — legal sites contain excavators too. The backend reports
   *evidence*; the legal/illegal judgement needs geofence context and is
   ultimately a human decision.
7. **A single environment serves backend, tools and future training.**
   PyTorch is large; duplicating it would cost gigabytes for no benefit.

---

## Licences

- Dataset: CC BY 4.0 (Roboflow)
- `yolov12/` and Ultralytics: **AGPL-3.0** — note this if you ever
  distribute the project
