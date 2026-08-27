# Architecture notes

Design decisions and the reasoning behind them. The user-facing guide is
the root [`README.md`](../README.md); this document is for *why*.

---

## 1. Four backend files, one job each

The inherited `backend/app.py` was 448 lines doing four unrelated jobs:
HTTP routing, configuration, class names, and model loading. That is why
the port was defined in three places (5004, 5000, 5000), the class list
in three places (6, 11, 3 classes), and the model path in two — each
disagreeing with the others.

| File | Owns | Must never contain |
|---|---|---|
| `config.py` | paths, ports, thresholds, modes | class names, HTTP |
| `taxonomy.py` | the six classes | paths, HTTP |
| `detector.py` | model loading, inference, drawing | routes, paths |
| `app.py` | HTTP routes | class names, paths, inference |

The rule: **if a value appears in two files, one of them is wrong.**

---

## 2. Two Ultralytics implementations — the important constraint

This is the least obvious thing in the project.

There are **two** copies of Ultralytics available:

| | Version | Where | Used for |
|---|---|---|---|
| **Vendored fork** | 8.3.63 | `yolov12/ultralytics/` | **inference on the current weights** |
| **pip package** | 8.3.253 | `.venv/lib/.../ultralytics/` | Phase 5 retraining |

They are not interchangeable, because the YOLOv12 attention block was
implemented differently:

| | `AAttn` projections | `pe` conv kernel |
|---|---|---|
| Fork 8.3.63 | `self.qk` + `self.v` (split) | 5 |
| pip 8.3.217+ | `self.qkv` (fused) | 7 |

An Ultralytics `.pt` file pickles the **model object**, so when it is
loaded the class definitions come from whichever `ultralytics` is on
`sys.path`. Load a fork-built checkpoint under pip Ultralytics and it
appears to work — names, `nc` and parameter count all read back
correctly — then dies at the first forward pass:

```
AttributeError: 'AAttn' object has no attribute 'qkv'. Did you mean: 'qk'?
```

### How the choice is made

`detector._import_yolo()` is the **only** place `sys.path` is modified,
it is driven by `GALAMSEY_YOLO_IMPL`, and it **logs what it did**:

```
INFO detector: Ultralytics 8.3.63 loaded from .../yolov12/ultralytics (implementation=vendored)
```

This matters because the previous author hit the same mechanism by
accident: the training notebook ran `pip install ultralytics` but sat
*inside* `yolov12/`, so `import ultralytics` silently picked up the
local folder. The checkpoint records `8.3.63` even though the notebook
installed a newer version. Silent shadowing like that is a classic
"works on my machine" bug — so here the choice is explicit and reported.

**Phase 5 removes this constraint:** retraining with pip Ultralytics
produces upstream-compatible weights, after which
`GALAMSEY_YOLO_IMPL=pip` becomes the default and `yolov12/` reverts to
being reference source only.

---

## 3. Class ids are the contract; names are decoration

A detector outputs an integer. `"Person"` exists in exactly two places:
`data.yaml`, and — frozen at training time — inside the `.pt` file.

If two components disagree about what integer 2 means, **nothing
errors**. The system prints a confident, wrong label. This project has a
live example: a neighbouring dataset's class 0 is *people* and class 2
is *excavators* — exactly the reverse of ours. Training on it would
teach the model that excavators are people, with no error anywhere.

Defences:

1. `taxonomy.py` is the single source; nothing else defines class names.
2. `taxonomy.verify_against_model()` runs before any model is used and
   **refuses** it on mismatch (verified against run 8's 12-class
   weights).
3. `tools/validate_dataset.py` checks `data.yaml` names against
   `taxonomy.py`.
4. A test asserts `frontend/config.js` matches `backend/taxonomy.py`.

### Why the typo is preserved

`ProcessingEqui**ment**` is wrong, and correcting it in `data.yaml`
would take ten seconds. We don't, because the trained weights embed the
same misspelling. Fixing one without retraining the other silently
desynchronises them — the exact failure mode above. The correction lives
in `DISPLAY_NAMES`, applied at render time only.

**Storage keeps the source spelling; presentation fixes it.** That
separation is worth internalising: it applies to any field where an
upstream system owns the canonical value.

---

## 4. Mock mode is tagged aggressively, on purpose

The inherited `_generate_dummy_detections()` returned:

```json
{"class_name": "excavator", "confidence": 0.87, "is_illegal": true}
```

Structurally identical to real output. Nothing said it was invented. A
screenshot of that is indistinguishable from working AI — which in a
final-year project is an integrity problem, not merely a bug.

Now synthetic data is marked at five layers: per-detection `source`,
per-detection `is_synthetic`, per-response `synthetic` + `warning`, a
`MOCK ` name prefix, and a red banner burned into rendered frames.

Redundant by design. Any single tag could be dropped by a future edit;
all five is unlikely.

There is also a mode that refuses to fake anything: with
`GALAMSEY_DETECTION_MODE=model`, a model that will not load yields mode
`unavailable` and **zero** detections. Explicitly asking for real AI and
silently receiving placeholders is never acceptable.

---

## 5. The backend reports evidence, not verdicts

`taxonomy.MINING_ACTIVITY_WEIGHT` ranks how strongly each class
indicates industrial mining. It deliberately does **not** say
"illegal".

The inherited code had `_is_illegal_mining()`, which decided legality by
string-matching class names — so any detected excavator was "illegal".
That is wrong on the facts: legal concessions use excavators too, and a
person or a truck is not evidence of a crime.

Whether mining is illegal depends on **where** it is happening — inside
a licensed concession, or inside a forest reserve or river buffer. That
is geofence information the frontend holds, and it is ultimately a human
and legal judgement. A vision model cannot determine it from pixels.

So: the backend reports *what it saw*; the frontend combines that with
*where*; a person decides.

---

## 6. No frontend build system

Plain HTML/CSS/JS, no npm, no bundler, no framework. Deliberate.

A build system earns its place with module bundling, JSX/TypeScript
compilation, tree shaking, or dozens of dependencies. This app has three
script files and two CDN links. Adding Vite would introduce
`node_modules`, a lock file, a compile step between editing and seeing
the result, and a new category of breakage — for no benefit.

It would also not hide the Maps key. Anything the browser can read, the
user can read; bundling makes secrets harder to *find*, never actually
secret.

Configuration is split instead: `config.js` (committed, no secrets) and
`config.local.js` (git-ignored, machine-specific).

---

## 7. `file://` cannot work, and why

Browsers classify `file://` as an **insecure context**, which disables:

| API | Consequence |
|---|---|
| `navigator.mediaDevices.getUserMedia` | webcam button does nothing |
| `navigator.geolocation` | map location tracking blocked |
| `BroadcastChannel` across tabs | map pop-out cannot sync |
| `fetch` to the backend | blocked — `file://` has an opaque `null` origin no server can allow |

`http://localhost` **is** a secure context — browsers make that
exception specifically so local development works without HTTPS. Hence
`tools/serve_frontend.sh`.

Port 5501 was not arbitrary: the inherited `frontend/.vscode/settings.json`
set Live Server to 5501, so the original author served it too.

---

## 8. The camera is a peripheral, not a prerequisite

The inherited code called `initialize_camera()` at import time. On a
machine with no camera — or with permission denied, which is the default
on macOS — the server logged errors before serving a single request.

Now the camera opens lazily on the first `/video_feed` request. The
server starts, `/health` returns 200, and `/detect` runs real inference
on uploaded images with no camera anywhere. `/video_feed` returns
**503** (server fine, this resource unavailable) rather than 500 (server
broken), which is the semantically correct code and lets monitoring tell
the difference.

Supported inputs, in the order they were built:

1. no camera at all — full API except `/video_feed`
2. uploaded image via `POST /detect`
3. local webcam via `/video_feed`
4. *(future)* Raspberry Pi / drone feed — not implemented

---

## 9. Security decisions

| Was | Now | Why |
|---|---|---|
| `host='0.0.0.0'` | `127.0.0.1` | the API was reachable by every machine on the network |
| `debug=True` | `False` | the Werkzeug debugger is a remote code-execution console; its reloader also constructed the detector twice, loading the model twice and fighting over the camera |
| `CORS(app)` | explicit origin allow-list | any website could call this API from a visitor's browser |
| API key in HTML | git-ignored `config.local.js` | keeps it out of Git — see the caveat below |

**The key caveat:** a browser Maps key cannot be secret, and the
existing one is still in commit `48bff44`. Git history is permanent;
removing a secret later does not remove it from an earlier commit. The
only real remedy is **rotation in Google Cloud Console**, plus referrer
and API restrictions and a quota cap.

---

## 10. What must not be edited

| Path | Why |
|---|---|
| `yolov12/` | vendored upstream fork of `sunsmarterjie/yolov12`. Editing it means you can no longer diff against upstream or take fixes. If a project-specific change is genuinely needed, wrap it in our code instead. |
| `illegal mining activities/*/labels/` | ground truth. Changing labels invalidates every metric ever measured. |
| `backend/models/archive/` | historical evidence. |
| `~/Downloads/yolov12/` | the only record of prior training, including a verified clean dataset backup in `illegal_mining.zip`. |

---

## 11. Phase 5 — what comes next, and why in this order

1. **Recover the 403 images.** A neighbouring dataset holds 403
   photographs absent from ours, unusable only because their class ids
   follow a different schema. Establish the source project's true class
   list, derive a verified remapping, convert polygon annotations to
   boxes, de-duplicate by perceptual hash, then validate. This is a
   *label* problem, not an *image* problem — and it is worth roughly 7×
   the current training data.
2. **Retrain with pip Ultralytics**, which also removes the vendored-fork
   constraint (§2).
3. **Fix the `ProcessingEquiment` typo** at the same time, in data and
   weights together.
4. **Address class imbalance** — 61.5× between `Person` and `Vehicle`.
5. **Persist detection history** to `backend/data/`; it is in memory now.
6. **Decide on geofence drawing** — Google removed the Drawing library,
   so it needs a replacement or removal from the UI.

Order matters: retraining before recovering the data would waste the
run, and fixing the typo before retraining would desynchronise the
weights.
