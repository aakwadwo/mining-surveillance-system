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

This is the least obvious thing in the project. Phase 5 defused it,
but the mechanism is still worth understanding.

There are **two** copies of Ultralytics available:

| | Version | Where | Used for |
|---|---|---|---|
| **pip package** | 8.3.253 | `.venv/lib/.../ultralytics/` | **everything, since Phase 5** |
| **Vendored fork** | 8.3.63 | `yolov12/ultralytics/` | ONLY the historical Run 6 / Run 8 weights |

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

**Phase 5 removed this constraint.** `galamsey-phase5.pt` was trained
with pip Ultralytics, so `GALAMSEY_YOLO_IMPL` now defaults to `pip` and
the production path never touches the fork. `yolov12/` is retained
because the two historical checkpoints still need it — and because
deleting vendored upstream source you cannot re-derive is a bad trade.

One extra wrinkle when running the fork today: Ultralytics 8.3.63
computes average precision with `np.trapz`, which NumPy 2.0 removed.
Inference never touches it, which is why prediction works and only
`val()` fails. `tools/evaluate_model.py` aliases
`np.trapz = np.trapezoid` in OUR code before importing the fork. The
fork's source is not modified.

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

### The typo, and when it was safe to fix

`ProcessingEqui**ment**` was wrong, and correcting it in `data.yaml`
would have taken ten seconds. Phase 4 deliberately did **not**, because
the trained weights embedded the same misspelling — fixing one without
retraining the other silently desynchronises them, which is the exact
failure mode above.

Phase 5 retrains, which is precisely the moment the correction became
safe. It was applied in ONE commit across the taxonomy module, both
`data.yaml` files, the frontend config, the tests and the docs. Class
IDs never moved; only the string changed.

Run 6 still carries the old spelling and is retained as evidence, so
`LEGACY_NAME_ALIASES` lets `verify_against_model()` accept it. That
mechanism is strictly for **documented renames where the id and the
concept are unchanged** — a test asserts it cannot hide a genuine
mismatch (class 3 renamed to `WashPlant` is still rejected). The moment
an alias is used to silence a real disagreement, the guard is worthless.

**The general lesson:** when an upstream system owns a canonical value,
fix it at the moment you regenerate the derived artefacts, not before.

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
| `illegal mining activities/*/labels/` | ground truth for the Phase 4 baseline. Changing labels invalidates every metric ever measured. |
| `dataset-phase5/` | GENERATED. Edit `tools/recover_dataset.py` and rebuild; hand-edits will be silently overwritten. |
| `~/Downloads/galamsey.v2i.yolov12/` | the pristine source of the recovered 449 photographs. |
| `backend/models/archive/` | historical evidence. |
| `~/Downloads/yolov12/` | the only record of prior training, including a verified clean dataset backup in `illegal_mining.zip`. |

---

## 10b. Live detection: three layers, three owners

Phase 6 connected the camera to the model. The design rule is that each
layer has exactly one owner:

| Layer | Owner | Rate |
|---|---|---|
| the picture | the **browser** (`<video>`, hardware accelerated) | 30-60 fps |
| the inference | the **backend** (`/api/detect` -> YOLOv12) | ~1 per 1.5 s |
| the boxes | the **frontend** (absolutely-positioned divs) | on each result |

Sending frames to Python to have rectangles drawn on them would collapse
all three into one, make the video jerky, and produce boxes that cannot
be restyled, hidden or made accessible.

### The in-flight guard

`setInterval(send, 100)` posts faster than ~110 ms inference can drain.
The queue grows without limit, memory climbs, and the boxes drift
further behind the picture - a classic unbounded-producer bug. The loop
is therefore `capture -> infer -> render -> WAIT -> repeat` with an
explicit `inFlight` flag. Under load the system degrades by *slowing
down*, which is what you want, rather than by falling over.

### The coordinate trap

Three coordinate spaces, all different: the frame POSTed, the `<video>`
element's box, and the picture actually painted inside it. Because the
video is `object-fit: cover`, the painted picture is generally BIGGER
than the element and centred, with parts cropped off.

Naive scaling (`x * elementWidth / sourceWidth`) ignores the crop and is
wrong everywhere - but only slightly, so it looks plausible and survives
casual review. The correct transform uses `max` (fill) rather than `min`
(fit), plus a centring offset that goes negative when cropped.

This is worth remembering generally: a transform that is *almost* right
is harder to catch than one that is obviously broken.

### Detection is not a legal verdict

The overlay, the report and the alerts all read from the same response,
so they cannot disagree. But "Site Type" reports what the model SAW, and
alerts fire only on machinery/processing equipment above 0.5 confidence
- never on `Person` alone. A person at a mining site is not evidence of
a crime, and a system that implies otherwise is worse than one that says
nothing. See section 5.

---

## 10c. Persistence: SQLite, and why nothing bigger

`GET /detections` used to be an in-memory deque - a live buffer
pretending to be history. It is now SQLite at
`backend/data/detections.db`.

SQLite ships with Python: no new dependency, no server, one file. A
database server would add installation, configuration, running and
backup burden for a single-user dashboard holding a few thousand rows.
The moment to reach for one is when multiple writers on different
machines need the same data - not before.

Two policies keep the store meaningful rather than merely large:

  * **events, not frames** - only cycles that detected something, with
    consecutive identical results inside 10 s collapsed into one event
  * **conclusions, not pixels** - classes, confidences, boxes, timing
    and location are stored; camera frames never are. That is a privacy
    decision as much as a storage one.

Retention is capped at 5000 rows. An explicit, documented limit beats
unbounded growth that nobody notices until the disk is full.

---

## 11. What Phase 5 did, and what Phase 6 should do

### Done in Phase 5

1. **Recovered 449 photographs.** The contaminated files were traced to
   a pristine `galamsey-ev0ma` v2 export whose own `data.yaml` declared
   the 7-class source taxonomy — so the mapping rests on documented
   evidence, not on guessing from a few crops. Polygons were converted
   to boxes, duplicates removed by perceptual hash.
2. **Retrained with pip Ultralytics**, removing the fork constraint.
3. **Fixed the spelling** in the same change as the retrain.
4. **Measured honestly.** Test mAP50 0.200 → 0.503, every class up; but
   the two classes whose data did not grow (`MiningTool`, `Structure`)
   regressed on the validation split and are noise-dominated.

The controlled part matters: hyperparameters were held **identical** to
Run 6, so the dataset was the only variable. Changing both at once would
have made the improvement unattributable.

### Phase 6 (done)

Connected the camera to the model, rendered real boxes, drove the whole
dashboard from one inference result, and made history persistent. See
sections 10b and 10c.

### Phase 7, in order

1. **Annotate `Structure` and `MiningTool`.** They are the only classes
   still failing, they are the two the recovered export did not cover,
   and the fix is data, not code. `Structure` has 10 training images.
2. **Then** consider splitting `ProcessingEquipment` back into its four
   source concepts (§ docs/TAXONOMY_MAPPING.md 4.1) — only once there is
   enough data per concept to learn from.
3. **Then** try a larger backbone (`yolo12s`). Doing this before fixing
   the data would buy a smaller improvement for a much slower model.
4. Persist detection history; replace or remove geofence drawing.

Order still matters. Capacity cannot compensate for absent data, and
re-splitting a class before you have data for the split just recreates
the `Structure` problem four times over.
