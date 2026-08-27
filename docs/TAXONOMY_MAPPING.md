# Taxonomy mapping — `galamsey-ev0ma` v2 → project 6-class taxonomy

Provenance record for the 475 photographs recovered in Phase 5.
**Every mapping below is justified by documented evidence, not by
guessing from a handful of images.**

---

## 1. Source identification

The 475 files that contaminated Dataset B were traced to a pristine,
untouched export still present on this machine:

```
/Users/kwadwo/Downloads/galamsey.v2i.yolov12/
```

| Evidence | Value |
|---|---|
| Roboflow workspace | `galamsey-zutez` — **the same workspace as Dataset A** |
| Roboflow project | `galamsey-ev0ma`, version 2 |
| URL | https://universe.roboflow.com/galamsey-zutez/galamsey-ev0ma/dataset/2 |
| Licence | CC BY 4.0 |
| Export timestamp (README) | **2025-11-20 5:13 pm GMT** |
| mtime of the 475 contaminated files | **2025-11-20 17:13:46** |
| README image count | **"The dataset includes 475 images."** |
| Byte comparison vs contaminated copy | **identical** — 475/475 images and 475/475 labels |

Four independent signals agree: workspace, timestamp to the second,
exact file count, and byte-level equality. The identification is
certain.

**Consequence:** the recovery pipeline reads the *pristine* export and
never touches the contaminated Dataset B copy.

### Declared source taxonomy

From the export's own `data.yaml` — this is the authoritative record,
not an inference:

```yaml
nc: 7
names: ['PERSON', 'chanfine machine', 'excavator', 'galamsey boat',
        'truck', 'washing machine', 'water pump']
```

### Independent visual verification

Every class was sampled and inspected. All seven declared names match
what the pixels show:

| id | declared name | verified content | objects | images |
|---|---|---|---|---|
| 0 | `PERSON` | people at mining sites | 518 | 81 |
| 1 | `chanfine machine` | small diesel pump/wash engines | 36 | 19 |
| 2 | `excavator` | tracked excavators | 331 | 301 |
| 3 | `galamsey boat` | floating river dredging platforms | 68 | 28 |
| 4 | `truck` | tipper and haul trucks | 153 | 121 |
| 5 | `washing machine` | sluice boxes / wash plants | 79 | 56 |
| 6 | `water pump` | pump rigs | 5 | 5 |
| | | **total** | **1,190** | **475** |

> A *chanfan* (locally "chanfine") is the small diesel engine that
> drives water pumps and washing rigs on Ghanaian galamsey sites. A
> *galamsey boat* is a floating dredging and washing platform used in
> rivers.

---

## 2. Target taxonomy

The project's six classes, defined in `backend/taxonomy.py`. Class 3's
written definition is decisive for several mappings below:

| id | name | definition |
|---|---|---|
| 0 | `ExcavationMachinery` | Excavators, bulldozers and other earth-moving machinery |
| 1 | `MiningTool` | Hand tools and small mining implements: shovels, pans, sluices |
| 2 | `Person` | People present at the site |
| 3 | `ProcessingEquipment` | **Washing plants, sluice boxes, dredges and pumping rigs** |
| 4 | `Structure` | Buildings, shelters and site structures |
| 5 | `Vehicle` | **Road** vehicles: trucks, cars, motorbikes |

---

## 3. The mapping

| src id | source name | → | tgt id | target name | evidence / reasoning | confidence |
|---|---|---|---|---|---|---|
| 0 | `PERSON` | → | **2** | `Person` | Identical concept. Visual check: people. | **HIGH** |
| 1 | `chanfine machine` | → | **3** | `ProcessingEquipment` | A chanfan is the diesel engine driving pumps and wash rigs. Class 3's definition explicitly covers "pumping rigs". Visual check: pump engines. | **HIGH** |
| 2 | `excavator` | → | **0** | `ExcavationMachinery` | Identical concept; class 0's definition names excavators first. Visual check: tracked excavators. | **HIGH** |
| 3 | `galamsey boat` | → | **3** | `ProcessingEquipment` | A floating **dredge**. Class 3's definition explicitly includes "dredges". Deliberately NOT class 5, whose definition is restricted to *road* vehicles. See §4. | **MEDIUM-HIGH** |
| 4 | `truck` | → | **5** | `Vehicle` | Class 5's definition names trucks first. Visual check: tipper/haul trucks. | **HIGH** |
| 5 | `washing machine` | → | **3** | `ProcessingEquipment` | A wash plant / sluice. Class 3's definition names "washing plants, sluice boxes" first. | **HIGH** |
| 6 | `water pump` | → | **3** | `ProcessingEquipment` | Class 3's definition explicitly covers "pumping rigs". | **HIGH** |

**No source class is forced into an unrelated target class.** Every
mapping is supported by the target class's own written definition.

---

## 4. Explicit taxonomy decisions

### 4.1 Four source classes collapse into `ProcessingEquipment`

`chanfine machine` (36), `galamsey boat` (68), `washing machine` (79)
and `water pump` (5) all map to class 3 — **188 objects merged into one
class**.

This loses real distinctions the source project cared about. It is
accepted deliberately because:

- All four are ore-processing or water-handling equipment, which is
  exactly what class 3 was defined to mean *before* this dataset was
  discovered — the definition was not bent to fit.
- Keeping the project at six classes preserves compatibility with
  Dataset A and keeps the Run 6 comparison meaningful. Changing the
  class count would make the baseline comparison impossible.
- Three of the four are individually tiny (36, 79, 5 objects). Trained
  as separate classes they would almost certainly learn nothing, exactly
  as `Structure` and `Vehicle` failed to in Run 6.

**Recorded as a future taxonomy extension:** if the dataset grows, split
class 3 into `WashPlant`, `Dredge` and `Pump`. That is a Phase 6+
decision requiring a full retrain, and it is noted here so the
information is not lost.

### 4.2 `galamsey boat` is not a `Vehicle`

A boat is a vehicle in ordinary English. It is **not** class 5 here
because class 5's definition says *road* vehicles, and because the
object's function on a galamsey site is dredging and washing ore, not
transport. Classifying it with trucks would mix "moving things around"
with "extracting gold", which is the distinction the taxonomy exists to
make. Marked MEDIUM-HIGH rather than HIGH because this is a judgement
call, and it is the one mapping most worth revisiting.

### 4.3 Two target classes receive nothing

No source class maps to `MiningTool` (1) or `Structure` (4) — the
`galamsey-ev0ma` project simply did not annotate them.

This is not an error, but it has a real consequence: adding 475 images
that contain **zero** `MiningTool` and `Structure` instances makes those
two classes *rarer in proportion*, even though their absolute counts are
unchanged. Quantified in Step 8 of the Phase 5 report.

It also means those 475 images contain unannotated hand tools and
structures, which the model will see as background. This is a known
limitation of merging datasets with different annotation scope, and it
is recorded rather than hidden.

---

## 5. Annotation format conversion

Source labels are **polygons**:

```
class_id  x1 y1  x2 y2  x3 y3  x4 y4  x5 y5 ...
```

The project uses **YOLO bounding boxes**:

```
class_id  x_center  y_center  width  height
```

Conversion takes the axis-aligned bounding box of the polygon:

```
x_min = min(xs)   x_max = max(xs)
y_min = min(ys)   y_max = max(ys)
cx = (x_min + x_max) / 2      w = x_max - x_min
cy = (y_min + y_max) / 2      h = y_max - y_min
```

This is the same transformation Ultralytics applies internally via
`segments2boxes`. Doing it explicitly means the stored labels are
honest about what they are, and the dataset validator can enforce a
single format.

Rows already in 5-field box format pass through unchanged.

---

## 6. What was rejected, and why

The recovery pipeline rejects rather than repairs. Counts are in the
Phase 5 report and in `tools/recover_dataset.py`'s output:

| Rejection | Reason |
|---|---|
| class id outside 0–6 | not present in the declared source taxonomy |
| fewer than 3 polygon points | cannot form an area |
| coordinates outside 0–1 | not normalised; likely pixel values |
| zero-area box after conversion | produces NaN loss |
| unreadable image | would fail mid-training |
| perceptual duplicate of a Dataset A image | would cause train/test leakage |

---

## 7. Reproducing this

```bash
./.venv/bin/python tools/recover_dataset.py --report-only   # inspect, write nothing
./.venv/bin/python tools/recover_dataset.py                 # build dataset-phase5/
./.venv/bin/python tools/validate_dataset.py dataset-phase5
```

Nothing in `/Users/kwadwo/Downloads/galamsey.v2i.yolov12/`,
`illegal mining activities/`, or the contaminated Dataset B copy is
modified. The pipeline only reads.
