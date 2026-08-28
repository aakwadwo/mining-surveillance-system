"""
Galamsey AI Detector — Phase 4 verification suite
=================================================

Run:
    ./.venv/bin/python -m pytest tests/ -v

Deliberately small. Every test here guards a bug this project actually
had — not a bug someone imagined. If a test fails, something we already
fixed once has come back.

Grouped as:
    A  environment
    B  dataset
    C  taxonomy
    D  backend API
    E  camera independence
    F  frontend static checks
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import config  # noqa: E402
import taxonomy  # noqa: E402


# ===============================================================
# A. Environment
# ===============================================================
class TestEnvironment:
    def test_running_inside_project_venv(self):
        """Guards against accidentally using conda base or the dead backend/venv."""
        assert "/.venv/" in sys.executable or sys.prefix.endswith(".venv"), (
            f"tests are running under {sys.executable}; "
            "use ./.venv/bin/python -m pytest"
        )

    def test_python_version(self):
        assert sys.version_info[:2] >= (3, 11), "project targets Python 3.11+"

    @pytest.mark.parametrize("module", ["torch", "torchvision", "cv2", "numpy", "flask", "flask_cors"])
    def test_imports(self, module):
        """The inherited backend/venv failed here: macOS quarantine blocked every native lib."""
        __import__(module)

    def test_torch_can_compute(self):
        """Importing torch is not enough; the native library must actually run."""
        import torch

        assert (torch.ones(2, 2) @ torch.ones(2, 2)).sum().item() == 8.0

    def test_accelerator_reported(self):
        import torch

        # Not asserting MPS is present (CI may be CPU-only), just that the
        # check works and the detector's device resolution is sane.
        assert isinstance(torch.backends.mps.is_available(), bool)


# ===============================================================
# B. Dataset
# ===============================================================
class TestDataset:
    def test_data_yaml_paths_resolve(self):
        """The inherited data.yaml used '../train/images' and pointed one level too high."""
        import yaml

        cfg = yaml.safe_load(config.DATASET_YAML.read_text())
        for key in ("train", "val", "test"):
            resolved = (config.DATASET_YAML.parent / cfg[key]).resolve()
            assert resolved.exists(), f"data.yaml {key}: '{cfg[key]}' -> {resolved} missing"

    def test_data_yaml_matches_taxonomy(self):
        import yaml

        cfg = yaml.safe_load(config.DATASET_YAML.read_text())
        assert cfg["nc"] == taxonomy.NUM_CLASSES
        assert list(cfg["names"]) == list(taxonomy.DATASET_NAMES)

    def test_validator_passes(self):
        """The full structural validator must give this dataset a clean bill of health."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tools" / "validate_dataset.py")],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "RESULT: PASSED" in result.stdout

    def test_validator_rejects_a_bad_dataset(self, tmp_path):
        """A validator that never fails is worthless. Build a broken dataset and check it is caught."""
        for split, folder in (("train", "train"), ("valid", "valid"), ("test", "test")):
            (tmp_path / folder / "images").mkdir(parents=True)
            (tmp_path / folder / "labels").mkdir(parents=True)
        # class id 9 does not exist in a 6-class dataset
        (tmp_path / "train" / "labels" / "a.txt").write_text("9 0.5 0.5 0.2 0.2\n")
        (tmp_path / "train" / "images" / "a.jpg").write_bytes(b"not really a jpeg")
        (tmp_path / "data.yaml").write_text(
            "train: train/images\nval: valid/images\ntest: test/images\n"
            f"nc: {taxonomy.NUM_CLASSES}\nnames: {list(taxonomy.DATASET_NAMES)}\n"
        )
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tools" / "validate_dataset.py"), str(tmp_path)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "class id 9 outside" in result.stdout


# ===============================================================
# C. Taxonomy
# ===============================================================
class TestTaxonomy:
    def test_six_classes_ids_frozen(self):
        assert taxonomy.NUM_CLASSES == 6
        assert taxonomy.DATASET_NAMES[0] == "ExcavationMachinery"
        assert taxonomy.DATASET_NAMES[2] == "Person"
        assert taxonomy.DATASET_NAMES[5] == "Vehicle"

    def test_class3_spelling_corrected_in_phase5(self):
        """
        Phase 4 preserved the upstream typo because the Run 6 weights carried
        it. Phase 5 retrains, so the fix was applied everywhere at once:
        taxonomy, both data.yaml files, frontend config, tests and docs.
        Ids did not move; only the string changed.
        """
        assert taxonomy.DATASET_NAMES[3] == "ProcessingEquipment"
        assert taxonomy.DISPLAY_NAMES[3] == "Processing Equipment"

    def test_run6_legacy_spelling_still_accepted(self):
        """
        Run 6 is historical evidence and must remain loadable. Its old
        spelling is a DOCUMENTED alias, not a silenced mismatch.
        """
        names = dict(enumerate(taxonomy.DATASET_NAMES))
        names[3] = "ProcessingEquiment"
        assert taxonomy.verify_against_model(names) == []

    def test_aliases_do_not_hide_real_mismatches(self):
        """An alias must never let a genuinely different taxonomy through."""
        names = dict(enumerate(taxonomy.DATASET_NAMES))
        names[3] = "WashPlant"
        assert taxonomy.verify_against_model(names)

    def test_verify_against_model_accepts_matching_names(self):
        assert taxonomy.verify_against_model(dict(enumerate(taxonomy.DATASET_NAMES))) == []

    def test_verify_against_model_rejects_run8_taxonomy(self):
        """
        This is THE guard. run 8's weights carry 12 classes with a different
        class 3 spelling. Without this check the system would happily print
        wrong labels while nothing errored.
        """
        run8 = {
            0: "ExcavationMachinery", 1: "MiningTool", 2: "Person",
            3: "ProcessingEquipment", 4: "Structure", 5: "Vehicle",
            6: "chanfine machine", 7: "excavator", 8: "galamsey boat",
            9: "truck", 10: "washing machine", 11: "water pump",
        }
        problems = taxonomy.verify_against_model(run8)
        assert problems, "run 8's 12-class taxonomy must be rejected"
        assert any("count mismatch" in p for p in problems)

    def test_frontend_config_matches_backend_taxonomy(self):
        """The frontend must not drift back into inventing its own class names."""
        text = (PROJECT_ROOT / "frontend" / "config.js").read_text()
        found = re.findall(r"\{\s*id:\s*(\d+),\s*name:\s*'([^']+)'", text)
        assert len(found) == taxonomy.NUM_CLASSES
        for cid, name in found:
            assert taxonomy.DATASET_NAMES[int(cid)] == name


# ===============================================================
# D. Backend API
# ===============================================================
@pytest.fixture(scope="module")
def client():
    """Flask's test client: exercises the real app without opening a socket."""
    import app as backend_app

    backend_app.app.config["TESTING"] = True
    with backend_app.app.test_client() as c:
        yield c


class TestBackendAPI:
    def test_config_is_valid(self):
        assert config.validate() == []

    def test_paths_are_absolute_and_cwd_independent(self):
        """The inherited code used relative paths and silently never found the model."""
        assert config.PROJECT_ROOT.is_absolute()
        assert config.MODEL_PATH.is_absolute()
        assert config.DATASET_YAML.exists()

    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.get_json()
        assert body["status"] == "healthy"
        assert body["detection"]["mode"] in ("model", "mock")
        assert body["taxonomy"]["num_classes"] == taxonomy.NUM_CLASSES

    def test_health_declares_whether_output_is_synthetic(self, client):
        """A response must always state whether it is real AI output."""
        body = client.get("/health").get_json()
        assert "synthetic" in body["detection"]
        if body["detection"]["synthetic"]:
            assert body["detection"]["warning"]

    def test_index_lists_endpoints(self, client):
        body = client.get("/").get_json()
        for route in ("GET /health", "POST /api/detections", "POST /detect"):
            assert route in body["endpoints"]

    def test_unknown_route_404s_with_json(self, client):
        r = client.get("/definitely-not-a-route")
        assert r.status_code == 404
        assert "error" in r.get_json()

    def test_statistics_shape(self, client):
        body = client.get("/statistics").get_json()
        assert set(body["detections_by_class"]) == set(taxonomy.DATASET_NAMES)

    def test_config_get_and_post(self, client):
        assert client.get("/config").status_code == 200
        assert client.post("/config", json={"threshold": 0.4}).status_code == 200
        assert client.post("/config", json={"threshold": 5}).status_code == 400
        assert client.post("/config", json={"threshold": "abc"}).status_code == 400
        assert client.post("/config", json={}).status_code == 400
        client.post("/config", json={"threshold": 0.25})   # restore

    def test_post_api_detections_accepts_frontend_contract(self, client):
        """frontend/DETECTION_API.md specifies this endpoint; it never existed."""
        r = client.post("/api/detections", json={
            "timestamp": "2026-01-01T00:00:00Z",
            "location": {"lat": 5.6037, "lng": -0.187},
            "detections": [{"type": "Person", "confidence": 0.9}],
        })
        assert r.status_code == 201
        assert r.get_json()["success"] is True

    @pytest.mark.parametrize("payload", [
        {"detections": "not a list"},
        {"detections": [], "location": {"lat": "x", "lng": 1}},
        [],
    ])
    def test_post_api_detections_rejects_bad_payloads(self, client, payload):
        assert client.post("/api/detections", json=payload).status_code == 400

    def test_detect_endpoint_runs_on_a_real_image(self, client):
        """Proves the inference path works with no camera involved."""
        images = sorted(
            p for p in (config.DATASET_DIR / "test" / "images").iterdir()
            if p.suffix.lower() == ".jpg" and not p.name.startswith(".")
        )
        assert images, "dataset test images missing"
        with images[0].open("rb") as fh:
            r = client.post("/detect", data={"image": (fh, images[0].name)},
                            content_type="multipart/form-data")
        assert r.status_code == 200
        body = r.get_json()
        assert "synthetic" in body and "detections" in body
        for d in body["detections"]:
            assert d["source"] in ("model", "mock")
            assert "is_synthetic" in d

    def test_detect_rejects_junk(self, client):
        r = client.post("/detect", data={"image": (__import__("io").BytesIO(b"nope"), "x.jpg")},
                        content_type="multipart/form-data")
        assert r.status_code == 400
        assert client.post("/detect", data={}, content_type="multipart/form-data").status_code == 400


# ===============================================================
# E. Camera independence
# ===============================================================
class TestCameraOptional:
    def test_backend_healthy_without_camera(self, client):
        """
        The inherited app grabbed the camera at import time, so on a machine
        with no camera it errored before serving anything. A camera is a
        peripheral, not a prerequisite.
        """
        body = client.get("/health").get_json()
        assert body["status"] == "healthy"
        assert body["camera"]["open"] is False

    def test_video_feed_degrades_gracefully(self, client):
        """
        With a camera: 200 and an MJPEG stream.
        Without one: 503 (server fine, this resource unavailable) plus a
        JSON explanation. NEVER 500 - the inherited code returned 500,
        which says "the server is broken" when it simply has no webcam.
        """
        r = client.get("/video_feed")
        assert r.status_code in (200, 503), f"unexpected {r.status_code}"
        if r.status_code == 503:
            assert "error" in r.get_json()
        else:
            assert "multipart/x-mixed-replace" in r.headers["Content-Type"]
        r.close()


# ===============================================================
# F. Frontend static checks
# ===============================================================
FRONTEND = PROJECT_ROOT / "frontend"


class TestFrontend:
    def test_expected_files_exist(self):
        for name in ("index.html", "script.js", "styles.css", "config.js",
                     "config.local.example.js", "map-popup.html", "map-popup.js"):
            assert (FRONTEND / name).exists(), f"missing frontend/{name}"

    def test_no_api_key_in_tracked_files(self):
        """Guards the secret from creeping back into a committed file."""
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.split()
        offenders = []
        for rel in tracked:
            p = PROJECT_ROOT / rel
            try:
                if re.search(r"AIza[A-Za-z0-9_\-]{20,}", p.read_text(errors="ignore")):
                    offenders.append(rel)
            except (OSError, UnicodeDecodeError):
                continue
        assert not offenders, f"Google API key found in tracked files: {offenders}"

    def test_config_local_is_git_ignored(self):
        r = subprocess.run(["git", "check-ignore", "frontend/config.local.js"],
                           cwd=PROJECT_ROOT, capture_output=True)
        assert r.returncode == 0, "frontend/config.local.js MUST be git-ignored"

    def test_index_html_tags_balanced(self):
        """#settingsModal was never closed, which swallowed the loading screen."""
        html = (FRONTEND / "index.html").read_text()
        html = re.sub(r"<!--.*?-->", "", html, flags=re.S)
        void = {"area", "base", "br", "col", "embed", "hr", "img", "input",
                "link", "meta", "param", "source", "track", "wbr"}
        stack: list[str] = []
        for m in re.finditer(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)([^>]*?)(/?)>", html):
            closing, tag, _, self_close = m.groups()
            tag = tag.lower()
            if tag in void or self_close == "/":
                continue
            if not closing:
                stack.append(tag)
            else:
                assert stack and stack[-1] == tag, f"unbalanced </{tag}>; open: {stack[-3:]}"
                stack.pop()
        assert not stack, f"unclosed tags: {stack}"

    def test_body_is_visible_without_javascript(self):
        """body{opacity:0} meant a JS failure produced a permanently blank page."""
        css = re.sub(r"/\*.*?\*/", "",
                     (FRONTEND / "styles.css").read_text(), flags=re.S)
        body_rule = re.search(r"^body \{(.*?)\}", css, re.S | re.M).group(1)
        opacity = re.search(r"opacity:\s*([\d.]+)", body_rule)
        assert opacity and float(opacity.group(1)) == 1.0

    def test_main_grid_has_no_fixed_oversized_column(self):
        """`70rem 1fr` overflowed every laptop between 1201px and ~1650px."""
        css = (FRONTEND / "styles.css").read_text()
        # Strip comments FIRST: our own explanatory comment contains
        # "body{overflow-x:hidden}", whose closing brace would otherwise
        # truncate the rule match.
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        rule = re.search(r"^\.main-content \{(.*?)\}", css, re.S | re.M).group(1)
        cols = re.search(r"grid-template-columns:\s*([^;]+);", rule).group(1)
        assert "70rem" not in cols
        assert "minmax" in cols

    def test_update_statistics_is_defined(self):
        """It was called twice and defined nowhere."""
        js = (FRONTEND / "script.js").read_text()
        assert re.search(r"^\s+async updateStatistics\(\)", js, re.M), \
            "updateStatistics() is called by startRealTimeUpdates but not defined"

    def test_no_hardcoded_drone_address_in_api_calls(self):
        """Detection traffic must go to the configured backend, not a phantom drone."""
        js = (FRONTEND / "script.js").read_text()
        assert "${this.settings.droneIP}:${this.settings.dronePort}/api/detections" not in js

    def test_footer_credits_the_right_model(self):
        assert "YOLOv8" not in (FRONTEND / "index.html").read_text()


# ===============================================================
# G. Phase 5 — recovered dataset and retrained model
# ===============================================================
PHASE5_DATASET = PROJECT_ROOT / "dataset-phase5"
PHASE5_MODEL = PROJECT_ROOT / "backend" / "models" / "galamsey-phase5.pt"


class TestPhase5Dataset:
    def test_dataset_exists(self):
        assert PHASE5_DATASET.is_dir(), "run tools/recover_dataset.py"
        assert (PHASE5_DATASET / "data.yaml").exists()
        assert (PHASE5_DATASET / "MANIFEST.json").exists()

    def test_validator_passes(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tools" / "validate_dataset.py"),
             str(PHASE5_DATASET)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout[-3000:]
        assert "RESULT: PASSED" in result.stdout

    def test_data_yaml_matches_taxonomy(self):
        import yaml

        cfg = yaml.safe_load((PHASE5_DATASET / "data.yaml").read_text())
        assert cfg["nc"] == taxonomy.NUM_CLASSES
        assert list(cfg["names"]) == list(taxonomy.DATASET_NAMES)

    def test_no_cross_split_leakage(self):
        """
        The single most damaging dataset bug. Verified here by the phash
        recorded in the manifest, independently of the build script.
        """
        manifest = json.loads((PHASE5_DATASET / "MANIFEST.json").read_text())
        split_of: dict[str, str] = {}
        leaks = []
        for entry in manifest["files"]:
            h = entry["phash"]
            if h in split_of and split_of[h] != entry["split"]:
                leaks.append(f"{entry['output_name']}: {split_of[h]} and {entry['split']}")
            split_of[h] = entry["split"]
        assert not leaks, f"cross-split leakage: {leaks[:5]}"

    def test_every_photograph_is_unique(self):
        manifest = json.loads((PHASE5_DATASET / "MANIFEST.json").read_text())
        hashes = [e["phash"] for e in manifest["files"]]
        assert len(hashes) == len(set(hashes)), "duplicate photographs in the dataset"

    def test_manifest_records_provenance(self):
        manifest = json.loads((PHASE5_DATASET / "MANIFEST.json").read_text())
        origins = {e["origin"] for e in manifest["files"]}
        assert origins == {"dataset-a", "galamsey-ev0ma-v2"}
        assert manifest["class_map_source_to_target"] == {
            "0": 2, "1": 3, "2": 0, "3": 3, "4": 5, "5": 3, "6": 3
        }

    def test_dataset_a_still_intact(self):
        """Phase 5 must not have modified the authoritative baseline."""
        a = PROJECT_ROOT / "illegal mining activities"
        counts = {s: len([p for p in (a / s / "images").iterdir()
                          if not p.name.startswith(".")]) for s in ("train", "valid", "test")}
        assert counts == {"train": 61, "valid": 17, "test": 9}

    def test_all_labels_are_boxes_not_polygons(self):
        """Source annotations were polygons; conversion must be complete."""
        for split in ("train", "valid", "test"):
            for lbl in (PHASE5_DATASET / split / "labels").iterdir():
                if lbl.suffix != ".txt":
                    continue
                for i, line in enumerate(lbl.read_text().splitlines(), 1):
                    if line.strip():
                        assert len(line.split()) == 5, f"{lbl.name}:{i} is not a 5-field box"


@pytest.mark.skipif(not PHASE5_MODEL.exists(), reason="Phase 5 model not yet trained")
class TestPhase5Model:
    def test_loads_without_the_vendored_fork(self):
        """
        The whole point of retraining with pip Ultralytics: the new weights
        must run under the maintained package, with no sys.path tricks.
        """
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys, os;"
             "from ultralytics import YOLO;"
             "import ultralytics;"
             "assert 'yolov12' not in os.path.dirname(ultralytics.__file__);"
             f"m = YOLO(r'{PHASE5_MODEL}');"
             "import numpy as np;"
             "r = m.predict(np.zeros((64,64,3), dtype='uint8'), verbose=False);"
             "print('OK', len(m.names))"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "OK 6" in result.stdout

    def test_embedded_names_match_taxonomy_exactly(self):
        """No legacy alias needed - the new model uses the corrected spelling."""
        result = subprocess.run(
            [sys.executable, "-c",
             f"from ultralytics import YOLO; import json;"
             f"print(json.dumps(dict(YOLO(r'{PHASE5_MODEL}').names)))"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        names = {int(k): v for k, v in json.loads(result.stdout.strip().splitlines()[-1]).items()}
        assert names == dict(enumerate(taxonomy.DATASET_NAMES))
        assert taxonomy.verify_against_model(names) == []


# ===============================================================
# H. Phase 6 — the real inference API (/api/detect)
# ===============================================================
def _a_real_test_image() -> Path:
    """A genuine photograph from the Phase 5 test split."""
    d = PROJECT_ROOT / "dataset-phase5" / "test" / "images"
    images = sorted(p for p in d.iterdir()
                    if p.suffix.lower() == ".jpg" and not p.name.startswith("."))
    assert images, "no test images found"
    return images[0]


@pytest.fixture(scope="module")
def detect_response(client):
    """
    One real inference, reused across assertions.

    Deliberately a REAL photograph through the REAL model - these tests
    are worthless if they assert against a fixture of canned numbers.
    """
    img = _a_real_test_image()
    with img.open("rb") as fh:
        r = client.post("/api/detect", data={"image": (fh, img.name)},
                        content_type="multipart/form-data")
    assert r.status_code == 200, r.get_data(as_text=True)[:500]
    return r.get_json(), img


class TestDetectAPI:
    def test_success_and_not_synthetic(self, detect_response):
        """The default path must run the real model, never mock data."""
        body, _ = detect_response
        assert body["success"] is True
        assert body["synthetic"] is False, "live detection must not be synthetic"
        assert "warning" not in body or body["warning"] is None

    def test_returns_detections(self, detect_response):
        body, _ = detect_response
        assert isinstance(body["detections"], list)
        assert body["detections"], "the model found nothing in a known-populated image"

    def test_every_detection_has_the_contract_fields(self, detect_response):
        body, _ = detect_response
        for d in body["detections"]:
            assert set(d) == {"class_id", "class_name", "confidence", "bbox"}

    def test_bbox_has_named_corners(self, detect_response):
        body, _ = detect_response
        for d in body["detections"]:
            assert set(d["bbox"]) == {"x1", "y1", "x2", "y2"}

    def test_image_dimensions_match_the_actual_file(self, detect_response):
        """Reported dimensions must describe the image that was sent."""
        body, img = detect_response
        from PIL import Image

        with Image.open(img) as im:
            width, height = im.size
        assert body["image_width"] == width
        assert body["image_height"] == height

    def test_boxes_lie_inside_the_image(self, detect_response):
        """A box outside the frame means the coordinate space is wrong."""
        body, _ = detect_response
        w, h = body["image_width"], body["image_height"]
        for d in body["detections"]:
            b = d["bbox"]
            assert 0 <= b["x1"] < b["x2"] <= w, f"bad x range {b} for width {w}"
            assert 0 <= b["y1"] < b["y2"] <= h, f"bad y range {b} for height {h}"

    def test_confidence_is_a_probability(self, detect_response):
        body, _ = detect_response
        for d in body["detections"]:
            assert 0.0 < d["confidence"] <= 1.0

    def test_confidence_respects_the_threshold(self, detect_response):
        body, _ = detect_response
        for d in body["detections"]:
            assert d["confidence"] >= config.CONFIDENCE_THRESHOLD - 1e-6

    def test_classes_are_the_project_taxonomy(self, detect_response):
        """
        Guards the failure this project has been bitten by: a model and an
        app disagreeing about what an integer means.
        """
        body, _ = detect_response
        for d in body["detections"]:
            assert taxonomy.is_valid_class_id(d["class_id"])
            assert d["class_name"] == taxonomy.dataset_name(d["class_id"])
            assert d["class_name"] in taxonomy.DATASET_NAMES

    def test_inference_time_is_reported_and_plausible(self, detect_response):
        body, _ = detect_response
        t = body["inference_time_ms"]
        assert isinstance(t, (int, float))
        assert 0 < t < 60_000, f"implausible inference time {t} ms"

    # -- failure modes -------------------------------------------
    def test_missing_image_is_400_not_500(self, client):
        r = client.post("/api/detect", data={}, content_type="multipart/form-data")
        assert r.status_code == 400
        assert r.get_json()["success"] is False

    def test_empty_upload_is_400(self, client):
        import io

        r = client.post("/api/detect", data={"image": (io.BytesIO(b""), "empty.jpg")},
                        content_type="multipart/form-data")
        assert r.status_code == 400

    def test_undecodable_bytes_are_400(self, client):
        import io

        r = client.post("/api/detect",
                        data={"image": (io.BytesIO(b"this is not an image"), "x.jpg")},
                        content_type="multipart/form-data")
        assert r.status_code == 400
        assert "decode" in r.get_json()["error"].lower()

    def test_json_body_instead_of_an_image_is_400(self, client):
        r = client.post("/api/detect", json={"oops": True})
        assert r.status_code == 400

    def test_a_bad_request_never_returns_500(self, client):
        """Malformed input must be distinguishable from a broken server."""
        import io

        for data, ctype in (
            ({}, "multipart/form-data"),
            ({"image": (io.BytesIO(b""), "a.jpg")}, "multipart/form-data"),
            ({"image": (io.BytesIO(b"\x00\x01\x02"), "a.jpg")}, "multipart/form-data"),
            ({"wrongfield": (io.BytesIO(b"x"), "a.jpg")}, "multipart/form-data"),
        ):
            r = client.post("/api/detect", data=data, content_type=ctype)
            assert r.status_code == 400, f"expected 400, got {r.status_code}"

    def test_model_unavailable_returns_503(self, client, monkeypatch):
        """
        A missing model is a 503 (server fine, this resource unavailable),
        never a 500 and never silently-substituted fake detections.
        """
        import app as backend_app

        monkeypatch.setattr(backend_app.detector, "mode", "unavailable")
        monkeypatch.setattr(backend_app.detector, "model_error", "simulated for test")
        img = _a_real_test_image()
        with img.open("rb") as fh:
            r = client.post("/api/detect", data={"image": (fh, img.name)},
                            content_type="multipart/form-data")
        assert r.status_code == 503
        body = r.get_json()
        assert body["success"] is False
        assert "unavailable" in body["error"]

    def test_raw_image_body_is_accepted(self, client):
        img = _a_real_test_image()
        r = client.post("/api/detect", data=img.read_bytes(), content_type="image/jpeg")
        assert r.status_code == 200
        assert r.get_json()["success"] is True

    # -- backwards compatibility ---------------------------------
    def test_legacy_detect_alias_keeps_its_old_shape(self, client):
        """/detect must not break for anything written against Phase 4."""
        img = _a_real_test_image()
        with img.open("rb") as fh:
            r = client.post("/detect", data={"image": (fh, img.name)},
                            content_type="multipart/form-data")
        assert r.status_code == 200
        body = r.get_json()
        assert "image" in body and "width" in body["image"]
        assert isinstance(body["detections"][0]["bbox"], list)
        assert len(body["detections"][0]["bbox"]) == 4

    def test_both_routes_agree(self, client):
        """Same model, same code path - they must report the same thing."""
        img = _a_real_test_image()
        with img.open("rb") as fh:
            new = client.post("/api/detect", data={"image": (fh, img.name)},
                              content_type="multipart/form-data").get_json()
        with img.open("rb") as fh:
            old = client.post("/detect", data={"image": (fh, img.name)},
                              content_type="multipart/form-data").get_json()
        assert len(new["detections"]) == old["count"]
        assert new["image_width"] == old["image"]["width"]
        for a, b in zip(new["detections"], old["detections"]):
            assert a["class_id"] == b["class_id"]
            assert [a["bbox"]["x1"], a["bbox"]["y1"], a["bbox"]["x2"], a["bbox"]["y2"]] == b["bbox"]

    def test_index_advertises_the_endpoint(self, client):
        assert "POST /api/detect" in client.get("/").get_json()["endpoints"]


# ===============================================================
# I. Phase 6 — persistent detection history
# ===============================================================
class TestDetectionHistoryStore:
    """
    Tests the store directly, against a temporary database.

    Kept separate from the API tests so a failure tells you WHICH layer
    broke: the storage or the route that uses it.
    """

    @pytest.fixture
    def store(self, tmp_path):
        from history import DetectionHistory

        return DetectionHistory(tmp_path / "test.db", max_events=50)

    def test_creates_its_database_file(self, store):
        assert store.db_path.exists()
        assert store.count() == 0

    def test_records_and_reads_back(self, store):
        event_id = store.record(
            [{"class_id": 2, "class_name": "Person", "confidence": 0.9,
              "bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}],
            {"source": "test", "image_width": 640, "image_height": 480,
             "inference_time_ms": 12.3},
        )
        assert event_id is not None
        assert store.count() == 1

        event = store.recent(10)[0]
        assert event["detection_count"] == 1
        assert event["image_width"] == 640
        assert event["inference_time_ms"] == 12.3
        assert event["detections"][0]["class_name"] == "Person"
        assert event["detections"][0]["bbox"] == {"x1": 1, "y1": 2, "x2": 3, "y2": 4}

    def test_survives_reopening_the_database(self, tmp_path):
        """The whole point: a new process must see the old events."""
        from history import DetectionHistory

        db = tmp_path / "persist.db"
        first = DetectionHistory(db)
        first.record([{"class_name": "Vehicle", "confidence": 0.7}], {"source": "test"})
        assert first.count() == 1

        # A completely separate instance, as a restarted backend would be.
        second = DetectionHistory(db)
        assert second.count() == 1
        assert second.recent(1)[0]["detections"][0]["class_name"] == "Vehicle"

    def test_empty_frames_are_not_events(self, store):
        """~2400 empty frames an hour must not become 2400 rows."""
        assert store.record([], {"source": "test"}) is None
        assert store.count() == 0

    def test_consecutive_identical_results_are_deduplicated(self, store):
        payload = [{"class_name": "Person", "confidence": 0.9}]
        assert store.record(payload, {"source": "test"}) is not None
        assert store.record(payload, {"source": "test"}) is None
        assert store.record(payload, {"source": "test"}) is None
        assert store.count() == 1

    def test_a_different_result_is_a_new_event(self, store):
        store.record([{"class_name": "Person", "confidence": 0.9}], {"source": "t"})
        store.record([{"class_name": "Person", "confidence": 0.9},
                      {"class_name": "Vehicle", "confidence": 0.8}], {"source": "t"})
        assert store.count() == 2

    def test_force_bypasses_both_filters(self, store):
        """Explicit client submissions are deliberate acts, not sampling."""
        assert store.record([], {"source": "frontend"}, force=True) is not None
        payload = [{"class_name": "Person", "confidence": 0.9}]
        store.record(payload, {"source": "frontend"}, force=True)
        store.record(payload, {"source": "frontend"}, force=True)
        assert store.count() == 3

    def test_signature_accepts_either_field_name(self, store):
        """/api/detect sends class_name; the frontend contract sends type."""
        assert store._signature([{"class_name": "Person"}]) == "Person:1"
        assert store._signature([{"type": "Person"}]) == "Person:1"
        assert store._signature([{"class_name": "Person"}, {"type": "Person"}]) == "Person:2"

    def test_retention_cap_is_enforced(self, store):
        """A long-running server must not fill the disk."""
        for i in range(80):
            store.record([{"class_name": f"C{i}", "confidence": 0.5}], {"source": "t"})
        assert store.count() <= store.max_events

    def test_summary_aggregates(self, store):
        store.record([{"class_name": "Person"}, {"class_name": "Person"}], {"source": "t"})
        store.record([{"class_name": "Vehicle"}], {"source": "t"})
        s = store.summary()
        assert s["total_events"] == 2
        assert s["total_objects"] == 3
        assert s["by_class_recent"]["Person"] == 2

    def test_clear_empties_it(self, store):
        store.record([{"class_name": "Person"}], {"source": "t"})
        assert store.clear() == 1
        assert store.count() == 0


class TestHistoryAPI:
    def test_detections_endpoint_reports_persistence(self, client):
        body = client.get("/detections").get_json()
        assert body["persistent"] is True
        assert "total_stored" in body
        assert body["storage"].endswith(".db")

    def test_inference_creates_a_persisted_event(self, client):
        before = client.get("/detections").get_json()["total_stored"]
        img = _a_real_test_image()
        with img.open("rb") as fh:
            client.post("/api/detect", data={"image": (fh, img.name)},
                        content_type="multipart/form-data")
        after = client.get("/detections").get_json()["total_stored"]
        assert after >= before, "inference should not lose events"

    def test_statistics_includes_history_totals(self, client):
        h = client.get("/statistics").get_json()["history"]
        for key in ("total_events", "total_objects", "storage", "max_events"):
            assert key in h

    def test_frontend_submission_returns_an_event_id(self, client):
        r = client.post("/api/detections", json={
            "location": {"lat": 5.6, "lng": -0.19},
            "detections": [{"type": "Vehicle", "confidence": 0.9}],
        })
        assert r.status_code == 201
        assert r.get_json()["event_id"] is not None


# ===============================================================
# J. Phase 6 audit fixes — upload limits and stale-state clearing
# ===============================================================
class TestUploadLimits:
    """
    Without a cap, one POST makes the server buffer the whole upload in
    memory - the cheapest possible denial of service, and easy to trigger
    by accident with a high-resolution phone photo.
    """

    def test_body_size_limit_is_configured(self):
        import app as backend_app

        assert backend_app.app.config["MAX_CONTENT_LENGTH"] == config.MAX_UPLOAD_BYTES
        assert config.MAX_UPLOAD_BYTES > 0

    def test_oversized_body_is_413_not_500(self, client):
        """
        413 means "your upload was too big"; 500 means "the server broke".
        They must not be confused - this exact bug appeared while fixing
        it, because Werkzeug's HTTPException was caught by the generic
        handler before Flask's error handler could see it.
        """
        import io

        payload = b"\xff\xd8\xff" + b"\x00" * (config.MAX_UPLOAD_BYTES + 1024)
        r = client.post("/api/detect",
                        data={"image": (io.BytesIO(payload), "big.jpg")},
                        content_type="multipart/form-data")
        assert r.status_code == 413, f"expected 413, got {r.status_code}"
        assert r.get_json()["success"] is False

    def test_decompression_bomb_is_rejected_on_DECODED_size(self, client):
        """
        A flat image compresses to almost nothing but decodes to hundreds
        of MB. Guarding file size alone would let it straight through, so
        the check must be on pixel count after decoding.
        """
        import io

        import cv2
        import numpy as np

        side = int(config.MAX_IMAGE_PIXELS ** 0.5) + 500
        flat = np.full((side, side, 3), 128, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", flat, [cv2.IMWRITE_JPEG_QUALITY, 50])
        assert ok
        assert len(buf.tobytes()) < config.MAX_UPLOAD_BYTES, "must pass the SIZE check"

        r = client.post("/api/detect",
                        data={"image": (io.BytesIO(buf.tobytes()), "bomb.jpg")},
                        content_type="multipart/form-data")
        assert r.status_code == 413
        assert "pixel" in r.get_json()["error"].lower()

    def test_a_normal_frame_is_unaffected(self, client):
        img = _a_real_test_image()
        with img.open("rb") as fh:
            r = client.post("/api/detect", data={"image": (fh, img.name)},
                            content_type="multipart/form-data")
        assert r.status_code == 200
        assert r.get_json()["success"] is True

    def test_filename_is_reduced_to_a_basename(self):
        """
        Defence in depth: the filename only reaches history metadata today,
        but a traversal string must never be sitting in the database
        waiting for some future feature to use it as a path.
        """
        import app as backend_app

        assert backend_app._safe_filename("../../etc/passwd") == "passwd"
        assert backend_app._safe_filename("/tmp/x.jpg") == "x.jpg"
        assert backend_app._safe_filename(None) is None
        assert len(backend_app._safe_filename("a" * 500)) <= 120

    def test_limits_are_reported_by_config_endpoint(self, client):
        body = client.get("/config").get_json()
        assert body["max_upload_bytes"] == config.MAX_UPLOAD_BYTES
        assert body["max_image_pixels"] == config.MAX_IMAGE_PIXELS


class TestFrontendStaleStateHandling:
    """
    Static checks on live-detection.js. The behaviour itself was verified
    in a browser; these guard against the logic being removed later.
    """

    SRC = (PROJECT_ROOT / "frontend" / "live-detection.js").read_text()

    def test_stale_clearing_exists(self):
        """
        If the backend dies, the last boxes used to stay on screen looking
        live. Showing stale results as current is the same integrity
        problem as showing mock data as real.
        """
        assert "markStale()" in self.SRC
        assert "STALE_AFTER_ERRORS" in self.SRC

    def test_stale_is_triggered_from_the_error_path(self):
        assert "this.markStale();" in self.SRC

    def test_stale_clears_boxes_and_blanks_the_report(self):
        block = self.SRC[self.SRC.index("markStale()"):self.SRC.index("stop() {")]
        assert "this.clearBoxes()" in block
        assert "lastResult = null" in block
        for field in ("siteType", "environment", "confidenceValue", "suggestedAction"):
            assert field in block, f"{field} not reset on stale"

    def test_single_request_in_flight_guard_exists(self):
        assert "this.inFlight" in self.SRC
        assert "if (this.inFlight)" in self.SRC

    def test_overlay_uses_cover_aware_scaling(self):
        """Math.min here instead of Math.max would misplace every box."""
        assert "Math.max(boxW / srcW, boxH / srcH)" in self.SRC

    def test_resize_listener_recomputes_boxes(self):
        assert "addEventListener('resize'" in self.SRC

    def test_mock_output_is_never_shown_as_real(self):
        assert "result.synthetic" in self.SRC
        assert "MOCK mode" in self.SRC
