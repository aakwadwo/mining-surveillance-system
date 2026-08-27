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

    def test_upstream_typo_preserved_in_data_but_fixed_for_display(self):
        """Ids and the wire name are frozen; the typo is corrected only for humans."""
        assert taxonomy.DATASET_NAMES[3] == "ProcessingEquiment"   # dataset + weights
        assert taxonomy.DISPLAY_NAMES[3] == "Processing Equipment"  # UI only

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
