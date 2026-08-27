"""
Galamsey AI Detector — persistent detection history
===================================================

Detection events survive a backend restart.

------------------------------------------------------------------
WHY SQLITE
------------------------------------------------------------------
The project had no database. Adding PostgreSQL or MongoDB for this
would mean a server to install, run, configure and back up - real
operational weight for a single-user dashboard that records a few
thousand rows.

SQLite is part of the Python standard library. It adds ZERO new
dependencies, stores everything in one file you can copy or delete, and
handles far more concurrency than this application will ever generate.
If the project ever needs multiple writers across machines, that is the
moment to reach for a server, not before.

------------------------------------------------------------------
EVENTS, NOT FRAMES
------------------------------------------------------------------
The live loop runs an inference roughly every 1.5 seconds. Writing a row
per cycle would mean ~2,400 rows an hour, almost all of them saying the
same thing, and the images themselves would dwarf everything if we
stored those too.

So two rules:

  1. Only cycles that actually DETECTED something are recorded. A frame
     with nothing in it is not an event.
  2. Consecutive results with the same class signature inside
     DEDUP_WINDOW_SECONDS are treated as the same ongoing event and
     skipped. Standing in front of the camera for a minute produces one
     event, not forty.

Raw frames are never stored. We keep what the model concluded - classes,
confidences, boxes, dimensions, timing - not the pixels. Storing every
camera frame would be a privacy problem as much as a disk problem, and
nothing in this project needs it.

------------------------------------------------------------------
RETENTION
------------------------------------------------------------------
A server left running for weeks should not fill the disk. The table is
trimmed to MAX_EVENTS rows (newest kept). That is a deliberate,
documented limit rather than unbounded growth nobody notices until it
matters.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_EVENTS = 5000
DEDUP_WINDOW_SECONDS = 10.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS detection_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,
    source            TEXT,
    synthetic         INTEGER NOT NULL DEFAULT 0,
    detection_count   INTEGER NOT NULL,
    class_signature   TEXT    NOT NULL,
    image_width       INTEGER,
    image_height      INTEGER,
    inference_time_ms REAL,
    latitude          REAL,
    longitude         REAL,
    detections_json   TEXT    NOT NULL,
    metadata_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON detection_events(timestamp);
CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT);
"""


class DetectionHistory:
    """SQLite-backed store of detection events."""

    def __init__(self, db_path: Path, max_events: int = MAX_EVENTS) -> None:
        self.db_path = Path(db_path)
        self.max_events = max_events
        self._lock = threading.Lock()
        self._last_signature: str | None = None
        self._last_time: float = 0.0
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # -- connection ----------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """
        A short-lived connection per operation.

        Flask runs threaded, and a SQLite connection belongs to the
        thread that created it. Opening per call sidesteps that entirely
        and costs microseconds for a local file.
        """
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),),
            )
        logger.info("detection history ready at %s", self.db_path)

    # -- writing -------------------------------------------------
    @staticmethod
    def _signature(detections: list[dict[str, Any]]) -> str:
        """
        A stable fingerprint of WHAT was seen, ignoring exact positions.

        "2 Person, 1 Vehicle" is the same ongoing event whether the
        people shuffled a few pixels or not, so boxes are excluded.
        """
        counts: dict[str, int] = {}
        for d in detections:
            # /api/detect sends class_name; records submitted by the
            # frontend use "type" (the shape in frontend/DETECTION_API.md).
            # Accept both rather than silently recording "?".
            name = str(d.get("class_name") or d.get("type") or "unknown")
            counts[name] = counts.get(name, 0) + 1
        return ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))

    def record(
        self,
        detections: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        *,
        synthetic: bool = False,
        force: bool = False,
    ) -> int | None:
        """
        Persist one detection event.

        Returns the new row id, or None if the event was skipped as an
        empty frame or a duplicate of the one before it.

        `force=True` bypasses both filters - used for records submitted
        explicitly by a client, which are deliberate acts rather than
        an automatic sampling loop.
        """
        metadata = metadata or {}

        if not detections and not force:
            return None

        signature = self._signature(detections)
        now = datetime.now(timezone.utc)

        if not force:
            with self._lock:
                same = signature == self._last_signature
                recent = (now.timestamp() - self._last_time) < DEDUP_WINDOW_SECONDS
                if same and recent:
                    self._last_time = now.timestamp()   # extend the window
                    return None

        location = metadata.get("location") or {}
        row = (
            now.isoformat(),
            str(metadata.get("source", "unknown")),
            1 if synthetic else 0,
            len(detections),
            signature,
            metadata.get("image_width"),
            metadata.get("image_height"),
            metadata.get("inference_time_ms"),
            location.get("lat"),
            location.get("lng"),
            json.dumps(detections),
            json.dumps(metadata),
        )

        try:
            with self._lock, self._connect() as conn:
                cursor = conn.execute(
                    "INSERT INTO detection_events (timestamp, source, synthetic,"
                    " detection_count, class_signature, image_width, image_height,"
                    " inference_time_ms, latitude, longitude, detections_json,"
                    " metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    row,
                )
                event_id = cursor.lastrowid
                conn.execute(
                    "DELETE FROM detection_events WHERE id <= "
                    "(SELECT MAX(id) FROM detection_events) - ?",
                    (self.max_events,),
                )
                self._last_signature = signature
                self._last_time = now.timestamp()
            return event_id
        except sqlite3.Error as exc:
            # History is valuable but not worth failing a detection over.
            logger.error("could not persist detection event: %s", exc)
            return None

    # -- reading -------------------------------------------------
    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Most recent events, oldest first (matching the old API order)."""
        limit = max(0, min(int(limit), self.max_events))
        if limit == 0:
            return []
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM detection_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._to_dict(r) for r in reversed(rows)]

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "source": row["source"],
            "synthetic": bool(row["synthetic"]),
            "detection_count": row["detection_count"],
            "class_signature": row["class_signature"],
            "image_width": row["image_width"],
            "image_height": row["image_height"],
            "inference_time_ms": row["inference_time_ms"],
            "location": (
                {"lat": row["latitude"], "lng": row["longitude"]}
                if row["latitude"] is not None else None
            ),
            "detections": json.loads(row["detections_json"]),
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]

    def summary(self) -> dict[str, Any]:
        """Aggregates computed in SQL rather than by loading every row."""
        with self._lock, self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]
            objects = conn.execute(
                "SELECT COALESCE(SUM(detection_count), 0) FROM detection_events"
            ).fetchone()[0]
            first = conn.execute(
                "SELECT MIN(timestamp) FROM detection_events"
            ).fetchone()[0]
            last = conn.execute(
                "SELECT MAX(timestamp) FROM detection_events"
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT detections_json FROM detection_events "
                "ORDER BY id DESC LIMIT 500"
            ).fetchall()

        by_class: dict[str, int] = {}
        for r in rows:
            for d in json.loads(r["detections_json"]):
                name = str(d.get("class_name") or d.get("type") or "unknown")
                by_class[name] = by_class.get(name, 0) + 1

        return {
            "total_events": total,
            "total_objects": objects,
            "first_event": first,
            "last_event": last,
            "by_class_recent": by_class,
            "storage": str(self.db_path),
            "max_events": self.max_events,
        }

    def clear(self) -> int:
        with self._lock, self._connect() as conn:
            removed = conn.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]
            conn.execute("DELETE FROM detection_events")
            self._last_signature = None
            self._last_time = 0.0
        return removed
