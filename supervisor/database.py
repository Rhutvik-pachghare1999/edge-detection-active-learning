# SQLite wrapper for AECS-SDC.
# Keeps things simple — no ORM, just plain sqlite3 with a dataclass for events.

import os
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Event:
    frame_id:      str
    timestamp:     float
    c_teacher:     float
    c_student:     float
    delta:         float
    harvested:     bool
    scenario:      str           = "unknown"
    clip_path:     Optional[str] = None
    entropy:       float         = 0.0
    sonar_m:       float         = 0.0
    accel_mag:     float         = 0.0
    teacher_labels: str          = ""   # JSON: list of {label, name, score, cx, cy, bw, bh}


class Database:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_id       TEXT    NOT NULL,
                    timestamp      REAL    NOT NULL,
                    c_teacher      REAL    NOT NULL,
                    c_student      REAL    NOT NULL,
                    delta          REAL    NOT NULL,
                    harvested      INTEGER NOT NULL,
                    scenario       TEXT    DEFAULT 'unknown',
                    clip_path      TEXT,
                    entropy        REAL    DEFAULT 0.0,
                    sonar_m        REAL    DEFAULT 0.0,
                    accel_mag      REAL    DEFAULT 0.0,
                    teacher_labels TEXT    DEFAULT ''
                )
            """)
            for col, coltype, default in [("sonar_m", "REAL", "0.0"), ("accel_mag", "REAL", "0.0"), ("teacher_labels", "TEXT", "''")]:
                try:
                    conn.execute(f"ALTER TABLE events ADD COLUMN {col} {coltype} DEFAULT {default}")
                except Exception:
                    pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS config_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   REAL NOT NULL,
                    tau         REAL NOT NULL,
                    sensitivity TEXT NOT NULL
                )
            """)
            conn.commit()

    def log_event(self, event: Event):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO events "
                "(frame_id, timestamp, c_teacher, c_student, delta, "
                " harvested, scenario, clip_path, entropy, sonar_m, accel_mag, teacher_labels) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.frame_id, event.timestamp,
                    event.c_teacher, event.c_student, event.delta,
                    int(event.harvested), event.scenario,
                    event.clip_path, event.entropy,
                    event.sonar_m, event.accel_mag, event.teacher_labels,
                ),
            )
            conn.commit()

    def log_config_change(self, tau: float, sensitivity: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO config_log (timestamp, tau, sensitivity) VALUES (?, ?, ?)",
                (time.time(), tau, sensitivity),
            )
            conn.commit()

    def update_clip_path(self, frame_id: str, clip_path: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE events SET clip_path = ? WHERE frame_id = ?",
                (clip_path, frame_id),
            )
            conn.commit()

    def get_results_by_scenario(self) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    scenario,
                    COUNT(*)                              AS total_frames,
                    SUM(harvested)                        AS harvested_frames,
                    ROUND(100.0*SUM(harvested)/COUNT(*), 2) AS harvest_rate_pct,
                    ROUND(AVG(delta), 4)                  AS mean_delta,
                    ROUND(MAX(delta), 4)                  AS max_delta
                FROM events
                GROUP BY scenario
                ORDER BY scenario
            """).fetchall()
        return [dict(r) for r in rows]

    def get_recent_events(self, n: int = 100) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]
