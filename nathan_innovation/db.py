import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).parent / "attendance.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                person_id       TEXT NOT NULL,
                emotion         TEXT,
                liveness_passed INTEGER,
                confidence      REAL NOT NULL
            )
        """)


def log_attendance(
    person_id: str,
    emotion: str | None = None,
    liveness_passed: bool | None = None,
    confidence: float = 0.0,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    liveness_int = None if liveness_passed is None else int(liveness_passed)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO attendance_events (timestamp, person_id, emotion, liveness_passed, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, person_id, emotion, liveness_int, confidence),
        )


def get_events(date_str: str | None = None) -> list[dict]:
    with _connect() as conn:
        if date_str is None:
            rows = conn.execute(
                "SELECT * FROM attendance_events ORDER BY timestamp"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM attendance_events WHERE timestamp LIKE ? ORDER BY timestamp",
                (f"{date_str}%",),
            ).fetchall()
    return [dict(r) for r in rows]


init_db()
