"""
Seed the attendance database with 7 days of realistic fake data.
Run once before app.py exists to populate the dashboard for demo purposes.

    python -m nathan_innovation.demo_seed
"""

import random
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nathan_innovation.db import _DB_PATH, init_db

EMPLOYEES = ["Alice Chen", "Bob Nguyen", "Claire Smith", "David Park", "Emma Wilson"]
EMOTIONS = ["happy", "neutral", "sad", "angry", "surprise", "fear", "disgust"]
EMOTION_WEIGHTS = [0.35, 0.30, 0.12, 0.08, 0.07, 0.05, 0.03]


def _random_arrival(day: date) -> datetime:
    base_hour = 8
    minute_offset = random.randint(0, 120)
    return datetime(
        day.year, day.month, day.day,
        base_hour + minute_offset // 60,
        minute_offset % 60,
        random.randint(0, 59),
        tzinfo=timezone.utc,
    )


def seed(clear_existing: bool = False) -> None:
    init_db()

    if clear_existing:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute("DELETE FROM attendance_events")
        print("Cleared existing attendance data.")

    today = date.today()
    rows_inserted = 0

    with sqlite3.connect(_DB_PATH) as conn:
        for offset in range(6, -1, -1):
            day = today - timedelta(days=offset)
            if day.weekday() >= 5:
                continue
            present = random.sample(EMPLOYEES, k=random.randint(3, len(EMPLOYEES)))
            for employee in present:
                arrival = _random_arrival(day)
                emotion = random.choices(EMOTIONS, weights=EMOTION_WEIGHTS, k=1)[0]
                confidence = round(random.uniform(0.62, 0.97), 4)
                liveness = 1 if random.random() > 0.05 else 0
                conn.execute(
                    "INSERT INTO attendance_events "
                    "(timestamp, person_id, emotion, liveness_passed, confidence) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (arrival.strftime("%Y-%m-%dT%H:%M:%S"), employee, emotion, liveness, confidence),
                )
                rows_inserted += 1

    print(f"Seeded {rows_inserted} attendance events over 7 days.")
    print(f"Database: {_DB_PATH}")


if __name__ == "__main__":
    seed()
