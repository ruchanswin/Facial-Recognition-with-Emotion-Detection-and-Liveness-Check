"""
Dashboard query helpers for per-person valence timeline.
Used by cb_dashboard() in gradio_app.py.
"""

from datetime import datetime

from dhruv_innovation.emotion_valence import VALENCE_MAP

try:
    from nathan_innovation.db import get_events
except ImportError:
    import sqlite3
    from pathlib import Path
    _DB = Path("nathan_innovation/attendance.db")

    def get_events(date_str=None):
        if not _DB.exists():
            return []
        conn = sqlite3.connect(str(_DB))
        conn.row_factory = sqlite3.Row
        if date_str is None:
            rows = conn.execute(
                "SELECT * FROM attendance_events ORDER BY timestamp"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM attendance_events "
                "WHERE timestamp LIKE ? ORDER BY timestamp",
                (f"{date_str}%",),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


def person_valence_timeline(person_id: str, date_str: str) -> list[dict]:
    """
    Time-ordered list of {time, valence} for one person on one day.
    Used to draw the per-person sentiment line in the dashboard.
    """
    events = [
        e for e in get_events(date_str)
        if e["person_id"] == person_id and e.get("emotion")
    ]
    result = []
    for e in events:
        label   = (e["emotion"] or "").split()[-1].lower()
        valence = VALENCE_MAP.get(label, 0.0)
        t       = datetime.fromisoformat(e["timestamp"]).strftime("%H:%M")
        result.append({"time": t, "valence": round(valence, 3)})
    return result


def team_sentiment_summary(date_str: str) -> dict:
    """
    Team-wide sentiment for a day:
      mean_valence : float in [-1, +1]
      counts       : {positive, neutral, stressed}
      label        : "Positive" | "Neutral" | "Stressed"
      color        : hex string for the KPI card
    """
    events   = get_events(date_str)
    valences = []
    counts   = {"positive": 0, "neutral": 0, "stressed": 0}

    for e in events:
        label = (e.get("emotion") or "").split()[-1].lower()
        v     = VALENCE_MAP.get(label, 0.0)
        valences.append(v)
        if v >= 0.25:
            counts["positive"] += 1
        elif v <= -0.25:
            counts["stressed"] += 1
        else:
            counts["neutral"] += 1

    mean_v = round(sum(valences) / len(valences), 3) if valences else 0.0

    if mean_v >= 0.25:
        label_str = "Positive"
        color     = "#00E676"   # SUCCESS green
    elif mean_v <= -0.25:
        label_str = "Stressed"
        color     = "#FF1744"   # DANGER red
    else:
        label_str = "Neutral"
        color     = "#FFB300"   # WARN amber

    return {
        "mean_valence": mean_v,
        "counts":       counts,
        "label":        label_str,
        "color":        color,
    }