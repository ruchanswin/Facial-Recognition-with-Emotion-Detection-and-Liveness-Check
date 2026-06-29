from datetime import date, datetime, timedelta

from nathan_innovation.db import get_events


def _today() -> str:
    return date.today().isoformat()


def attendance_summary(date_str: str | None = None) -> list[str]:
    date_str = date_str or _today()
    events = get_events(date_str)
    seen: dict[str, str] = {}
    for e in events:
        if e["person_id"] not in seen:
            seen[e["person_id"]] = e["timestamp"]
    return sorted(seen.keys())


def emotion_distribution(date_str: str | None = None) -> dict[str, int]:
    events = get_events(date_str or _today())
    counts: dict[str, int] = {}
    for e in events:
        label = e["emotion"] or "unknown"
        counts[label] = counts.get(label, 0) + 1
    return counts


def daily_counts(n_days: int = 7) -> list[tuple[str, int]]:
    result = []
    today = date.today()
    for offset in range(n_days - 1, -1, -1):
        day = (today - timedelta(days=offset)).isoformat()
        events = get_events(day)
        unique = len({e["person_id"] for e in events})
        result.append((day, unique))
    return result


def arrivals_by_hour(date_str: str | None = None) -> dict[int, int]:
    events = get_events(date_str or _today())
    seen_first: dict[str, int] = {}
    for e in events:
        pid = e["person_id"]
        if pid not in seen_first:
            hour = datetime.fromisoformat(e["timestamp"]).hour
            seen_first[pid] = hour
    counts: dict[int, int] = {}
    for hour in seen_first.values():
        counts[hour] = counts.get(hour, 0) + 1
    return counts


def avg_confidence(date_str: str | None = None) -> float:
    events = get_events(date_str or _today())
    if not events:
        return 0.0
    return sum(e["confidence"] for e in events) / len(events)
