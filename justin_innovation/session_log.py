"""CSV session logging for attendance + emotion events."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_session_stats(
    rows: list[dict[str, str]],
    gate_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Aggregate attendance/emotion metrics from CSV rows."""
    gate_stats = gate_stats or {}
    stats: dict[str, Any] = {
        "total_events": len(rows),
        "unique_attendees": [],
        "spoof_count": 0,
        "liveness_unknown": 0,
        "emotion_distribution": {},
        "per_person": {},
        "gate_stats": dict(gate_stats),
    }

    if not rows:
        return stats

    smooth_counts: Counter[str] = Counter()
    per_person_rows: dict[str, list[dict[str, str]]] = defaultdict(list)

    for r in rows:
        person = r.get("person_id", "")
        per_person_rows[person].append(r)
        emo = r.get("emotion_smooth") or r.get("emotion_raw") or "unknown"
        smooth_counts[emo] += 1

        live = r.get("liveness_passed", "")
        if live == "0":
            stats["spoof_count"] += 1
        elif live == "":
            stats["liveness_unknown"] += 1

    stats["unique_attendees"] = sorted(per_person_rows.keys())
    stats["emotion_distribution"] = dict(smooth_counts.most_common())

    for person, person_rows in per_person_rows.items():
        times = [_parse_ts(r["timestamp"]) for r in person_rows]
        times = [t for t in times if t is not None]
        dwell_sec = 0.0
        if len(times) >= 2:
            dwell_sec = (max(times) - min(times)).total_seconds()
        elif len(times) == 1:
            dwell_sec = 0.0

        emo_counter = Counter(
            r.get("emotion_smooth") or r.get("emotion_raw") or "unknown" for r in person_rows
        )
        sims = [_parse_float(r.get("similarity", "0")) for r in person_rows]
        stats["per_person"][person] = {
            "log_count": len(person_rows),
            "dwell_seconds": round(dwell_sec, 1),
            "first_seen": person_rows[0].get("timestamp", ""),
            "last_seen": person_rows[-1].get("timestamp", ""),
            "dominant_emotion": emo_counter.most_common(1)[0][0] if emo_counter else "n/a",
            "avg_similarity": round(sum(sims) / len(sims), 4) if sims else 0.0,
        }

    return stats


class SessionLogger:
    """
    Append attendance/emotion rows to CSV with per-person cooldown.

    Optional quality gates (similarity, emotion confidence, liveness) reduce
    low-confidence log noise.
    """

    FIELDNAMES = [
        "timestamp",
        "person_id",
        "similarity",
        "liveness_passed",
        "emotion_raw",
        "emotion_conf_raw",
        "emotion_smooth",
        "emotion_conf_smooth",
    ]

    def __init__(
        self,
        log_path: str = "logs/attendance_session.csv",
        cooldown_sec: float = 10.0,
        *,
        gate_enabled: bool = True,
        min_similarity: float | None = 0.55,
        min_emotion_conf: float | None = 0.20,
        require_liveness: bool = False,
    ):
        self.log_path = Path(log_path)
        self.cooldown_sec = cooldown_sec
        self.gate_enabled = gate_enabled
        self.min_similarity = min_similarity
        self.min_emotion_conf = min_emotion_conf
        self.require_liveness = require_liveness
        self._last_log_time: dict[str, float] = {}
        self.gate_stats: dict[str, int] = {
            "accepted": 0,
            "rejected_unknown": 0,
            "rejected_cooldown": 0,
            "rejected_similarity": 0,
            "rejected_emotion_conf": 0,
            "rejected_liveness": 0,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def configure_gates(
        self,
        *,
        gate_enabled: bool | None = None,
        min_similarity: float | None = None,
        min_emotion_conf: float | None = None,
        require_liveness: bool | None = None,
    ) -> None:
        if gate_enabled is not None:
            self.gate_enabled = gate_enabled
        if min_similarity is not None:
            self.min_similarity = min_similarity
        if min_emotion_conf is not None:
            self.min_emotion_conf = min_emotion_conf
        if require_liveness is not None:
            self.require_liveness = require_liveness

    def passes_gate(
        self,
        person_id: str,
        similarity: float,
        emotion_conf: float,
        *,
        liveness_passed: bool | None = None,
    ) -> tuple[bool, str]:
        if person_id in ("Unknown", "Face ID unavailable", ""):
            return False, "unknown"

        if not self.gate_enabled:
            if self.require_liveness and liveness_passed is not True:
                return False, "liveness"
            return True, ""

        if self.min_similarity is not None and similarity < self.min_similarity:
            return False, "similarity"
        if self.min_emotion_conf is not None and emotion_conf < self.min_emotion_conf:
            return False, "emotion_conf"
        if self.require_liveness and liveness_passed is not True:
            return False, "liveness"
        return True, ""

    def _record_rejection(self, reason: str) -> None:
        key = {
            "unknown": "rejected_unknown",
            "cooldown": "rejected_cooldown",
            "similarity": "rejected_similarity",
            "emotion_conf": "rejected_emotion_conf",
            "liveness": "rejected_liveness",
        }.get(reason, "rejected_unknown")
        self.gate_stats[key] = self.gate_stats.get(key, 0) + 1

    def _ensure_header(self) -> None:
        if self.log_path.exists() and self.log_path.stat().st_size > 0:
            with self.log_path.open(newline="", encoding="utf-8") as f:
                first_row = next(csv.reader(f), None)
            if first_row == self.FIELDNAMES:
                return
        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def maybe_log(
        self,
        person_id: str,
        similarity: float,
        emotion_raw: str,
        emotion_conf_raw: float,
        emotion_smooth: str,
        emotion_conf_smooth: float,
        *,
        liveness_passed: bool | None = None,
    ) -> bool:
        ok, reason = self.passes_gate(
            person_id,
            similarity,
            emotion_conf_smooth if emotion_conf_smooth > 0 else emotion_conf_raw,
            liveness_passed=liveness_passed,
        )
        if not ok:
            self._record_rejection(reason)
            return False

        now = time.time()
        last = self._last_log_time.get(person_id, 0.0)
        if now - last < self.cooldown_sec:
            self._record_rejection("cooldown")
            return False

        self._last_log_time[person_id] = now
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        live_str = "" if liveness_passed is None else ("1" if liveness_passed else "0")

        row = {
            "timestamp": ts,
            "person_id": person_id,
            "similarity": f"{similarity:.4f}",
            "liveness_passed": live_str,
            "emotion_raw": emotion_raw or "n/a",
            "emotion_conf_raw": f"{emotion_conf_raw:.4f}",
            "emotion_smooth": emotion_smooth or "n/a",
            "emotion_conf_smooth": f"{emotion_conf_smooth:.4f}",
        }
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)
        self.gate_stats["accepted"] = self.gate_stats.get("accepted", 0) + 1
        return True

    def read_rows(self) -> list[dict[str, str]]:
        if not self.log_path.exists():
            return []
        with self.log_path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def session_stats(self) -> dict[str, Any]:
        return compute_session_stats(self.read_rows(), self.gate_stats)

    def export_summary(
        self,
        path: str | Path | None = None,
        *,
        fmt: str = "markdown",
    ) -> Path:
        """
        Write session summary to markdown or JSON.

        Default path: logs/session_YYYYMMDD_HHMMSS_summary.<ext>
        """
        rows = self.read_rows()
        stats = compute_session_stats(rows, self.gate_stats)
        fmt = fmt.lower()
        ext = "json" if fmt == "json" else "md"

        if path is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = self.log_path.parent / f"session_{stamp}_summary.{ext}"
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "log_path": str(self.log_path.resolve()),
                "gates": {
                    "enabled": self.gate_enabled,
                    "min_similarity": self.min_similarity,
                    "min_emotion_conf": self.min_emotion_conf,
                    "require_liveness": self.require_liveness,
                    "cooldown_sec": self.cooldown_sec,
                },
                **stats,
            }
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return out

        lines = [
            "# FaceAttend session summary",
            "",
            f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"- **Log file:** `{self.log_path.resolve()}`",
            "",
            "## Quality-aware logging",
            "",
            f"| Setting | Value |",
            f"| ------- | ----- |",
            f"| Gates enabled | {self.gate_enabled} |",
            f"| Min similarity | {self.min_similarity} |",
            f"| Min emotion confidence | {self.min_emotion_conf} |",
            f"| Require liveness | {self.require_liveness} |",
            f"| Cooldown (s) | {self.cooldown_sec} |",
            "",
            "### Gate statistics (this session)",
            "",
        ]
        for key, val in sorted(self.gate_stats.items()):
            lines.append(f"- **{key}:** {val}")

        lines.extend(["", "## Attendance", "", f"- **Total log events:** {stats['total_events']}"])
        lines.append(f"- **Unique attendees:** {len(stats['unique_attendees'])}")
        if stats["unique_attendees"]:
            lines.append(f"- **Names:** {', '.join(stats['unique_attendees'])}")
        lines.append(f"- **Spoof-flagged events:** {stats['spoof_count']}")
        lines.append(f"- **Liveness not recorded:** {stats['liveness_unknown']}")

        lines.extend(["", "## Emotion distribution (smoothed)", ""])
        if stats["emotion_distribution"]:
            for emo, count in stats["emotion_distribution"].items():
                pct = 100.0 * count / max(stats["total_events"], 1)
                lines.append(f"- **{emo}:** {count} ({pct:.1f}%)")
        else:
            lines.append("_No events recorded._")

        lines.extend(["", "## Per-person", ""])
        for person, info in sorted(stats["per_person"].items()):
            lines.append(f"### {person}")
            lines.append(f"- Logs: {info['log_count']}")
            lines.append(f"- Dwell (first→last log): {info['dwell_seconds']} s")
            lines.append(f"- First seen: {info['first_seen']}")
            lines.append(f"- Last seen: {info['last_seen']}")
            lines.append(f"- Dominant emotion: {info['dominant_emotion']}")
            lines.append(f"- Avg similarity: {info['avg_similarity']}")
            lines.append("")

        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def print_summary(self) -> None:
        rows = self.read_rows()
        stats = compute_session_stats(rows, self.gate_stats)
        print(f"\n{'=' * 60}")
        print(f"Session log: {self.log_path.resolve()}")
        print(f"Total events: {stats['total_events']}")

        if self.gate_stats:
            print("\nGate statistics:")
            for key, val in sorted(self.gate_stats.items()):
                print(f"  {key:24s}: {val}")

        if not rows:
            print("No attendance events recorded this session.")
            print(f"{'=' * 60}\n")
            return

        print(f"Unique attendees: {len(stats['unique_attendees'])} — {', '.join(stats['unique_attendees'])}")
        print(f"Spoof-flagged events: {stats['spoof_count']}")

        print("\nEmotion distribution (smoothed labels):")
        for emo, count in stats["emotion_distribution"].items():
            pct = 100.0 * count / len(rows)
            print(f"  {emo:12s}: {count:4d} ({pct:5.1f}%)")

        print("\nPer-person:")
        for person, info in sorted(stats["per_person"].items()):
            print(
                f"  {person:18s}: {info['dominant_emotion']} "
                f"({info['log_count']} logs, dwell {info['dwell_seconds']}s)"
            )

        print(f"{'=' * 60}\n")
