"""
Evaluate emotion label flicker across rolling-window sizes.

Replays emotion_raw (and optional confidence) sequences from a CSV through
EmotionSmoother at several window sizes and reports label-change rate.

Typical input: logs/attendance_session.csv (sparse, one row per cooldown log)
or a denser trace CSV with columns: emotion_raw, emotion_conf_raw, timestamp.

Usage:
    py -m justin_innovation.eval_window --csv logs/attendance_session.csv
    py -m justin_innovation.eval_window --csv logs/attendance_session.csv --windows 5 15 30 --out justin_innovation/window_eval.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime

from justin_innovation.emotion_smoother import EmotionSmoother


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_emotion_sequence(csv_path: Path) -> tuple[list[str], list[float], float]:
    """Return labels, confidences, and duration in minutes."""
    labels: list[str] = []
    confs: list[float] = []
    times: list[datetime] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("emotion_raw") or row.get("emotion") or "").strip().lower()
            if not raw or raw == "n/a":
                continue
            labels.append(raw.split()[-1])
            try:
                confs.append(float(row.get("emotion_conf_raw") or row.get("emotion_conf") or 0.5))
            except ValueError:
                confs.append(0.5)
            ts = _parse_ts(row.get("timestamp", ""))
            if ts:
                times.append(ts)

    if len(times) >= 2:
        duration_min = max((times[-1] - times[0]).total_seconds() / 60.0, 1e-6)
    else:
        # Sparse log: treat each row as ~0.5s apart for a conservative rate estimate
        duration_min = max(len(labels) * 0.5 / 60.0, 1e-6)

    return labels, confs, duration_min


def label_change_count(labels: list[str]) -> int:
    if len(labels) < 2:
        return 0
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])


def flicker_per_minute(labels: list[str], duration_min: float) -> float:
    return label_change_count(labels) / duration_min


def smooth_sequence(
    labels: list[str], confs: list[float], window_size: int
) -> list[str]:
    smoother = EmotionSmoother(window_size=window_size)
    return [smoother.update(lbl, conf) for lbl, conf in zip(labels, confs)]


def evaluate_windows(
    labels: list[str],
    confs: list[float],
    duration_min: float,
    windows: list[int],
) -> dict[str, dict[str, float]]:
    raw_flicker = flicker_per_minute(labels, duration_min)
    results: dict[str, dict[str, float]] = {
        "raw": {
            "window": 0,
            "changes": float(label_change_count(labels)),
            "flicker_per_min": raw_flicker,
        }
    }
    for w in windows:
        smoothed = smooth_sequence(labels, confs, w)
        results[f"window_{w}"] = {
            "window": w,
            "changes": float(label_change_count(smoothed)),
            "flicker_per_min": flicker_per_minute(smoothed, duration_min),
        }
    return results


def plot_results(results: dict[str, dict[str, float]], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    names = []
    flickers = []
    for key in sorted(results.keys(), key=lambda k: results[k]["window"]):
        w = results[key]["window"]
        names.append("raw" if w == 0 else f"w={w}")
        flickers.append(results[key]["flicker_per_min"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, flickers, color=["#FF7043"] + ["#42A5F5"] * (len(names) - 1))
    ax.set_ylabel("Label changes per minute")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, flickers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_report(results: dict[str, dict[str, float]], duration_min: float, n_samples: int) -> None:
    print(f"\n{'=' * 60}")
    print("Emotion smoothing window evaluation")
    print(f"Samples: {n_samples}  |  Duration (est.): {duration_min:.2f} min")
    print(f"{'=' * 60}")
    raw = results["raw"]["flicker_per_min"]
    print(f"{'Mode':<12} {'Changes':>8} {'Flicker/min':>12} {'vs raw':>10}")
    print("-" * 46)
    for key in sorted(results.keys(), key=lambda k: results[k]["window"]):
        r = results[key]
        w = r["window"]
        label = "raw" if w == 0 else f"window={int(w)}"
        vs = "" if w == 0 else f"{(1 - r['flicker_per_min'] / raw) * 100:+.0f}%" if raw > 0 else "n/a"
        print(f"{label:<12} {int(r['changes']):>8} {r['flicker_per_min']:>12.2f} {vs:>10}")
    print(f"{'=' * 60}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare emotion flicker across smoother window sizes")
    p.add_argument("--csv", default="logs/attendance_session.csv", help="CSV with emotion_raw column")
    p.add_argument("--windows", type=int, nargs="+", default=[5, 15, 30])
    p.add_argument("--out", default="justin_innovation/window_eval.png", help="Output plot path")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    labels, confs, duration_min = load_emotion_sequence(csv_path)
    if len(labels) < 2:
        raise SystemExit(
            f"Need at least 2 emotion rows in {csv_path}. "
            "Run a live session first or use a denser emotion trace CSV."
        )

    results = evaluate_windows(labels, confs, duration_min, args.windows)
    print_report(results, duration_min, len(labels))

    if not args.no_plot:
        plot_results(results, Path(args.out), f"Flicker rate · {csv_path.name}")
        print(f"Plot saved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
