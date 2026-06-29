"""
Demo — Emotion Smoothing, Quality-Aware Session Logging & Window Evaluation

Shows:
  PART 1 - Synthetic webcam sequence: raw label flicker vs rolling-window smoother
  PART 2 - Quality-gated attendance logging (gates on vs off)
  PART 3 - Session summary export (markdown + gate statistics)
  PART 4 - Flicker-rate comparison across window sizes {5, 15, 30}

Run (no model or live camera required):
    python -m justin_innovation.demo

Optional — include real session CSV from FaceAttend:
    python -m justin_innovation.demo --csv logs/attendance_session.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from justin_innovation import EmotionSmoother, SessionLogger
from justin_innovation.eval_window import evaluate_windows, load_emotion_sequence, print_report


# ---------------------------------------------------------------------------
# Synthetic data — mimics noisy per-frame emotion classifier output
# ---------------------------------------------------------------------------

FLICKERY_SEQUENCE: list[tuple[str, float]] = [
    ("happy", 0.41),
    ("fear", 0.28),
    ("happy", 0.35),
    ("surprise", 0.22),
    ("happy", 0.44),
    ("neutral", 0.31),
    ("happy", 0.48),
    ("sad", 0.19),
    ("happy", 0.52),
    ("happy", 0.55),
    ("happy", 0.61),
    ("fear", 0.24),
    ("happy", 0.58),
    ("happy", 0.63),
    ("happy", 0.67),
    ("happy", 0.70),
    ("neutral", 0.33),
    ("happy", 0.72),
    ("happy", 0.75),
    ("happy", 0.78),
]


def _flicker_count(labels: list[str]) -> int:
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])


# ---------------------------------------------------------------------------
# PART 1 — Emotion smoothing
# ---------------------------------------------------------------------------

def run_smoothing_demo(window_size: int = 15) -> tuple[list[str], list[str], list[float]]:
    print("=" * 66)
    print("  PART 1 - Emotion Smoothing (Rolling Majority Vote)")
    print(f"  Window size: {window_size} frames")
    print("=" * 66)

    smoother = EmotionSmoother(window_size=window_size)
    raw_labels: list[str] = []
    smooth_labels: list[str] = []
    confs: list[float] = []

    print(f"\n  {'#':>3}  {'Raw':<10}  {'Conf':>5}  {'Smoothed':<10}  Match")
    print(f"  {'-' * 42}")

    for i, (label, conf) in enumerate(FLICKERY_SEQUENCE, 1):
        raw_labels.append(label)
        confs.append(conf)
        smooth = smoother.update(label, conf)
        smooth_labels.append(smooth)
        match = "yes" if label == smooth else "no"
        print(f"  {i:>3}  {label:<10}  {conf:>5.2f}  {smooth:<10}  {match}")

    raw_changes = _flicker_count(raw_labels)
    smooth_changes = _flicker_count(smooth_labels)
    reduction = (1 - smooth_changes / raw_changes) * 100 if raw_changes else 0.0

    print(f"\n  Label changes (sequence):  raw {raw_changes}  ->  smoothed {smooth_changes}")
    print(f"  Flicker reduction: {reduction:.0f}%")
    print("\n  Key insight:")
    print("  Raw classifier jumps between happy / fear / surprise every few frames.")
    print("  Smoothed label stabilises once the window is mostly 'happy'.")
    print()
    return raw_labels, smooth_labels, confs


# ---------------------------------------------------------------------------
# PART 2 — Quality-gated logging
# ---------------------------------------------------------------------------

def run_gating_demo() -> SessionLogger:
    print("=" * 66)
    print("  PART 2 - Quality-Aware Attendance Logging")
    print("  Same synthetic 'check-in' attempts — gates OFF vs ON")
    print("=" * 66)

    attempts = [
        ("justin", 0.82, 0.55, True, "high sim + conf, live"),
        ("justin", 0.48, 0.60, True, "low similarity"),
        ("justin", 0.80, 0.12, True, "low emotion confidence"),
        ("justin", 0.79, 0.58, False, "liveness failed"),
        ("Unknown", 0.90, 0.90, True, "unknown identity"),
        ("justin", 0.85, 0.62, True, "valid (2nd log — cooldown)"),
    ]

    tmp = Path(tempfile.mkdtemp(prefix="justin_demo_"))
    path_off = tmp / "ungated.csv"
    path_on = tmp / "gated.csv"

    logger_off = SessionLogger(
        str(path_off),
        cooldown_sec=0.0,
        gate_enabled=False,
        require_liveness=False,
    )
    logger_on = SessionLogger(
        str(path_on),
        cooldown_sec=0.0,
        gate_enabled=True,
        min_similarity=0.55,
        min_emotion_conf=0.20,
        require_liveness=True,
    )

    print(f"\n  {'Person':<10}  {'Sim':>5}  {'Conf':>5}  {'Live':>5}  "
          f"{'Ungated':>8}  {'Gated':>8}  Note")
    print(f"  {'-' * 62}")

    for person, sim, conf, live, note in attempts:
        ok_off = logger_off.maybe_log(
            person, sim, "happy", conf, "happy", conf, liveness_passed=live
        )
        ok_on = logger_on.maybe_log(
            person, sim, "happy", conf, "happy", conf, liveness_passed=live
        )
        print(
            f"  {person:<10}  {sim:>5.2f}  {conf:>5.2f}  "
            f"{'yes' if live else 'no':>5}  "
            f"{'LOG' if ok_off else 'skip':>8}  "
            f"{'LOG' if ok_on else 'skip':>8}  {note}"
        )

    print(f"\n  Ungated CSV rows : {len(logger_off.read_rows())}")
    print(f"  Gated CSV rows   : {len(logger_on.read_rows())}")
    print("\n  Gate rejections (gated logger):")
    for key, val in sorted(logger_on.gate_stats.items()):
        if key != "accepted" and val:
            print(f"    {key}: {val}")
    print(f"    accepted: {logger_on.gate_stats.get('accepted', 0)}")
    print("\n  Key insight:")
    print("  Quality-aware logging pairs with quality-aware registration (Anh):")
    print("  only high-confidence identity + emotion events enter the audit trail.")
    print()
    return logger_on


# ---------------------------------------------------------------------------
# PART 3 — Session summary export
# ---------------------------------------------------------------------------

def run_summary_demo(gated_logger: SessionLogger | None = None) -> Path:
    print("=" * 66)
    print("  PART 3 - End-of-Session Summary Export")
    print("=" * 66)

    if gated_logger is not None:
        logger = gated_logger
    else:
        out_dir = Path("justin_innovation") / "demo_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        demo_csv = out_dir / "demo_session.csv"
        logger = SessionLogger(
            str(demo_csv),
            cooldown_sec=0.0,
            gate_enabled=True,
            min_similarity=0.55,
            min_emotion_conf=0.20,
            require_liveness=False,
        )
        logger.maybe_log("justin", 0.88, "happy", 0.45, "happy", 0.52, liveness_passed=True)
        logger.maybe_log("alice", 0.76, "neutral", 0.38, "neutral", 0.40, liveness_passed=True)
        logger.maybe_log("justin", 0.90, "happy", 0.62, "happy", 0.65, liveness_passed=True)

    summary_dir = Path("justin_innovation") / "demo_output"
    summary_dir.mkdir(parents=True, exist_ok=True)
    md_path = logger.export_summary(summary_dir / "demo_session_summary.md")
    json_path = logger.export_summary(summary_dir / "demo_session_summary.json", fmt="json")

    print(f"\n  Markdown : {md_path.resolve()}")
    print(f"  JSON     : {json_path.resolve()}")
    print("\n  Preview (first lines of markdown):")
    for line in md_path.read_text(encoding="utf-8").splitlines()[:12]:
        print(f"    {line}")
    print("    ...")
    print("\n  In FaceAttend: stopping the camera auto-writes logs/session_*_summary.md")
    print()
    return md_path


# ---------------------------------------------------------------------------
# PART 4 — Window size evaluation
# ---------------------------------------------------------------------------

def run_window_eval_demo(
    labels: list[str] | None = None,
    confs: list[float] | None = None,
    csv_path: Path | None = None,
) -> None:
    print("=" * 66)
    print("  PART 4 - Smoothing Window Evaluation (Flicker Rate)")
    print("=" * 66)

    if csv_path and csv_path.exists():
        labels, confs, duration_min = load_emotion_sequence(csv_path)
        print(f"\n  Using real CSV: {csv_path.resolve()}")
    else:
        labels = labels or [lbl for lbl, _ in FLICKERY_SEQUENCE]
        confs = confs or [c for _, c in FLICKERY_SEQUENCE]
        duration_min = max(len(labels) * 0.5 / 60.0, 1e-6)
        print("\n  Using synthetic sequence from PART 1")

    if len(labels) < 2:
        print("  Skipped — need at least 2 emotion samples.")
        return

    windows = [5, 15, 30]
    results = evaluate_windows(labels, confs, duration_min, windows)
    print_report(results, duration_min, len(labels))

    out_dir = Path("justin_innovation") / "demo_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "demo_window_eval.png"
    try:
        from justin_innovation.eval_window import plot_results

        plot_results(results, plot_path, "Justin innovation — flicker vs window size")
        print(f"  Plot saved: {plot_path.resolve()}")
    except Exception as e:
        print(f"  (Plot skipped: {e})")

    print("  Compare with Dhruv's eval_smoother.py for fixed vs adaptive valence windows.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Justin innovation demo (COS30082)")
    p.add_argument(
        "--csv",
        default=None,
        help="Optional logs/attendance_session.csv for PART 4 real-data replay",
    )
    args = p.parse_args()

    print()
    print("=" * 66)
    print("  Emotion Smoothing & Session Logging — Demo")
    print("  COS30082  |  Individual Innovation  |  Justin")
    print("=" * 66)
    print()

    _, _, confs = run_smoothing_demo()
    raw_labels = [lbl for lbl, _ in FLICKERY_SEQUENCE]
    gated_logger = run_gating_demo()
    run_summary_demo(gated_logger)

    csv_path = Path(args.csv) if args.csv else None
    run_window_eval_demo(raw_labels, confs, csv_path=csv_path)

    print("  Demo complete.")
    print("  Next steps:")
    print("    - Live demo : python gradio_app.py  (Settings -> quality-aware logging)")
    print("    - Eval plot : python -m justin_innovation.eval_window --csv logs/attendance_session.csv")
    print()


if __name__ == "__main__":
    main()
