"""
Evaluation - Fixed Window vs Adaptive Valence Smoother

Loads the real attendance session CSV (logs/attendance_session.csv),
replays every person's emotion sequence through both smoothers, and
produces:

  1. Console table - per-person stability metrics
  2. Per-person valence timeline plot (PNG)
  3. Aggregate comparison plot: signal variance + reaction latency

Run:
    python -m dhruv_innovation.eval_smoother
    python -m dhruv_innovation.eval_smoother --csv logs/attendance_session.csv
    python -m dhruv_innovation.eval_smoother --csv logs/attendance_session.csv --out_dir dhruv_innovation/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dhruv_innovation.emotion_valence import ValenceSmoother, VALENCE_MAP
from justin_innovation import EmotionSmoother


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session_csv(csv_path: str) -> dict[str, list[tuple[str, float]]]:
    """
    Returns {person_id: [(emotion_label, confidence), ...]} in time order.
    Uses emotion_smooth + emotion_conf_smooth columns.
    Falls back to emotion_raw if smooth columns are missing.
    """
    data: dict[str, list[tuple[str, float]]] = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            person = row.get("person_id", "").strip()
            if not person:
                continue

            # Prefer raw emotion + raw confidence for fair replay
            label = (row.get("emotion_raw") or row.get("emotion_smooth") or "").strip().lower()
            try:
                conf = float(row.get("emotion_conf_raw") or row.get("emotion_conf_smooth") or 0)
            except ValueError:
                conf = 0.0

            if label and label not in ("n/a", "unknown", ""):
                data[person].append((label, conf))

    return dict(data)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(valence_series: list[float]) -> dict:
    """
    Compute stability metrics for a valence time series.
    - variance:         lower = more stable signal
    - mean_abs_delta:   average frame-to-frame change (lower = smoother)
    - reaction_latency: frames until a sustained shift (+/-0.3) is reflected
    """
    arr = np.array(valence_series)
    if len(arr) < 2:
        return {"variance": 0.0, "mean_abs_delta": 0.0, "reaction_latency": 0}

    variance = float(np.var(arr))
    deltas = np.abs(np.diff(arr))
    mean_abs_delta = float(np.mean(deltas))

    # Reaction latency: find first sustained positive-to-negative or v.v. shift
    shift_threshold = 0.30
    latency = 0
    for i in range(1, len(arr)):
        if abs(arr[i] - arr[0]) >= shift_threshold:
            latency = i
            break

    return {
        "variance":        round(variance, 6),
        "mean_abs_delta":  round(mean_abs_delta, 6),
        "reaction_latency": latency,
    }


def replay_sequence(
    sequence: list[tuple[str, float]],
    base_window: int = 15,
) -> tuple[list[float], list[float], list[int]]:
    """
    Replay a sequence through both smoothers.
    Returns (fixed_valences, adaptive_valences, adaptive_windows).
    fixed_valences: derive valence from EmotionSmoother's majority-vote label.
    """
    fixed_smoother   = EmotionSmoother(window_size=base_window)
    valence_smoother = ValenceSmoother(base_window=base_window)

    fixed_valences    = []
    adaptive_valences = []
    adaptive_windows  = []

    for label, conf in sequence:
        fixed_label = fixed_smoother.update(label, conf)
        fixed_valences.append(VALENCE_MAP.get(fixed_label, 0.0))

        vs_out = valence_smoother.update(label, conf)
        adaptive_valences.append(vs_out["valence"])
        adaptive_windows.append(vs_out["window"])

    return fixed_valences, adaptive_valences, adaptive_windows


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(
    results: dict[str, dict],
    base_window: int,
) -> None:
    print()
    print("=" * 80)
    print("  Fixed Window vs Adaptive Valence Smoother - Evaluation Report")
    print(f"  Base window: {base_window} frames  |  Metric: lower variance = more stable")
    print("=" * 80)
    print()
    print(f"  {'Person':<14}  {'Frames':>6}  "
          f"{'Fixed Var':>10}  {'Adapt Var':>10}  {'Improvement':>12}  "
          f"{'Fixed Δ/frame':>14}  {'Adapt Δ/frame':>14}  {'Avg win':>8}")
    print(f"  {'-' * 98}")

    total_fixed_var   = 0.0
    total_adapt_var   = 0.0
    total_fixed_delta = 0.0
    total_adapt_delta = 0.0
    n = 0

    for person, r in sorted(results.items()):
        fv = r["fixed_metrics"]["variance"]
        av = r["adapt_metrics"]["variance"]
        fd = r["fixed_metrics"]["mean_abs_delta"]
        ad = r["adapt_metrics"]["mean_abs_delta"]
        aw = r["avg_adaptive_window"]
        nf = r["n_frames"]

        improvement = ((fv - av) / fv * 100) if fv > 1e-9 else 0.0
        flag = " ✓" if av <= fv else " ✗"

        print(f"  {person:<14}  {nf:>6}  "
              f"{fv:>10.6f}  {av:>10.6f}  {improvement:>+11.1f}%{flag}  "
              f"{fd:>14.6f}  {ad:>14.6f}  {aw:>8.1f}")

        total_fixed_var   += fv
        total_adapt_var   += av
        total_fixed_delta += fd
        total_adapt_delta += ad
        n += 1

    if n:
        avg_improvement = (total_fixed_var - total_adapt_var) / total_fixed_var * 100 \
                          if total_fixed_var > 1e-9 else 0.0
        print(f"  {'─' * 98}")
        print(f"  {'AVERAGE':<14}  {'':>6}  "
              f"{total_fixed_var/n:>10.6f}  {total_adapt_var/n:>10.6f}  "
              f"{avg_improvement:>+11.1f}%   "
              f"{total_fixed_delta/n:>14.6f}  {total_adapt_delta/n:>14.6f}")
    print()

    wins = sum(1 for r in results.values()
               if r["adapt_metrics"]["variance"] <= r["fixed_metrics"]["variance"])
    print(f"  Adaptive smoother ≤ Fixed variance:  {wins}/{n} persons "
          f"({wins/n*100:.0f}%)")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_timelines(
    results: dict[str, dict],
    out_path: str,
) -> None:
    """Per-person valence timeline: fixed vs adaptive on same axes."""
    persons = sorted(results.keys())
    n = len(persons)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, person in zip(axes_flat, persons):
        r = results[person]
        fixed_v  = r["fixed_valences"]
        adapt_v  = r["adaptive_valences"]
        x = list(range(len(fixed_v)))

        ax.plot(x, fixed_v,  color="#F44336", linewidth=1.8,
                linestyle="--", label="Fixed window", alpha=0.85)
        ax.plot(x, adapt_v, color="#2196F3", linewidth=2.0,
                label="Adaptive (ours)", alpha=0.95)
        ax.axhline(0.25,  color="#4CAF50", linestyle=":", linewidth=0.8,
                   label="Positive threshold")
        ax.axhline(-0.25, color="#FF9800", linestyle=":", linewidth=0.8,
                   label="Stress threshold")
        ax.axhline(0,     color="#9E9E9E", linestyle="-",  linewidth=0.5)

        fv = r["fixed_metrics"]["variance"]
        av = r["adapt_metrics"]["variance"]
        ax.set_title(f"{person}  |  var fixed={fv:.4f}  adapt={av:.4f}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Valence", fontsize=8)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for ax in axes_flat[len(persons):]:
        ax.set_visible(False)

    fig.suptitle(
        "Valence Timeline: Fixed Window vs Adaptive Smoother\n"
        "Dhruv Patel - COS30082 Individual Innovation",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Timeline plot saved → {out_path}")


def plot_aggregate(
    results: dict[str, dict],
    out_path: str,
    base_window: int,
) -> None:
    """Bar chart comparing variance and mean-abs-delta across persons."""
    persons   = sorted(results.keys())
    fixed_var = [results[p]["fixed_metrics"]["variance"]       for p in persons]
    adapt_var = [results[p]["adapt_metrics"]["variance"]       for p in persons]
    fixed_dlt = [results[p]["fixed_metrics"]["mean_abs_delta"] for p in persons]
    adapt_dlt = [results[p]["adapt_metrics"]["mean_abs_delta"] for p in persons]

    x = np.arange(len(persons))
    w = 0.38

    fig = plt.figure(figsize=(max(10, len(persons) * 2), 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.45)

    # --- Variance ---
    ax1 = fig.add_subplot(gs[0])
    b1 = ax1.bar(x - w/2, fixed_var, w, label="Fixed window",     color="#F44336", alpha=0.82)
    b2 = ax1.bar(x + w/2, adapt_var, w, label="Adaptive (ours)",  color="#2196F3", alpha=0.82)
    ax1.set_title("Valence Signal Variance  (lower = more stable)",
                  fontweight="bold", fontsize=11)
    ax1.set_ylabel("Variance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(persons)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.0005,
                     f"{h:.4f}", ha="center", va="bottom", fontsize=7)

    # --- Mean abs delta ---
    ax2 = fig.add_subplot(gs[1])
    b3 = ax2.bar(x - w/2, fixed_dlt, w, label="Fixed window",     color="#F44336", alpha=0.82)
    b4 = ax2.bar(x + w/2, adapt_dlt, w, label="Adaptive (ours)",  color="#2196F3", alpha=0.82)
    ax2.set_title("Mean Absolute Frame-to-Frame Valence Change  (lower = smoother)",
                  fontweight="bold", fontsize=11)
    ax2.set_ylabel("Mean |Δ valence|")
    ax2.set_xticks(x)
    ax2.set_xticklabels(persons)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.0002,
                     f"{h:.4f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        f"Adaptive vs Fixed Window Smoother - Aggregate Comparison\n"
        f"Base window = {base_window} frames  |  Dhruv Patel - COS30082",
        fontsize=12, fontweight="bold",
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Aggregate plot saved → {out_path}")


def plot_adaptive_windows(
    results: dict[str, dict],
    out_path: str,
) -> None:
    """Show how the adaptive window size changes over time per person."""
    persons = sorted(results.keys())
    n = len(persons)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, person in zip(axes_flat, persons):
        r = results[person]
        windows = r["adaptive_windows"]
        x = list(range(len(windows)))
        ax.step(x, windows, color="#7B1FA2", linewidth=1.8, where="post")
        ax.axhline(15, color="#9E9E9E", linestyle="--", linewidth=0.8,
                   label="Base window (15)")
        ax.axhline(7,  color="#4CAF50", linestyle=":", linewidth=0.8,
                   label="Min window (7)")
        ax.axhline(25, color="#FF9800", linestyle=":", linewidth=0.8,
                   label="Max window (25)")
        ax.set_title(f"{person}  |  avg={r['avg_adaptive_window']:.1f} frames",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Window size", fontsize=8)
        ax.set_ylim(0, 30)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(persons):]:
        ax.set_visible(False)

    fig.suptitle(
        "Adaptive Window Size Over Time\n"
        "Window shrinks when model is confident, grows when uncertain",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Adaptive window plot saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark fixed vs adaptive valence smoother on session CSV"
    )
    p.add_argument(
        "--csv",
        default="logs/attendance_session.csv",
        help="Path to attendance session CSV",
    )
    p.add_argument(
        "--base_window",
        type=int,
        default=15,
        help="Base window size for both smoothers (default: 15)",
    )
    p.add_argument(
        "--out_dir",
        default="dhruv_innovation",
        help="Output directory for PNG plots",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("  Fixed Window vs Adaptive Valence Smoother - Evaluation")
    print("  COS30082  |  Individual Innovation  |  Dhruv Patel")
    print("=" * 70)

    # Load data
    if not os.path.exists(args.csv):
        print(f"\n  [WARN] CSV not found at {args.csv}")
        print("  Generating synthetic session data for evaluation...\n")
        session_data = _generate_synthetic_session()
    else:
        print(f"\n  Loading session data from: {args.csv}")
        session_data = load_session_csv(args.csv)
        print(f"  Found {len(session_data)} persons: {', '.join(sorted(session_data))}\n")

    if not session_data:
        print("  No data to evaluate.")
        return

    # Replay each person's sequence
    results = {}
    for person, sequence in session_data.items():
        if len(sequence) < 3:
            continue
        fixed_v, adapt_v, adapt_w = replay_sequence(sequence, args.base_window)
        results[person] = {
            "n_frames":           len(sequence),
            "fixed_valences":     fixed_v,
            "adaptive_valences":  adapt_v,
            "adaptive_windows":   adapt_w,
            "fixed_metrics":      compute_metrics(fixed_v),
            "adapt_metrics":      compute_metrics(adapt_v),
            "avg_adaptive_window": float(np.mean(adapt_w)) if adapt_w else args.base_window,
        }

    if not results:
        print("  No persons with enough frames (need ≥ 3).")
        return

    # Console report
    print_report(results, args.base_window)

    # Plots
    out_dir = Path(args.out_dir)
    plot_timelines(
        results,
        str(out_dir / "eval_valence_timelines.png"),
    )
    plot_aggregate(
        results,
        str(out_dir / "eval_aggregate_comparison.png"),
        args.base_window,
    )
    plot_adaptive_windows(
        results,
        str(out_dir / "eval_adaptive_windows.png"),
    )

    print()
    print("  Evaluation complete.")
    print(f"  Plots saved to: {out_dir.resolve()}")
    print()


# ---------------------------------------------------------------------------
# Synthetic session generator (fallback when CSV not present)
# ---------------------------------------------------------------------------

def _generate_synthetic_session() -> dict[str, list[tuple[str, float]]]:
    """Generate a plausible synthetic session so the eval runs standalone."""
    import random
    rng = random.Random(42)

    def make_sequence(emotions_phases: list[tuple[str, float, int]]) -> list:
        seq = []
        for emotion, conf_mean, n in emotions_phases:
            for _ in range(n):
                c = float(np.clip(rng.gauss(conf_mean, 0.05), 0.10, 0.95))
                seq.append((emotion, c))
        return seq

    return {
        "Alice": make_sequence([("happy", 0.80, 5), ("neutral", 0.55, 5), ("happy", 0.85, 5)]),
        "Bob":   make_sequence([("angry", 0.60, 6), ("sad",     0.55, 4), ("neutral", 0.40, 5)]),
        "Claire":make_sequence([("neutral", 0.40, 5), ("happy", 0.70, 5), ("surprise", 0.60, 5)]),
        "David": make_sequence([("sad", 0.65, 7), ("fear", 0.55, 3), ("sad", 0.70, 5)]),
        "Emma":  make_sequence([("happy", 0.88, 6), ("happy", 0.82, 4), ("neutral", 0.50, 5)]),
    }


if __name__ == "__main__":
    main()