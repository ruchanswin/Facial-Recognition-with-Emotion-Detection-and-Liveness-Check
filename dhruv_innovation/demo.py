"""
Demo - Confidence-Weighted Emotion Valence Scoring

Shows three things:
  PART 1 - Synthetic sequence: how adaptive window reacts faster than fixed window
            when the model is confident, and stabilises more when uncertain.
  PART 2 - Side-by-side comparison table of fixed vs adaptive smoothing.
  PART 3 - Team sentiment summary from a synthetic multi-person session.

Run (no model or data required):
    python -m dhruv_innovation.demo
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dhruv_innovation.emotion_valence import ValenceSmoother, VALENCE_MAP
from justin_innovation import EmotionSmoother


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(valence: float, width: int = 30) -> str:
    """ASCII bar: centre=neutral, left=stressed, right=positive."""
    mid = width // 2
    pos = int((valence + 1.0) / 2.0 * width)
    pos = max(0, min(width - 1, pos))
    bar = ["-"] * width
    bar[mid] = "|"
    bar[pos] = "█"
    return "".join(bar)


def _sentiment_color(sentiment: str) -> str:
    return {"positive": "▲", "neutral": "◆", "stressed": "▼"}.get(sentiment, "?")


# ---------------------------------------------------------------------------
# PART 1 - Synthetic scenario walkthrough
# ---------------------------------------------------------------------------

def run_synthetic_demo() -> None:
    print("=" * 66)
    print("  PART 1 - Synthetic Scenario: Emotion Shift Detection")
    print("  Sequence: starts angry/uncertain → shifts to happy/confident")
    print("=" * 66)

    # Simulate a person who starts stressed then becomes happy
    # Each tuple: (emotion_label, confidence)
    sequence = [
        # Uncertain start - model not sure
        ("angry",   0.22),
        ("angry",   0.18),
        ("sad",     0.21),
        ("angry",   0.19),
        ("neutral", 0.25),
        ("angry",   0.20),
        ("sad",     0.23),
        # Transition - confidence rising
        ("neutral", 0.40),
        ("neutral", 0.45),
        ("happy",   0.50),
        # Happy confirmed with high confidence
        ("happy",   0.72),
        ("happy",   0.80),
        ("happy",   0.85),
        ("happy",   0.78),
        ("happy",   0.90),
    ]

    fixed_smoother   = EmotionSmoother(window_size=15)
    valence_smoother = ValenceSmoother(base_window=15)

    print(f"\n  {'#':>3}  {'Label':<10}  {'Conf':>5}  "
          f"{'Fixed smoothed':<14}  {'Valence':>8}  {'Sentiment':<10}  "
          f"{'Window':>6}  Bar [-1 ← 0 → +1]")
    print(f"  {'-' * 90}")

    for i, (label, conf) in enumerate(sequence, 1):
        fixed_out  = fixed_smoother.update(label, conf)
        vs_out     = valence_smoother.update(label, conf)

        valence   = vs_out["valence"]
        sentiment = vs_out["sentiment"]
        window    = vs_out["window"]
        icon      = _sentiment_color(sentiment)

        print(f"  {i:>3}  {label:<10}  {conf:>5.2f}  "
              f"{fixed_out:<14}  {valence:>+8.4f}  "
              f"{icon} {sentiment:<8}  {window:>6}  {_bar(valence)}")

    print()
    print("  Key observations:")
    print("  • Frames 1-7: low confidence → window EXPANDS (smooths noise)")
    print("  • Frames 11-15: high confidence → window SHRINKS (reacts fast)")
    print("  • Fixed smoother still shows 'angry' long after emotion shifted")
    print()


# ---------------------------------------------------------------------------
# PART 2 - Fixed vs Adaptive comparison table
# ---------------------------------------------------------------------------

def run_comparison_demo() -> None:
    print("=" * 66)
    print("  PART 2 - Fixed Window vs Adaptive Window: Side-by-side")
    print("  Scenario: sudden positive shift after prolonged stress")
    print("=" * 66)

    # Phase 1: stressed (long)
    # Phase 2: sudden happy burst with high confidence
    sequence = (
        [("angry",   0.20)] * 8 +
        [("sad",     0.18)] * 4 +
        [("happy",   0.88)] * 6   # sudden confident positive shift
    )

    fixed_smoother   = EmotionSmoother(window_size=15)
    valence_smoother = ValenceSmoother(base_window=15)

    print(f"\n  {'#':>3}  {'Input':<10}  {'Fixed label':<14}  "
          f"{'Adapt valence':>14}  {'Adapt window':>13}  Phase")
    print(f"  {'-' * 72}")

    phases = ["stressed"] * 12 + ["↑ SHIFT"] * 6
    for i, ((label, conf), phase) in enumerate(zip(sequence, phases), 1):
        fixed_out = fixed_smoother.update(label, conf)
        vs_out    = valence_smoother.update(label, conf)
        valence   = vs_out["valence"]
        window    = vs_out["window"]
        print(f"  {i:>3}  {label:<10}  {fixed_out:<14}  "
              f"{valence:>+14.4f}  {window:>13}  {phase}")

    print()
    print("  Key insight:")
    print("  • Fixed smoother keeps reporting 'angry' until window flushes (~15 frames)")
    print("  • Adaptive smoother detects the positive shift within 3-4 frames")
    print("    because high-confidence happy frames shrink the window immediately")
    print()


# ---------------------------------------------------------------------------
# PART 3 - Team sentiment summary
# ---------------------------------------------------------------------------

def run_team_demo() -> None:
    print("=" * 66)
    print("  PART 3 - Team Sentiment Summary (Multi-Person)")
    print("  Simulates 5 employees with different emotional states")
    print("=" * 66)

    team_data = {
        "Alice":  [("happy",   0.82), ("happy",   0.75), ("neutral", 0.60)],
        "Bob":    [("angry",   0.55), ("sad",     0.60), ("angry",   0.70)],
        "Claire": [("neutral", 0.45), ("neutral", 0.50), ("happy",   0.40)],
        "David":  [("sad",     0.65), ("fear",    0.50), ("sad",     0.70)],
        "Emma":   [("happy",   0.90), ("happy",   0.85), ("surprise",0.55)],
    }

    print(f"\n  {'Person':<10}  {'Final Valence':>14}  {'Sentiment':<12}  "
          f"{'Adapt Window':>13}  {'Dominant Emotion'}")
    print(f"  {'-' * 72}")

    all_valences = []
    for person, frames in team_data.items():
        vs = ValenceSmoother(base_window=10)
        result = None
        for label, conf in frames:
            result = vs.update(label, conf)
        valence   = result["valence"]
        sentiment = result["sentiment"]
        window    = result["window"]
        dom_label = result["label"]
        icon      = _sentiment_color(sentiment)
        all_valences.append(valence)
        print(f"  {person:<10}  {valence:>+14.4f}  {icon} {sentiment:<10}  "
              f"{window:>13}  {dom_label}")

    team_mean = sum(all_valences) / len(all_valences)
    if team_mean >= 0.25:
        team_label = "Positive ▲"
        note = "Team morale looks good today!"
    elif team_mean <= -0.25:
        team_label = "Stressed ▼"
        note = "Consider a check-in - team stress detected."
    else:
        team_label = "Neutral ◆"
        note = "Team sentiment is balanced."

    print(f"\n  {'─' * 50}")
    print(f"  Team mean valence : {team_mean:+.4f}")
    print(f"  Team sentiment    : {team_label}")
    print(f"  HR note           : {note}")
    print()


# ---------------------------------------------------------------------------
# PART 4 - VALENCE MAP reference (Russell 1980 justification)
# ---------------------------------------------------------------------------

def print_valence_map() -> None:
    print("=" * 66)
    print("  PART 4 - Valence Map Reference (Russell 1980 Circumplex)")
    print("  Maps emotion labels to valence in [-1, +1]")
    print("=" * 66)
    print()
    print(f"  {'Emotion':<12}  {'Valence':>8}  {'Circumplex position'}")
    print(f"  {'-' * 50}")

    references = {
        "happy":    "High valence, high arousal (Q1)",
        "surprise": "Ambiguous - slightly positive, high arousal",
        "neutral":  "Midpoint - no valence",
        "shy":      "Slightly positive, low arousal",
        "confused": "Slightly negative, uncertain arousal",
        "sad":      "Negative valence, low arousal (Q3)",
        "fear":     "Negative valence, high arousal (Q2)",
        "angry":    "Strong negative valence, high arousal",
        "disgust":  "Strongest negative, moderate arousal",
        "unknown":  "Default - treated as neutral",
    }

    for emotion, note in references.items():
        v = VALENCE_MAP.get(emotion, 0.0)
        bar = _bar(v, width=20)
        print(f"  {emotion:<12}  {v:>+8.3f}  {bar}  {note}")

    print()
    print("  Source: Russell, J.A. (1980). A circumplex model of affect.")
    print("  Journal of Personality and Social Psychology, 39(6), 1161-1178.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 66)
    print("  Confidence-Weighted Emotion Valence Scoring - Demo")
    print("  COS30082  |  Individual Innovation  |  Dhruv Patel")
    print("=" * 66)
    print()

    print_valence_map()
    run_synthetic_demo()
    run_comparison_demo()
    run_team_demo()

    print("  Demo complete.")
    print("  Run eval_smoother.py to benchmark against real session data.")
    print()


if __name__ == "__main__":
    main()