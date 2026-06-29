"""
Individual Innovation - Confidence-Weighted Emotion Valence Scoring
Author: Dhruv Patel

Problem:
    EmotionSmoother (justin_innovation/emotion_smoother.py) uses a FIXED window=15
    majority vote and throws away confidence entirely.
    A 90%-confident "angry" and a 20%-confident "angry" are treated
    identically. The system also never answers "is the mood good or bad?"

This module adds three things:
  1. VALENCE MAP  - maps 7 emotion labels → float in [-1, +1]
                    based on Russell's (1980) circumplex model of affect.
  2. VALENCE SMOOTHER - confidence-weighted rolling average (not majority
                    vote). High-confidence frames pull the score harder.
  3. ADAPTIVE WINDOW - window shrinks when confidence is high (react
                    faster), grows when confidence is low (stabilise more).
"""

from collections import deque

# Russell (1980) circumplex valence scores
VALENCE_MAP: dict[str, float] = {
    "happy":    1.0,
    "surprise": 0.3,   # ambiguous - slightly positive
    "neutral":  0.0,
    "shy":      0.1,
    "confused": -0.4,
    "sad":      -0.6,
    "fear":     -0.7,
    "angry":    -0.8,
    "disgust":  -0.9,
    "unknown":   0.0,
}

# Thresholds for the categorical label shown in the UI
_POS_THRESH  =  0.25
_NEG_THRESH  = -0.25


class ValenceSmoother:
    """
    Drop-in companion to EmotionSmoother.

    Instead of majority vote it computes:
        valence = Σ(valence(label_i) × conf_i) / Σ(conf_i)
    over an adaptive rolling window.

    Usage (in _camera_loop, right after smoother_snap.update()):
        vs_result = valence_smoother_snap.update(raw, conf)
        result["valence"]   = vs_result["valence"]
        result["sentiment"] = vs_result["sentiment"]
        result["vs_window"] = vs_result["window"]
    """

    def __init__(self, base_window: int = 15):
        self.base_window = base_window
        # Store up to 2× base so the adaptive expansion has room
        self._buffer: deque[tuple[str, float]] = deque(maxlen=base_window * 2)

    # ------------------------------------------------------------------
    # private
    # ------------------------------------------------------------------

    def _adaptive_window(self) -> int:
        """Return current effective window size based on mean confidence."""
        if not self._buffer:
            return self.base_window
        mean_conf = sum(c for _, c in self._buffer) / len(self._buffer)
        if mean_conf > 0.60:
            # Model is confident → react faster
            return max(7, self.base_window // 2)
        elif mean_conf < 0.30:
            # Model is uncertain → smooth harder
            return min(25, self.base_window * 2)
        return self.base_window

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def update(self, label: str, confidence: float) -> dict:
        """
        Push one frame observation and return the current sentiment state.

        Parameters
        ----------
        label      : raw emotion string e.g. "angry" or "😠 angry"
        confidence : model confidence in [0, 1]

        Returns
        -------
        dict with keys:
            valence   : float in [-1, +1]
            sentiment : "positive" | "neutral" | "stressed"
            label     : str  - highest-confidence label in window
            window    : int  - current adaptive window size
        """
        clean = label.split()[-1].lower()
        self._buffer.append((clean, float(confidence)))

        window = self._adaptive_window()
        recent = list(self._buffer)[-window:]

        total_conf = sum(c for _, c in recent)
        if total_conf < 1e-6:
            valence = 0.0
        else:
            valence = sum(
                VALENCE_MAP.get(lbl, 0.0) * c for lbl, c in recent
            ) / total_conf

        # Sentiment label
        if valence >= _POS_THRESH:
            sentiment = "positive"
        elif valence <= _NEG_THRESH:
            sentiment = "stressed"
        else:
            sentiment = "neutral"

        # Best label = highest confidence in the window
        best_label = max(recent, key=lambda x: x[1])[0]

        return {
            "valence":   round(valence, 4),
            "sentiment": sentiment,
            "label":     best_label,
            "window":    window,
        }

    def reset(self) -> None:
        self._buffer.clear()