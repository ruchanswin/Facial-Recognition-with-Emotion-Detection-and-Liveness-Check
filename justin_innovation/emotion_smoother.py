"""Temporal smoothing for real-time emotion labels (reduces frame-to-frame flicker)."""

from collections import Counter, deque


class EmotionSmoother:
    """
    Rolling-window smoother over recent (label, confidence) pairs.

    Uses majority vote; ties are broken by summed confidence in the window.
    """

    def __init__(self, window_size: int = 15, ws: int = None):
        if ws is not None:
            window_size = ws
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self._buffer: deque[tuple[str, float]] = deque(maxlen=window_size)

    def update(self, label: str, confidence: float) -> str:
        self._buffer.append((label, confidence))
        if len(self._buffer) == 1:
            return label

        counts = Counter(lbl for lbl, _ in self._buffer)
        top_count = counts.most_common(1)[0][1]
        candidates = [lbl for lbl, c in counts.items() if c == top_count]

        if len(candidates) == 1:
            return candidates[0]

        scores: dict[str, float] = {}
        for lbl, conf in self._buffer:
            if lbl in candidates:
                scores[lbl] = scores.get(lbl, 0.0) + conf
        return max(scores, key=scores.get)

    def mean_confidence(self, label: str) -> float:
        confs = [c for lbl, c in self._buffer if lbl == label]
        return sum(confs) / len(confs) if confs else 0.0

    def reset(self) -> None:
        self._buffer.clear()
