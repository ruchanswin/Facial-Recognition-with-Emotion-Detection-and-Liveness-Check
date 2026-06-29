import os
import datetime
import numpy as np
import cv2
from scipy.stats import ks_2samp


class MLOpsDriftMonitor:
    """
    Real-time MLOps monitoring pipeline for:
    - Data skew detection
    - Model drift detection

    Uses:
    - Kolmogorov-Smirnov statistical testing
    - Mean brightness deviation checks
    - Confidence degradation monitoring
    """

    
    def __init__(
      self,
      log_path="logs/mlops_drift_alerts.log",
      buffer_size=4,                   
      skew_p_threshold=0.05,           
      drift_p_threshold=0.05,           
      brightness_shift_threshold=12,     
      confidence_drop_threshold=0.15,
    ):

        self.log_path = log_path
        self.buffer_size = buffer_size

        self.skew_p_threshold = skew_p_threshold
        self.drift_p_threshold = drift_p_threshold

        self.brightness_shift_threshold = brightness_shift_threshold
        self.confidence_drop_threshold = confidence_drop_threshold

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Baseline reference distributions
        self.reference_brightness = []
        self.reference_confidences = []

        # Live telemetry buffers
        self.current_brightness = []
        self.current_confidences = []

        self.baseline_established = False

    # BASELINE CREATION
    def establish_baseline_reference(
        self,
        initial_frames: list,
        target_confidence=0.85
    ):

        self.reference_brightness.clear()
        self.reference_confidences.clear()

        # Fallback synthetic baseline
        if not initial_frames:

            self.reference_brightness = list(
                np.random.normal(120, 15, 200)
            )

            self.reference_confidences = list(
                np.clip(
                    np.random.normal(target_confidence, 0.04, 200),
                    0,
                    1
                )
            )

        else:

            for frame in initial_frames:

                brightness = self.compute_frame_brightness(frame)

                self.reference_brightness.append(brightness)

                # Add slight realistic confidence variance
                noisy_confidence = np.clip(
                    target_confidence + np.random.normal(0, 0.03),
                    0,
                    1
                )

                self.reference_confidences.append(noisy_confidence)

        # Precompute baseline stats
        self.reference_brightness_mean = np.mean(
            self.reference_brightness
        )

        self.reference_brightness_std = np.std(
            self.reference_brightness
        )

        self.reference_confidence_mean = np.mean(
            self.reference_confidences
        )

        self.baseline_established = True

        self.log_event(
            "Baseline reference successfully established."
        )

    # FRAME METRICS
    @staticmethod
    def compute_frame_brightness(frame: np.ndarray) -> float:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # DO NOT round brightness
        return float(np.mean(gray))

    # TELEMETRY COLLECTION
    def collect_live_telemetry(
        self,
        frame: np.ndarray,
        confidence: float
    ) -> dict | None:

        if not self.baseline_established:
            return None

        brightness = self.compute_frame_brightness(frame)

        self.current_brightness.append(brightness)
        self.current_confidences.append(
            float(np.clip(confidence, 0, 1))
        )

        # remove the oldest frame if we exceed buffer size
        if len(self.current_brightness) > self.buffer_size:
            self.current_brightness.pop(0)
            self.current_confidences.pop(0)

        # Execute statistical audit on every single frame once window is primed
        if len(self.current_brightness) == self.buffer_size:
            return self.execute_statistical_audit()

        return None
    
    # DRIFT ANALYSIS
    def execute_statistical_audit(self) -> dict:

        alerts = {
            "data_skew": False,
            "model_drift": False,
            "msg": ""
        }

        # Compute current stats
        current_brightness_mean = np.mean(
            self.current_brightness
        )

        current_confidence_mean = np.mean(
            self.current_confidences
        )

        brightness_shift = abs(
            current_brightness_mean
            - self.reference_brightness_mean
        )

        confidence_drop = (
            self.reference_confidence_mean
            - current_confidence_mean
        )

        # KS Statistical Tests
        _, p_val_skew = ks_2samp(
            self.reference_brightness,
            self.current_brightness
        )

        _, p_val_drift = ks_2samp(
            self.reference_confidences,
            self.current_confidences
        )

        # DATA SKEW DETECTION
        if (
            p_val_skew < self.skew_p_threshold
            and brightness_shift > self.brightness_shift_threshold
        ):

            alerts["data_skew"] = True

            msg = (
                "DATA SKEW: Significant lighting "
                "distribution deviation detected."
            )

            self.log_event(
                f"WARNING: {msg} "
                f"(p={p_val_skew:.4f}, "
                f"brightness_shift={brightness_shift:.2f})"
            )

            alerts["msg"] += msg + " "

        # MODEL DRIFT DETECTION
        if (
            p_val_drift < self.drift_p_threshold
            and confidence_drop > self.confidence_drop_threshold
        ):

            alerts["model_drift"] = True

            msg = (
                "MODEL DRIFT: Confidence scores "
                "degrading below baseline."
            )

            self.log_event(
                f"CRITICAL: {msg} "
                f"(p={p_val_drift:.4f}, "
                f"confidence_drop={confidence_drop:.3f})"
            )

            alerts["msg"] += msg

        # NORMAL STATUS
        if not alerts["data_skew"] and not alerts["model_drift"]:

            self.log_event(
                "Telemetry stable. No significant drift detected."
            )

        return alerts

    # LOGGING
    def log_event(self, message: str):

        timestamp = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        formatted_message = (
            f"[{timestamp}] {message}"
        )

        print(f"[MLOps Engine] {formatted_message}")

        with open(self.log_path, "a") as log_file:
            log_file.write(formatted_message + "\n")