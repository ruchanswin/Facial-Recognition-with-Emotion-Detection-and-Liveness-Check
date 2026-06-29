import cv2
import numpy as np
import tensorflow as tf
import time
import math

from anti_spoof.loader import load_liveness_model


class LivenessDetector:
    """
    Behavioural-fusion liveness detector.

    Combines CNN texture analysis with three behavioural signals:
      - Blink detection (eye state via Haar cascade)
      - Head turn detection (frontal vs. profile cascade + eye-centre offset)
      - Mouth-action detection (open/close geometry ratio)

    Supports two usage modes:
      1. Pipeline mode  — call is_real(face_image, full_frame) to get a
                          (bool, float) verdict compatible with pipeline.py's
                          _safe_is_real() helper.  No drawing is performed.
      2. Standalone mode — call process_frame(frame, show_debug) to get a
                          fully annotated BGR frame (original behaviour).
    """

    def __init__(self, model_path='saved_models/antispoof_v2.h5'):
        self.model = load_liveness_model(model_path)

        # Core cascades
        self.face_cascade    = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade     = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

        # Behavioural signal timeouts (seconds)
        self.timeouts      = {"blink": 15, "mouth_action": 25, "turn": 35}
        self.max_distance  = 100   # px — centroid tracking radius
        self.min_face_size = 120   # px — ignore tiny detections

        # Per-user state memory  {uid: {...}}
        self.user_states = {}
        self.next_id     = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _track_and_assign_id(self, centroid):
        """Assign an existing tracking ID or create a new one."""
        for uid, state in self.user_states.items():
            prev = state["centroid"]
            if math.hypot(centroid[0] - prev[0], centroid[1] - prev[1]) < self.max_distance:
                self.user_states[uid]["centroid"] = centroid
                return uid

        uid = self.next_id
        self.user_states[uid] = {
            "blink": 0, "turn": 0, "mouth_action": 0,
            "centroid": centroid,
            "last_turn_dir": "Center",
            "last_eye_state": "Open",
        }
        self.next_id += 1
        return uid

    def _cnn_score(self, frame_bgr, x, y, w, h, padding=0.2):
        """
        Run the CNN texture classifier on a padded face crop.
        Returns (prediction_real: bool, cnn_confidence: float 0-100).
        """
        p_x = int(max(0, x - w * padding))
        p_y = int(max(0, y - h * padding))
        p_w = int(w * (1 + 2 * padding))
        p_h = int(h * (1 + 2 * padding))
        crop = frame_bgr[p_y:p_y + p_h, p_x:p_x + p_w]

        pred_val = 1.0          # default: assume spoof when crop is empty
        if crop.size > 0:
            rgb      = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized  = cv2.resize(rgb, (224, 224))
            inp      = np.expand_dims(resized.astype("float32") / 255.0, axis=0)
            pred_val = float(self.model.predict(inp, verbose=0)[0][0])

        is_real_cnn    = pred_val < 0.5
        cnn_confidence = (1.0 - pred_val) * 100.0      # higher = more real
        return is_real_cnn, cnn_confidence

    def _update_behavioural_state(self, uid, face_type, roi_gray, w, h, current_time):
        """Update blink / turn / mouth signals for one tracked face."""
        state = self.user_states[uid]

        # --- Head turn ---
        if face_type == "profile":
            state["turn"]          = current_time
            state["last_turn_dir"] = "Sharp Profile"
        else:
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            if len(eyes) == 0:
                # Eyes not visible → blink
                state["blink"]          = current_time
                state["last_eye_state"] = "Blink"
            else:
                state["last_eye_state"] = "Open"
                if len(eyes) == 2:
                    centers = sorted([ex + ew // 2 for (ex, ey, ew, eh) in eyes])
                    if centers[1] < w * 0.48:
                        state["turn"]          = current_time
                        state["last_turn_dir"] = "Right"
                    elif centers[0] > w * 0.52:
                        state["turn"]          = current_time
                        state["last_turn_dir"] = "Left"
                    else:
                        state["last_turn_dir"] = "Center"

        # --- Mouth action ---
        mouths = self.mouth_cascade.detectMultiScale(roi_gray[int(h * 2 / 3):, :], 1.5, 25)
        if len(mouths) > 0:
            mx, my, mw, mh = mouths[0]
            ratio = mh / float(mw)
            if ratio > 0.40 or ratio < 0.18:
                state["mouth_action"] = current_time

    def _compute_verdict(self, uid, cnn_real, cnn_confidence, current_time):
        """
        Combine CNN + behavioural signals into a final liveness decision.

        Returns (is_real: bool, liveness_score: float 0-100).
        """
        state = self.user_states[uid]
        fresh = {
            k: (state[k] != 0 and current_time - state[k] < self.timeouts[k])
            for k in self.timeouts
        }
        active_behavioural = sum(fresh.values())
        total_score        = active_behavioural + (1 if cnn_real else 0)

        is_real = (total_score >= 2 or (fresh["blink"] and cnn_real)) and fresh["blink"]

        boosted = cnn_confidence + active_behavioural * 10
        liveness_score = max(0.0, min(100.0, boosted))

        return is_real, liveness_score, fresh

    def _detect_all_faces(self, gray, frame_width):
        """Return list of ((x,y,w,h), face_type) from all cascades."""
        frontal  = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        profile  = self.profile_cascade.detectMultiScale(gray, 1.3, 5)

        gray_flipped   = cv2.flip(gray, 1)
        profile_flip   = self.profile_cascade.detectMultiScale(gray_flipped, 1.3, 5)

        detections = (
            [(f, "frontal") for f in frontal] +
            [(p, "profile") for p in profile]
        )
        for (px, py, pw, ph) in profile_flip:
            detections.append(((frame_width - px - pw, py, pw, ph), "profile"))

        return detections

    # ------------------------------------------------------------------
    # Pipeline-compatible API
    # ------------------------------------------------------------------

    def is_real(self, face_image, full_frame=None, threshold: float = 0.5):
        """
        Pipeline-compatible liveness check.

        Parameters
        ----------
        face_image : np.ndarray (BGR)
            Pre-cropped face image (from MTCNN / detect_and_crop_both).
            Used for the CNN texture pass when full_frame is not supplied.
        full_frame : np.ndarray (BGR) | None
            The original video frame.  When provided, behavioural signals
            (blink, head turn, mouth) are extracted and fused with the CNN
            score for a stronger verdict.  When None the method falls back
            to CNN-only classification, matching the old predict.py behaviour.
        threshold : float
            Minimum liveness score (0-1) to be considered live.
            Maps to the 0-100 internal scale as threshold * 100.

        Returns
        -------
        (is_live : bool, prob_real : float)
            prob_real is normalised to [0, 1] for compatibility with
            pipeline.py's _safe_is_real() helper.
        """
        # ---- CNN-only fallback (no full frame supplied) ----------------
        if full_frame is None:
            face = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_LINEAR)
            inp  = np.expand_dims(face.astype("float32") / 255.0, axis=0)
            p_spoof  = float(self.model.predict(inp, verbose=0)[0][0])
            prob_real = 1.0 - p_spoof
            return prob_real >= threshold, prob_real

        # ---- Behavioural-fusion path (full frame supplied) -------------
        current_time = time.time()
        gray         = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
        detections   = self._detect_all_faces(gray, full_frame.shape[1])

        best_is_real  = False
        best_score    = 0.0

        seen_ids = []
        for (x, y, w, h), face_type in detections:
            if w < self.min_face_size or h < self.min_face_size:
                continue

            centroid    = (int(x + w / 2), int(y + h / 2))
            uid         = self._track_and_assign_id(centroid)
            if uid in seen_ids:
                continue
            seen_ids.append(uid)

            cnn_real, cnn_conf = self._cnn_score(full_frame, x, y, w, h)

            roi_gray = gray[y:y + h, x:x + w]
            self._update_behavioural_state(uid, face_type, roi_gray, w, h, current_time)

            is_real, liveness_score, _ = self._compute_verdict(uid, cnn_real, cnn_conf, current_time)

            if liveness_score > best_score:
                best_is_real = is_real
                best_score   = liveness_score

        # Purge stale IDs
        self.user_states = {uid: s for uid, s in self.user_states.items() if uid in seen_ids}

        prob_real = best_score / 100.0
        return prob_real >= threshold, prob_real

    # ------------------------------------------------------------------
    # Standalone / debug mode  (original process_frame behaviour)
    # ------------------------------------------------------------------

    def process_frame(self, frame, show_debug=False):
        """Annotate frame in-place and return it.  Original standalone behaviour."""
        current_time = time.time()
        gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections   = self._detect_all_faces(gray, frame.shape[1])

        current_frame_faces = []

        for (x, y, w, h), face_type in detections:
            if w < self.min_face_size or h < self.min_face_size:
                continue

            centroid    = (int(x + w / 2), int(y + h / 2))
            uid         = self._track_and_assign_id(centroid)
            if uid in current_frame_faces:
                continue
            current_frame_faces.append(uid)

            cnn_real, cnn_conf = self._cnn_score(frame, x, y, w, h)

            roi_gray = gray[y:y + h, x:x + w]
            if face_type == "profile":
                self.user_states[uid]["turn"]          = current_time
                self.user_states[uid]["last_turn_dir"] = "Sharp Profile"

            self._update_behavioural_state(uid, face_type, roi_gray, w, h, current_time)

            is_real, liveness_score, fresh = self._compute_verdict(uid, cnn_real, cnn_conf, current_time)
            state = self.user_states[uid]

            # HUD
            color = (0, 255, 0) if is_real else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID {uid}: {'REAL' if is_real else 'ANALYZING'}",
                        (x, y - 10), 0, 0.6, color, 2)

            if not show_debug:
                cv2.putText(frame, f"Liveness: {liveness_score:.1f}%",
                            (x, y + h + 20), 0, 0.5, color, 1)

            if show_debug:
                cv2.rectangle(frame, (x + w + 5, y - 20), (x + w + 230, y + 140), (0, 0, 0), -1)
                for idx, (check, is_fresh) in enumerate(fresh.items()):
                    rem = max(0, int(self.timeouts[check] - (current_time - state[check]))) if state[check] != 0 else 0
                    cv2.putText(frame, f"{check.upper()}: {rem}s",
                                (x + w + 10, y + idx * 20), 0, 0.4,
                                (0, 255, 0) if is_fresh else (0, 0, 255), 1)
                cv2.putText(frame, f"TOTAL SCORE: {liveness_score:.1f}% REAL",
                            (x + w + 10, y + 65), 0, 0.4, color, 1)
                cv2.putText(frame, f"BASE CONFIDENCE: {cnn_conf:.1f}%",
                            (x + w + 10, y + 85), 0, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"LOOKING: {state['last_turn_dir']}",
                            (x + w + 10, y + 105), 0, 0.4, (240, 240, 10), 1)
                cv2.putText(frame, f"EYES: {state['last_eye_state']}",
                            (x + w + 10, y + 125), 0, 0.4, (10, 240, 240), 1)

        self.user_states = {uid: s for uid, s in self.user_states.items() if uid in current_frame_faces}
        return frame


if __name__ == '__main__':
    # detector = LivenessDetector('liveness_model.h5')
    detector = LivenessDetector('antispoof_v3.h5')
    cap = cv2.VideoCapture(0)

    debug_mode = True
    print("Liveness System Running. Press 'd' to toggle debug panel. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detector.process_frame(frame, show_debug=debug_mode)
        cv2.imshow('Final Liveness - Behavioural Fusion Class', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug UI Panel set to: {debug_mode}")

    cap.release()
    cv2.destroyAllWindows()
