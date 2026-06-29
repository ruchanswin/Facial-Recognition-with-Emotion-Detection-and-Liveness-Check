"""
Face recognition pipeline for processing video frames.
This module defines the main pipeline function `run_pipeline` that takes a BGR frame and runs it through the full face recognition process:
1. Detect and crop the face using MTCNN.
2. Perform liveness detection on the cropped face (throttled to every N frames).
3. Extract face embeddings and identify the person (using backbone features for Approach A).
4. Predict emotion from the face crop (throttled to every M frames).

"""

import torch

from utils.face_detector import detect_and_crop_both
from face_recognition.model import FaceModel
from emotion.inference import EmotionPredictor
from anti_spoof.predict import LivenessDetector
from core.registry import identify

_frame_counter = 0
_cached_liveness = (None, 0.0)
_cached_emotion  = (None, 0.0, None)


# helpers
def _safe_is_real(detector: LivenessDetector, 
                frame_rgb,
                full_frame=None, 
                threshold: float = 0.5):
    """
    Normalise LivenessDetector.is_real() return value to (bool, float).

    Score is always P(real) in [0, 1] when a (bool, float) tuple is returned.
    """
    result = detector.is_real(frame_rgb, full_frame=full_frame, threshold=threshold)

    if isinstance(result, (tuple, list)) and len(result) >= 2:
        is_live, score = result[0], result[1]
    elif isinstance(result, bool):
        is_live = result
        score = 1.0 if result else 0.0
    else:
        # bare float (probability of being real)
        score = float(result)
        is_live = score > 0.5
    return bool(is_live), float(score)

def load_models(face_model_path: str,
                emotion_model_path: str,
                antispoof_model_path: str) -> dict:
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    # Face recognition model — auto-detect checkpoint format
    ckpt = torch.load(face_model_path, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", 4000)
    face_model = FaceModel(num_classes=num_classes)
    if "backbone_state_dict" in ckpt:
        # ArcFace checkpoint — only backbone weights are stored
        face_model.backbone.load_state_dict(ckpt["backbone_state_dict"])
    else:
        # Classification or triplet checkpoint — full model weights
        face_model.load_state_dict(ckpt["model_state_dict"])
    face_model.set_mode("embedding")
    face_model.eval().to(device)

    emotion  = EmotionPredictor(emotion_model_path, device)
    liveness = LivenessDetector(antispoof_model_path)

    return {
        "face":    face_model,
        "emotion": emotion,
        "liveness": liveness,
        "device":  device,
    }


def run_pipeline(frame_bgr, models: dict, threshold: float = 0.55,
                 liveness_every: int = 5, emotion_every: int = 3,
                 enforce_liveness: bool = True,
                 liveness_threshold: float = 0.5) -> dict:
    """
    Run the full face recognition pipeline on one BGR frame.

    Returns a dict with keys:
        face_crop      : torch.Tensor | None
        name           : str | None
        similarity     : float
        emotion        : str | None   (e.g. "😊 happy")
        emotion_conf   : float
        is_live        : bool | None
        liveness_score : float
    """
    result = {
        "face_crop":      None,
        "name":           None,
        "similarity":     0.0,
        "emotion":        None,
        "emotion_conf":   0.0,
        "is_live":        None,
        "liveness_score": 0.0,
    }

    global _frame_counter, _cached_liveness, _cached_emotion
    _frame_counter += 1

    # 1. Detect & crop face
    tensor, face_raw = detect_and_crop_both(frame_bgr)
    if tensor is None:
        return result
    result["face_crop"] = tensor

    # 2. Liveness check on cropped face
    if _frame_counter % liveness_every == 1 or _cached_liveness[0] is None:
        _cached_liveness = _safe_is_real(
            models["liveness"],
            face_raw,
            full_frame=frame_bgr,
            threshold=liveness_threshold
        )
    is_live, score = _cached_liveness
    result["is_live"]        = is_live
    result["liveness_score"] = score

    if enforce_liveness and not is_live:
        return result  # skip identity + emotion when spoof is enforced

    # 3. Face embedding → identity lookup
    with torch.no_grad():
        emb = models["face"].get_backbone_features(
            tensor.unsqueeze(0).to(models["device"])
        )
    name, sim = identify(emb.cpu(), threshold)
    result["name"]       = name
    result["similarity"] = sim

    # 4. Emotion
    if _frame_counter % emotion_every == 1 or _cached_emotion[0] is None:
        _cached_emotion = models["emotion"].predict_tensor(tensor)
    label, conf, icon = _cached_emotion
    result["emotion"]      = f"{icon} {label}"
    result["emotion_conf"] = conf

    return result