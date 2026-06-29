"""
This module manages the face registry for enrollment and identification.
The registry is stored as a simple pickle file mapping names to face embeddings.
The `enroll` function takes a name and a list of face crops, scores their quality
with `FaceQualityScorer`, and computes a weighted average embedding to store in the registry.
The `identify` function takes a face embedding and compares it against the registry to find the best match.
"""

import cv2, pickle, torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from utils.face_detector import detect_and_crop_both
from anh_innovation.quality_registration import FaceQualityScorer

REGISTRY_PATH = Path("registered_faces.pkl")

_scorer = FaceQualityScorer()
_SOFTMAX_TEMP = 5.0
_MIN_QUALITY  = 0.15

def load_registry():
    if REGISTRY_PATH.exists():
        return pickle.loads(REGISTRY_PATH.read_bytes())
    return {}

def save_registry(db):
    REGISTRY_PATH.write_bytes(pickle.dumps(db))

def enroll(name: str, frames: list, face_model, device):
    """
    Returns (success: bool, report: str).
    report contains the quality-aware registration summary for display in the UI.
    """
    embeddings, quality_scores = [], []
    no_face_count = 0

    for frame in frames:
        tensor, crop_raw = detect_and_crop_both(frame)
        if tensor is None:
            no_face_count += 1
            continue
        # score the actual detected face crop, not the full frame
        crop_bgr = cv2.cvtColor(crop_raw, cv2.COLOR_RGB2BGR)
        score = _scorer.score(crop_bgr)
        with torch.no_grad():
            emb = face_model.get_backbone_features(tensor.unsqueeze(0).to(device))
        embeddings.append(emb.cpu())
        quality_scores.append(score)

    if not embeddings:
        return False, ""

    scores_t  = torch.tensor(quality_scores, dtype=torch.float32)
    good_mask = scores_t >= _MIN_QUALITY
    if not good_mask.any():
        msg = (f"All {len(quality_scores)} frame(s) scored below quality threshold "
               f"({_MIN_QUALITY}) — try better lighting or a clearer photo.")
        print(f"  [Quality] {msg}")
        return False, msg

    good_embs   = torch.stack(embeddings)[good_mask]
    good_scores = scores_t[good_mask]
    weights     = torch.softmax(good_scores * _SOFTMAX_TEMP, dim=0)

    # Build report string (also printed to terminal)
    total_submitted = len(frames)
    total_detected  = len(quality_scores)
    total_kept      = good_mask.sum().item()

    lines = []
    lines.append(f"── Quality-Aware Registration: '{name}' ──")
    lines.append(f"{total_submitted} frame(s) submitted  →  "
                 f"{total_detected} face(s) detected  →  "
                 f"{total_kept} kept  (threshold ≥ {_MIN_QUALITY})")
    if no_face_count > 0:
        lines.append(f"  ✗ {no_face_count} frame(s) skipped — no face detected (blocked/blurry)")
    lines.append(f"{'Frame':>5}  {'Quality':>7}  {'Weight':>7}  Status")
    lines.append("─" * 42)
    w_idx = 0
    for i, score in enumerate(quality_scores):
        kept = score >= _MIN_QUALITY
        w    = f"{weights[w_idx].item()*100:>5.1f}%  " if kept else "    --  "
        if kept:
            w_idx += 1
        status = "✓ keep" if kept else "✗ discard (low quality)"
        lines.append(f"{i+1:>5}  {score:>7.3f}  {w}{status}")
    lines.append("─" * 42)
    lines.append(f"Top-frame weight: {weights.max().item()*100:.1f}%  |  "
                 f"Equal-weight: {100/len(good_scores):.1f}%")

    report = "\n".join(lines)
    print("\n" + report + "\n")

    weights_exp = weights.unsqueeze(1).unsqueeze(1)
    final       = F.normalize((good_embs * weights_exp).sum(0), p=2, dim=-1)

    db = load_registry()
    db[name] = final
    save_registry(db)
    return True, report

def identify(embedding, threshold=0.55):
    db = load_registry()
    if not db:
        return "Unknown", 0.0
    best_name, best_sim = "Unknown", 0.0
    emb_1d = embedding.squeeze()
    for name, ref_emb in db.items():
        sim = torch.nn.functional.cosine_similarity(
            emb_1d.unsqueeze(0), ref_emb.squeeze().unsqueeze(0)
        ).item()
        if sim > best_sim:
            best_sim, best_name = sim, name
    return (best_name, best_sim) if best_sim > threshold else ("Unknown", best_sim)