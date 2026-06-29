"""
Demo -- Quality-Aware Multi-Frame Face Registration
Author: Anh Phan  |  COS30082 Individual Innovation

Run (no model required -- synthetic mode):
    python -m anh_innovation.demo

Run with a trained model (quantitative comparison on real data):
    python -m anh_innovation.demo --model_path saved_models/face_classif_model.pth
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from anh_innovation.quality_registration import FaceQualityScorer, QualityAwareRegistration


def _make_face(brightness: int = 128, blur_ksize: int = 1, seed: int = 0) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    face = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
    target = np.full_like(face, brightness)
    face   = cv2.addWeighted(face, 0.5, target, 0.5, 0)
    if blur_ksize > 1:
        face = cv2.GaussianBlur(face, (blur_ksize, blur_ksize), 0)
    return face


def run_synthetic_demo() -> None:
    print("=" * 62)
    print("  PART 1 - Synthetic Quality Scoring")
    print("  Shows how frames are scored and weighted before embedding.")
    print("=" * 62)

    synthetic_frames = [
        ("Sharp + well-lit",   _make_face(128,  1, seed=1)),
        ("Sharp + well-lit",   _make_face(140,  1, seed=2)),
        ("Slightly blurry",    _make_face(128,  5, seed=3)),
        ("Very blurry",        _make_face(128, 21, seed=4)),
        ("Dark",               _make_face( 30,  1, seed=5)),
        ("Very dark",          _make_face( 10,  3, seed=6)),
        ("Overexposed",        _make_face(230,  1, seed=7)),
        ("Blurry + dark",      _make_face( 40, 15, seed=8)),
    ]

    scorer  = FaceQualityScorer()
    labels  = [lbl for lbl, _ in synthetic_frames]
    crops   = [img for _, img in synthetic_frames]
    scores  = scorer.score_batch(crops)
    temp    = 5.0
    weights = torch.softmax(torch.tensor(scores) * temp, dim=0).numpy()

    print(f"\n  {'#':>3}  {'Description':<22}  {'Sharp':>6}  {'Bright':>6}  {'Total':>6}  {'Weight':>7}  Status")
    print(f"  {'-'*70}")

    threshold = 0.15
    for i, (label, crop, score, weight) in enumerate(zip(labels, crops, scores, weights)):
        s      = scorer.sharpness_score(crop)
        b      = scorer.brightness_score(crop)
        status = "keep" if score >= threshold else "DISCARD"
        w_str  = f"{weight*100:>6.2f}%" if status == "keep" else f"{'--':>7}"
        print(f"  {i+1:>3}  {label:<22}  {s:>6.3f}  {b:>6.3f}  {score:>6.3f}  {w_str}  {status}")

    good = scores >= threshold
    print(f"\n  Frames kept  : {good.sum()}/{len(crops)}")
    print(f"  Softmax temp : {temp}  (higher T -> best frame dominates more)")
    print(f"\n  Key insight:")
    print(f"    Frames 1+2 hold {weights[good][:2].sum()*100:.1f}% of total weight.")
    print(f"    Equal-weight gives each frame exactly {100/len(crops):.1f}%.")
    print()


def run_model_demo(model_path: str, data_root: str, n_identities: int = 15,
                   n_reg_frames: int = 8, seed: int = 42) -> None:
    import random
    import torch.nn as nn
    from pathlib import Path

    print("=" * 62)
    print("  PART 2 - Quantitative Comparison on Real Identities")
    print(f"  Model : {model_path}")
    print("=" * 62)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    arch = ckpt.get("backbone_arch", "resnet18")

    if arch == "convnext_tiny":
        from torchvision.models import convnext_tiny
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Identity()
        model.load_state_dict(ckpt["backbone_state_dict"])
    else:
        from face_recognition.model import FaceModel
        model = FaceModel(num_classes=ckpt.get("num_classes", 4000))
        if "backbone_state_dict" in ckpt:
            model.backbone.load_state_dict(ckpt["backbone_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])

    model.eval().to(device)
    reg_qw = QualityAwareRegistration(model, device, arch=arch)
    reg_eq = QualityAwareRegistration(model, device, arch=arch)
    scorer = FaceQualityScorer()

    val_dir = Path(data_root) / "classification_data" / "train_data"
    if not val_dir.exists():
        print(f"  Data not found at {val_dir} -- skipping Part 2.")
        return

    random.seed(seed)
    id_dirs  = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    selected = random.sample(id_dirs, min(n_identities, len(id_dirs)))

    wins_qw, n_tested = 0, 0
    print(f"\n  {'Identity':<14} {'QW-sim':>7} {'EQ-sim':>7} {'Delta':>7} {'Avg-Q':>7} Winner")
    print(f"  {'-'*56}")

    for id_dir in selected:
        imgs = sorted(id_dir.glob("*.jpg")) + sorted(id_dir.glob("*.png"))
        if len(imgs) < n_reg_frames + 1:
            continue
        crops = [cv2.resize(cv2.imread(str(p)), (112, 112))
                 for p in imgs[:n_reg_frames] if cv2.imread(str(p)) is not None]
        if not crops:
            continue
        verify_img = cv2.imread(str(imgs[-1]))
        if verify_img is None:
            continue

        avg_q      = float(scorer.score_batch(crops).mean())
        emb_qw, _  = reg_qw.register(id_dir.name, crops, verbose=False)
        emb_eq     = reg_eq.register_equal_weight(id_dir.name, crops)
        verify_emb = reg_qw._embed([cv2.resize(verify_img, (112, 112))])[0].numpy()

        sim_qw = float(np.dot(verify_emb, emb_qw))
        sim_eq = float(np.dot(verify_emb, emb_eq))
        delta  = sim_qw - sim_eq
        winner = "QW +" if delta > 0 else ("EQ +" if delta < 0 else "tie")
        if delta > 0:
            wins_qw += 1
        n_tested += 1
        print(f"  {id_dir.name[:14]:<14} {sim_qw:>7.4f} {sim_eq:>7.4f} "
              f"{delta:>+7.4f} {avg_q:>7.3f} {winner}")

    print(f"\n  {'='*56}")
    if n_tested > 0:
        print(f"  Quality-weighted won {wins_qw}/{n_tested} ({wins_qw/n_tested*100:.0f}%)")
    print(f"  QW = quality-weighted  |  EQ = equal-weight  |  Delta = QW - EQ")
    print()


def run_contamination_demo(model_path: str, data_root: str, n_identities: int = 15,
                           n_clean: int = 5, n_bad: int = 3, seed: int = 42) -> None:
    """Part 3: mix good frames with artificially bad ones — QW should win clearly."""
    import random
    import torch.nn as nn
    from pathlib import Path

    print("=" * 62)
    print("  PART 3 - Contamination Scenario (Real-World Simulation)")
    print(f"  {n_clean} clean frames + {n_bad} blurry/dark frames per identity")
    print("=" * 62)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    arch = ckpt.get("backbone_arch", "resnet18")

    if arch == "convnext_tiny":
        from torchvision.models import convnext_tiny
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Identity()
        model.load_state_dict(ckpt["backbone_state_dict"])
    else:
        from face_recognition.model import FaceModel
        model = FaceModel(num_classes=ckpt.get("num_classes", 4000))
        if "backbone_state_dict" in ckpt:
            model.backbone.load_state_dict(ckpt["backbone_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])

    model.eval().to(device)
    reg_qw = QualityAwareRegistration(model, device, arch=arch)
    reg_eq = QualityAwareRegistration(model, device, arch=arch)
    scorer = FaceQualityScorer()

    val_dir = Path(data_root) / "classification_data" / "train_data"
    if not val_dir.exists():
        print(f"  Data not found at {val_dir} -- skipping Part 3.")
        return

    random.seed(seed)
    id_dirs  = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    selected = random.sample(id_dirs, min(n_identities, len(id_dirs)))

    wins_qw, n_tested = 0, 0
    print(f"\n  {'Identity':<14} {'QW-sim':>7} {'EQ-sim':>7} {'Delta':>7} {'BadQ':>6} Winner")
    print(f"  {'-'*58}")

    for id_dir in selected:
        imgs = sorted(id_dir.glob("*.jpg")) + sorted(id_dir.glob("*.png"))
        if len(imgs) < n_clean + 1:
            continue

        clean_crops = [cv2.resize(cv2.imread(str(p)), (112, 112))
                       for p in imgs[:n_clean] if cv2.imread(str(p)) is not None]
        if len(clean_crops) < n_clean:
            continue

        # Bad frames: blur + darken together so quality scorer discards them
        # (blur alone ~0.30 quality; blur+dark drops to ~0.05, below 0.15 threshold)
        base = clean_crops[0]
        bad_crops = []
        for i in range(n_bad):
            blurred = cv2.GaussianBlur(base, (51, 51), 0)
            bad = (blurred * 0.15).astype(np.uint8)   # blurry + dark → quality ~0.05
            bad_crops.append(bad)

        mixed = clean_crops + bad_crops
        random.shuffle(mixed)

        bad_q = float(scorer.score_batch(bad_crops).mean())

        verify_img = cv2.imread(str(imgs[-1]))
        if verify_img is None:
            continue
        verify_crop = cv2.resize(verify_img, (112, 112))

        emb_qw, _  = reg_qw.register(id_dir.name, mixed, verbose=False)
        emb_eq     = reg_eq.register_equal_weight(id_dir.name, mixed)
        verify_emb = reg_qw._embed([verify_crop])[0].numpy()

        sim_qw = float(np.dot(verify_emb, emb_qw))
        sim_eq = float(np.dot(verify_emb, emb_eq))
        delta  = sim_qw - sim_eq
        winner = "QW +" if delta > 0 else ("EQ +" if delta < 0 else "tie")
        if delta > 0:
            wins_qw += 1
        n_tested += 1
        print(f"  {id_dir.name[:14]:<14} {sim_qw:>7.4f} {sim_eq:>7.4f} "
              f"{delta:>+7.4f} {bad_q:>6.3f} {winner}")

    print(f"\n  {'='*58}")
    if n_tested > 0:
        pct = wins_qw / n_tested * 100
        print(f"  Quality-weighted won {wins_qw}/{n_tested} ({pct:.0f}%)"
              f"  [vs Part 2 without contamination]")
    print(f"  Bad frame avg quality: ~0.05-0.15 (below discard threshold 0.15)")
    print(f"  QW = quality-weighted  |  EQ = equal-weight  |  Delta = QW - EQ")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Demo: Quality-Aware Registration")
    p.add_argument("--model_path",   default=None)
    p.add_argument("--data_root",    default="data/11-785-fall-20-homework-2-part-2")
    p.add_argument("--n_identities", type=int, default=15)
    p.add_argument("--n_frames",     type=int, default=8)
    args = p.parse_args()

    print()
    print("=" * 64)
    print("  Quality-Aware Multi-Frame Face Registration -- Demo")
    print("  COS30082  |  Individual Innovation  |  Anh Phan")
    print("=" * 64)
    print()

    run_synthetic_demo()

    if args.model_path:
        run_model_demo(args.model_path, args.data_root, args.n_identities, args.n_frames)
        run_contamination_demo(args.model_path, args.data_root, args.n_identities)
    else:
        print("  (Pass --model_path to run quantitative comparison on real data.)")
        print()


if __name__ == "__main__":
    main()
