"""
Contamination Benchmark -- Quality-Aware vs Equal-Weight Registration
Author: Anh Phan  |  COS30082 Individual Innovation

Experiment: vary % of synthetically degraded frames in the registration set.
Expected: EQ similarity drops steeply; QW degrades much more slowly.

Run:
    python -m anh_innovation.benchmark
    python -m anh_innovation.benchmark --model_path saved_models/face_arcface_model.pth
"""

import argparse
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from anh_innovation.quality_registration import QualityAwareRegistration


def degrade(img: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (31, 31), 0)
    shift   = random.choice([-90, 90])
    return np.clip(blurred.astype(int) + shift, 0, 255).astype(np.uint8)


def trial(reg_qw, reg_eq, good_crops, verify_crop, n_bad, rng):
    n_good = len(good_crops) - n_bad
    chosen = rng.sample(good_crops, n_good) if n_good > 0 else []
    bad    = [degrade(rng.choice(good_crops)) for _ in range(n_bad)]
    crops  = chosen + bad
    rng.shuffle(crops)

    emb_qw, _ = reg_qw.register("__t__", crops, verbose=False)
    emb_eq    = reg_eq.register_equal_weight("__t__", crops)
    v_emb     = reg_qw._embed([verify_crop])[0].numpy()
    return float(np.dot(v_emb, emb_qw)), float(np.dot(v_emb, emb_eq))


def run_benchmark(model_path, data_root, n_identities=20, n_reg_frames=8,
                  n_trials=3, seed=42, out_path="anh_innovation/benchmark_results.png"):
    from pathlib import Path

    random.seed(seed)
    rng = random.Random(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\nModel  : {model_path}\n")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    arch = ckpt.get("backbone_arch", "resnet18")

    if arch == "convnext_tiny":
        from torchvision.models import convnext_tiny
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Identity()
        model.load_state_dict(ckpt["backbone_state_dict"])
        print("Backbone: ConvNeXt-Tiny (768-dim)")
    else:
        from face_recognition.model import FaceModel
        model = FaceModel(num_classes=ckpt.get("num_classes", 4000))
        if "backbone_state_dict" in ckpt:
            model.backbone.load_state_dict(ckpt["backbone_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        print("Backbone: ResNet-18 (512-dim)")

    model.eval().to(device)
    reg_qw = QualityAwareRegistration(model, device, arch=arch)
    reg_eq = QualityAwareRegistration(model, device, arch=arch)

    val_dir  = Path(data_root) / "classification_data" / "train_data"
    id_dirs  = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    selected = random.sample(id_dirs, min(n_identities, len(id_dirs)))

    bad_counts = list(range(n_reg_frames))
    contam_pct = [b / n_reg_frames * 100 for b in bad_counts]
    mean_qw = np.zeros(len(bad_counts))
    mean_eq = np.zeros(len(bad_counts))
    tested  = 0

    print("  Contamination ->  " + "  ".join(f"{int(p):>5}%" for p in contam_pct))
    print("  " + "-" * (18 + 9 * len(bad_counts)))

    for id_dir in selected:
        imgs = sorted(id_dir.glob("*.jpg")) + sorted(id_dir.glob("*.png"))
        if len(imgs) < n_reg_frames + 1:
            continue
        crops = [cv2.resize(cv2.imread(str(p)), (112, 112))
                 for p in imgs[:n_reg_frames] if cv2.imread(str(p)) is not None]
        if len(crops) < n_reg_frames:
            continue
        verify_img = cv2.imread(str(imgs[-1]))
        if verify_img is None:
            continue
        verify_crop = cv2.resize(verify_img, (112, 112))

        row_qw, row_eq = [], []
        for n_bad in bad_counts:
            sq_list, se_list = [], []
            for _ in range(n_trials):
                sq, se = trial(reg_qw, reg_eq, crops, verify_crop, n_bad, rng)
                sq_list.append(sq); se_list.append(se)
            row_qw.append(np.mean(sq_list))
            row_eq.append(np.mean(se_list))

        mean_qw += np.array(row_qw)
        mean_eq += np.array(row_eq)
        tested  += 1
        deltas   = [f"{(q-e):>+5.3f}" for q, e in zip(row_qw, row_eq)]
        print(f"  {id_dir.name:<14}  " + "  ".join(deltas))

    if tested == 0:
        print("No identities found. Check --data_root.")
        return

    mean_qw /= tested
    mean_eq /= tested

    print(f"\n  {'='*56}")
    print(f"  Average over {tested} identities ({n_trials} trials each)")
    print(f"  {'Contam%':>8}  {'QW (ours)':>10}  {'EQ (baseline)':>14}  {'Gain':>6}")
    print(f"  {'-'*44}")
    for pct, qw, eq in zip(contam_pct, mean_qw, mean_eq):
        print(f"  {pct:>7.1f}%  {qw:>10.4f}  {eq:>14.4f}  {qw-eq:>+6.4f}")
    print(f"  {'='*56}\n")

    plt.figure(figsize=(8, 5))
    plt.plot(contam_pct, mean_qw, "o-",  color="#2196F3", linewidth=2.2,
             markersize=7, label="Quality-Weighted (ours)")
    plt.plot(contam_pct, mean_eq, "s--", color="#F44336", linewidth=2.2,
             markersize=7, label="Equal-Weight (standard)")
    plt.fill_between(contam_pct, mean_qw, mean_eq,
                     where=(np.array(mean_qw) > np.array(mean_eq)),
                     alpha=0.12, color="#2196F3", label="QW advantage")
    plt.xlabel("Contamination Level (% bad frames in registration set)", fontsize=12)
    plt.ylabel("Cosine Similarity (higher = better match)", fontsize=12)
    plt.title("Quality-Aware Registration vs Equal-Weight Baseline\n"
              "Effect of Degraded Frames on Verification Performance", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(contam_pct, [f"{int(p)}%" for p in contam_pct])
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   default="saved_models/face_classif_model.pth")
    p.add_argument("--data_root",    default="data/11-785-fall-20-homework-2-part-2")
    p.add_argument("--n_identities", type=int, default=20)
    p.add_argument("--n_frames",     type=int, default=8)
    p.add_argument("--n_trials",     type=int, default=3)
    p.add_argument("--out",          default="anh_innovation/benchmark_results.png")
    args = p.parse_args()
    run_benchmark(args.model_path, args.data_root, args.n_identities,
                  args.n_frames, args.n_trials, out_path=args.out)


if __name__ == "__main__":
    main()
