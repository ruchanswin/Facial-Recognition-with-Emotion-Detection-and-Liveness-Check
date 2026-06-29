"""
Face verification AUC evaluation — cosine and Euclidean similarity on pair list.

Usage:
  # Approach A (classification)
  python -m face_recognition.evaluate --model_path saved_models/face_classif_model.pth

  # Approach B (triplet)
  python -m face_recognition.evaluate --model_path saved_models/face_triplet_model.pth \
      --embedding_type backbone --label "Approach B (Triplet)"

  # Approach C (ArcFace)
  python -m face_recognition.evaluate --model_path saved_models/face_arcface_model.pth \
      --label "Approach C (ArcFace)"

  # All at once
  python -m face_recognition.evaluate_all
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from face_recognition.model import FaceModel


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate face verification AUC")
    p.add_argument("--model_path", default="saved_models/face_classif_model.pth")
    p.add_argument("--data_root", default="data/11-785-fall-20-homework-2-part-2")
    p.add_argument("--pairs_file", default=None, help="Defaults to verification_pairs_val.txt in data_root")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--save_dir", default="saved_models")
    p.add_argument("--label", default="Approach A (Classification)", help="Legend label for ROC plot")
    p.add_argument("--roc_out", default=None, help="Path to save ROC PNG (default: save_dir/roc_classif.png)")
    p.add_argument("--val_acc", action="store_true",
                   help="Also compute closed-set classification accuracy on val_data")
    p.add_argument(
        "--embedding_type",
        choices=["backbone", "head"],
        default="backbone",
        help=(
            "backbone: 512-dim raw backbone features (Approach A). "
            "head: 128-dim L2-normalised embedding head output (Approach B / triplet model)."
        ),
    )
    return p.parse_args()


def load_model(model_path: str, device: torch.device):
    """Auto-detects checkpoint format (ArcFace vs classification/triplet). Returns (model, arch)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    arch = ckpt.get("backbone_arch", "resnet18")

    if arch == "convnext_tiny":
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Identity()
        model.load_state_dict(ckpt["backbone_state_dict"])
        print(f"Backbone: ConvNeXt-Tiny (768-dim)")
    else:
        num_classes = ckpt.get("num_classes", 4000)
        model = FaceModel(num_classes=num_classes)
        if "backbone_state_dict" in ckpt:
            model.backbone.load_state_dict(ckpt["backbone_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        print(f"Backbone: ResNet-18 (512-dim)")

    model.eval()
    model.to(device)
    return model, arch


def parse_pairs(pairs_file: str):
    pairs = []
    with open(pairs_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            path1, path2, label = parts[0], parts[1], int(parts[2])
            pairs.append((path1, path2, label))
    return pairs


def extract_embeddings(model, image_paths, data_root, batch_size, device,
                       embedding_type="backbone", arch="resnet18"):
    """
    Returns dict mapping image path -> embedding tensor.
    embedding_type='backbone': 512-dim raw features (Approach A/C).
    embedding_type='head': 128-dim normalised vectors (Approach B triplet).
    """
    eval_transform = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = {}
    paths = list(image_paths)

    for start in tqdm(range(0, len(paths), batch_size), desc="Extracting embeddings"):
        batch_paths = paths[start : start + batch_size]
        tensors = []
        valid_paths = []
        for p in batch_paths:
            full_path = os.path.join(data_root, p)
            try:
                img = Image.open(full_path).convert("RGB")
                tensors.append(eval_transform(img))
                valid_paths.append(p)
            except Exception as e:
                print(f"  Warning: could not load {full_path}: {e}")

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            if arch == "convnext_tiny":
                features = F.normalize(model(batch), p=2, dim=1)
            elif embedding_type == "head":
                features = model.get_embedding(batch)
            else:
                features = model.backbone(batch)

        for path, emb in zip(valid_paths, features.cpu()):
            embeddings[path] = emb

    return embeddings


def compute_scores(pairs, embeddings):
    cosine_scores = []
    euclid_scores = []
    labels = []
    missing = 0

    for path1, path2, label in pairs:
        if path1 not in embeddings or path2 not in embeddings:
            missing += 1
            continue
        e1 = embeddings[path1]
        e2 = embeddings[path2]

        # Cosine similarity — normalize per vector then take dot product
        e1_norm = F.normalize(e1.unsqueeze(0), p=2, dim=1).squeeze(0)
        e2_norm = F.normalize(e2.unsqueeze(0), p=2, dim=1).squeeze(0)
        cos_sim = torch.dot(e1_norm, e2_norm).item()

        # Euclidean distance on raw features → convert to similarity score
        dist = torch.norm(e1 - e2).item()
        euclid_sim = 1.0 / (1.0 + dist)

        cosine_scores.append(cos_sim)
        euclid_scores.append(euclid_sim)
        labels.append(label)

    if missing:
        print(f"  Warning: {missing} pairs skipped (missing embeddings)")

    return np.array(cosine_scores), np.array(euclid_scores), np.array(labels)


def plot_roc(metrics, save_path, title="ROC Curve — Face Verification"):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc_val) in metrics.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC plot saved to: {save_path}")


def compute_val_accuracy(model, arch, ckpt, data_root, batch_size, device):
    """Closed-set classification accuracy on val_data.

    Approach A: full FaceModel classifier head (model_state_dict checkpoint).
    Approach C: cosine nearest-class-centre using ArcFace weight matrix.
    Approach B: N/A — triplet model has no classifier head.
    """
    val_dir = os.path.join(data_root, "classification_data", "val_data")
    if not os.path.isdir(val_dir):
        print(f"  Val directory not found: {val_dir}")
        return None

    val_tf = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"  Val samples: {len(val_ds)}  |  Classes: {len(val_ds.classes)}")

    has_arcface   = "arcface_state_dict" in ckpt
    has_full_model = "model_state_dict" in ckpt

    if not has_arcface and not has_full_model:
        print("  Val accuracy: N/A — triplet model has no classifier head.")
        return None

    correct, total = 0, 0

    if has_arcface:
        weight = ckpt["arcface_state_dict"]["weight"].to(device)
        w_norm = F.normalize(weight, p=2, dim=1)
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="  val acc (ArcFace)"):
                imgs, labels = imgs.to(device), labels.to(device)
                if arch == "convnext_tiny":
                    emb = F.normalize(model(imgs), p=2, dim=1)
                else:
                    emb = F.normalize(model.backbone(imgs), p=2, dim=1)
                preds = torch.mm(emb, w_norm.t()).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
    else:
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="  val acc (Classification)"):
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

    acc = correct / total * 100
    print(f"\n  Val Accuracy: {correct}/{total} = {acc:.2f}%")
    return acc


def main():
    args = parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    pairs_file = args.pairs_file or os.path.join(args.data_root, "verification_pairs_val.txt")
    model_stem = os.path.splitext(os.path.basename(args.model_path))[0]
    if "arcface" in model_stem:
        default_roc = "roc_arcface.png"
    elif args.embedding_type == "head":
        default_roc = "roc_triplet.png"
    else:
        default_roc = "roc_classif.png"
    roc_out = args.roc_out or os.path.join(args.save_dir, default_roc)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    model, arch = load_model(args.model_path, device)

    print(f"Parsing pairs from: {pairs_file}")
    pairs = parse_pairs(pairs_file)
    print(f"  {len(pairs)} pairs loaded")

    unique_paths = set()
    for p1, p2, _ in pairs:
        unique_paths.add(p1)
        unique_paths.add(p2)
    print(f"  {len(unique_paths)} unique images")

    embeddings = extract_embeddings(model, unique_paths, args.data_root, args.batch_size, device, args.embedding_type, arch)

    cosine_scores, euclid_scores, labels = compute_scores(pairs, embeddings)

    cosine_auc = roc_auc_score(labels, cosine_scores)
    euclid_auc = roc_auc_score(labels, euclid_scores)

    fpr_cos, tpr_cos, _ = roc_curve(labels, cosine_scores)
    fpr_euc, tpr_euc, _ = roc_curve(labels, euclid_scores)

    print(f"\n{'='*50}")
    print(f"  {args.label}")
    print(f"  Cosine similarity AUC :  {cosine_auc:.4f}")
    print(f"  Euclidean distance AUC:  {euclid_auc:.4f}")
    print(f"{'='*50}\n")

    metrics = {
        f"{args.label} (cosine)":   (fpr_cos, tpr_cos, cosine_auc),
        f"{args.label} (euclidean)": (fpr_euc, tpr_euc, euclid_auc),
    }
    plot_roc(metrics, roc_out)

    if args.val_acc:
        print(f"\nComputing val accuracy on: {args.data_root}")
        ckpt_raw = torch.load(args.model_path, map_location=device, weights_only=False)
        compute_val_accuracy(model, arch, ckpt_raw, args.data_root, args.batch_size, device)


if __name__ == "__main__":
    main()
