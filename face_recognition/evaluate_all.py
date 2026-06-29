"""Evaluate all three approaches and produce individual + comparison ROC plots."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from face_recognition.model import FaceModel

APPROACHES = [
    {
        "label":          "Approach A: Classification",
        "model_path":     "saved_models/face_classif_model.pth",
        "embedding_type": "backbone",
        "roc_out":        "saved_models/roc_classif.png",
        "color":          "#4FC3F7",
    },
    {
        "label":          "Approach B: Triplet Loss",
        "model_path":     "saved_models/face_triplet_model.pth",
        "embedding_type": "head",
        "roc_out":        "saved_models/roc_triplet.png",
        "color":          "#FFB300",
    },
    {
        "label":          "Approach C: ArcFace",
        "model_path":     "saved_models/face_arcface_model.pth",
        "embedding_type": "backbone",
        "roc_out":        "saved_models/roc_arcface.png",
        "color":          "#00E676",
    },
]

EVAL_TF = transforms.Compose([
    transforms.Resize(112),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default="data/11-785-fall-20-homework-2-part-2")
    p.add_argument("--pairs_file",  default=None)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--save_dir",    default="saved_models")
    return p.parse_args()


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", 4000)
    model = FaceModel(num_classes=num_classes)
    if "backbone_state_dict" in ckpt:
        model.backbone.load_state_dict(ckpt["backbone_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model


def extract_embeddings(model, paths, data_root, batch_size, device, embedding_type):
    embeddings = {}
    path_list = list(paths)
    for start in tqdm(range(0, len(path_list), batch_size), desc="  embeddings", leave=False):
        batch_paths = path_list[start: start + batch_size]
        tensors, valid = [], []
        for p in batch_paths:
            try:
                img = Image.open(os.path.join(data_root, p)).convert("RGB")
                tensors.append(EVAL_TF(img))
                valid.append(p)
            except Exception:
                pass
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            if embedding_type == "head":
                feats = model.get_embedding(batch)
            else:
                feats = F.normalize(model.backbone(batch), p=2, dim=1)
        for path, emb in zip(valid, feats.cpu()):
            embeddings[path] = emb
    return embeddings


def compute_cosine_auc(pairs, embeddings):
    scores, labels, missing = [], [], 0
    for p1, p2, label in pairs:
        if p1 not in embeddings or p2 not in embeddings:
            missing += 1
            continue
        sim = torch.dot(embeddings[p1], embeddings[p2]).item()
        scores.append(sim)
        labels.append(label)
    if missing:
        print(f"    Warning: {missing} pairs skipped (missing embeddings)")
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)
    auc = roc_auc_score(labels_arr, scores_arr)
    fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
    return auc, fpr, tpr


def plot_single(fpr, tpr, auc, label, save_path):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Face Verification")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison(results, save_path):
    plt.figure(figsize=(7, 6))
    for cfg, auc, fpr, tpr in results:
        plt.plot(fpr, tpr, color=cfg["color"], linewidth=2,
                 label=f"{cfg['label']}  (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison — All Approaches")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison ROC saved to: {save_path}")


def main():
    args = parse_args()

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}\n")

    pairs_file = args.pairs_file or os.path.join(args.data_root, "verification_pairs_val.txt")
    pairs = []
    with open(pairs_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                pairs.append((parts[0], parts[1], int(parts[2])))
    unique_paths = {p for pair in pairs for p in (pair[0], pair[1])}
    print(f"Pairs: {len(pairs)}  |  Unique images: {len(unique_paths)}\n")

    os.makedirs(args.save_dir, exist_ok=True)

    results = []
    summary_rows = []

    for cfg in APPROACHES:
        if not os.path.exists(cfg["model_path"]):
            print(f"[SKIP] {cfg['label']} — {cfg['model_path']} not found\n")
            continue

        print(f"=== {cfg['label']} ===")
        model = load_model(cfg["model_path"], device)
        embeddings = extract_embeddings(
            model, unique_paths, args.data_root,
            args.batch_size, device, cfg["embedding_type"],
        )
        auc, fpr, tpr = compute_cosine_auc(pairs, embeddings)
        print(f"  Cosine AUC: {auc:.4f}")

        plot_single(fpr, tpr, auc, cfg["label"], cfg["roc_out"])
        print(f"  ROC saved : {cfg['roc_out']}\n")

        results.append((cfg, auc, fpr, tpr))
        summary_rows.append((cfg["label"], auc))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Combined plot
    if results:
        comparison_path = os.path.join(args.save_dir, "roc_comparison.png")
        plot_comparison(results, comparison_path)

    # Summary table
    print("=" * 50)
    print(f"  {'Approach':<35} {'Cosine AUC':>10}")
    print("=" * 50)
    for name, auc in sorted(summary_rows, key=lambda x: -x[1]):
        print(f"  {name:<35} {auc:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
