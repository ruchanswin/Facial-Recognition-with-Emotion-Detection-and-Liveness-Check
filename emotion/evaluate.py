"""
Emotion classification evaluation (FER2013 or FANE prepared splits).

Reports accuracy, per-class metrics, saves a confusion matrix and CSV results.
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from emotion.dataset import EMOTION_CLASSES, get_test_loader
from emotion.model import (
    default_confusion_matrix_path,
    default_emotion_model_path,
    default_evaluation_csv_path,
    load_emotion_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate emotion classifier")
    p.add_argument(
        "--dataset",
        choices=("fer2013", "fane_split"),
        default="fer2013",
        help="Must match checkpoint: fane uses data_root/test/",
    )
    p.add_argument("--data_root", default="data/fer2013")
    p.add_argument(
        "--model_path",
        default=None,
        help="Defaults to saved_models/fane_emotion_model.pth or fer2013_emotion_model.pth by --dataset",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", default="saved_models")
    p.add_argument(
        "--confusion_out",
        default=None,
        help="Defaults to save_dir/<dataset>_emotion_model_confusion_matrix.png",
    )
    p.add_argument(
        "--csv_out",
        default=None,
        help="Defaults to save_dir/<dataset>_emotion_model_evaluation_results.csv",
    )
    return p.parse_args()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_preds = []
    all_labels = []

    model.eval()
    for imgs, labels in tqdm(loader, desc="Evaluating"):
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: str) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Emotion Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def save_evaluation_csv(
    save_path: str,
    *,
    dataset: str,
    model_path: str,
    data_root: str,
    class_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cm: np.ndarray,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    overall_accuracy = float((y_true == y_pred).mean())

    fieldnames = [
        "dataset",
        "model_path",
        "data_root",
        "class",
        "precision",
        "recall",
        "f1_score",
        "support",
        "class_accuracy",
        "correct",
        "total",
    ]

    rows: list[dict[str, str | int | float]] = []
    for label in class_names:
        metrics = report[label]
        idx = class_names.index(label)
        total = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        class_acc = correct / total if total else 0.0
        rows.append(
            {
                "dataset": dataset,
                "model_path": model_path,
                "data_root": data_root,
                "class": label,
                "precision": round(metrics["precision"], 6),
                "recall": round(metrics["recall"], 6),
                "f1_score": round(metrics["f1-score"], 6),
                "support": int(metrics["support"]),
                "class_accuracy": round(class_acc, 6),
                "correct": correct,
                "total": total,
            }
        )

    for summary_label in ("macro avg", "weighted avg"):
        metrics = report[summary_label]
        rows.append(
            {
                "dataset": dataset,
                "model_path": model_path,
                "data_root": data_root,
                "class": summary_label,
                "precision": round(metrics["precision"], 6),
                "recall": round(metrics["recall"], 6),
                "f1_score": round(metrics["f1-score"], 6),
                "support": int(metrics["support"]),
                "class_accuracy": "",
                "correct": "",
                "total": "",
            }
        )

    rows.append(
        {
            "dataset": dataset,
            "model_path": model_path,
            "data_root": data_root,
            "class": "overall",
            "precision": "",
            "recall": "",
            "f1_score": "",
            "support": len(y_true),
            "class_accuracy": round(overall_accuracy, 6),
            "correct": int((y_true == y_pred).sum()),
            "total": len(y_true),
        }
    )

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluation results saved to: {save_path}")


def main():
    args = parse_args()
    device = select_device()
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = args.model_path or default_emotion_model_path(args.dataset, args.save_dir)
    confusion_out = args.confusion_out or default_confusion_matrix_path(args.dataset, args.save_dir)
    csv_out = args.csv_out or default_evaluation_csv_path(args.dataset, args.save_dir)

    print(f"Loading model from: {model_path}")
    model, class_names = load_emotion_model(model_path, device)
    if class_names != list(EMOTION_CLASSES):
        print(f"Checkpoint class order: {', '.join(class_names)}")

    loader = get_test_loader(
        args.data_root,
        args.batch_size,
        args.num_workers,
        dataset=[args.dataset],
        class_names=list(class_names),
    )
    y_true, y_pred = collect_predictions(model, loader, device)

    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall accuracy: {accuracy:.4f}\n")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print("Per-class accuracy:")
    for idx, label in enumerate(class_names):
        total = cm[idx].sum()
        class_acc = cm[idx, idx] / total if total else 0.0
        print(f"  {label:8s}: {class_acc:.4f} ({cm[idx, idx]}/{total})")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
    plot_confusion_matrix(cm, class_names, confusion_out)
    save_evaluation_csv(
        csv_out,
        dataset=args.dataset,
        model_path=model_path,
        data_root=args.data_root,
        class_names=class_names,
        y_true=y_true,
        y_pred=y_pred,
        cm=cm,
    )


if __name__ == "__main__":
    main()
