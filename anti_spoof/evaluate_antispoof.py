"""
evaluate_antispoof.py
---------------------
Evaluates the trained MobileNetV2 anti-spoofing model (antispoof_v3.h5) on the
'eval' split of the nguyenkhoa/antispoofing-3 HuggingFace dataset.

Outputs
-------
- Console: accuracy, precision, recall, F1, AUC, TP/FP/TN/FN counts
- results/antispoof_confusion_matrix.png  — saved confusion matrix figure
- results/antispoof_roc_curve.png         — saved ROC curve figure

Usage
-----
  python evaluate_antispoof.py
  python evaluate_antispoof.py --max_samples 500   # quick run for testing
"""

import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

# local project imports 
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_liveness_model

try:
    from loader import load_liveness_model
except ImportError:
    # fallback: use build + load_weights only
    import tensorflow as tf
    def load_liveness_model(path):
        m = build_liveness_model()
        m.load_weights(path)
        return m


# helpers

def preprocess(image_pil):
    """Convert a PIL image to a normalised (224, 224, 3) float32 array."""
    arr = np.array(image_pil.convert("RGB"))
    arr = cv2.resize(arr, (224, 224))
    return arr.astype("float32") / 255.0


def collect_predictions(model, split="eval", max_samples=None, threshold=0.5):
    """
    Stream the HuggingFace dataset split and collect ground-truth labels,
    predicted probabilities, and binary predictions.

    Returns
    -------
    y_true   : np.ndarray of int  (0 = live, 1 = spoof)
    y_prob   : np.ndarray of float (P(spoof) from the sigmoid output)
    y_pred   : np.ndarray of int  (thresholded at `threshold`)
    """
    print(f"Loading '{split}' split from nguyenkhoa/antispoofing-3 ...")
    dataset = load_dataset("nguyenkhoa/antispoofing-3", split=split, streaming=True)

    y_true, y_prob = [], []
    count = 0

    for ex in dataset:
        if ex.get("cropped_image") is None:
            continue

        label   = 0 if ex["label"] == 0 else 1
        img_arr = preprocess(ex["cropped_image"])
        batch   = np.expand_dims(img_arr, axis=0)

        p_spoof = float(model.predict(batch, verbose=0)[0][0])
        y_true.append(label)
        y_prob.append(p_spoof)

        count += 1
        if count % 50 == 0:
            print(f"  evaluated {count} samples ...", flush=True)
        if max_samples and count >= max_samples:
            break

    print(f"  done — {count} samples evaluated.\n")

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return y_true, y_prob, y_pred


# plotting 

def plot_confusion_matrix(y_true, y_pred, save_path="results/antispoof_confusion_matrix.png"):
    cm  = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Live (0)", "Spoof (1)"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Anti-Spoofing Confusion Matrix\n(MobileNetV2, eval split)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")
    return cm


def plot_roc_curve(y_true, y_prob, auc_score, save_path="results/antispoof_roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Anti-Spoofing (eval split)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved → {save_path}")


# main 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  default="saved_models/antispoof_v2.h5")
    parser.add_argument("--split",       default="eval",
                        help="Dataset split to evaluate on (default: eval)")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="P(spoof) threshold for binary decision (default: 0.5)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of eval samples (useful for a quick test run)")
    args = parser.parse_args()

    # load model 
    print(f"Loading model from {args.model_path} ...")
    model = load_liveness_model(args.model_path)
    print("Model loaded.\n")

    # collect predictions 
    y_true, y_prob, y_pred = collect_predictions(
        model,
        split=args.split,
        max_samples=args.max_samples,
        threshold=args.threshold,
    )

    # metrics 
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    # confusion_matrix layout: rows=actual, cols=predicted
    # For binary (0=live, 1=spoof):
    #   TN = live predicted live
    #   FP = live predicted spoof
    #   FN = spoof predicted live
    #   TP = spoof predicted spoof
    tn, fp, fn, tp = cm.ravel()

    # Per-class recall
    live_recall  = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # specificity / live recall
    spoof_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # sensitivity / spoof recall

    # print report 
    sep = "=" * 52
    print(sep)
    print("  ANTI-SPOOFING EVALUATION REPORT")
    print(f"  Model : {args.model_path}")
    print(f"  Split : {args.split}   |   Samples : {len(y_true)}")
    print(f"  Threshold : {args.threshold}")
    print(sep)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}   (spoof class)")
    print(f"  Recall    : {rec:.4f}   (spoof class / sensitivity)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  AUC       : {auc:.4f}")
    print()
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"             Pred Live  Pred Spoof")
    print(f"  Act Live   {tn:>9}  {fp:>10}")
    print(f"  Act Spoof  {fn:>9}  {tp:>10}")
    print()
    print(f"  True  Positives (spoof correctly rejected) : {tp}")
    print(f"  True  Negatives (live  correctly accepted) : {tn}")
    print(f"  False Positives (live  flagged as spoof)   : {fp}  ← false rejection rate")
    print(f"  False Negatives (spoof accepted as live)   : {fn}  ← spoof pass-through rate")
    print()
    print(f"  Live  recall (specificity) : {live_recall:.4f}  ({live_recall*100:.2f}%)")
    print(f"  Spoof recall (sensitivity) : {spoof_recall:.4f}  ({spoof_recall*100:.2f}%)")
    print(sep)

    # class distribution 
    n_live  = int((y_true == 0).sum())
    n_spoof = int((y_true == 1).sum())
    print(f"\n  Class distribution in eval split:")
    print(f"    Live  : {n_live}  ({n_live/len(y_true)*100:.1f}%)")
    print(f"    Spoof : {n_spoof}  ({n_spoof/len(y_true)*100:.1f}%)")
    print()

    # save plots 
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_prob, auc)

    print("\nAll done. Copy the numbers above into Section 6.3 of your report.")


if __name__ == "__main__":
    main()
