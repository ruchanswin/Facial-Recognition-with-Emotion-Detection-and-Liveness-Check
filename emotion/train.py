"""
Step 6 — Emotion classification training (FER2013 or FANE).

FER2013: stratified fraction of train/ is held out as validation.

FANE: expects output from emotion.prepare_fane_splits with separate train/, val/, and test/ folders.

Saves best validation checkpoint (saved_models/fane_emotion_model.pth or fer2013_emotion_model.pth).
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from emotion.dataset import (
    get_fane_train_val_loaders,
    get_train_val_loaders,
)
from emotion.model import EmotionModel, default_emotion_model_path


def parse_args():
    p = argparse.ArgumentParser(description="Train emotion classifier (FER2013 or FANE)")
    p.add_argument(
        "--dataset",
        choices=("fer2013", "fane_split"),
        default="fer2013",
        help="fer2013: train/ holds data, val split from train. fane_split: prepared train/, val/, test/ folders.",
    )
    p.add_argument("--data_root", default="data/fer2013")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.15, help="Fraction of train/ held out for validation")
    p.add_argument("--seed", type=int, default=42, help="Random seed for the stratified train/val split")
    p.add_argument(
        "--imbalance_strategy",
        choices=("sqrt_class_weights", "class_weights", "weighted_sampler", "none"),
        default="weighted_sampler",
        help="How to compensate for class imbalance during training",
    )
    p.add_argument("--save_dir", default="saved_models")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument(
        "--freeze_layers",
        default="conv1,bn1,layer1,layer2",
        help="Comma-separated ResNet sub-modules to freeze during fine-tuning",
    )
    return p.parse_args()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def freeze_backbone_layers(model: EmotionModel, freeze_layers: str) -> None:
    freeze = [s.strip() for s in freeze_layers.split(",") if s.strip()]
    for name, param in model.named_parameters():
        if any(name.startswith(f"backbone.{layer}") for layer in freeze):
            param.requires_grad = False


def main():
    args = parse_args()
    device = select_device()
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    use_weighted_sampler = args.imbalance_strategy == "weighted_sampler"
    class_weight_power = 0.5 if args.imbalance_strategy == "sqrt_class_weights" else 1.0
    if args.dataset == "fane_split":
        train_loader, val_loader, class_weights, class_names = get_fane_train_val_loaders(
            args.data_root,
            args.batch_size,
            args.num_workers,
            use_weighted_sampler=use_weighted_sampler,
            class_weight_power=class_weight_power,
        )
    else:
        train_loader, val_loader, class_weights, class_names = get_train_val_loaders(
            args.data_root,
            args.batch_size,
            args.num_workers,
            val_split=args.val_split,
            seed=args.seed,
            use_weighted_sampler=use_weighted_sampler,
            class_weight_power=class_weight_power,
        )

    print(f"Dataset: {args.dataset}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"Train images: {len(train_loader.dataset):,} | Val images: {len(val_loader.dataset):,}")
    print(f"Imbalance strategy: {args.imbalance_strategy}")
    print(
        "Class weights: "
        + ", ".join(f"{label}={weight:.3f}" for label, weight in zip(class_names, class_weights.tolist()))
    )

    model = EmotionModel(num_classes=len(class_names)).to(device)
    freeze_backbone_layers(model, args.freeze_layers)

    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    train_count = sum(p.numel() for p in trainable)
    print(f"Trainable params: {train_count:,}  |  Frozen params: {frozen_count:,}")

    weighted_loss_strategies = {"sqrt_class_weights", "class_weights"}
    loss_weights = class_weights.to(device) if args.imbalance_strategy in weighted_loss_strategies else None
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.label_smoothing)
    optimizer = Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", ckpt.get("best_test_acc", 0.0))
        print(f"Resumed from epoch {ckpt['epoch']}, best val acc: {best_val_acc:.4f}")

    save_path = default_emotion_model_path(args.dataset, args.save_dir)
    print(f"Checkpoint file: {save_path}")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        flag = " *" if val_acc > best_val_acc else ""
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}{flag}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "class_names": list(class_names),
                    "dataset": args.dataset,
                    "val_split": args.val_split,
                    "seed": args.seed,
                    "imbalance_strategy": args.imbalance_strategy,
                },
                save_path,
            )

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
