"""
Step 4 — Approach A: Classification-based face recognition training.

Trains FaceModel as a 4000-class classifier on train_data/, validates top-1
accuracy on val_data/, and saves the best checkpoint to saved_models/.
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
from face_recognition.dataset import get_train_loader, get_val_loader
from face_recognition.model import FaceModel


def parse_args():
    p = argparse.ArgumentParser(description="Train classification-based face recognition")
    p.add_argument("--data_root", default="data/11-785-fall-20-homework-2-part-2")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_dir", default="saved_models")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--freeze_layers", default="conv1,bn1,layer1,layer2",
                   help="Comma-separated backbone sub-modules to freeze (partial fine-tuning)")
    return p.parse_args()


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


def main():
    args = parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    train_loader = get_train_loader(args.data_root, args.batch_size, args.num_workers)
    val_loader = get_val_loader(args.data_root, args.batch_size, args.num_workers)

    # Dataset has 4000 identity folders in train_data/
    num_classes = len(train_loader.dataset.classes)
    print(f"Num classes: {num_classes}")

    model = FaceModel(num_classes=num_classes).to(device)
    model.set_mode("classifier")

    # Partial fine-tuning: freeze early backbone layers, fine-tune the rest.
    # Early layers (conv1, bn1, layer1, layer2) learn generic edges/textures from
    # ImageNet and transfer well without retraining. Later layers (layer3, layer4)
    # and both heads learn face-specific features and are fully updated.
    freeze = [s.strip() for s in args.freeze_layers.split(",") if s.strip()]
    for name, param in model.named_parameters():
        if any(name.startswith(f"backbone.{f}") for f in freeze):
            param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    train_count = sum(p.numel() for p in trainable)
    print(f"Trainable params: {train_count:,}  |  Frozen params: {frozen_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}, best val acc: {best_val_acc:.4f}")

    save_path = os.path.join(args.save_dir, "face_classif_model.pth")

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
                    "num_classes": num_classes,
                },
                save_path,
            )

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
