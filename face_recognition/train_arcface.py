"""
Approach C: ArcFace loss with ResNet-18 or ConvNeXt-Tiny backbone.

Usage:
  python -m face_recognition.train_arcface --arch resnet18 \
      --init_from saved_models/face_classif_model.pth
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def build_backbone(arch: str = "convnext_tiny") -> tuple[nn.Module, int]:
    """Returns (backbone, embedding_dim). resnet18=512-dim, convnext_tiny=768-dim."""
    if arch == "resnet18":
        from face_recognition.model import FaceModel
        fm = FaceModel(num_classes=1)
        return fm.backbone, 512
    backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    embedding_dim = backbone.classifier[2].in_features
    backbone.classifier[2] = nn.Identity()
    return backbone, embedding_dim


# ---------------------------------------------------------------------------
# ArcFace loss head
# ---------------------------------------------------------------------------

class ArcFaceHead(nn.Module):
    """
    Angular margin cross-entropy loss head.
      1. cos(θ) = normalize(embeddings) @ normalize(W).T
      2. Add margin m to the target class angle: cos(θ + m)
      3. Scale by s and apply cross-entropy.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ):
        super().__init__()
        self.margin = margin
        self.scale  = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w         = F.normalize(self.weight, p=2, dim=1)
        cos_theta = torch.mm(embeddings, w.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sin_theta   = (1.0 - cos_theta ** 2).clamp(min=1e-12).sqrt()
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        logits = (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta) * self.scale
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_one_epoch(
    backbone: nn.Module,
    arcface:  ArcFaceHead,
    loader:   DataLoader,
    optimizer: torch.optim.Optimizer,
    device:   torch.device,
) -> float:
    backbone.train()
    arcface.train()

    total_loss = 0.0
    n_batches  = 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        features = F.normalize(backbone(imgs), p=2, dim=1)
        loss = arcface(features, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(arcface.parameters()), max_norm=5.0
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Verification AUC evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_verification_auc(
    backbone:   nn.Module,
    pairs_file: str,
    data_root:  str,
    batch_size: int,
    device:     torch.device,
) -> float:
    backbone.eval()

    eval_tf = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    pairs: list[tuple[str, str, int]] = []
    with open(pairs_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                pairs.append((parts[0], parts[1], int(parts[2])))

    unique_paths = {p for pair in pairs for p in (pair[0], pair[1])}
    paths        = list(unique_paths)

    embeddings: dict[str, torch.Tensor] = {}
    for start in tqdm(range(0, len(paths), batch_size), desc="  eval", leave=False):
        batch_paths = paths[start : start + batch_size]
        tensors, valid = [], []
        for p in batch_paths:
            try:
                img = Image.open(os.path.join(data_root, p)).convert("RGB")
                tensors.append(eval_tf(img))
                valid.append(p)
            except Exception:
                pass
        if tensors:
            batch = torch.stack(tensors).to(device)
            embs  = F.normalize(backbone(batch), p=2, dim=1).cpu()
            for path, emb in zip(valid, embs):
                embeddings[path] = emb

    scores, labels = [], []
    for p1, p2, label in pairs:
        if p1 in embeddings and p2 in embeddings:
            scores.append(torch.dot(embeddings[p1], embeddings[p2]).item())
            labels.append(label)

    if not labels:
        return 0.0
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ArcFace + ConvNeXt-Tiny face recognition (Approach C)"
    )
    p.add_argument("--data_root",        default="data/11-785-fall-20-homework-2-part-2")
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--batch_size",       type=int,   default=128)
    p.add_argument("--margin",           type=float, default=0.5)
    p.add_argument("--scale",            type=float, default=64.0)
    p.add_argument("--backbone_lr",      type=float, default=1e-4)
    p.add_argument("--head_lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",     type=float, default=5e-4)
    p.add_argument("--num_workers",      type=int,   default=2)
    p.add_argument("--save_dir",         default="saved_models")
    p.add_argument("--arch",             default="convnext_tiny",
                   choices=["convnext_tiny", "resnet18"],
                   help="Backbone architecture")
    p.add_argument("--init_from",        default=None,
                   help="Warm-start backbone from a classification checkpoint (.pth)")
    p.add_argument("--resume",           default=None, help="Resume from ArcFace checkpoint")
    p.add_argument("--override_backbone_lr", type=float, default=None,
                   help="Override backbone LR after resuming (e.g. 1e-5 for fine-tuning)")
    p.add_argument("--eval_every",       type=int,   default=5,
                   help="Compute verification AUC every N epochs (0 = only final)")
    p.add_argument("--eval_batch_size",  type=int,   default=64)
    p.add_argument("--pairs_file",       default=None)
    p.add_argument("--device",           default=None,
                   help="Force device: cuda, cpu, mps (auto-detect if omitted)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    #  Dataset 
    train_tf = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dir     = os.path.join(args.data_root, "classification_data", "train_data")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    num_classes   = len(train_dataset.classes)
    arch_label = "ConvNeXt-Tiny (768-dim, 2022)" if args.arch == "convnext_tiny" else "ResNet-18 (512-dim)"
    print(f"Backbone : {arch_label}")
    print(f"Loss     : ArcFace  margin={args.margin}  scale={args.scale}")
    print(f"Identities: {num_classes}  |  Train: {len(train_dataset)}")

    pin        = torch.cuda.is_available()
    persistent = args.num_workers > 0
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )

    #  Model 
    backbone, embedding_dim = build_backbone(args.arch)
    if args.init_from:
        ckpt_init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        backbone.load_state_dict(ckpt_init["model_state_dict"], strict=False)
        print(f"Warm-start backbone from: {args.init_from}")
    backbone = backbone.to(device)
    arcface  = ArcFaceHead(embedding_dim, num_classes, args.margin, args.scale).to(device)
    print(f"Embedding dim: {embedding_dim}")
    print(f"Backbone params : {sum(p.numel() for p in backbone.parameters()):,}")

    #  Optimiser 
    optimizer = Adam(
        [
            {"params": backbone.parameters(), "lr": args.backbone_lr},
            {"params": arcface.parameters(),  "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    #  Resume 
    start_epoch = 0
    best_auc    = 0.0
    save_path   = os.path.join(args.save_dir, "face_arcface_model.pth")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        backbone.load_state_dict(ckpt["backbone_state_dict"])
        arcface.load_state_dict(ckpt["arcface_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auc    = ckpt.get("best_auc", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}, best AUC: {best_auc:.4f}")
        if args.override_backbone_lr is not None:
            optimizer.param_groups[0]["lr"] = args.override_backbone_lr
            print(f"Backbone LR overridden -> {args.override_backbone_lr}")
        for _ in range(start_epoch):
            scheduler.step()

    pairs_file = args.pairs_file or os.path.join(args.data_root, "verification_pairs_val.txt")
    print()

    #  Training loop 
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(backbone, arcface, loader, optimizer, device)
        scheduler.step()

        do_eval = (
            args.eval_every > 0 and (epoch + 1) % args.eval_every == 0
        ) or epoch == args.epochs - 1

        auc_str = ""
        if do_eval:
            auc     = eval_verification_auc(
                backbone, pairs_file, args.data_root, args.eval_batch_size, device
            )
            flag    = " *" if auc > best_auc else ""
            auc_str = f"  AUC {auc:.4f}{flag}"
            if auc > best_auc:
                best_auc = auc
                torch.save(
                    {
                        "epoch":                epoch,
                        "backbone_state_dict":  backbone.state_dict(),
                        "arcface_state_dict":   arcface.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_auc":             best_auc,
                        "num_classes":          num_classes,
                        "embedding_dim":        embedding_dim,
                        "backbone_arch":        args.arch,
                    },
                    save_path,
                )

        print(f"Epoch {epoch + 1:3d}/{args.epochs} | loss {avg_loss:.4f}{auc_str}")

    print(f"\nBest cosine AUC: {best_auc:.4f}")
    print(f"Model saved to:  {save_path}")
    print(
        f"\nTo evaluate:\n"
        f"  python -m face_recognition.evaluate "
        f"--model_path {save_path} "
        f'--label "Approach C (ArcFace + ConvNeXt)"'
    )


if __name__ == "__main__":
    main()
