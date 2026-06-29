"""
Approach B: Metric learning with semi-hard triplet loss.

Strategy:
  - Uses 512-dim L2-normalised backbone features (no embedding head).
  - Semi-hard mining: negatives where d(a,p_mean) < d(a,n) < d(a,p_mean) + margin.
  - Falls back to hardest negative when no semi-hard negative exists.
  - Low LR (1e-5) preserves backbone features from warm-start.

Usage:
  python -m face_recognition.train_triplet --init_from saved_models/face_classif_model.pth
"""

import argparse
import os
import random
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from face_recognition.model import FaceModel


# ---------------------------------------------------------------------------
# P×K batch sampler
# ---------------------------------------------------------------------------

class PKBatchSampler(Sampler):
    """Yields batches of P*K indices: P identities with K images each."""

    def __init__(self, dataset, P: int = 16, K: int = 8):
        self.P = P
        self.K = K
        self.label_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            self.label_to_indices[label].append(idx)
        self.labels = list(self.label_to_indices.keys())
        self.num_batches = len(dataset) // (P * K)

    def __iter__(self):
        for _ in range(self.num_batches):
            selected = random.sample(self.labels, min(self.P, len(self.labels)))
            batch: list[int] = []
            for lbl in selected:
                pool = self.label_to_indices[lbl]
                chosen = random.sample(pool, self.K) if len(pool) >= self.K \
                         else random.choices(pool, k=self.K)
                batch.extend(chosen)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


# ---------------------------------------------------------------------------
# Semi-hard triplet loss
# ---------------------------------------------------------------------------

def semi_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> tuple[torch.Tensor, int, int]:
    dot  = torch.mm(embeddings, embeddings.t())
    sq   = dot.diagonal().unsqueeze(1)
    dist = (sq + sq.t() - 2.0 * dot).clamp(min=1e-12).sqrt()

    same         = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff         = ~same
    eye          = torch.eye(same.size(0), dtype=torch.bool, device=embeddings.device)
    same_no_self = same & ~eye

    pos_count     = same_no_self.float().sum(dim=1).clamp(min=1)
    pos_dist_mean = (dist * same_no_self.float()).sum(dim=1) / pos_count

    d_ap      = pos_dist_mean.unsqueeze(1)
    semi_mask = diff & (dist > d_ap) & (dist < d_ap + margin)

    semi_dist         = dist.masked_fill(~semi_mask, float("inf"))
    best_semi_hard, _ = semi_dist.min(dim=1)
    has_semi_hard     = semi_mask.any(dim=1)

    hard_neg_dist  = dist.masked_fill(~diff, float("inf"))
    hardest_neg, _ = hard_neg_dist.min(dim=1)

    d_an     = torch.where(has_semi_hard, best_semi_hard, hardest_neg)
    loss_per = (pos_dist_mean - d_an + margin).clamp(min=0.0)

    valid    = same_no_self.any(dim=1)
    n_valid  = int(valid.sum().item())
    n_active = int(((loss_per > 0) & valid).sum().item())

    if n_valid == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True), 0, 0

    return loss_per[valid].mean(), n_valid, n_active


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: FaceModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    margin: float,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss   = 0.0
    total_valid  = 0
    total_active = 0
    n_batches    = 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = F.normalize(model.backbone(imgs), p=2, dim=1)
        loss, n_valid, n_active = semi_hard_triplet_loss(embeddings, labels, margin)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss   += loss.item()
        total_valid  += n_valid
        total_active += n_active
        n_batches    += 1

    avg_loss    = total_loss / max(n_batches, 1)
    active_frac = total_active / max(total_valid, 1)
    return avg_loss, active_frac


@torch.no_grad()
def eval_verification_auc(
    model: FaceModel,
    pairs_file: str,
    data_root: str,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()

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
            embs  = F.normalize(model.backbone(batch), p=2, dim=1).cpu()
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
    p = argparse.ArgumentParser(description="Train triplet-loss face recognition (Approach B)")
    p.add_argument("--data_root",   default="data/11-785-fall-20-homework-2-part-2")
    p.add_argument("--epochs",      type=int,   default=40)
    p.add_argument("--P",           type=int,   default=16,   help="Identities per batch")
    p.add_argument("--K",           type=int,   default=8,    help="Images per identity")
    p.add_argument("--margin",      type=float, default=0.2,  help="Triplet margin")
    p.add_argument("--lr",          type=float, default=1e-5, help="Backbone learning rate")
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--save_dir",    default="saved_models")
    p.add_argument("--resume",      default=None, help="Resume from checkpoint")
    p.add_argument("--init_from",   default=None,
                   help="Warm-start backbone from Approach A classification checkpoint")
    p.add_argument("--eval_every",  type=int,   default=5,
                   help="AUC evaluation interval in epochs (0 = only final)")
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--pairs_file",  default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    #  Dataset & loader 
    train_dir = os.path.join(args.data_root, "classification_data", "train_data")
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset    = datasets.ImageFolder(train_dir, transform=transform)
    num_classes = len(dataset.classes)
    print(f"Identities: {num_classes}  |  Batch: {args.P}P x {args.K}K = {args.P * args.K}")
    print(f"margin={args.margin}  lr={args.lr}  epochs={args.epochs}")

    pk_sampler = PKBatchSampler(dataset, P=args.P, K=args.K)
    pin        = torch.cuda.is_available()
    persistent = args.num_workers > 0
    loader = DataLoader(
        dataset,
        batch_sampler=pk_sampler,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )

    #  Model 
    model = FaceModel(num_classes=num_classes).to(device)
    model.set_mode("embedding")

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Warm start from: {args.init_from}")
    else:
        print("Training from ImageNet-pretrained weights (no warm start)")

    n_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"Backbone params: {n_params:,}\n")

    optimizer = Adam(model.backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    #  Resume 
    start_epoch = 0
    best_auc    = 0.0
    save_path   = os.path.join(args.save_dir, "face_triplet_model.pth")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auc    = ckpt.get("best_auc", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}, best AUC: {best_auc:.4f}")
        for _ in range(start_epoch):
            scheduler.step()

    pairs_file = args.pairs_file or os.path.join(args.data_root, "verification_pairs_val.txt")

    #  Training loop 
    for epoch in range(start_epoch, args.epochs):
        train_loss, active_frac = train_one_epoch(
            model, loader, optimizer, args.margin, device
        )
        scheduler.step()

        do_eval = (
            args.eval_every > 0 and (epoch + 1) % args.eval_every == 0
        ) or epoch == args.epochs - 1

        auc_str = ""
        if do_eval:
            auc     = eval_verification_auc(
                model, pairs_file, args.data_root, args.eval_batch_size, device
            )
            flag    = " *" if auc > best_auc else ""
            auc_str = f"  AUC {auc:.4f}{flag}"
            if auc > best_auc:
                best_auc = auc
                torch.save(
                    {
                        "epoch":                epoch,
                        "model_state_dict":     model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_auc":             best_auc,
                        "num_classes":          num_classes,
                        "embedding_dim":        512,
                    },
                    save_path,
                )

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"loss {train_loss:.4f}  active {active_frac:.1%}"
            + auc_str
        )

    print(f"\nBest cosine AUC: {best_auc:.4f}")
    print(f"Model saved to:  {save_path}")
    print(
        f"\nTo evaluate:\n"
        f"  python -m face_recognition.evaluate "
        f"--model_path {save_path} "
        f"--embedding_type backbone "
        f'--label "Approach B (Triplet)"'
    )


if __name__ == "__main__":
    main()
