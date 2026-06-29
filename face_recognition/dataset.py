import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _pin_memory() -> bool:
    # MPS does not support pinned memory; only enable for CUDA
    return torch.cuda.is_available()


def get_train_loader(data_root: str, batch_size: int = 64, num_workers: int = 2) -> DataLoader:
    train_dir = os.path.join(data_root, "classification_data", "train_data")
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    persistent = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )


def get_val_loader(data_root: str, batch_size: int = 64, num_workers: int = 2) -> DataLoader:
    val_dir = os.path.join(data_root, "classification_data", "val_data")
    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(val_dir, transform=transform)
    persistent = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
