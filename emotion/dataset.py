import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


EMOTION_CLASSES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _pin_memory() -> bool:
    return torch.cuda.is_available()


def _ensure_split_exists(split_dir: str, class_names: list[str]) -> None:
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Dataset split folder not found: {split_dir}")

    missing = [label for label in class_names if not os.path.isdir(os.path.join(split_dir, label))]
    if missing:
        raise FileNotFoundError(
            f"Missing class subfolders under {split_dir}: {', '.join(missing)}. "
            f"Expected {[os.path.join(split_dir, n) for n in class_names[:3]]} ..."
        )


def infer_class_names_from_split(train_split_dir: str) -> list[str]:
    """Sorted subfolder names under train/ define class order for FANE-style datasets."""
    if not os.path.isdir(train_split_dir):
        raise FileNotFoundError(f"Train split not found: {train_split_dir}")
    names = sorted(
        d.name
        for d in Path(train_split_dir).iterdir()
        if d.is_dir()
    )
    if not names:
        raise RuntimeError(f"No class subfolders found under {train_split_dir}")
    return names


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(72),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FolderEmotionDataset(Dataset):
    """Flat face images under root/<class_name>/, with explicit class ordering."""

    def __init__(
        self,
        root: str,
        class_names: list[str],
        transform=None,
        samples: list[tuple[str, int]] | None = None,
    ):
        self.root = root
        self.classes = list(class_names)
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.transform = transform

        _ensure_split_exists(root, self.classes)

        if samples is None:
            self.samples = self._collect_samples()
        else:
            self.samples = samples

        if not self.samples:
            raise RuntimeError(f"No emotion images found under {root}")

    def _collect_samples(self) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        for label in self.classes:
            class_dir = Path(self.root) / label
            target = self.class_to_idx[label]
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((str(path), target))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class FER2013Dataset(FolderEmotionDataset):
    """FER2013 with fixed label order from the README."""

    def __init__(
        self,
        root: str,
        transform=None,
        samples: list[tuple[str, int]] | None = None,
    ):
        super().__init__(
            root,
            class_names=list(EMOTION_CLASSES),
            transform=transform,
            samples=samples,
        )


def _class_counts(samples: list[tuple[str, int]]) -> Counter:
    return Counter(target for _, target in samples)


def get_class_weights(samples: list[tuple[str, int]], num_classes: int, power: float = 1.0) -> torch.Tensor:
    counts = _class_counts(samples)
    total = sum(counts[i] for i in range(num_classes))
    weights = []
    for idx in range(num_classes):
        cnt = counts.get(idx, 0)
        if cnt:
            weights.append((total / (num_classes * cnt)) ** power)
        else:
            weights.append(0.0)
    return torch.tensor(weights, dtype=torch.float32)


def _stratified_split(
    samples: list[tuple[str, int]],
    val_split: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")

    rng = random.Random(seed)
    by_class: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    train_samples: list[tuple[str, int]] = []
    val_samples: list[tuple[str, int]] = []
    for class_samples in by_class.values():
        rng.shuffle(class_samples)
        val_count = max(1, int(round(len(class_samples) * val_split)))
        val_samples.extend(class_samples[:val_count])
        train_samples.extend(class_samples[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def _weighted_sampler(samples: list[tuple[str, int]]) -> WeightedRandomSampler:
    counts = _class_counts(samples)
    sample_weights = [1.0 / counts[target] for _, target in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def _loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool = False, sampler=None) -> DataLoader:
    persistent = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )


def get_train_val_loaders(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 2,
    val_split: float = 0.15,
    seed: int = 42,
    use_weighted_sampler: bool = False,
    class_weight_power: float = 1.0,
) -> tuple[DataLoader, DataLoader, torch.Tensor, list[str]]:
    train_dir = os.path.join(data_root, "train")
    full_dataset = FER2013Dataset(train_dir, transform=None)
    train_samples, val_samples = _stratified_split(full_dataset.samples, val_split, seed)
    num_classes = len(EMOTION_CLASSES)

    train_dataset = FER2013Dataset(train_dir, transform=train_transform(), samples=train_samples)
    val_dataset = FER2013Dataset(train_dir, transform=eval_transform(), samples=val_samples)
    sampler = _weighted_sampler(train_samples) if use_weighted_sampler else None

    train_loader = _loader(train_dataset, batch_size, num_workers, shuffle=True, sampler=sampler)
    val_loader = _loader(val_dataset, batch_size, num_workers, shuffle=False)
    class_weights = get_class_weights(train_samples, num_classes=num_classes, power=class_weight_power)
    return train_loader, val_loader, class_weights, list(EMOTION_CLASSES)


def get_fane_train_val_loaders(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 2,
    use_weighted_sampler: bool = False,
    class_weight_power: float = 1.0,
) -> tuple[DataLoader, DataLoader, torch.Tensor, list[str]]:
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"FANE expects prepared splits: missing {val_dir}. "
            "Run: py -m emotion.prepare_fane_splits --source ... --output ..."
        )

    class_names = infer_class_names_from_split(train_dir)
    missing_val = [
        cn for cn in class_names if not os.path.isdir(os.path.join(val_dir, cn))
    ]
    if missing_val:
        raise FileNotFoundError(f"Validation split missing folders: {', '.join(missing_val)}")

    train_dataset = FolderEmotionDataset(train_dir, class_names, transform=train_transform())
    val_dataset = FolderEmotionDataset(val_dir, class_names, transform=eval_transform())
    sampler = _weighted_sampler(train_dataset.samples) if use_weighted_sampler else None

    train_loader = _loader(train_dataset, batch_size, num_workers, shuffle=True, sampler=sampler)
    val_loader = _loader(val_dataset, batch_size, num_workers, shuffle=False)
    num_classes = len(class_names)
    class_weights = get_class_weights(
        train_dataset.samples, num_classes=num_classes, power=class_weight_power
    )
    return train_loader, val_loader, class_weights, class_names


def get_train_loader(data_root: str, batch_size: int = 64, num_workers: int = 2) -> DataLoader:
    dataset = FER2013Dataset(os.path.join(data_root, "train"), transform=train_transform())
    return _loader(dataset, batch_size, num_workers, shuffle=True)


def get_test_loader(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 2,
    *,
    dataset: list[str] = ["fane_split", "fer2013"],
    class_names: list[str] | None = None,
) -> DataLoader:
    if "fane_split" in dataset:
        test_dir = os.path.join(data_root, "test")
        names = class_names or infer_class_names_from_split(os.path.join(data_root, "train"))
        emotion_dataset = FolderEmotionDataset(test_dir, names, transform=eval_transform())
    if "fer2013" in dataset:
        test_dir = os.path.join(data_root, "test")
        emotion_dataset = FER2013Dataset(test_dir, transform=eval_transform())
    return _loader(emotion_dataset, batch_size, num_workers, shuffle=False)
