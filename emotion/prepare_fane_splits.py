"""
Prepare FANE (or any class-folder emotion dataset) into train/val/test splits.

The raw download is usually a single root with one subfolder per emotion class.
This script performs a per-class random split and writes:

  output_root/train/<class>/*
  output_root/val/<class>/*
  output_root/test/<class>/*

Uses symlinks when possible (Unix or Windows Developer Mode), otherwise copies files.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Stratified train/val/test split for FANE-style folder dataset")
    p.add_argument("--source", required=True, help="Raw dataset root: <source>/<class_name>/images")
    p.add_argument("--output", required=True, help="Output root containing train/, val/, test/")
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", help="Always copy files instead of symlinks")
    return p.parse_args()


def _list_class_images(class_dir: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(class_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(path)
    return out


def _try_link_or_copy(src: Path, dst: Path, force_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst, target_is_directory=False)
    except OSError:
        shutil.copy2(src, dst)


def _split_three_way(
    paths: list[Path],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    n = len(paths)
    if n == 0:
        return [], [], []

    rng.shuffle(paths)
    n_test = max(0, round(n * test_frac))
    n_val = max(0, round(n * val_frac))
    n_train = n - n_val - n_test

    # Ensure coverage when the dataset is tiny
    if n_train <= 0 and n > 0:
        if n_val > 1:
            n_val -= 1
            n_train = 1
        elif n_test > 1:
            n_test -= 1
            n_train = 1
        else:
            n_train = n

    if n_train < 1:
        n_train = min(1, n)
        rem = n - n_train
        n_val = rem // 2
        n_test = rem - n_val

    train_paths = paths[:n_train]
    val_paths = paths[n_train : n_train + n_val]
    test_paths = paths[n_train + n_val :]
    return train_paths, val_paths, test_paths


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)

    frac_sum = args.train_frac + args.val_frac + args.test_frac
    if abs(frac_sum - 1.0) > 1e-4:
        raise SystemExit(f"train_frac + val_frac + test_frac must sum to 1.0 (got {frac_sum})")

    if not source.is_dir():
        raise SystemExit(f"Source not found: {source}")

    class_dirs = sorted([p for p in source.iterdir() if p.is_dir()])
    if not class_dirs:
        raise SystemExit(f"No class subfolders under {source}")

    rng = random.Random(args.seed)
    splits = {"train": 0, "val": 0, "test": 0}

    print("Classes (alphabetical order, used as label indices):")
    for d in class_dirs:
        print(f"  - {d.name}")

    for class_dir in class_dirs:
        label_name = class_dir.name
        images = _list_class_images(class_dir)
        if not images:
            print(f"  Warning: no images under {class_dir}, skipping")
            continue

        for subset_name in ("train", "val", "test"):
            (output / subset_name / label_name).mkdir(parents=True, exist_ok=True)

        train_p, val_p, test_p = _split_three_way(
            images, args.train_frac, args.val_frac, args.test_frac, rng
        )

        for subset_name, plist in ("train", train_p), ("val", val_p), ("test", test_p):
            for src in plist:
                try:
                    rel = src.relative_to(class_dir)
                except ValueError:
                    rel = Path(src.name)
                dst = output / subset_name / label_name / rel
                _try_link_or_copy(src, dst, args.copy)
                splits[subset_name] += 1

    print(f"\nSplit complete under {output.resolve()}")
    print(f"  train: {splits['train']} files")
    print(f"  val:   {splits['val']} files")
    print(f"  test:  {splits['test']} files")
    print("\nTrain with:")
    print(f"  py -m emotion.train --dataset fane --data_root {output}")


if __name__ == "__main__":
    main()
