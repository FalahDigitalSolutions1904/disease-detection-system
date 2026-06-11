"""
preprocess_data.py
==================
Data preprocessing script that splits raw data into train/val/test sets
and applies CLAHE enhancement to all images.

Usage:
    python -m src.preprocess_data --mode skin_lesion --val_split 0.15 --test_split 0.10
    python -m src.preprocess_data --mode skin_lesion --dry_run
"""

import os
import cv2
import shutil
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from src.utils import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, ANALYSIS_MODES, DEFAULT_MODE,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, IMAGE_SIZE,
)
from src.data_loader import apply_clahe


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw medical images into train/val/test splits.")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE,
                        choices=list(ANALYSIS_MODES.keys()))
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Fraction of data to reserve for validation.")
    parser.add_argument("--test_split", type=float, default=0.10,
                        help="Fraction of data to reserve for testing.")
    parser.add_argument("--apply_clahe", action="store_true", default=True,
                        help="Apply CLAHE contrast enhancement during preprocessing.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Count files but do not copy anything.")
    return parser.parse_args()


def get_image_paths(directory: Path) -> list:
    """Recursively find all image files in a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    return [p for p in directory.rglob("*") if p.suffix.lower() in extensions]


def preprocess_and_save(src_path: Path, dst_path: Path, apply_clahe_flag: bool = True):
    """Read, optionally CLAHE-enhance, resize, and save an image."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"  WARNING: could not read {src_path}, skipping.")
        return False
    if apply_clahe_flag:
        img = apply_clahe(img)
    img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst_path), img_resized)
    return True


def split_class_images(images: list, val_split: float, test_split: float):
    """Shuffle and split image paths into train/val/test."""
    np.random.seed(42)
    indices = np.random.permutation(len(images))
    n_test = max(1, int(len(images) * test_split))
    n_val = max(1, int(len(images) * val_split))

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    return (
        [images[i] for i in train_idx],
        [images[i] for i in val_idx],
        [images[i] for i in test_idx],
    )


def main():
    args = parse_args()
    np.random.seed(42)

    mode_cfg = ANALYSIS_MODES[args.mode]

    # Locate raw data
    if args.mode == "skin_lesion":
        raw_train = RAW_DATA_DIR / "isic" / "Train"
        raw_test_dir = RAW_DATA_DIR / "isic" / "Test"
    else:
        raw_train = RAW_DATA_DIR / args.mode / "train"
        raw_test_dir = RAW_DATA_DIR / args.mode / "test"

    if not raw_train.exists():
        print(f"Raw data directory not found: {raw_train}")
        print("Please place your data in data/raw/<mode>/train/<class_name>/")
        return

    # Discover class directories
    class_dirs = sorted([d for d in raw_train.iterdir() if d.is_dir()])
    if not class_dirs:
        print("No class subdirectories found in raw training data.")
        return

    print(f"\nMode        : {mode_cfg['name']}")
    print(f"Classes     : {[d.name for d in class_dirs]}")
    print(f"Val split   : {args.val_split:.0%}")
    print(f"Test split  : {args.test_split:.0%}")
    print(f"Apply CLAHE : {args.apply_clahe}")
    print(f"Dry run     : {args.dry_run}\n")

    out_base = PROCESSED_DATA_DIR / args.mode
    total_saved = defaultdict(int)

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = get_image_paths(class_dir)

        if not images:
            print(f"  {class_name}: no images found, skipping.")
            continue

        train_imgs, val_imgs, test_imgs = split_class_images(
            images, args.val_split, args.test_split
        )

        print(f"  {class_name}: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")

        if args.dry_run:
            continue

        for split, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            for src_path in split_imgs:
                dst_path = out_base / split / class_name / src_path.name
                if preprocess_and_save(src_path, dst_path, args.apply_clahe):
                    total_saved[split] += 1

    if not args.dry_run:
        print(f"\nProcessed images saved to: {out_base}")
        for split, count in total_saved.items():
            print(f"  {split}: {count} images")
    else:
        print("\nDry run complete — no files were written.")


if __name__ == "__main__":
    main()
