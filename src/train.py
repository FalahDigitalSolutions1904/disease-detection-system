"""
train.py
========
Two-phase training script for the Disease Detection AI.

Phase 1: Train the custom classification head with frozen EfficientNetB4 base.
Phase 2: Fine-tune the top 30 layers of the base model with cosine-decay LR.

Handles class imbalance via class-weighted cross-entropy and label smoothing.
Uses recall-oriented checkpointing (minimising false negatives is critical
in medical screening).

Usage:
    python -m src.train                         # Full training on processed data
    python -m src.train --mode skin_lesion      # Train skin lesion model
    python -m src.train --dummy                 # Quick verification (synthetic data)
    python -m src.train --epochs1 5 --epochs2 3 # Custom epoch counts
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Import configs from utils to maintain the requested file structure
from src.utils import (
    IMAGE_SIZE, BATCH_SIZE, EPOCHS_PHASE1, EPOCHS_PHASE2,
    LEARNING_RATE_PHASE1, LEARNING_RATE_PHASE2, FINE_TUNE_LAYERS,
    MODELS_DIR, WEIGHTS_PATH, PROCESSED_DATA_DIR, ANALYSIS_MODES, DEFAULT_MODE,
)
from src.data_loader import compute_class_weights, build_train_dataset, build_eval_dataset
from src.model import build_model, unfreeze_top_layers


def parse_args():
    parser = argparse.ArgumentParser(description="Train Disease Detection AI model.")
    parser.add_argument("--dummy", action="store_true",
                        help="Run with synthetic data (no real data required).")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE,
                        choices=list(ANALYSIS_MODES.keys()),
                        help="Which clinical mode / dataset to train on.")
    parser.add_argument("--epochs1", type=int, default=EPOCHS_PHASE1,
                        help="Phase 1 epochs (frozen base).")
    parser.add_argument("--epochs2", type=int, default=EPOCHS_PHASE2,
                        help="Phase 2 epochs (fine-tuning).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training data generators.")
    return parser.parse_args()


def generate_synthetic_dataset(num_samples: int = 64, num_classes: int = 4) -> tf.data.Dataset:
    """Generate a synthetic tf.data.Dataset for test/verification runs."""
    images = np.random.uniform(0, 255, (num_samples, *IMAGE_SIZE, 3)).astype(np.float32)
    labels = np.zeros((num_samples, num_classes), dtype=np.float32)
    for i in range(num_samples):
        labels[i, np.random.randint(0, num_classes)] = 1.0

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset.shuffle(num_samples).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def cosine_warmup_schedule(epoch: int, lr: float, warmup: int = 3, total: int = 20) -> float:
    """Cosine LR schedule with linear warmup for stable training."""
    if epoch < warmup:
        return lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1 + np.cos(np.pi * progress))


def main():
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mode_cfg = ANALYSIS_MODES[args.mode]
    class_names = mode_cfg["classes"]
    num_classes = len(class_names)

    train_dir = PROCESSED_DATA_DIR / args.mode / "train"
    val_dir = PROCESSED_DATA_DIR / args.mode / "val"

    # ─── 1. Prepare Datasets ────────────────────────────────────────────────────
    class_weights = None
    if args.dummy or not train_dir.exists():
        if not args.dummy:
            print(f"\n⚠️  Training data not found at: {train_dir}")
            print(f"   Falling back to synthetic dummy data for demo/verification...\n")
        else:
            print("⚠️  --dummy flag set. Using synthetic dataset...")

        train_dataset = generate_synthetic_dataset(128, num_classes)
        val_dataset = generate_synthetic_dataset(32, num_classes)
        class_weights = {i: 1.0 for i in range(num_classes)}
    else:
        print(f"📂 Loading training data from: {train_dir}")
        train_dataset = build_train_dataset(str(train_dir))
        val_dataset = build_eval_dataset(str(val_dir))

        # Determine actual num_classes from data
        class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        num_classes = len(class_dirs)
        class_names = [d.name for d in class_dirs]

        class_weights = compute_class_weights(str(train_dir))
        print(f"⚖️  Class weights: {dict(zip(class_names, class_weights.values()))}")

    # ─── 2. Build Model ─────────────────────────────────────────────────────────
    print(f"🏗️  Building EfficientNetB4 [{num_classes} classes: {class_names}]...")
    model = build_model(num_classes=num_classes, weights="imagenet" if not args.dummy else None)
    model.summary(print_fn=lambda x: None)  # Build graph silently

    total_params = model.count_params()
    print(f"   Total parameters : {total_params:,}")
    print(f"   Trainable params : {sum(tf.size(v).numpy() for v in model.trainable_variables):,}")

    # ─── 3. Callbacks ───────────────────────────────────────────────────────────
    mode_weights_path = str(MODELS_DIR / f"checkpoint.weights.h5")
    checkpoint_path = mode_weights_path.replace(".h5", ".temp_checkpoint.weights.h5")

    callbacks_p1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall", patience=6, mode="max",
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3,
            min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_recall", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: cosine_warmup_schedule(epoch, LEARNING_RATE_PHASE1,
                                                     warmup=3, total=args.epochs1),
            verbose=0,
        ),
    ]

    # ─── PHASE 1: Train Custom Head ─────────────────────────────────────────────
    print(f"\n🚀 Phase 1 — Training Classification Head ({args.epochs1} epochs max)...")
    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs1,
        class_weight=class_weights,
        callbacks=callbacks_p1,
        verbose=1,
    )

    # Load best Phase 1 weights before fine-tuning
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    # ─── PHASE 2: Fine-Tune Top Layers ──────────────────────────────────────────
    print(f"\n🚀 Phase 2 — Fine-tuning top {FINE_TUNE_LAYERS} layers ({args.epochs2} epochs max)...")
    model = unfreeze_top_layers(model, num_layers=FINE_TUNE_LAYERS, lr=LEARNING_RATE_PHASE2)

    trainable_after = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"   Trainable params after unfreeze: {trainable_after:,}")

    callbacks_p2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall", patience=5, mode="max",
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2,
            min_lr=1e-8, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_recall", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs2,
        class_weight=class_weights,
        callbacks=callbacks_p2,
        verbose=1,
    )

    # ─── Save Final Weights ──────────────────────────────────────────────────────
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    model.save_weights(mode_weights_path)
    print(f"\n✅ Training complete! Weights saved → {mode_weights_path}")

    # Cleanup temporary checkpoint files
    for path in [checkpoint_path, checkpoint_path + ".index"]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
