"""
data_loader.py
==============
Handles all image preprocessing and dataset pipeline logic:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Advanced data augmentation with MixUp support
  - Class-balanced sampling via computed class weights
  - Separate train vs. eval dataset builders
  - Single-image preprocessing for inference
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import defaultdict

# Import configs from utils to maintain the requested file structure
from src.utils import (
    IMAGE_SIZE, BATCH_SIZE,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
)


# ─── CLAHE Preprocessing ───────────────────────────────────────────────────────

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: tuple = CLAHE_TILE_GRID,
) -> np.ndarray:
    """
    Apply CLAHE to the L-channel of an LAB image.

    Enhances local contrast, which is critical for medical images where
    subtle intensity variations carry diagnostic information.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ─── Class Weight Computation ──────────────────────────────────────────────────

def compute_class_weights(data_dir: str) -> dict:
    """
    Compute class weights from a directory of labelled images.

    Addresses the severe class imbalance problem by assigning higher weights
    to under-represented classes during training.

    Args:
        data_dir: Path to training data directory with subdirectories per class.

    Returns:
        Dictionary mapping class index → weight (float).
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return {}

    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    class_counts = {}

    for idx, class_dir in enumerate(class_dirs):
        count = len([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ])
        class_counts[idx] = max(count, 1)

    total = sum(class_counts.values())
    n_classes = len(class_counts)

    # sklearn-style balanced class weights
    class_weights = {
        idx: total / (n_classes * count)
        for idx, count in class_counts.items()
    }
    return class_weights


# ─── Augmentation Layer ─────────────────────────────────────────────────────────

def get_augmentation_layer() -> tf.keras.Sequential:
    """
    Return a Keras augmentation pipeline for training.

    Medical imaging augmentations are conservative — no extreme distortions,
    since pathological features must remain diagnostically valid.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.10),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ], name="augmentation")


# ─── MixUp Augmentation ────────────────────────────────────────────────────────

def mixup_batch(images: tf.Tensor, labels: tf.Tensor, alpha: float = 0.2) -> tuple:
    """
    Apply MixUp augmentation to a batch.

    MixUp linearly interpolates between random pairs of training examples,
    encouraging the model to behave linearly between training examples.
    Particularly effective for improving generalisation on medical imaging.

    Args:
        images: Batch of images (B, H, W, C).
        labels: One-hot labels (B, num_classes).
        alpha:  Beta distribution concentration parameter.

    Returns:
        Mixed images and labels.
    """
    batch_size = tf.shape(images)[0]
    lam = tf.cast(
        tf.random.stateless_uniform([], seed=(0, 1), minval=0, maxval=1),
        tf.float32,
    )
    lam = tf.maximum(lam, 1.0 - lam)  # Keep lam >= 0.5 for stability

    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam * images + (1.0 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1.0 - lam) * tf.gather(labels, indices)
    return mixed_images, mixed_labels


# ─── CLAHE Batch Wrapper ────────────────────────────────────────────────────────

def tf_clahe_batch(images: tf.Tensor, labels: tf.Tensor) -> tuple:
    """TensorFlow wrapper to apply CLAHE to a batch of images."""
    def _clahe_batch(batch_np):
        enhanced = []
        for img in batch_np:
            img_uint8 = img.astype(np.uint8)
            img_enhanced = apply_clahe(img_uint8)
            enhanced.append(img_enhanced.astype(np.float32))
        return np.stack(enhanced, axis=0)

    enhanced_images = tf.py_function(_clahe_batch, [images], tf.float32)
    enhanced_images.set_shape(images.shape)
    return enhanced_images, labels


# ─── Dataset Builders ──────────────────────────────────────────────────────────

def build_train_dataset(
    data_dir: str,
    image_size: tuple = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    apply_clahe_flag: bool = True,
    use_mixup: bool = True,
) -> tf.data.Dataset:
    """
    Build a training tf.data.Dataset with augmentation, CLAHE, and MixUp.

    Args:
        data_dir:         Path to training split directory.
        image_size:       Target (H, W) for resizing.
        batch_size:       Batch size.
        apply_clahe_flag: Whether to apply CLAHE enhancement.
        use_mixup:        Whether to apply MixUp augmentation.

    Returns:
        Augmented, batched, prefetched tf.data.Dataset.
    """
    augmentation = get_augmentation_layer()

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )

    if apply_clahe_flag:
        dataset = dataset.map(tf_clahe_batch, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if use_mixup:
        dataset = dataset.map(
            lambda x, y: mixup_batch(x, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.prefetch(tf.data.AUTOTUNE)


def build_eval_dataset(
    data_dir: str,
    image_size: tuple = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    apply_clahe_flag: bool = True,
) -> tf.data.Dataset:
    """
    Build an evaluation tf.data.Dataset (no augmentation, no shuffling).

    Args:
        data_dir:         Path to validation/test split directory.
        image_size:       Target (H, W) for resizing.
        batch_size:       Batch size.
        apply_clahe_flag: Whether to apply CLAHE enhancement.

    Returns:
        Batched, prefetched tf.data.Dataset.
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    if apply_clahe_flag:
        dataset = dataset.map(tf_clahe_batch, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.prefetch(tf.data.AUTOTUNE)


# ─── Legacy Wrapper ─────────────────────────────────────────────────────────────

def build_dataset(
    data_dir: str,
    image_size: tuple = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    apply_clahe_flag: bool = True,
) -> tf.data.Dataset:
    """Backward-compatible wrapper — delegates to build_eval_dataset."""
    return build_eval_dataset(data_dir, image_size, batch_size, apply_clahe_flag)


# ─── Single Image Preprocessing ────────────────────────────────────────────────

def preprocess_single_image(
    image_bgr: np.ndarray,
    apply_clahe_flag: bool = True,
    image_size: tuple = IMAGE_SIZE,
) -> np.ndarray:
    """
    Preprocess a single image for inference.

    Args:
        image_bgr:        BGR image (as read by OpenCV).
        apply_clahe_flag: Whether to apply CLAHE enhancement.
        image_size:       Target (H, W) for resizing.

    Returns:
        Preprocessed float32 array with batch dimension (1, H, W, C).
    """
    if apply_clahe_flag:
        image_bgr = apply_clahe(image_bgr)

    img_resized = cv2.resize(image_bgr, image_size, interpolation=cv2.INTER_AREA)
    img_array = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)
    return img_array
