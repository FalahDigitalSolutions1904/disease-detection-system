"""
evaluate.py
===========
Evaluation script for the Disease Detection AI.

Loads the trained model and runs inference on the test split,
computing and logging:
  - Classification report (Precision, Recall, F1-Score)
  - Confusion matrix saved to models/confusion_matrix.png
  - ROC curves saved to models/roc_curves.png
  - Expected Calibration Error (ECE)
  - Results serialized to models/evaluation_results.json

Usage:
    python -m src.evaluate --mode skin_lesion
    python -m src.evaluate --mode chest_xray
    python -m src.evaluate --dummy
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report

from src.utils import (
    IMAGE_SIZE, BATCH_SIZE, MODELS_DIR, PROCESSED_DATA_DIR,
    ANALYSIS_MODES, DEFAULT_MODE,
    plot_confusion_matrix, plot_roc_curves, expected_calibration_error,
)
from src.data_loader import build_eval_dataset
from src.model import build_model
from src.utils import mc_dropout_predict


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Disease Detection AI model.")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE,
                        choices=list(ANALYSIS_MODES.keys()),
                        help="Clinical mode / dataset to evaluate.")
    parser.add_argument("--dummy", action="store_true",
                        help="Run with synthetic data (no real data required).")
    parser.add_argument("--mc_passes", type=int, default=10,
                        help="Number of MC-Dropout passes for uncertainty estimation.")
    return parser.parse_args()


def generate_synthetic_test_data(num_samples: int = 64, num_classes: int = 4):
    """Generate synthetic predictions and labels for testing the evaluation pipeline."""
    y_true = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        y_true[i, np.random.randint(0, num_classes)] = 1.0

    # Simulated model outputs (logits -> softmax)
    raw_logits = np.random.randn(num_samples, num_classes).astype(np.float32)
    y_pred = tf.nn.softmax(raw_logits).numpy()
    return y_true, y_pred


def main():
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mode_cfg = ANALYSIS_MODES[args.mode]
    class_names = mode_cfg["classes"]
    num_classes = len(class_names)

    test_dir = PROCESSED_DATA_DIR / args.mode / "test"

    # ─── 1. Load Data ─────────────────────────────────────────────────────────
    if args.dummy or not test_dir.exists():
        print("Using synthetic evaluation data (no test split found).")
        y_true, y_pred = generate_synthetic_test_data(64, num_classes)
    else:
        print(f"Loading test data from: {test_dir}")
        # Detect actual classes from directory
        class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
        num_classes = len(class_dirs)
        class_names = [d.name for d in class_dirs]

        # Build the model and load weights
        model = build_model(num_classes=num_classes, weights="imagenet")
        weights_path = MODELS_DIR / "checkpoint.weights.h5"
        if weights_path.exists():
            try:
                model.load_weights(str(weights_path))
                print(f"Loaded weights from {weights_path}")
            except Exception as e:
                print(f"Warning: could not load weights — {e}")

        test_dataset = build_eval_dataset(str(test_dir))

        all_preds, all_labels = [], []
        for images, labels in test_dataset:
            preds = model(images, training=False).numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

        y_true = np.concatenate(all_labels, axis=0)
        y_pred = np.concatenate(all_preds, axis=0)

    # ─── 2. Metrics ───────────────────────────────────────────────────────────
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)

    report = classification_report(
        y_true_idx, y_pred_idx,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    ece = expected_calibration_error(y_true, y_pred)
    overall_accuracy = float(report.get("accuracy", 0.0))

    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print(classification_report(y_true_idx, y_pred_idx, target_names=class_names, zero_division=0))
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  Overall Accuracy               : {overall_accuracy:.4f}")
    print("="*60 + "\n")

    # ─── 3. Visualizations ────────────────────────────────────────────────────
    cm_path = str(MODELS_DIR / "confusion_matrix.png")
    roc_path = str(MODELS_DIR / "roc_curves.png")

    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)
    print(f"Confusion matrix saved → {cm_path}")

    plot_roc_curves(y_true, y_pred, class_names, save_path=roc_path)
    print(f"ROC curves saved → {roc_path}")

    # ─── 4. Save JSON Results ─────────────────────────────────────────────────
    results = {
        "mode": args.mode,
        "num_classes": num_classes,
        "class_names": class_names,
        "accuracy": overall_accuracy,
        "ece": ece,
        "per_class_metrics": {
            cls: {
                "precision": report[cls]["precision"] if cls in report else 0,
                "recall": report[cls]["recall"] if cls in report else 0,
                "f1-score": report[cls]["f1-score"] if cls in report else 0,
                "support": report[cls]["support"] if cls in report else 0,
            }
            for cls in class_names
        },
    }

    results_path = MODELS_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
