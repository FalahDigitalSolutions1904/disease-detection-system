"""
utils.py
========
Central configuration and helper functions for:
  - Project paths and hyperparameter settings
  - Monte Carlo Dropout inference for predictive uncertainty estimation
  - Expected Calibration Error (ECE) and clinical risk stratification
  - Plotting training history, confusion matrices, and ROC curves
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ─── Project Paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "checkpoint.weights.h5"

# ─── Image Settings ────────────────────────────────────────────────────────────

IMAGE_SIZE = (380, 380)
IMAGE_SHAPE = (*IMAGE_SIZE, 3)
BATCH_SIZE = 16

# ─── Analysis Modes & Class Definitions ─────────────────────────────────────────

ANALYSIS_MODES = {
    "chest_xray": {
        "name": "Chest X-Ray Analysis",
        "icon": "🫁",
        "classes": ["Normal", "Pneumonia"],
        "description": "Detects Pneumonia vs Normal findings from frontal chest radiographs.",
        "datasets": [
            "Chest X-Ray Images (Pneumonia) — Kaggle / HuggingFace",
            "5,863 JPEG images, 2 classes: Normal & Pneumonia",
        ],
        "hf_source": "keremberke/chest-xray-classification",
    },
    "retinal_scan": {
        "name": "Retinal Scan Analysis",
        "icon": "👁️",
        "classes": ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"],
        "description": "Grades diabetic retinopathy severity from retinal fundus photographs.",
        "datasets": [
            "APTOS 2019 Blindness Detection — HuggingFace",
            "3,662 retinal fundus images, 5 severity grades",
        ],
        "hf_source": "HarryAhnHS/aptos-2019-blindness-detection",
    },
    "skin_lesion": {
        "name": "Skin Lesion Analysis",
        "icon": "🔬",
        "classes": [
            "actinic keratosis",
            "basal cell carcinoma",
            "dermatofibroma",
            "melanoma",
            "nevus",
            "pigmented benign keratosis",
            "seborrheic keratosis",
            "squamous cell carcinoma",
            "vascular lesion"
        ],
        "description": "Classifies dermatoscopic images into 9 classes of skin lesions.",
        "datasets": [
            "Skin Cancer ISIC dataset",
            "9 diagnostic classes",
        ],
        "hf_source": "marmal88/skin_cancer",
    },
}

DEFAULT_MODE = "skin_lesion"

# ─── Training Hyperparameters ──────────────────────────────────────────────────

LEARNING_RATE_PHASE1 = 1e-3        # Frozen base phase
LEARNING_RATE_PHASE2 = 1e-5        # Fine-tuning phase
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 15
DROPOUT_RATE = 0.4                 # Dropout rate for regularization and MC-Dropout
FINE_TUNE_LAYERS = 30              # Number of layers to unfreeze in base
LABEL_SMOOTHING = 0.1              # Label smoothing coefficient
WARMUP_EPOCHS = 3                  # Linear warmup epochs

# ─── MC-Dropout / Uncertainty ──────────────────────────────────────────────────

MC_DROPOUT_PASSES = 30
UNCERTAINTY_THRESHOLDS = {
    "low": 0.3,                    # Below -> "Low Uncertainty (High Confidence)"
    "high": 0.8,                   # Above -> "High Uncertainty (Refer to Specialist)"
}

# ─── Clinical Risk Levels ──────────────────────────────────────────────────────

CLINICAL_RISK_LEVELS = {
    "low": {
        "label": "Low Risk",
        "color": "#4CAF50",
        "icon": "✅",
        "action": "No immediate action required. Consider routine follow-up.",
    },
    "moderate": {
        "label": "Moderate Risk",
        "color": "#FFC107",
        "icon": "⚠️",
        "action": "Further investigation recommended. Schedule specialist consultation.",
    },
    "high": {
        "label": "High Risk",
        "color": "#FF5722",
        "icon": "🚨",
        "action": "Urgent specialist referral recommended. Do not delay evaluation.",
    },
    "critical": {
        "label": "Critical — Immediate Attention",
        "color": "#F44336",
        "icon": "🆘",
        "action": "URGENT: Immediate specialist review required. Potential life-threatening finding.",
    },
}

# ─── CLAHE Defaults ────────────────────────────────────────────────────────────

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# ─── Grad-CAM ──────────────────────────────────────────────────────────────────

GRADCAM_ALPHA = 0.4                # Heatmap overlay blend factor


# ─── MC-Dropout Inference & Uncertainty ────────────────────────────────────────

def mc_dropout_predict(
    model: tf.keras.Model,
    img_array: np.ndarray,
    n_passes: int = MC_DROPOUT_PASSES,
) -> dict:
    """
    Run Monte Carlo Dropout inference over N stochastic forward passes.
    Enables dropout layers during inference to capture epistemic uncertainty.
    """
    predictions = np.stack(
        [model(img_array, training=True).numpy() for _ in range(n_passes)],
        axis=0,
    )  # Shape: (n_passes, 1, num_classes)

    predictions = predictions[:, 0, :]  # Shape: (n_passes, num_classes)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)

    # Predictive entropy: H = -sum(p * log(p))
    eps = 1e-8
    entropy = -np.sum(mean_pred * np.log(mean_pred + eps))

    # Calculate single-prediction reliability score
    # Lower variance across passes = higher reliability
    mean_std = np.mean(std_pred)
    reliability = max(0.0, 1.0 - (mean_std / 0.5))

    return {
        "mean": mean_pred,
        "std": std_pred,
        "predictive_entropy": float(entropy),
        "raw_passes": predictions,
        "reliability": float(reliability),
    }


def uncertainty_label(entropy: float, low_thresh: float = 0.3, high_thresh: float = 0.8) -> str:
    """Return a human-readable uncertainty level label based on Shannon entropy."""
    if entropy < low_thresh:
        return "Low (High Confidence)"
    elif entropy < high_thresh:
        return "Moderate"
    else:
        return "High (Low Confidence — Refer to Specialist)"


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate the Expected Calibration Error (ECE) for evaluation datasets.
    """
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        true_labels = np.argmax(y_true, axis=1)
    else:
        true_labels = y_true.flatten()

    confidences = np.max(y_pred, axis=1)
    predictions = np.argmax(y_pred, axis=1)
    accuracies = (predictions == true_labels)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)


def determine_clinical_risk(pred_class: str, confidence: float, uncertainty_lvl: str) -> str:
    """
    Determine the clinical risk level based on the prediction class,
    confidence score, and uncertainty estimation.
    """
    pred_class_lower = pred_class.lower()
    is_normal = "normal" in pred_class_lower or "healthy" in pred_class_lower or "no dr" in pred_class_lower or "benign" in pred_class_lower or "nevus" in pred_class_lower
    
    # Critical: High confidence disease prediction (severe case)
    if not is_normal and confidence >= 0.85:
        return "critical"
    # High: Moderate confidence disease or high uncertainty disease
    elif not is_normal:
        return "high"
    # Moderate: Normal/healthy prediction but high uncertainty (potential false negative risk)
    elif is_normal and uncertainty_lvl.lower().startswith("high"):
        return "moderate"
    # Low: Normal/healthy prediction with low/moderate uncertainty
    else:
        return "low"


# ─── Plotting & Logging Helpers ─────────────────────────────────────────────────

def plot_history(history, save_path: str = None):
    """Plot training and validation history for loss, accuracy, and recall."""
    plt.figure(figsize=(18, 5))
    metrics = ['loss', 'accuracy', 'recall']
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        if metric in history.history:
            plt.plot(history.history[metric], label='Train')
        if f'val_{metric}' in history.history:
            plt.plot(history.history[f'val_{metric}'], label='Val')
        plt.title(f'Model {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, save_path: str = None):
    """Compute and plot confusion matrix with normalized values."""
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true

    y_pred_indices = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=cm,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_roc_curves(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, save_path: str = None):
    """Compute and plot ROC curves (one-vs-rest for multi-class)."""
    # Force one-hot encoding if integer array
    if len(y_true.shape) == 1 or y_true.shape[1] == 1:
        num_classes = len(class_names)
        y_true_onehot = np.eye(num_classes)[y_true.flatten()]
    else:
        y_true_onehot = y_true

    plt.figure(figsize=(8, 6))
    
    # Check if binary or multi-class
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true_onehot[:, 1], y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    else:
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, label=f'{class_names[i]} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
