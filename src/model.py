"""
model.py
========
Defines the EfficientNetB4-based architecture with a custom classification head
and embeds the Grad-CAM / Grad-CAM++ explainability logic.
Supports:
  - Transfer learning from ImageNet weights
  - Unfreezing top layers for fine-tuning
  - Monte Carlo Dropout for uncertainty estimation
  - Label smoothing for better calibration
  - Grad-CAM and Grad-CAM++ visual localization heatmaps
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Import configs from utils to maintain the requested file structure
from src.utils import DROPOUT_RATE, LABEL_SMOOTHING


class F1Score(tf.keras.metrics.Metric):
    """Custom stateful F1-Score metric for Keras."""
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


# ─── Model Factory ─────────────────────────────────────────────────────────────

def build_model(
    num_classes: int,
    dropout_rate: float = DROPOUT_RATE,
    input_shape: tuple = (380, 380, 3),
    weights: str = "imagenet",
    use_label_smoothing: bool = True,
) -> Model:
    """
    Build EfficientNetB4 with a custom classification head.

    Architecture:
        Input → EfficientNetB4 (frozen) → GlobalAveragePooling2D
              → BatchNorm → Dropout(0.4) → Dense(512, swish)
              → BatchNorm → Dropout(0.3) → Dense(num_classes, softmax)
    """
    base = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    base.trainable = False  # Freeze during initial training phase

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)

    # ─── Custom Classification Head ──────────────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x, training=True)     # training=True -> MC-Dropout at inference
    x = layers.Dense(512, activation="swish")(x)           # swish > relu for EfficientNet
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate * 0.75)(x, training=True)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="EfficientNetB4_DiseaseDetector")

    loss = (
        tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
        if use_label_smoothing
        else "categorical_crossentropy"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            F1Score(name="f1_score"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def unfreeze_top_layers(model: Model, num_layers: int = 30, lr: float = 1e-5) -> Model:
    """
    Unfreeze the top N layers of the EfficientNetB4 base model for fine-tuning.
    """
    base = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or (
            hasattr(layer, "layers") and not isinstance(layer, tf.keras.layers.InputLayer)
        ):
            base = layer
            break
    if base is None:
        raise ValueError("Base model not found in model layers.")

    # Keep BatchNorm frozen to preserve ImageNet statistics
    for layer in base.layers[-num_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            F1Score(name="f1_score"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ─── Grad-CAM & Grad-CAM++ Interpretability Logic ──────────────────────────────

def get_gradcam_models(model: tf.keras.Model, last_conv_layer_name: str) -> tuple:
    """
    Construct the models needed for Grad-CAM, handling nested architectures.
    Returns: (grad_model, is_nested, base_model, base_grad_model)
    """
    try:
        model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        return grad_model, False, None, None
    except ValueError:
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) or hasattr(layer, 'layers'):
                try:
                    sub_layer = layer.get_layer(last_conv_layer_name)
                    base_grad_model = tf.keras.Model(
                        inputs=layer.inputs,
                        outputs=[sub_layer.output, layer.output]
                    )
                    return None, True, layer, base_grad_model
                except ValueError:
                    continue
        raise ValueError(f"Layer {last_conv_layer_name} not found in model.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = None,
    pred_index: int = None,
    variant: str = "gradcam++",
) -> np.ndarray:
    """
    Compute a Grad-CAM or Grad-CAM++ heatmap for a given image and model.
    """
    # Auto-detect last conv layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.Model) or hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D) or "conv" in sub_layer.name.lower():
                        last_conv_layer_name = sub_layer.name
                        break
            elif isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
                last_conv_layer_name = layer.name
                break
            if last_conv_layer_name:
                break

    if last_conv_layer_name is None:
        raise ValueError("Could not auto-detect a convolutional layer in the model.")

    grad_model, is_nested, base_model, base_grad_model = get_gradcam_models(model, last_conv_layer_name)

    if not is_nested:
        with tf.GradientTape(persistent=True) as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
    else:
        with tf.GradientTape(persistent=True) as tape:
            # Reconstruct forward pass up to base model
            x = img_array
            for layer in model.layers:
                if layer == base_model:
                    break
                if not isinstance(layer, tf.keras.layers.InputLayer):
                    x = layer(x)
            
            # Pass through base grad model
            conv_outputs, base_outputs = base_grad_model(x)
            
            # Pass through the rest of the model head
            x = base_outputs
            head_started = False
            for layer in model.layers:
                if head_started:
                    x = layer(x)
                if layer == base_model:
                    head_started = True
            predictions = x
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

    # Compute gradients of class channel with respect to conv outputs
    grads = tape.gradient(class_channel, conv_outputs)

    if variant.lower() == "gradcam++":
        conv_outputs_val = conv_outputs[0]  # Shape: (H, W, C)
        grads_val = grads[0]                # Shape: (H, W, C)

        # Focus only on positive gradients
        relu_grads = tf.maximum(grads_val, 0.0)

        # Higher-order derivatives approximation
        grads_power_2 = tf.square(grads_val)
        grads_power_3 = tf.pow(grads_val, 3)

        # Sum of activations per channel
        sum_activations = tf.reduce_sum(conv_outputs_val, axis=(0, 1))

        # Compute alpha coefficients
        eps = 1e-8
        alpha_denom = 2 * grads_power_2 + sum_activations * grads_power_3 + eps
        alpha = grads_power_2 / alpha_denom

        # Channel weights
        weights = tf.reduce_sum(alpha * relu_grads, axis=(0, 1))

        # Weighted sum of conv outputs
        heatmap = tf.reduce_sum(weights * conv_outputs_val, axis=-1)
    else:
        # Standard Grad-CAM
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs_val = conv_outputs[0]
        heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

    # Resize the heatmap to the spatial dimensions of the input image
    heatmap_np = heatmap.numpy()
    heatmap_resized = cv2.resize(heatmap_np, (img_array.shape[2], img_array.shape[1]))

    heatmap_resized = np.maximum(heatmap_resized, 0)
    max_val = np.max(heatmap_resized)
    if max_val > 0:
        heatmap_resized = heatmap_resized / max_val
    return heatmap_resized


def overlay_heatmap(heatmap: np.ndarray, original_img: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap onto the original image.
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap_colored, alpha, original_img, 1 - alpha, 0)
