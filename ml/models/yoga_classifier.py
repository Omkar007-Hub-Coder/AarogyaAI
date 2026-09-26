"""
ml/models/yoga_classifier.py

Image-based yoga pose classifier using EfficientNetB0 transfer learning.

Architecture:
  EfficientNetB0 (ImageNet pretrained, top removed)
  → GlobalAveragePooling2D
  → Dropout(0.3)
  → Dense(256, relu)
  → Dropout(0.2)
  → Dense(num_classes, softmax)

Training strategy:
  Phase 1: Train head only (EfficientNet frozen) — 10 epochs
  Phase 2: Fine-tune top 20 layers of EfficientNet — 10 more epochs

Input:  (224, 224, 3) float32, ImageNet-normalized
Output: (num_classes,) softmax probabilities
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_INPUT_SIZE = (224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────

def build_yoga_classifier(num_classes: int, fine_tune: bool = False):
    """
    Build an EfficientNetB0-based classifier.

    Parameters
    ----------
    num_classes : int
        Number of yoga pose classes (82 for Yoga-82).
    fine_tune : bool
        If True, unfreeze top 20 layers of EfficientNet for phase-2 training.

    Returns
    -------
    keras.Model
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        raise ImportError(
            "TensorFlow is required for the image classifier. "
            "Install it with: pip install tensorflow"
        )

    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*MODEL_INPUT_SIZE, 3),
    )

    # Phase 1: freeze entire base
    base.trainable = fine_tune
    if fine_tune:
        # Unfreeze only the top 20 layers
        for layer in base.layers[:-20]:
            layer.trainable = False
        for layer in base.layers[-20:]:
            layer.trainable = True

    inputs = keras.Input(shape=(*MODEL_INPUT_SIZE, 3), name="image_input")
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="pose_output")(x)

    model = keras.Model(inputs, outputs, name="yoga_classifier_efficientnet")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Dataset pipeline (tf.data)
# ─────────────────────────────────────────────────────────────────────────────

def build_tf_dataset(
    image_label_pairs: list[tuple[str, int]],
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True,
):
    """
    Build a tf.data.Dataset from (path, int_label) pairs.

    Data augmentation (training only, applied after batching on the batch tensor):
      - RandomFlip (horizontal)
      - RandomRotation ±15°
      - RandomZoom ±10%
      - RandomBrightness ±15%

    Augmentation is implemented as a Keras Sequential preprocessing model
    applied to each batch (not inside map()). This is the correct pattern for
    stateful Keras augmentation layers — they must NOT be instantiated inside
    tf.data.map() because doing so creates new layer objects on every graph
    trace, breaking determinism and adding overhead.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required for dataset pipeline.")

    paths  = [p for p, _ in image_label_pairs]
    labels = [lbl for _, lbl in image_label_pairs]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)

    def load_and_preprocess(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, MODEL_INPUT_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        # ImageNet normalization
        mean = tf.constant([0.485, 0.456, 0.406])
        std  = tf.constant([0.229, 0.224, 0.225])
        img = (img - mean) / std
        return img, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    if augment:
        # Build augmentation pipeline once, apply to each batch.
        # These layers are defined outside map() so they are traced once.
        augmenter = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),    # ±~15°
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomBrightness(0.15),
        ], name="augmentation")

        def apply_augment(images, labels):
            return augmenter(images, training=True), labels

        ds = ds.map(apply_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_yoga_classifier(model, class_names: list[str], save_dir: str | Path) -> None:
    """Save model weights (.keras) and class-name mapping."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "yoga_classifier.keras"
    model.save(model_path)
    meta = {"class_names": class_names, "num_classes": len(class_names)}
    with open(save_dir / "yoga_classifier_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved yoga classifier to {model_path}")


def load_yoga_classifier(save_dir: str | Path):
    """Load model and class names from a saved directory."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required.")
    save_dir = Path(save_dir)
    model = tf.keras.models.load_model(save_dir / "yoga_classifier.keras")
    with open(save_dir / "yoga_classifier_meta.json") as f:
        meta = json.load(f)
    return model, meta["class_names"]


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_pose(
    image_path: str | Path,
    model,
    class_names: list[str],
    top_k: int = 5,
) -> dict:
    """
    Classify a single yoga pose image.

    Returns
    -------
    {
      "predicted_pose": str,
      "confidence": float,
      "top_k": [{"pose": str, "confidence": float}, ...]
    }
    """
    from ml.preprocessing.image_utils import preprocess_for_model
    arr = preprocess_for_model(image_path)
    if arr is None:
        return {"error": f"Could not load image: {image_path}"}

    import numpy as np
    batch = arr[np.newaxis, ...]          # (1, 224, 224, 3)
    probs = model.predict(batch, verbose=0)[0]   # (num_classes,)

    top_indices = np.argsort(probs)[::-1][:top_k]
    return {
        "predicted_pose": class_names[top_indices[0]],
        "confidence": float(probs[top_indices[0]]),
        "top_k": [
            {"pose": class_names[i], "confidence": float(probs[i])}
            for i in top_indices
        ],
    }
