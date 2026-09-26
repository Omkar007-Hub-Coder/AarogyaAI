#!/usr/bin/env python3
"""
scripts/infer_yoga_image.py

Standalone image inference script for the EfficientNetB0 Yoga-82 classifier.
Run exclusively under .venv-tf (Python 3.9 + TF 2.15).

Usage:
    TF_CPP_MIN_LOG_LEVEL=3 .venv-tf/bin/python scripts/infer_yoga_image.py \
        --image <path> [--top-k 5] [--models-dir <path>]

Prints a single JSON object to stdout and exits with code 0 on success,
or prints {"error": "..."} and exits with code 1 on failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True,  help="Path to input image")
    parser.add_argument("--top-k",      type=int, default=5)
    parser.add_argument("--models-dir", default=str(ROOT / "models"))
    args = parser.parse_args()

    image_path  = Path(args.image)
    models_dir  = Path(args.models_dir)
    top_k       = args.top_k

    if not image_path.exists():
        print(json.dumps({"error": f"Image file not found: {image_path}"}))
        sys.exit(1)

    # Try best checkpoint first, then final saved model
    model_candidates = [
        models_dir / "yoga_classifier_best.keras",
        models_dir / "yoga_classifier.keras",
    ]
    model_path = next((p for p in model_candidates if p.exists()), None)
    if model_path is None:
        print(json.dumps({"error": "No trained yoga classifier found in models/"}))
        sys.exit(1)

    meta_path = models_dir / "yoga_classifier_meta.json"
    if not meta_path.exists():
        # Fall back to class_to_idx mapping
        meta_path = None

    try:
        import tensorflow as tf  # noqa: F401 — imported for side effects (GPU init etc.)
        import numpy as np

        # Suppress Metal/hardware noise
        tf.get_logger().setLevel("ERROR")

        # Load class names
        if meta_path and meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            class_names: list[str] = meta["class_names"]
        else:
            # Fall back to sorted class_to_idx keys
            idx_path = models_dir / "yoga82_class_to_idx.json"
            if idx_path.exists():
                with open(idx_path) as f:
                    c2i = json.load(f)
                class_names = [k for k, _ in sorted(c2i.items(), key=lambda x: x[1])]
            else:
                print(json.dumps({"error": "No class mapping found (yoga_classifier_meta.json or yoga82_class_to_idx.json)"}))
                sys.exit(1)

        # Load model
        model = tf.keras.models.load_model(str(model_path))

        # Preprocess image
        from ml.preprocessing.image_utils import preprocess_for_model
        arr = preprocess_for_model(str(image_path))
        if arr is None:
            print(json.dumps({"error": f"Could not decode image: {image_path}"}))
            sys.exit(1)

        batch = arr[np.newaxis, ...]           # (1, 224, 224, 3)
        probs = model.predict(batch, verbose=0)[0]   # (num_classes,)

        top_indices = np.argsort(probs)[::-1][:top_k]
        result = {
            "predicted_pose": class_names[top_indices[0]],
            "confidence":     float(probs[top_indices[0]]),
            "top_k": [
                {"pose": class_names[int(i)], "confidence": float(probs[i])}
                for i in top_indices
            ],
            "model_file": model_path.name,
            "num_classes": len(class_names),
        }
        print(json.dumps(result))
        sys.exit(0)

    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
