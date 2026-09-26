"""
ml/training/train_skeleton.py

Train skeleton-based pose classifiers using BlazePose keypoint features.
Compares RandomForest, SVM, GradientBoosting. Saves the best model.

Usage (from AarogyaAI/ root with venv active):
    python -m ml.training.train_skeleton

Requires:
    data/processed/skeleton/train_features.npy
    data/processed/skeleton/train_labels.npy
    data/processed/skeleton/test_features.npy
    data/processed/skeleton/test_labels.npy

Run scripts/preprocess.py first.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.models.skeleton_classifier import (
    build_skeleton_models,
    save_skeleton_classifier,
)
from ml.evaluation.metrics import (
    evaluate_classifier,
    cross_validate_classifier,
    save_metrics,
    print_metrics_summary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = ROOT / "data" / "processed" / "skeleton"
MODELS_DIR    = ROOT / "models"
METRICS_DIR   = ROOT / "models" / "metrics"


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-processed skeleton feature arrays."""
    required = [
        PROCESSED_DIR / "train_features.npy",
        PROCESSED_DIR / "train_labels.npy",
        PROCESSED_DIR / "test_features.npy",
        PROCESSED_DIR / "test_labels.npy",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run scripts/preprocess.py --skeleton first."
            )
    X_train = np.load(PROCESSED_DIR / "train_features.npy")
    y_train = np.load(PROCESSED_DIR / "train_labels.npy", allow_pickle=True)
    X_test  = np.load(PROCESSED_DIR / "test_features.npy")
    y_test  = np.load(PROCESSED_DIR / "test_labels.npy",  allow_pickle=True)
    return X_train, y_train, X_test, y_test


def run() -> None:
    logger.info("Loading skeleton data …")
    X_train, y_train, X_test, y_test = load_data()
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    models = build_skeleton_models()
    best_name, best_model, best_acc = None, None, -1.0
    all_metrics = {}

    for name, pipeline in models.items():
        logger.info(f"Training {name} …")
        pipeline.fit(X_train, y_train_enc)

        cv_metrics = cross_validate_classifier(
            pipeline, X_train, y_train_enc, cv=5, data_source="REAL"
        )
        test_metrics = evaluate_classifier(
            pipeline, X_test, y_test_enc,
            label_names=le.classes_.tolist(), data_source="REAL"
        )
        all_metrics[name] = {"cv": cv_metrics, "test": test_metrics}
        print_metrics_summary(test_metrics, title=f"Skeleton — {name} (test set)")

        acc = test_metrics["accuracy"]
        if acc > best_acc:
            best_acc   = acc
            best_name  = name
            best_model = pipeline

    logger.info(f"Best model: {best_name} (test accuracy={best_acc:.4f})")
    save_skeleton_classifier(best_model, le, best_name, MODELS_DIR)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(all_metrics, METRICS_DIR / "skeleton_classifier_metrics.json")
    logger.info("Skeleton classifier training complete.")


if __name__ == "__main__":
    run()
