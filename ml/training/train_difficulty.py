"""
ml/training/train_difficulty.py

Train yoga difficulty predictor on SYNTHETIC data.
Compares LogisticRegression, RandomForest, GradientBoosting.

⚠️  All metrics from SYNTHETIC data only.

Usage (from AarogyaAI/ root):
    python -m ml.training.train_difficulty
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.models.difficulty_predictor import (
    FEATURE_COLUMNS,
    build_difficulty_models,
    generate_synthetic_difficulty_data,
    save_difficulty_predictor,
)
from ml.evaluation.metrics import (
    cross_validate_classifier,
    evaluate_classifier,
    save_metrics,
    print_metrics_summary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR  = ROOT / "models"
METRICS_DIR = ROOT / "models" / "metrics"


def run(n_samples: int = 1500) -> None:
    logger.info("⚠️  Generating SYNTHETIC difficulty training data …")
    df = generate_synthetic_difficulty_data(n_samples=n_samples)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["difficulty_label"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42,
    )
    logger.info(f"Train={len(X_train)}, Test={len(X_test)}, Classes={le.classes_.tolist()}")

    models = build_difficulty_models()
    best_name, best_model, best_f1 = None, None, -1.0
    all_metrics = {}

    for name, pipeline in models.items():
        logger.info(f"Training {name} …")
        pipeline.fit(X_train, y_train)

        cv_metrics   = cross_validate_classifier(pipeline, X_train, y_train, cv=5, data_source="SYNTHETIC")
        test_metrics = evaluate_classifier(pipeline, X_test, y_test,
                                           label_names=le.classes_.tolist(), data_source="SYNTHETIC")
        all_metrics[name] = {"cv": cv_metrics, "test": test_metrics}
        print_metrics_summary(test_metrics, title=f"Difficulty — {name} (test set)")

        f1 = test_metrics["f1_score"]
        if f1 > best_f1:
            best_f1    = f1
            best_name  = name
            best_model = pipeline

    logger.info(f"Best model: {best_name} (f1={best_f1:.4f}) [SYNTHETIC]")
    save_difficulty_predictor(best_model, le, best_name, MODELS_DIR)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(all_metrics, METRICS_DIR / "difficulty_metrics.json")
    logger.info("Difficulty predictor training complete.")


if __name__ == "__main__":
    run()
