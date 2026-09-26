"""
ml/training/train_user_goal.py

Train user-goal classifiers on SYNTHETIC data.
Compares LogisticRegression, RandomForest, GradientBoosting.
Saves best model.

⚠️  All metrics produced by this script are from SYNTHETIC data.
    They must NOT be presented as real-world performance.

Usage (from AarogyaAI/ root):
    python -m ml.training.train_user_goal
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

from ml.models.user_goal_classifier import (
    FEATURE_COLUMNS,
    build_goal_models,
    generate_synthetic_profiles,
    save_goal_classifier,
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


def run(n_samples: int = 2000) -> None:
    logger.info("⚠️  Generating SYNTHETIC user profiles …")
    df = generate_synthetic_profiles(n_samples=n_samples)

    # Save synthetic data for transparency
    synth_dir = ROOT / "data" / "processed" / "user_profiles"
    synth_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(synth_dir / "synthetic_profiles.csv", index=False)
    logger.info(f"Saved synthetic profiles to {synth_dir / 'synthetic_profiles.csv'}")

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["goal_label"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    logger.info(f"Train={len(X_train)}, Test={len(X_test)}, Classes={le.classes_.tolist()}")

    models = build_goal_models()
    best_name, best_model, best_f1 = None, None, -1.0
    all_metrics = {}

    for name, pipeline in models.items():
        logger.info(f"Training {name} …")
        pipeline.fit(X_train, y_train)

        cv_metrics   = cross_validate_classifier(pipeline, X_train, y_train, cv=5, data_source="SYNTHETIC")
        test_metrics = evaluate_classifier(pipeline, X_test, y_test,
                                           label_names=le.classes_.tolist(), data_source="SYNTHETIC")
        all_metrics[name] = {"cv": cv_metrics, "test": test_metrics}
        print_metrics_summary(test_metrics, title=f"User Goal — {name} (test set)")

        f1 = test_metrics["f1_score"]
        if f1 > best_f1:
            best_f1    = f1
            best_name  = name
            best_model = pipeline

    logger.info(f"Best model: {best_name} (test f1={best_f1:.4f}) [SYNTHETIC]")
    save_goal_classifier(best_model, le, best_name, MODELS_DIR)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(all_metrics, METRICS_DIR / "user_goal_metrics.json")
    logger.info("User goal classifier training complete.")


if __name__ == "__main__":
    run()
