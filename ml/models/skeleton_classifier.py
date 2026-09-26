"""
ml/models/skeleton_classifier.py

Skeleton-based yoga pose classifier using BlazePose 33-keypoint features.

Input features (109-dim per sample):
  - 33 × (x, y, z) normalized skeleton coordinates (99 features)
  - 10 joint angles (degrees)

Models trained and compared:
  - RandomForest
  - SVM (RBF kernel, calibrated for probability output)
  - GradientBoosting

The best model by validation accuracy is selected.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def _make_hist_gb() -> HistGradientBoostingClassifier:
    """HistGradientBoostingClassifier is ~10-100× faster than the standard
    GradientBoostingClassifier on datasets with >1 k samples while achieving
    comparable accuracy.  It natively supports predict_proba.
    """
    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        max_depth=None,
        random_state=RANDOM_SEED,
    )


def build_skeleton_models() -> dict[str, Pipeline]:
    """Return a dict of named sklearn Pipelines for skeleton classification."""
    # SVC wrapped in CalibratedClassifierCV so predict_proba is available
    # without the deprecated probability=True flag (sklearn ≥ 1.9).
    svc_base = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=RANDOM_SEED)
    svc_calibrated = CalibratedClassifierCV(svc_base, cv=3, ensemble=False)

    return {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=4,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", svc_calibrated),
        ]),
        # HistGradientBoosting is ~10-100× faster than GradientBoosting on
        # large datasets while achieving comparable accuracy.
        "hist_gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", _make_hist_gb()),
        ]),
    }


def save_skeleton_classifier(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    model_name: str,
    save_dir: str | Path,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_dir / f"skeleton_{model_name}.joblib")
    joblib.dump(label_encoder, save_dir / "skeleton_label_encoder.joblib")
    classes = label_encoder.classes_.tolist()
    with open(save_dir / "skeleton_classes.json", "w") as f:
        json.dump({"classes": classes, "model_name": model_name}, f, indent=2)
    logger.info(f"Saved skeleton classifier ({model_name}) to {save_dir}")


def load_skeleton_classifier(save_dir: str | Path) -> tuple[Pipeline, LabelEncoder]:
    save_dir = Path(save_dir)
    with open(save_dir / "skeleton_classes.json") as f:
        meta = json.load(f)
    pipeline = joblib.load(save_dir / f"skeleton_{meta['model_name']}.joblib")
    label_encoder = joblib.load(save_dir / "skeleton_label_encoder.joblib")
    return pipeline, label_encoder


def predict_pose_from_skeleton(
    landmarks: np.ndarray,
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    top_k: int = 5,
) -> dict:
    """
    Predict pose from a (33, 3) or (33, 4) skeleton array.

    Returns
    -------
    {
      "predicted_pose": str,
      "confidence": float,
      "top_k": [{"pose": str, "confidence": float}, ...]
    }
    """
    from ml.preprocessing.skeleton_utils import extract_skeleton_features
    feat = extract_skeleton_features(landmarks).reshape(1, -1)
    proba = pipeline.predict_proba(feat)[0]
    top_indices = np.argsort(proba)[::-1][:top_k]
    return {
        "predicted_pose": label_encoder.inverse_transform([top_indices[0]])[0],
        "confidence": float(proba[top_indices[0]]),
        "top_k": [
            {
                "pose": label_encoder.inverse_transform([i])[0],
                "confidence": float(proba[i]),
            }
            for i in top_indices
        ],
    }
