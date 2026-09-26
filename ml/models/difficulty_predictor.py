"""
ml/models/difficulty_predictor.py

Yoga pose difficulty predictor.

Predicts: Beginner | Intermediate | Advanced

⚠️  DATA STATUS: SYNTHETIC PROTOTYPE
──────────────────────────────────────────────────────────────────────────────
No real user-performance dataset exists. Training data is generated
synthetically with domain rules. Metrics are SYNTHETIC and do not
reflect real-world performance.
──────────────────────────────────────────────────────────────────────────────

Features (8):
  yoga_experience (0-3), flexibility_score (1-5), strength_score (1-5),
  balance_score (1-5), fitness_level (0-3), bmi,
  session_duration_min, previous_performance_score (0-10)

Labels: Beginner | Intermediate | Advanced
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)
RANDOM_SEED = 42
DIFFICULTY_CLASSES = ["Beginner", "Intermediate", "Advanced"]

FEATURE_COLUMNS = [
    "yoga_experience",
    "flexibility_score",
    "strength_score",
    "balance_score",
    "fitness_level",
    "bmi",
    "session_duration_min",
    "previous_performance_score",
]


def generate_synthetic_difficulty_data(
    n_samples: int = 1500, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    ⚠️  SYNTHETIC DATA — domain rules, not clinical guidelines.
    """
    rng = np.random.default_rng(seed)

    yoga_experience   = rng.integers(0, 4, n_samples).astype(float)
    flexibility_score = rng.integers(1, 6, n_samples).astype(float)
    strength_score    = rng.integers(1, 6, n_samples).astype(float)
    balance_score     = rng.integers(1, 6, n_samples).astype(float)
    fitness_level     = rng.integers(0, 4, n_samples).astype(float)
    bmi               = rng.normal(24, 4, n_samples).clip(16, 40)
    session_duration  = rng.choice([20, 30, 45, 60, 90], n_samples).astype(float)
    performance_score = rng.uniform(0, 10, n_samples)

    labels = []
    for i in range(n_samples):
        composite = (
            yoga_experience[i] * 2.0
            + flexibility_score[i]
            + strength_score[i]
            + balance_score[i]
            + fitness_level[i] * 1.5
            + performance_score[i] * 0.5
        )
        if composite < 10:
            diff = "Beginner"
        elif composite < 18:
            diff = "Intermediate"
        else:
            diff = "Advanced"
        # Noise
        if rng.random() < 0.07:
            diff = rng.choice(DIFFICULTY_CLASSES)
        labels.append(diff)

    return pd.DataFrame({
        "yoga_experience":          yoga_experience,
        "flexibility_score":        flexibility_score,
        "strength_score":           strength_score,
        "balance_score":            balance_score,
        "fitness_level":            fitness_level,
        "bmi":                      bmi.round(2),
        "session_duration_min":     session_duration,
        "previous_performance_score": performance_score.round(2),
        "difficulty_label":         labels,
        "_synthetic":               True,
    })


def build_difficulty_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, C=1.0, random_state=RANDOM_SEED)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=4, random_state=RANDOM_SEED,
            )),
        ]),
    }


def save_difficulty_predictor(
    pipeline: Pipeline, label_encoder: LabelEncoder, model_name: str, save_dir: str | Path,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline,      save_dir / f"difficulty_{model_name}.joblib")
    joblib.dump(label_encoder, save_dir / "difficulty_label_encoder.joblib")
    meta = {
        "model_name": model_name,
        "classes": label_encoder.classes_.tolist(),
        "features": FEATURE_COLUMNS,
        "data_source": "SYNTHETIC",
    }
    with open(save_dir / "difficulty_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_difficulty_predictor(save_dir: str | Path) -> tuple[Pipeline, LabelEncoder, dict]:
    save_dir = Path(save_dir)
    with open(save_dir / "difficulty_meta.json") as f:
        meta = json.load(f)
    pipeline      = joblib.load(save_dir / f"difficulty_{meta['model_name']}.joblib")
    label_encoder = joblib.load(save_dir / "difficulty_label_encoder.joblib")
    return pipeline, label_encoder, meta


def predict_difficulty(
    profile: dict,
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
) -> dict:
    """
    Returns predicted difficulty + probability breakdown.
    data_source field always present to indicate data provenance.
    """
    row = [float(profile.get(f, 0)) for f in FEATURE_COLUMNS]
    feat = np.array(row).reshape(1, -1)
    proba = pipeline.predict_proba(feat)[0]
    classes = label_encoder.classes_
    return {
        "predicted_difficulty": label_encoder.inverse_transform([np.argmax(proba)])[0],
        "probabilities": {c: float(p) for c, p in zip(classes, proba)},
        "data_source": "SYNTHETIC",
    }
