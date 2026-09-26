"""
ml/models/user_goal_classifier.py

User health-goal classifier.

⚠️  DATA STATUS: SYNTHETIC PROTOTYPE
──────────────────────────────────────────────────────────────────────────────
No real labelled dataset of user profiles → yoga-goal mappings exists in this
project. The training data used here is procedurally generated with domain
rules.

IMPLICATIONS:
  - Metrics obtained from this data do NOT represent real-world performance.
  - The model must be retrained on real user data before any production use.
  - All evaluation output is explicitly marked SYNTHETIC in logs and outputs.
──────────────────────────────────────────────────────────────────────────────

Features (9):
  age, bmi, activity_level_enc, yoga_experience_enc,
  flexibility_score (1–5), strength_score (1–5),
  sleep_hours, session_duration_min, fitness_experience_enc

Labels (6):
  Flexibility | Strength | Mobility | Weight Management |
  Stress Reduction | General Fitness
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

GOAL_CLASSES = [
    "Flexibility",
    "Strength",
    "Mobility",
    "Weight Management",
    "Stress Reduction",
    "General Fitness",
]

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_profiles(n_samples: int = 2000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    ⚠️  SYNTHETIC DATA — not real user data.

    Generates user profiles with domain-rule-based goal labels.
    Rules are illustrative; they do not represent clinical guidelines.
    """
    rng = np.random.default_rng(seed)

    age               = rng.integers(18, 65, n_samples).astype(float)
    height_cm         = rng.normal(165, 10, n_samples)
    weight_kg         = rng.normal(68,  15, n_samples)
    bmi               = weight_kg / ((height_cm / 100) ** 2)
    activity_level    = rng.integers(0, 4, n_samples)   # 0=sedentary … 3=active
    yoga_experience   = rng.integers(0, 4, n_samples)   # 0=none … 3=advanced
    fitness_experience= rng.integers(0, 4, n_samples)
    flexibility_score = rng.integers(1, 6, n_samples).astype(float)  # 1–5
    strength_score    = rng.integers(1, 6, n_samples).astype(float)
    sleep_hours       = rng.normal(7, 1.5, n_samples).clip(4, 10)
    session_duration  = rng.choice([20, 30, 45, 60, 90], n_samples).astype(float)

    # Domain-rule labelling (SYNTHETIC — not clinical)
    goals = []
    for i in range(n_samples):
        if flexibility_score[i] <= 2 and yoga_experience[i] <= 1:
            g = "Flexibility"
        elif strength_score[i] <= 2 and activity_level[i] >= 2:
            g = "Strength"
        elif bmi[i] >= 27.5 and activity_level[i] <= 1:
            g = "Weight Management"
        elif sleep_hours[i] < 6 or (age[i] > 45 and activity_level[i] <= 1):
            g = "Stress Reduction"
        elif yoga_experience[i] >= 2 and flexibility_score[i] >= 3:
            g = "Mobility"
        else:
            g = "General Fitness"
        # Add controlled noise to avoid perfect rule separation
        if rng.random() < 0.08:
            g = rng.choice(GOAL_CLASSES)
        goals.append(g)

    return pd.DataFrame({
        "age":                 age,
        "bmi":                 bmi.round(2),
        "activity_level":      activity_level,
        "yoga_experience":     yoga_experience,
        "flexibility_score":   flexibility_score,
        "strength_score":      strength_score,
        "sleep_hours":         sleep_hours.round(1),
        "session_duration_min":session_duration,
        "fitness_experience":  fitness_experience,
        "goal_label":          goals,
        "_synthetic":          True,   # explicit marker
    })


FEATURE_COLUMNS = [
    "age", "bmi", "activity_level", "yoga_experience",
    "flexibility_score", "strength_score", "sleep_hours",
    "session_duration_min", "fitness_experience",
]


def build_goal_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, C=1.0, random_state=RANDOM_SEED,
            )),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=10,
                random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1,
                max_depth=4, random_state=RANDOM_SEED,
            )),
        ]),
    }


def save_goal_classifier(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    model_name: str,
    save_dir: str | Path,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline,      save_dir / f"goal_{model_name}.joblib")
    joblib.dump(label_encoder, save_dir / "goal_label_encoder.joblib")
    meta = {
        "model_name": model_name,
        "classes": label_encoder.classes_.tolist(),
        "features": FEATURE_COLUMNS,
        "data_source": "SYNTHETIC",
    }
    with open(save_dir / "goal_classifier_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved goal classifier ({model_name}) to {save_dir}")


def load_goal_classifier(save_dir: str | Path) -> tuple[Pipeline, LabelEncoder, dict]:
    save_dir = Path(save_dir)
    with open(save_dir / "goal_classifier_meta.json") as f:
        meta = json.load(f)
    pipeline      = joblib.load(save_dir / f"goal_{meta['model_name']}.joblib")
    label_encoder = joblib.load(save_dir / "goal_label_encoder.joblib")
    return pipeline, label_encoder, meta


def predict_user_goal(
    profile: dict,
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    top_k: int = 3,
) -> dict:
    """
    Predict the wellness goal for a user profile dict.

    Returns
    -------
    {
      "predicted_goal": str,
      "confidence": float,
      "top_k": [...],
      "data_source": "SYNTHETIC"
    }
    """
    row = [float(profile.get(f, 0)) for f in FEATURE_COLUMNS]
    feat = np.array(row).reshape(1, -1)
    proba = pipeline.predict_proba(feat)[0]
    top_indices = np.argsort(proba)[::-1][:top_k]
    return {
        "predicted_goal": label_encoder.inverse_transform([top_indices[0]])[0],
        "confidence": float(proba[top_indices[0]]),
        "top_k": [
            {"goal": label_encoder.inverse_transform([i])[0], "confidence": float(proba[i])}
            for i in top_indices
        ],
        "data_source": "SYNTHETIC",
    }
