"""
Feature engineering for AarogyaAI.

Input:  raw user profile row (pandas Series or dict)
Output: feature vector (numpy array) for model training/inference
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ACTIVITY_MAP = {"sedentary": 0, "light": 1, "moderate": 2, "active": 3}
GOAL_MAP = {"weight_loss": 0, "muscle_gain": 1, "flexibility": 2, "general_wellness": 3}
GENDER_MAP = {"male": 0, "female": 1, "other": 2}
DIET_MAP = {"none": 0, "vegetarian": 1, "vegan": 2, "non-vegetarian": 3}


def compute_bmi(height_cm: float, weight_kg: float) -> float:
    return weight_kg / ((height_cm / 100) ** 2)


def bmi_category(bmi: float) -> int:
    """0=underweight, 1=normal, 2=overweight, 3=obese"""
    if bmi < 18.5:
        return 0
    elif bmi < 25.0:
        return 1
    elif bmi < 30.0:
        return 2
    return 3


def build_feature_vector(row: dict | pd.Series) -> np.ndarray:
    """
    Convert a user profile row into a numeric feature vector.

    Features (in order):
        0: age (int)
        1: gender (encoded)
        2: height_cm (float)
        3: weight_kg (float)
        4: bmi (float)
        5: bmi_category (int 0-3)
        6: activity_level (encoded)
        7: health_goal (encoded)
        8: dietary_preference (encoded)
    """
    bmi = compute_bmi(float(row["height_cm"]), float(row["weight_kg"]))
    return np.array([
        float(row["age"]),
        float(GENDER_MAP.get(str(row["gender"]), 2)),
        float(row["height_cm"]),
        float(row["weight_kg"]),
        bmi,
        float(bmi_category(bmi)),
        float(ACTIVITY_MAP.get(str(row["activity_level"]), 0)),
        float(GOAL_MAP.get(str(row["health_goal"]), 3)),
        float(DIET_MAP.get(str(row.get("dietary_preference", "none")), 0)),
    ], dtype=np.float32)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build a 2D feature matrix from a DataFrame of user profiles."""
    return np.vstack([build_feature_vector(row) for _, row in df.iterrows()])


FEATURE_NAMES = [
    "age", "gender_enc", "height_cm", "weight_kg",
    "bmi", "bmi_category", "activity_level_enc", "health_goal_enc", "diet_enc",
]
