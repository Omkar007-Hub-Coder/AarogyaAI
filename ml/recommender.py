"""
Recommender module.

When trained models exist in models/, they are loaded and used.
Falls back to the rule-based catalog otherwise.

This module is imported by the backend service — keep the public interface stable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_model(name: str) -> Any | None:
    """Load a joblib model from the models/ directory if it exists."""
    try:
        import joblib
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            return joblib.load(path)
    except Exception as e:
        print(f"[recommender] Could not load model '{name}': {e}")
    return None


def predict_categories(feature_vector: np.ndarray, category: str) -> list[str]:
    """
    Use a trained model to return item names for a category.
    Returns empty list if no model is available (backend falls back to rule-based).
    """
    model = load_model(f"{category}_model")
    if model is None:
        return []
    # Expected: model.predict returns item indices / labels
    result = model.predict(feature_vector.reshape(1, -1))
    return list(result)
