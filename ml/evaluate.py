"""
evaluate.py — Model evaluation utilities.

Call after training to get real metrics from held-out test data.
No fake results are generated here; call only when a trained model exists.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    top_k_accuracy_score,
)
from sklearn.model_selection import cross_val_score


def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return accuracy and a classification report dict."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }


def cross_validate(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    """Return mean ± std cross-validation accuracy."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return {"mean_accuracy": float(scores.mean()), "std_accuracy": float(scores.std())}
