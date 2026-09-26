"""
ml/evaluation/metrics.py

Shared evaluation utilities for all classifiers.
Returns actual computed metrics — never fabricated values.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

logger = logging.getLogger(__name__)


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str] | None = None,
    data_source: str = "REAL",
) -> dict[str, Any]:
    """
    Compute accuracy, precision, recall, F1, confusion matrix.

    Parameters
    ----------
    data_source : str
        'REAL' or 'SYNTHETIC'. Always stored in output so callers know the
        provenance of the metrics.
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test,    y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test,        y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )

    result = {
        "data_source": data_source,
        "n_test_samples": len(y_test),
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1_score":  round(float(f1),   4),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    logger.info(
        f"[{data_source}] acc={acc:.4f} p={prec:.4f} r={rec:.4f} f1={f1:.4f} "
        f"(n={len(y_test)})"
    )
    return result


def cross_validate_classifier(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    data_source: str = "REAL",
) -> dict[str, Any]:
    """Stratified k-fold cross-validation."""
    import warnings
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        cv_results = cross_validate(
            model, X, y, cv=skf,
            scoring=["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"],
            return_train_score=False,
        )
    return {
        "data_source": data_source,
        "cv_folds": cv,
        "n_samples": len(y),
        "accuracy":  {"mean": round(float(cv_results["test_accuracy"].mean()),         4),
                      "std":  round(float(cv_results["test_accuracy"].std()),          4)},
        "f1_score":  {"mean": round(float(cv_results["test_f1_weighted"].mean()),      4),
                      "std":  round(float(cv_results["test_f1_weighted"].std()),       4)},
        "precision": {"mean": round(float(cv_results["test_precision_weighted"].mean()),4),
                      "std":  round(float(cv_results["test_precision_weighted"].std()),4)},
        "recall":    {"mean": round(float(cv_results["test_recall_weighted"].mean()),   4),
                      "std":  round(float(cv_results["test_recall_weighted"].std()),    4)},
    }


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {output_path}")


def print_metrics_summary(metrics: dict, title: str = "Evaluation") -> None:
    ds = metrics.get("data_source", "?")
    print(f"\n{'='*55}")
    print(f"  {title}  [data_source={ds}]")
    print(f"{'='*55}")
    print(f"  Samples  : {metrics.get('n_test_samples', metrics.get('n_samples', '?'))}")
    print(f"  Accuracy : {metrics.get('accuracy', '?')}")
    if "precision" in metrics:
        v = metrics["precision"]
        print(f"  Precision: {v['mean'] if isinstance(v, dict) else v}")
    if "recall" in metrics:
        v = metrics["recall"]
        print(f"  Recall   : {v['mean'] if isinstance(v, dict) else v}")
    if "f1_score" in metrics:
        v = metrics["f1_score"]
        print(f"  F1 Score : {v['mean'] if isinstance(v, dict) else v}")
    if ds == "SYNTHETIC":
        print("\n  ⚠️  SYNTHETIC DATA — metrics do not reflect real-world performance.")
    print(f"{'='*55}\n")
