#!/usr/bin/env python3
"""
scripts/train_skeleton_classifier.py

Train and evaluate a skeleton-based yoga pose classifier on the REAL
BlazePose Yoga-82 dataset (yoga-82-skeletons-normalized).

Pipeline
--------
1. Load .npy per-sample directory structure (train / valid / test splits).
2. Feature-engineer each sample → 109-dim vector via skeleton_utils.
3. Encode labels with LabelEncoder (sorted, deterministic).
4. Train three pipelines: RandomForest, SVM (RBF), GradientBoosting.
5. Select best model by validation accuracy (no fabrication).
6. Evaluate on untouched test split → accuracy, macro P/R/F1, per-class,
   confusion matrix, Top-1, Top-5.
7. Save model, label encoder, class mapping, metrics, per-class CSV,
   confusion matrix, dataset manifest.

Usage
-----
    python scripts/train_skeleton_classifier.py [--data-dir <path>] [--models-dir <path>]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.preprocessing.skeleton_utils import load_skeleton_npy_dir, SKELETON_FEATURE_DIM
from ml.models.skeleton_classifier import (
    build_skeleton_models,
    save_skeleton_classifier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_skeleton")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATA = ROOT / "data" / "raw" / "yoga-82-skeletons-normalized"
DEFAULT_MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train skeleton-based yoga classifier")
    p.add_argument("--data-dir",   default=str(DEFAULT_DATA),   help="Root of skeleton dataset")
    p.add_argument("--models-dir", default=str(DEFAULT_MODELS), help="Where to save trained model")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Top-K accuracy helper
# ─────────────────────────────────────────────────────────────────────────────

def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    """Fraction of samples where the true class is in the top-k predicted classes."""
    top_k_preds = np.argsort(proba, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k_preds[i] for i in range(len(y_true)))
    return correct / len(y_true)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    data_dir   = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    metrics_dir = models_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load all splits ───────────────────────────────────────────────────
    log.info("Loading TRAIN split …")
    t0 = time.time()
    X_train, y_train_raw = load_skeleton_npy_dir(str(data_dir / "train"))
    log.info(f"  → {len(X_train)} samples, {X_train.shape[1]}-dim features  ({time.time()-t0:.1f}s)")

    log.info("Loading VALID split …")
    t0 = time.time()
    X_valid, y_valid_raw = load_skeleton_npy_dir(str(data_dir / "valid"))
    log.info(f"  → {len(X_valid)} samples  ({time.time()-t0:.1f}s)")

    log.info("Loading TEST split …")
    t0 = time.time()
    X_test, y_test_raw = load_skeleton_npy_dir(str(data_dir / "test"))
    log.info(f"  → {len(X_test)} samples  ({time.time()-t0:.1f}s)")

    # ── 2. Encode labels ─────────────────────────────────────────────────────
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # Fit on all known classes (sorted, deterministic)
    all_labels = sorted(set(y_train_raw) | set(y_valid_raw) | set(y_test_raw))
    le.fit(all_labels)

    y_train = le.transform(y_train_raw)
    y_valid = le.transform(y_valid_raw)
    y_test  = le.transform(y_test_raw)

    n_classes = len(le.classes_)
    log.info(f"Classes: {n_classes}  |  feature dim: {SKELETON_FEATURE_DIM}")

    # ── 3. Dataset statistics ────────────────────────────────────────────────
    from collections import Counter
    train_counts = Counter(y_train_raw.tolist())
    valid_counts = Counter(y_valid_raw.tolist())
    test_counts  = Counter(y_test_raw.tolist())

    train_vals = list(train_counts.values())
    dataset_manifest = {
        "data_source": "REAL",
        "dataset": "yoga-82-skeletons-normalized (BlazePose Yoga-82, Kaggle)",
        "feature_dim": int(SKELETON_FEATURE_DIM),
        "n_classes": n_classes,
        "splits": {
            "train": {
                "n_samples": int(len(X_train)),
                "min_class": int(min(train_vals)),
                "max_class": int(max(train_vals)),
                "mean_class": round(sum(train_vals) / len(train_vals), 1),
                "class_counts": {k: int(v) for k, v in sorted(train_counts.items())},
            },
            "valid": {"n_samples": int(len(X_valid))},
            "test":  {"n_samples": int(len(X_test))},
        },
        "classes": list(le.classes_),
    }

    manifest_path = metrics_dir / "skeleton_dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(dataset_manifest, f, indent=2)
    log.info(f"Dataset manifest saved → {manifest_path}")

    # ── 4. Train all models on train split, select by val accuracy ───────────
    log.info("\n=== Training and comparing models ===")
    models = build_skeleton_models()
    val_results: dict[str, float] = {}

    for name, pipeline in models.items():
        log.info(f"  Fitting {name} …")
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0

        val_acc = pipeline.score(X_valid, y_valid)
        val_results[name] = val_acc
        log.info(f"  {name:25s}  val_acc={val_acc:.4f}  ({elapsed:.1f}s)")

    best_name = max(val_results, key=val_results.get)
    log.info(f"\nBest model by validation accuracy: {best_name} ({val_results[best_name]:.4f})")

    best_pipeline = models[best_name]

    # ── 5. Evaluate best model on TEST split ─────────────────────────────────
    log.info("\n=== Evaluating best model on TEST split ===")
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    y_pred = best_pipeline.predict(X_test)

    # Probabilities for Top-5
    y_proba = best_pipeline.predict_proba(X_test)

    top1 = accuracy_score(y_test, y_pred)
    top5 = top_k_accuracy(y_test, y_proba, k=5)

    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec  = recall_score(y_test,  y_pred, average="macro", zero_division=0)
    macro_f1   = f1_score(y_test,      y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    class_names = list(le.classes_)
    report_dict = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    log.info(f"  Top-1 accuracy : {top1:.4f}")
    log.info(f"  Top-5 accuracy : {top5:.4f}")
    log.info(f"  Macro precision: {macro_prec:.4f}")
    log.info(f"  Macro recall   : {macro_rec:.4f}")
    log.info(f"  Macro F1       : {macro_f1:.4f}")

    # Also compute validation metrics for all models (for the report)
    model_comparison = {}
    for name, pipeline in models.items():
        yt_pred = pipeline.predict(X_valid)
        model_comparison[name] = {
            "val_accuracy": round(float(val_results[name]), 4),
            "val_macro_f1": round(float(f1_score(y_valid, yt_pred, average="macro", zero_division=0)), 4),
        }

    # ── 6. Save all artifacts ─────────────────────────────────────────────────
    log.info("\n=== Saving artifacts ===")

    # 6a. Trained model + label encoder
    save_skeleton_classifier(best_pipeline, le, best_name, models_dir)
    log.info(f"  Model saved → {models_dir}/skeleton_{best_name}.joblib")

    # 6b. Evaluation metrics JSON
    metrics_payload = {
        "data_source": "REAL",
        "dataset": "yoga-82-skeletons-normalized",
        "model_selected": best_name,
        "model_comparison_on_valid": model_comparison,
        "model_config": str(best_pipeline),
        "n_classes": n_classes,
        "n_train_samples": int(len(X_train)),
        "n_valid_samples": int(len(X_valid)),
        "n_test_samples": int(len(X_test)),
        "feature_dim": int(SKELETON_FEATURE_DIM),
        "test_top1_accuracy": round(float(top1), 4),
        "test_top5_accuracy": round(float(top5), 4),
        "test_macro_precision": round(float(macro_prec), 4),
        "test_macro_recall": round(float(macro_rec), 4),
        "test_macro_f1": round(float(macro_f1), 4),
        "classification_report": report_dict,
    }

    metrics_path = metrics_dir / "skeleton_classifier_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    log.info(f"  Metrics JSON → {metrics_path}")

    # 6c. Per-class CSV
    import csv
    per_class_path = metrics_dir / "skeleton_per_class_metrics.csv"
    with open(per_class_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for cls in class_names:
            row = report_dict.get(cls, {})
            writer.writerow([
                cls,
                round(row.get("precision", 0.0), 4),
                round(row.get("recall", 0.0), 4),
                round(row.get("f1-score", 0.0), 4),
                int(row.get("support", 0)),
            ])
    log.info(f"  Per-class CSV → {per_class_path}")

    # 6d. Confusion matrix (numpy binary, and also as JSON-serializable)
    cm_path = metrics_dir / "skeleton_confusion_matrix.npy"
    np.save(str(cm_path), cm)
    log.info(f"  Confusion matrix → {cm_path}")

    # 6e. Confusion matrix as JSON (for portability)
    cm_json_path = metrics_dir / "skeleton_confusion_matrix.json"
    with open(cm_json_path, "w") as f:
        json.dump({
            "classes": class_names,
            "matrix": cm.tolist(),
        }, f)
    log.info(f"  Confusion matrix JSON → {cm_json_path}")

    log.info("\n=== Done ===")
    log.info(f"Selected model : {best_name}")
    log.info(f"Test Top-1     : {top1:.4f}")
    log.info(f"Test Top-5     : {top5:.4f}")
    log.info(f"Test Macro F1  : {macro_f1:.4f}")


if __name__ == "__main__":
    main()
