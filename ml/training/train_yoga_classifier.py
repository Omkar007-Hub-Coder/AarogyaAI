"""
ml/training/train_yoga_classifier.py

Train EfficientNetB0-based yoga pose image classifier (Yoga-82).

Training strategy:
  Phase 1 (10 epochs): Train head only (base frozen)
  Phase 2 (10 epochs): Fine-tune top-20 EfficientNet layers

Usage (from AarogyaAI/ root):
    python -m ml.training.train_yoga_classifier [--subset N] [--epochs-p1 E] [--epochs-p2 E]

  --subset N : Use only N images per class (useful on limited hardware).
               Full dataset if not specified.
  --epochs-p1: Phase 1 epochs (default 10)
  --epochs-p2: Phase 2 epochs (default 10)

Requirements:
  pip install tensorflow pillow
  data/processed/yoga82/train/, data/processed/yoga82/valid/, and
  data/processed/yoga82/test/ must exist.
  (run scripts/preprocess.py --yoga82 first)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = ROOT / "data" / "processed" / "yoga82"
MODELS_DIR    = ROOT / "models"
METRICS_DIR   = ROOT / "models" / "metrics"


def _check_tensorflow() -> bool:
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow {tf.__version__} found.")
        return True
    except ImportError:
        return False


def run(subset_per_class: int | None = None, epochs_p1: int = 10, epochs_p2: int = 10) -> None:
    if not _check_tensorflow():
        logger.error(
            "TensorFlow not installed. Install with:\n"
            "  pip install tensorflow\n"
            "Then re-run this script."
        )
        sys.exit(1)

    train_dir = PROCESSED_DIR / "train"
    val_dir   = PROCESSED_DIR / "valid"   # dataset ships with 'valid', not 'val'
    test_dir  = PROCESSED_DIR / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            logger.error(
                f"Directory not found: {d}\n"
                "Run: python scripts/preprocess.py --yoga82\n"
                "     (requires data/raw/yoga82/ to be populated)"
            )
            sys.exit(1)

    import tensorflow as tf
    from ml.models.yoga_classifier import (
        build_yoga_classifier,
        save_yoga_classifier,
        MODEL_INPUT_SIZE,
    )
    from ml.evaluation.metrics import save_metrics

    # ── Discover classes ─────────────────────────────────────────────────────
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    num_classes = len(class_names)
    logger.info(f"Found {num_classes} pose classes in {train_dir}")

    if num_classes == 0:
        logger.error("No class subdirectories found in train dir. Check your dataset.")
        sys.exit(1)

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    with open(MODELS_DIR / "yoga82_class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    # ── Build tf.data datasets ────────────────────────────────────────────────
    def _collect_pairs(split_dir: Path, limit: int | None) -> list[tuple[str, int]]:
        pairs = []
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            idx = class_to_idx.get(cls_dir.name)
            if idx is None:
                continue
            imgs = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
            if limit:
                imgs = imgs[:limit]
            pairs.extend([(str(p), idx) for p in imgs])
        return pairs

    train_pairs = _collect_pairs(train_dir, subset_per_class)
    val_pairs   = _collect_pairs(val_dir,   subset_per_class)
    test_pairs  = _collect_pairs(test_dir,  None)  # always use full test set; no subset
    logger.info(
        f"Train samples: {len(train_pairs)}, "
        f"Val samples: {len(val_pairs)}, "
        f"Test samples: {len(test_pairs)}"
    )

    if len(train_pairs) == 0:
        logger.error("No training images found. Ensure images are in data/processed/yoga82/train/<ClassName>/")
        sys.exit(1)

    # Build tf.data
    from ml.models.yoga_classifier import build_tf_dataset
    BATCH = 32
    train_ds = build_tf_dataset(train_pairs, batch_size=BATCH, augment=True,  shuffle=True)
    val_ds   = build_tf_dataset(val_pairs,   batch_size=BATCH, augment=False, shuffle=False)
    test_ds  = build_tf_dataset(test_pairs,  batch_size=BATCH, augment=False, shuffle=False)

    # Checkpoint path — saves best val_accuracy model during training
    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_path = str(MODELS_DIR / "yoga_classifier_best.keras")

    def _make_callbacks() -> list:
        """
        Create a fresh callbacks list.
        Called separately for Phase 1 and Phase 2 so that EarlyStopping's
        internal wait counter resets between phases.
        """
        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

    # ── Phase 1: Train head only ──────────────────────────────────────────────
    logger.info(f"Phase 1: training head ({epochs_p1} epochs, EfficientNet frozen) …")
    model = build_yoga_classifier(num_classes=num_classes, fine_tune=False)
    # Use legacy Adam on Apple Silicon (arm64) — standard Adam runs ~10× slower on M1/M2
    _Adam = tf.keras.optimizers.legacy.Adam
    model.compile(
        optimizer=_Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_p1, callbacks=_make_callbacks(), verbose=1,
    )

    # ── Phase 2: Fine-tune top-20 layers ─────────────────────────────────────
    if epochs_p2 > 0:
        logger.info(f"Phase 2: fine-tuning top-20 layers ({epochs_p2} epochs) …")
        # Save Phase 1 weights to a temporary file, then reload into fine-tune model.
        # Building a new graph with fine_tune=True changes trainability of base layers,
        # so we must transfer weights explicitly.
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            model.save_weights(tmp_path)
            model_ft = build_yoga_classifier(num_classes=num_classes, fine_tune=True)
            model_ft.compile(
                optimizer=_Adam(1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            model_ft.load_weights(tmp_path)
            h2 = model_ft.fit(
                train_ds, validation_data=val_ds,
                epochs=epochs_p2, callbacks=_make_callbacks(), verbose=1,
            )
            model = model_ft   # use fine-tuned model for saving and evaluation
        finally:
            os.unlink(tmp_path)

    # ── Save final model ──────────────────────────────────────────────────────
    save_yoga_classifier(model, class_names, MODELS_DIR)
    logger.info("Yoga classifier (final epoch) saved.")
    logger.info(f"Best-checkpoint (by val_accuracy) also saved to {ckpt_path}")

    # ── Evaluate on held-out TEST split (never seen during training) ──────────
    logger.info("Evaluating on held-out TEST split …")
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix, classification_report)

    def _collect_predictions(ds) -> tuple[list, list]:
        y_t, y_p = [], []
        for imgs, lbls in ds:
            preds = model.predict(imgs, verbose=0)
            y_p.extend(np.argmax(preds, axis=1).tolist())
            y_t.extend(lbls.numpy().tolist())
        return y_t, y_p

    test_true, test_pred = _collect_predictions(test_ds)
    val_true,  val_pred  = _collect_predictions(val_ds)

    from ml.evaluation.metrics import save_metrics, print_metrics_summary

    def _build_metrics(y_true, y_pred, split_name: str) -> dict:
        return {
            "data_source": "REAL",
            "dataset": "Yoga-82",
            "split": split_name,
            "n_samples": len(y_true),
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "f1_score":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
            ),
        }

    test_metrics = _build_metrics(test_true, test_pred, "test")
    val_metrics  = _build_metrics(val_true,  val_pred,  "valid")

    final_metrics = {
        "data_source": "REAL",
        "dataset": "Yoga-82",
        "training_config": {
            "epochs_phase1": epochs_p1,
            "epochs_phase2": epochs_p2,
            "batch_size": BATCH,
            "subset_per_class": subset_per_class,
        },
        "test": test_metrics,
        "valid": val_metrics,
    }

    print(f"\n[Yoga Classifier — TEST]  "
          f"accuracy={test_metrics['accuracy']:.4f}  "
          f"F1={test_metrics['f1_score']:.4f}  "
          f"(data_source=REAL, n={len(test_true)})")
    print(f"[Yoga Classifier — VALID] "
          f"accuracy={val_metrics['accuracy']:.4f}  "
          f"F1={val_metrics['f1_score']:.4f}  "
          f"(data_source=REAL, n={len(val_true)})")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(final_metrics, METRICS_DIR / "yoga_classifier_metrics.json")
    logger.info("Yoga classifier training and evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset",    type=int, default=None,
                        help="Max images per class (None = all)")
    parser.add_argument("--epochs-p1", type=int, default=10)
    parser.add_argument("--epochs-p2", type=int, default=10)
    args = parser.parse_args()
    run(subset_per_class=args.subset, epochs_p1=args.epochs_p1, epochs_p2=args.epochs_p2)
