"""
scripts/preprocess.py  (full replacement of previous stub)

Master preprocessing pipeline for AarogyaAI.

Sub-commands:
    --yoga82      Process Yoga-82 image dataset
    --skeleton    Process BlazePose skeleton CSVs
    --all         Run all available preprocessing steps

Usage (from AarogyaAI/ root with venv active):
    python scripts/preprocess.py --all
    python scripts/preprocess.py --yoga82
    python scripts/preprocess.py --skeleton

See data/README.md for required input file structure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# ─────────────────────────────────────────────────────────────────────────────
# Yoga-82 image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

YOGA82_RAW  = RAW_DIR  / "yoga82"
YOGA82_OUT  = PROCESSED_DIR / "yoga82"
IMAGE_SIZE  = (224, 224)
VAL_RATIO   = 0.10   # 10% of train → validation
RANDOM_SEED = 42


def _image_md5(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def preprocess_yoga82() -> None:
    """
    Process Yoga-82 raw images.

    The dataset ships with its own train/valid/test splits — these are used
    as-is. No re-splitting is performed (that would break the original
    evaluation protocol).

    Steps:
    1. Validate all three split directories exist
    2. Scan every image for corruption (PIL verify)
    3. Build a full MD5 index across all three splits
    4. Identify within-split duplicates (same bytes, different filename in same split)
       → keep the lexicographically first filename, skip the rest
    5. Identify cross-split duplicates (same image bytes appear in both train
       AND valid/test) → remove from train only; valid and test are kept pristine
       to preserve evaluation integrity
    6. Resize all kept images to 224×224 RGB and copy to data/processed/yoga82/
    7. Save metadata, class-to-index mapping, and a full split manifest
       recording every skipped file and the reason
    """
    import warnings
    from collections import defaultdict

    train_src = YOGA82_RAW / "yoga_train"
    valid_src = YOGA82_RAW / "yoga_valid"
    test_src  = YOGA82_RAW / "yoga_test"

    missing = [s for s in [train_src, valid_src, test_src] if not s.exists()]
    if missing:
        for m in missing:
            logger.error(f"Directory not found: {m}")
        return

    try:
        from PIL import Image as PILImage
    except ImportError:
        logger.error("Pillow required: pip install pillow")
        return

    # ── Step 1: Discover classes from train (authoritative) ──────────────────
    class_names = sorted([d.name for d in train_src.iterdir() if d.is_dir()])
    logger.info(f"Found {len(class_names)} classes")

    # ── Step 2 & 3: Scan all three splits, build hash index ──────────────────
    # hash → list of (split_name, class, Path)
    hash_index: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)

    raw_counts: dict[str, dict[str, int]] = {"train": {}, "valid": {}, "test": {}}
    corrupted_files: list[str] = []

    split_dirs = [("train", train_src), ("valid", valid_src), ("test", test_src)]
    for split_name, split_dir in split_dirs:
        for cls_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
            cls = cls_dir.name
            imgs = sorted(cls_dir.glob("*.jpg"))
            raw_counts[split_name][cls] = len(imgs)
            for img_path in imgs:
                if img_path.stat().st_size == 0:
                    corrupted_files.append(f"[zero-byte] {img_path}")
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        with PILImage.open(img_path) as im:
                            im.verify()
                    except Exception as e:
                        corrupted_files.append(f"[corrupt] {img_path}: {e}")
                        continue
                h = _image_md5(img_path)
                hash_index[h].append((split_name, cls, img_path))

    logger.info(
        f"Scanned: train={sum(raw_counts['train'].values())}, "
        f"valid={sum(raw_counts['valid'].values())}, "
        f"test={sum(raw_counts['test'].values())} | "
        f"corrupted={len(corrupted_files)}"
    )

    # ── Step 4 & 5: Determine which images to skip ───────────────────────────
    # For each hash group, decide what to keep.
    # Policy:
    #   - valid/test images are NEVER removed (evaluation integrity)
    #   - within-split duplicates in train/valid/test: keep first by sorted path, skip rest
    #   - cross-split duplicates: remove the train copy if the same hash
    #     appears in valid or test (prevents leakage into evaluation)

    skipped: list[dict] = []                 # full record of every skipped file
    keep: dict[str, set[Path]] = {"train": set(), "valid": set(), "test": set()}

    for h, entries in hash_index.items():
        by_split: dict[str, list[tuple[str, Path]]] = defaultdict(list)
        for split_name, cls, path in entries:
            by_split[split_name].append((cls, path))

        eval_splits = {"valid", "test"}
        in_eval = any(s in by_split for s in eval_splits)

        for split_name, cls_paths in by_split.items():
            # Sort for determinism; keep first, skip rest (within-split dups)
            cls_paths_sorted = sorted(cls_paths, key=lambda x: str(x[1]))
            for i, (cls, path) in enumerate(cls_paths_sorted):
                if i > 0:
                    skipped.append({
                        "path": str(path), "split": split_name, "cls": cls,
                        "reason": "within_split_duplicate", "md5": h
                    })
                    continue
                # Cross-split: skip train copy if same hash exists in valid/test
                if split_name == "train" and in_eval:
                    skipped.append({
                        "path": str(path), "split": split_name, "cls": cls,
                        "reason": "cross_split_duplicate_removed_from_train", "md5": h
                    })
                    continue
                keep[split_name].add(path)

    # ── Step 6: Resize and copy ───────────────────────────────────────────────
    def _process_split(paths: set[Path], split_name: str) -> dict[str, int]:
        """Resize + copy images; return per-class counts of written images."""
        per_class: dict[str, int] = defaultdict(int)
        failed = 0
        for img_path in sorted(paths):
            cls = img_path.parent.name
            out_dir = YOGA82_OUT / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / img_path.name
            if out_path.exists():
                per_class[cls] += 1
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with PILImage.open(img_path) as im:
                        im = im.convert("RGB").resize(IMAGE_SIZE, PILImage.BILINEAR)
                        im.save(out_path, "JPEG", quality=90)
                per_class[cls] += 1
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                failed += 1
        logger.info(
            f"[{split_name}] wrote {sum(per_class.values())} images "
            f"across {len(per_class)} classes  (failed={failed})"
        )
        return dict(per_class)

    logger.info("Processing train …")
    train_per_class = _process_split(keep["train"], "train")
    logger.info("Processing valid …")
    valid_per_class = _process_split(keep["valid"], "valid")
    logger.info("Processing test …")
    test_per_class  = _process_split(keep["test"],  "test")

    # ── Step 7: Save metadata & manifest ─────────────────────────────────────
    meta_dir  = PROCESSED_DIR / "metadata"
    splits_dir = PROCESSED_DIR / "splits"
    meta_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # class_to_idx — sorted alphabetically (matches class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    with open(meta_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(meta_dir / "yoga82_classes.txt", "w") as f:
        f.write("\n".join(class_names))

    # Per-class stats
    per_class_stats = {}
    for cls in class_names:
        per_class_stats[cls] = {
            "raw_train":  raw_counts["train"].get(cls, 0),
            "raw_valid":  raw_counts["valid"].get(cls, 0),
            "raw_test":   raw_counts["test"].get(cls, 0),
            "proc_train": train_per_class.get(cls, 0),
            "proc_valid": valid_per_class.get(cls, 0),
            "proc_test":  test_per_class.get(cls, 0),
        }

    dataset_stats = {
        "source_splits_used": ["yoga_train", "yoga_valid", "yoga_test"],
        "split_policy": "dataset_provided_splits_preserved",
        "image_size": list(IMAGE_SIZE),
        "n_classes": len(class_names),
        "raw_totals": {
            "train": sum(raw_counts["train"].values()),
            "valid": sum(raw_counts["valid"].values()),
            "test":  sum(raw_counts["test"].values()),
        },
        "processed_totals": {
            "train": sum(train_per_class.values()),
            "valid": sum(valid_per_class.values()),
            "test":  sum(test_per_class.values()),
        },
        "skipped_total": len(skipped),
        "skipped_within_split_dup": sum(1 for s in skipped if s["reason"] == "within_split_duplicate"),
        "skipped_cross_split_removed_from_train": sum(
            1 for s in skipped if s["reason"] == "cross_split_duplicate_removed_from_train"
        ),
        "corrupted": corrupted_files,
        "per_class": per_class_stats,
    }
    with open(meta_dir / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)

    # Full split manifest
    split_manifest = {
        "split_policy": "dataset_provided_splits_preserved",
        "note": "yoga_valid is the original validation set from Yoga-82, not a re-split of train",
        "processed_counts": {
            "train": sum(train_per_class.values()),
            "valid": sum(valid_per_class.values()),
            "test":  sum(test_per_class.values()),
        },
        "skipped": skipped,
    }
    with open(splits_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)

    logger.info(
        f"Yoga-82 preprocessing complete. "
        f"train={sum(train_per_class.values())}, "
        f"valid={sum(valid_per_class.values())}, "
        f"test={sum(test_per_class.values())} | "
        f"skipped={len(skipped)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton (BlazePose) preprocessing
# ─────────────────────────────────────────────────────────────────────────────

SKELETON_RAW = RAW_DIR  / "blazepose_yoga82"
SKELETON_OUT = PROCESSED_DIR / "skeleton"


def preprocess_skeleton() -> None:
    """
    Process BlazePose skeleton CSVs.
    Looks for:
      data/raw/blazepose_yoga82/keypoints_train.csv
      data/raw/blazepose_yoga82/keypoints_test.csv  (optional)
    """
    import joblib
    from sklearn.preprocessing import LabelEncoder
    from ml.preprocessing.skeleton_utils import load_skeleton_csv
    from ml.preprocessing.splits import make_tabular_splits

    train_csv = SKELETON_RAW / "keypoints_train.csv"
    test_csv  = SKELETON_RAW / "keypoints_test.csv"

    if not train_csv.exists():
        logger.error(
            f"Skeleton CSV not found: {train_csv}\n"
            "Download BlazePose Skeletons Yoga-82 from Kaggle and place "
            "keypoints_train.csv in data/raw/blazepose_yoga82/"
        )
        return

    logger.info(f"Loading skeleton train data from {train_csv} …")
    result = load_skeleton_csv(str(train_csv))
    if result is None:
        logger.error("Failed to load skeleton CSV.")
        return

    X_train_full, y_train_full = result
    logger.info(f"Loaded {len(X_train_full)} skeleton samples, {X_train_full.shape[1]} features")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train_full)

    # Stratified val split
    from sklearn.model_selection import train_test_split as sk_split
    X_tr, X_val, y_tr, y_val = sk_split(
        X_train_full, y_enc, test_size=0.10, stratify=y_enc, random_state=RANDOM_SEED
    )

    SKELETON_OUT.mkdir(parents=True, exist_ok=True)
    np.save(SKELETON_OUT / "train_features.npy", X_tr)
    np.save(SKELETON_OUT / "train_labels.npy",   y_tr)
    np.save(SKELETON_OUT / "val_features.npy",   X_val)
    np.save(SKELETON_OUT / "val_labels.npy",     y_val)

    if test_csv.exists():
        test_result = load_skeleton_csv(str(test_csv))
        if test_result is not None:
            X_test, y_test_raw = test_result
            y_test = le.transform(y_test_raw)
            np.save(SKELETON_OUT / "test_features.npy", X_test)
            np.save(SKELETON_OUT / "test_labels.npy",   y_test)
            logger.info(f"Test set: {len(X_test)} samples")

    joblib.dump(le, SKELETON_OUT / "label_encoder.joblib")

    stats = {
        "n_samples_train": int(len(X_tr)),
        "n_samples_val":   int(len(X_val)),
        "n_classes":       int(len(le.classes_)),
        "feature_dim":     int(X_tr.shape[1]),
        "classes":         le.classes_.tolist(),
    }
    with open(SKELETON_OUT / "skeleton_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Skeleton preprocessing complete. {stats}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AarogyaAI data preprocessing pipeline")
    parser.add_argument("--yoga82",   action="store_true", help="Process Yoga-82 images")
    parser.add_argument("--skeleton", action="store_true", help="Process skeleton CSVs")
    parser.add_argument("--all",      action="store_true", help="Run all steps")
    args = parser.parse_args()

    if not (args.yoga82 or args.skeleton or args.all):
        parser.print_help()
        sys.exit(0)

    if args.all or args.yoga82:
        logger.info("=== Yoga-82 image preprocessing ===")
        preprocess_yoga82()

    if args.all or args.skeleton:
        logger.info("=== Skeleton (BlazePose) preprocessing ===")
        preprocess_skeleton()


if __name__ == "__main__":
    main()
