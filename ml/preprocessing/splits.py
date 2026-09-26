"""
ml/preprocessing/splits.py

Reproducible train/val/test splits with data-leakage prevention.
All splits are stratified by class and saved to a manifest file.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def make_image_splits(
    image_paths: list[Path],
    labels: list[str],
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> dict[str, list[Any]]:
    """
    Stratified split of image paths into train / val / test.

    Returns dict with keys 'train', 'val', 'test', each containing
    a list of (str_path, label) tuples.
    """
    paths_str = [str(p) for p in image_paths]

    # First, carve out test set
    X_tv, X_test, y_tv, y_test = train_test_split(
        paths_str, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # Then split the remaining into train / val
    relative_val = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=relative_val,
        stratify=y_tv,
        random_state=RANDOM_SEED,
    )

    return {
        "train": list(zip(X_train, y_train)),
        "val":   list(zip(X_val,   y_val)),
        "test":  list(zip(X_test,  y_test)),
    }


def make_tabular_splits(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Stratified split for tabular (non-image) data."""
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=RANDOM_SEED,
    )
    relative_val = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=relative_val, stratify=y_tv, random_state=RANDOM_SEED,
    )
    return {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
        "test":  (X_test,  y_test),
    }


def save_split_manifest(splits: dict, output_path: str | Path) -> None:
    """Save a split manifest as JSON for full reproducibility."""
    manifest = {
        "random_seed": RANDOM_SEED,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "splits": {
            k: [(p, label) for p, label in v] for k, v in splits.items()
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_split_manifest(manifest_path: str | Path) -> dict:
    with open(manifest_path) as f:
        return json.load(f)
