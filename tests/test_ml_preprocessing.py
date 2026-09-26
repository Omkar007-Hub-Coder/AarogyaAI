"""
tests/test_ml_preprocessing.py

Tests for preprocessing utilities (splits, image utils).
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import tempfile
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.preprocessing.splits import (
    make_tabular_splits,
    make_image_splits,
    save_split_manifest,
    load_split_manifest,
)
from ml.preprocessing.image_utils import (
    normalize_image,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class TestTabularSplits:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(0, 1, (200, 5)).astype(np.float32)
        self.y = rng.integers(0, 3, 200)

    def test_split_sizes(self):
        splits = make_tabular_splits(self.X, self.y, val_ratio=0.1, test_ratio=0.1)
        total = sum(len(s[0]) for s in splits.values())
        assert total == 200

    def test_split_keys(self):
        splits = make_tabular_splits(self.X, self.y)
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_no_overlap(self):
        """Train, val, test must not share any samples."""
        splits = make_tabular_splits(self.X, self.y)
        # Use row content as identity
        def to_set(arr):
            return set(map(tuple, arr.tolist()))
        train_set = to_set(splits["train"][0])
        val_set   = to_set(splits["val"][0])
        test_set  = to_set(splits["test"][0])
        assert train_set.isdisjoint(val_set),  "Train/val overlap"
        assert train_set.isdisjoint(test_set), "Train/test overlap"
        assert val_set.isdisjoint(test_set),   "Val/test overlap"

    def test_reproducible(self):
        s1 = make_tabular_splits(self.X, self.y)
        s2 = make_tabular_splits(self.X, self.y)
        np.testing.assert_array_equal(s1["train"][0], s2["train"][0])


class TestImageSplits:
    def test_image_splits(self, tmp_path):
        paths = [tmp_path / f"{i}.jpg" for i in range(100)]
        labels = ["cat"] * 50 + ["dog"] * 50
        splits = make_image_splits(paths, labels, val_ratio=0.1, test_ratio=0.1)
        assert set(splits.keys()) == {"train", "val", "test"}
        total = sum(len(v) for v in splits.values())
        assert total == 100

    def test_manifest_roundtrip(self, tmp_path):
        paths = [Path(f"/fake/{i}.jpg") for i in range(50)]
        labels = ["a"] * 25 + ["b"] * 25
        splits = make_image_splits(paths, labels)
        manifest_path = tmp_path / "manifest.json"
        save_split_manifest(splits, manifest_path)
        loaded = load_split_manifest(manifest_path)
        assert loaded["random_seed"] == 42
        assert "train" in loaded["splits"]


class TestImageUtils:
    def test_normalize_output_range(self):
        """Normalized values should be within a reasonable range around 0."""
        arr = np.random.rand(224, 224, 3).astype(np.float32)
        normed = normalize_image(arr)
        # With ImageNet normalization, range roughly [-2.5, 2.5]
        assert normed.min() > -4.0
        assert normed.max() < 4.0

    def test_normalize_shape_preserved(self):
        arr = np.random.rand(224, 224, 3).astype(np.float32)
        normed = normalize_image(arr)
        assert normed.shape == (224, 224, 3)

    def test_normalize_uses_imagenet_stats(self):
        """Zero image normalized with ImageNet mean should equal -mean/std."""
        arr = np.zeros((1, 1, 3), dtype=np.float32)
        normed = normalize_image(arr)
        expected = -IMAGENET_MEAN / IMAGENET_STD
        np.testing.assert_allclose(normed[0, 0], expected, rtol=1e-5)
