"""
tests/test_ml_skeleton.py

Tests for skeleton feature engineering and classifier.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.preprocessing.skeleton_utils import (
    extract_skeleton_features,
    SKELETON_FEATURE_DIM,
    N_LANDMARKS,
)


def make_fake_landmarks(seed: int = 0) -> np.ndarray:
    """Create a deterministic (33, 3) landmark array."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (N_LANDMARKS, 3)).astype(np.float32)


class TestSkeletonFeatures:
    def test_output_shape(self):
        lm = make_fake_landmarks()
        feat = extract_skeleton_features(lm)
        assert feat.shape == (SKELETON_FEATURE_DIM,), \
            f"Expected ({SKELETON_FEATURE_DIM},), got {feat.shape}"

    def test_output_dtype(self):
        lm = make_fake_landmarks()
        feat = extract_skeleton_features(lm)
        assert feat.dtype == np.float32

    def test_no_nan_inf(self):
        lm = make_fake_landmarks()
        feat = extract_skeleton_features(lm)
        assert not np.any(np.isnan(feat)), "NaN in features"
        assert not np.any(np.isinf(feat)), "Inf in features"

    def test_accepts_4_column_landmarks(self):
        """BlazePose outputs (33, 4) — visibility column should be ignored."""
        rng = np.random.default_rng(1)
        lm4 = rng.uniform(-1, 1, (N_LANDMARKS, 4)).astype(np.float32)
        feat = extract_skeleton_features(lm4)
        assert feat.shape == (SKELETON_FEATURE_DIM,)

    def test_angles_in_valid_range(self):
        """Joint angles must be in [0, 180] degrees."""
        lm = make_fake_landmarks()
        feat = extract_skeleton_features(lm)
        # Angles are the last 10 features
        angles = feat[-10:]
        assert np.all(angles >= 0.0),   "Negative angle"
        assert np.all(angles <= 180.0), "Angle > 180°"

    def test_normalization_centers_on_hip(self):
        """After normalization, hip midpoint should be near origin."""
        lm = make_fake_landmarks(seed=42)
        feat = extract_skeleton_features(lm)
        # left_hip (idx 23) and right_hip (idx 24) in normalized space
        hip_left  = feat[23*3 : 23*3+3]
        hip_right = feat[24*3 : 24*3+3]
        hip_mid = (hip_left + hip_right) / 2.0
        assert np.linalg.norm(hip_mid) < 0.1, \
            f"Hip midpoint should be near origin, got {hip_mid}"

    def test_deterministic(self):
        """Same input → same output."""
        lm = make_fake_landmarks(seed=7)
        f1 = extract_skeleton_features(lm)
        f2 = extract_skeleton_features(lm)
        np.testing.assert_array_equal(f1, f2)
