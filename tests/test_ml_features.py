"""Tests for ML feature engineering."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from ml.features import build_feature_vector, compute_bmi, bmi_category, FEATURE_NAMES


SAMPLE_PROFILE = {
    "age": 28,
    "gender": "male",
    "height_cm": 175.0,
    "weight_kg": 72.0,
    "activity_level": "moderate",
    "health_goal": "muscle_gain",
    "dietary_preference": "non-vegetarian",
}


def test_compute_bmi():
    bmi = compute_bmi(175.0, 72.0)
    assert abs(bmi - 23.51) < 0.1


def test_bmi_category():
    assert bmi_category(17.0) == 0   # underweight
    assert bmi_category(22.0) == 1   # normal
    assert bmi_category(27.0) == 2   # overweight
    assert bmi_category(32.0) == 3   # obese


def test_feature_vector_shape():
    vec = build_feature_vector(SAMPLE_PROFILE)
    assert vec.shape == (len(FEATURE_NAMES),)


def test_feature_vector_dtype():
    vec = build_feature_vector(SAMPLE_PROFILE)
    assert vec.dtype == np.float32


def test_feature_vector_bmi_embedded():
    """BMI should be computed and embedded in the feature vector at index 4."""
    vec = build_feature_vector(SAMPLE_PROFILE)
    expected_bmi = compute_bmi(175.0, 72.0)
    assert abs(vec[4] - expected_bmi) < 0.01
