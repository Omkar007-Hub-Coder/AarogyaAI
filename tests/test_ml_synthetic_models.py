"""
tests/test_ml_synthetic_models.py

Tests for user-goal classifier and difficulty predictor.
These use SYNTHETIC data by design.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.models.user_goal_classifier import (
    generate_synthetic_profiles,
    build_goal_models,
    FEATURE_COLUMNS as GOAL_FEATURES,
    GOAL_CLASSES,
)
from ml.models.difficulty_predictor import (
    generate_synthetic_difficulty_data,
    build_difficulty_models,
    FEATURE_COLUMNS as DIFF_FEATURES,
    DIFFICULTY_CLASSES,
)
from ml.evaluation.metrics import evaluate_classifier, cross_validate_classifier


class TestSyntheticDataGeneration:
    def test_goal_profiles_shape(self):
        df = generate_synthetic_profiles(n_samples=500)
        assert len(df) == 500
        for col in GOAL_FEATURES:
            assert col in df.columns, f"Missing column: {col}"

    def test_goal_profiles_has_synthetic_flag(self):
        df = generate_synthetic_profiles(n_samples=100)
        assert "_synthetic" in df.columns
        assert df["_synthetic"].all(), "All rows must be marked synthetic"

    def test_goal_label_values(self):
        df = generate_synthetic_profiles(n_samples=300)
        labels = set(df["goal_label"].unique())
        assert labels.issubset(set(GOAL_CLASSES)), f"Unexpected labels: {labels - set(GOAL_CLASSES)}"

    def test_goal_class_distribution_not_degenerate(self):
        """No class should dominate > 70% of samples."""
        df = generate_synthetic_profiles(n_samples=1000)
        counts = df["goal_label"].value_counts(normalize=True)
        assert counts.max() < 0.70, f"Class distribution too skewed: {counts.to_dict()}"

    def test_difficulty_data_shape(self):
        df = generate_synthetic_difficulty_data(n_samples=400)
        assert len(df) == 400
        for col in DIFF_FEATURES:
            assert col in df.columns

    def test_difficulty_labels_valid(self):
        df = generate_synthetic_difficulty_data(n_samples=300)
        labels = set(df["difficulty_label"].unique())
        assert labels.issubset(set(DIFFICULTY_CLASSES))

    def test_reproducibility(self):
        df1 = generate_synthetic_profiles(n_samples=100, seed=42)
        df2 = generate_synthetic_profiles(n_samples=100, seed=42)
        assert (df1["goal_label"].values == df2["goal_label"].values).all()


class TestGoalClassifierTraining:
    """Train on small synthetic data — verifies pipeline works, not real-world performance."""

    def setup_method(self):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        df = generate_synthetic_profiles(n_samples=400)
        X = df[GOAL_FEATURES].values.astype(np.float32)
        self.le = LabelEncoder()
        y = self.le.fit_transform(df["goal_label"].values)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    def test_logistic_regression_trains(self):
        model = build_goal_models()["logistic_regression"]
        model.fit(self.X_train, self.y_train)
        m = evaluate_classifier(model, self.X_test, self.y_test,
                                 label_names=self.le.classes_.tolist(), data_source="SYNTHETIC")
        assert m["data_source"] == "SYNTHETIC"
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_random_forest_trains(self):
        model = build_goal_models()["random_forest"]
        model.fit(self.X_train, self.y_train)
        m = evaluate_classifier(model, self.X_test, self.y_test, data_source="SYNTHETIC")
        assert m["accuracy"] > 0.3, "RF accuracy suspiciously low even on synthetic data"

    def test_metrics_have_required_keys(self):
        model = build_goal_models()["logistic_regression"]
        model.fit(self.X_train, self.y_train)
        m = evaluate_classifier(model, self.X_test, self.y_test, data_source="SYNTHETIC")
        for key in ["accuracy", "precision", "recall", "f1_score", "confusion_matrix", "data_source"]:
            assert key in m, f"Missing key: {key}"


class TestDifficultyClassifierTraining:
    def setup_method(self):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        df = generate_synthetic_difficulty_data(n_samples=400)
        X = df[DIFF_FEATURES].values.astype(np.float32)
        self.le = LabelEncoder()
        y = self.le.fit_transform(df["difficulty_label"].values)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    def test_gradient_boosting_trains(self):
        model = build_difficulty_models()["gradient_boosting"]
        model.fit(self.X_train, self.y_train)
        m = evaluate_classifier(model, self.X_test, self.y_test, data_source="SYNTHETIC")
        assert m["data_source"] == "SYNTHETIC"
        assert m["accuracy"] > 0.3
