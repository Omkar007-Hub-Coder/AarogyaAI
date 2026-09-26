"""
tests/test_ml_recommendation.py

Tests for the yoga recommendation engine.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.recommendation.yoga_metadata import get_all_poses, poses_to_dataframe, YOGA_POSES
from ml.recommendation.engine import (
    pose_to_vector,
    user_to_vector,
    compute_suitability,
    recommend_poses,
    POSE_VECTOR_DIM,
    GOAL_INDEX,
)


SAMPLE_USER = {
    "yoga_experience": 1,
    "flexibility_score": 3.0,
    "strength_score": 3.0,
    "activity_level": 1,
    "session_duration_min": 30.0,
    "primary_goal": "General Fitness",
}


class TestYogaMetadata:
    def test_poses_not_empty(self):
        poses = get_all_poses()
        assert len(poses) >= 20, "Should have at least 20 poses"

    def test_pose_to_dataframe(self):
        df = poses_to_dataframe()
        assert len(df) == len(YOGA_POSES)
        assert "name" in df.columns
        assert "difficulty" in df.columns

    def test_difficulty_values_valid(self):
        valid = {"Beginner", "Intermediate", "Advanced"}
        for p in YOGA_POSES:
            assert p.difficulty in valid, f"{p.name}: invalid difficulty {p.difficulty}"

    def test_difficulty_num_consistent(self):
        mapping = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
        for p in YOGA_POSES:
            assert p.difficulty_num == mapping[p.difficulty], \
                f"{p.name}: difficulty_num mismatch"

    def test_goals_valid(self):
        valid_goals = set(GOAL_INDEX.keys())
        for p in YOGA_POSES:
            for g in p.goals:
                assert g in valid_goals, f"{p.name}: unknown goal '{g}'"

    def test_requirements_in_range(self):
        for p in YOGA_POSES:
            assert 1 <= p.flexibility_req <= 5, f"{p.name}: flex_req out of range"
            assert 1 <= p.strength_req <= 5,    f"{p.name}: str_req out of range"
            assert 0 <= p.experience_req <= 3,  f"{p.name}: exp_req out of range"


class TestVectors:
    def test_pose_vector_shape(self):
        for p in YOGA_POSES:
            v = pose_to_vector(p)
            assert v.shape == (POSE_VECTOR_DIM,), f"{p.name}: wrong vector shape"

    def test_pose_vector_dtype(self):
        v = pose_to_vector(YOGA_POSES[0])
        assert v.dtype == np.float32

    def test_user_vector_shape(self):
        v = user_to_vector(SAMPLE_USER)
        assert v.shape == (POSE_VECTOR_DIM,)

    def test_user_vector_no_nan(self):
        v = user_to_vector(SAMPLE_USER)
        assert not np.any(np.isnan(v))

    def test_missing_user_keys_handled(self):
        """user_to_vector should not crash on partial input."""
        v = user_to_vector({"primary_goal": "Flexibility"})
        assert v.shape == (POSE_VECTOR_DIM,)


class TestSuitabilityScore:
    def test_score_in_unit_interval(self):
        u = user_to_vector(SAMPLE_USER)
        for p in YOGA_POSES:
            score = compute_suitability(u, pose_to_vector(p))
            assert 0.0 <= score <= 1.0, \
                f"{p.name}: score {score} out of [0,1]"

    def test_beginner_scores_beginner_poses_higher(self):
        """A beginner user should score beginner poses higher than advanced ones on average."""
        beginner_user = {
            "yoga_experience": 0, "flexibility_score": 1.0,
            "strength_score": 1.0, "activity_level": 0,
            "session_duration_min": 20, "primary_goal": "General Fitness",
        }
        u = user_to_vector(beginner_user)
        beginner_poses = [p for p in YOGA_POSES if p.difficulty == "Beginner"]
        advanced_poses = [p for p in YOGA_POSES if p.difficulty == "Advanced"]
        avg_beginner = np.mean([compute_suitability(u, pose_to_vector(p)) for p in beginner_poses])
        avg_advanced = np.mean([compute_suitability(u, pose_to_vector(p)) for p in advanced_poses])
        assert avg_beginner > avg_advanced, \
            f"Beginner user avg_beginner={avg_beginner:.3f} should > avg_advanced={avg_advanced:.3f}"


class TestRecommendations:
    def test_returns_list(self):
        recs = recommend_poses(SAMPLE_USER, top_n=5)
        assert isinstance(recs, list)
        assert len(recs) == 5

    def test_sorted_by_score(self):
        recs = recommend_poses(SAMPLE_USER, top_n=10)
        scores = [r.suitability_score for r in recs]
        assert scores == sorted(scores, reverse=True), "Recommendations not sorted"

    def test_to_dict_keys(self):
        recs = recommend_poses(SAMPLE_USER, top_n=3)
        required_keys = {"pose_name", "suitability_score", "difficulty",
                         "duration_min", "target_areas", "category", "reason"}
        for r in recs:
            d = r.to_dict()
            assert required_keys.issubset(d.keys()), f"Missing keys: {required_keys - d.keys()}"

    def test_difficulty_filter(self):
        recs = recommend_poses(SAMPLE_USER, top_n=20, difficulty_filter="Beginner")
        for r in recs:
            assert r.difficulty == "Beginner", f"{r.pose_name} is not Beginner"

    def test_reason_not_empty(self):
        recs = recommend_poses(SAMPLE_USER, top_n=5)
        for r in recs:
            assert len(r.reason) > 0, "Empty reason"

    def test_top_n_capped_to_catalog_size(self):
        recs = recommend_poses(SAMPLE_USER, top_n=9999)
        assert len(recs) == len(YOGA_POSES)
