"""
tests/test_personalization.py

Tests for the personalization + Ahar layer:
  - Profile validation & BMI
  - TDEE / macro estimation
  - Yoga ranking
  - Ahar filtering (allergy, dietary)
  - Ahar ranking
  - Feedback recording + score adjustment
  - End-to-end personalization API
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pytest
import numpy as np

from ml.personalization.profile import (
    compute_bmi,
    bmi_category,
    validate_and_build_profile,
    estimate_tdee,
    ProfileValidationError,
    UserProfileFull,
)
from ml.recommendation.ahar_engine import (
    recommend_ahar,
    _passes_dietary_filter,
    _passes_allergy_filter,
)
from ml.recommendation.ahar_metadata import get_all_foods, FoodItem
from ml.recommendation.engine import recommend_poses, user_to_vector, GOAL_INDEX


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

VALID_PROFILE_DATA = {
    "name": "Test User",
    "age": 30,
    "height_cm": 170.0,
    "weight_kg": 70.0,
    "activity_level": "moderate",
    "primary_goal": "General Fitness",
    "gender": "male",
    "yoga_experience": "beginner",
    "fitness_experience": "none",
    "flexibility_score": 3.0,
    "strength_score": 3.0,
    "balance_score": 3.0,
    "session_duration_min": 30.0,
    "days_per_week": 3,
    "dietary_preference": "vegetarian",
    "food_preferences": [],
    "allergies": [],
    "sleep_hours": 7.0,
    "previous_performance_score": 5.0,
}


@pytest.fixture
def valid_profile() -> UserProfileFull:
    return validate_and_build_profile(VALID_PROFILE_DATA)


# ─────────────────────────────────────────────────────────────────────────────
# BMI
# ─────────────────────────────────────────────────────────────────────────────

class TestBMI:
    def test_compute_bmi_formula(self):
        bmi = compute_bmi(170.0, 70.0)
        expected = 70.0 / (1.70 ** 2)
        assert abs(bmi - expected) < 0.01

    def test_bmi_underweight(self):
        assert bmi_category(17.0) == "Underweight"

    def test_bmi_normal(self):
        assert bmi_category(22.0) == "Normal weight"

    def test_bmi_overweight(self):
        assert bmi_category(27.0) == "Overweight"

    def test_bmi_obese(self):
        assert bmi_category(32.0) == "Obese"

    def test_bmi_boundary_normal(self):
        # Exactly 18.5 → Normal weight
        assert bmi_category(18.5) == "Normal weight"

    def test_bmi_boundary_overweight(self):
        # Exactly 25.0 → Overweight
        assert bmi_category(25.0) == "Overweight"

    def test_bmi_invalid_height(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            compute_bmi(0.0, 70.0)


# ─────────────────────────────────────────────────────────────────────────────
# Profile validation
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileValidation:
    def test_valid_profile_builds(self, valid_profile):
        assert valid_profile.name == "Test User"
        assert valid_profile.age == 30

    def test_bmi_computed(self, valid_profile):
        assert valid_profile.bmi > 0
        assert valid_profile.bmi_category in {"Underweight","Normal weight","Overweight","Obese"}

    def test_experience_encoded(self, valid_profile):
        assert valid_profile.yoga_experience_enc == 1  # "beginner" → 1

    def test_activity_encoded(self, valid_profile):
        assert valid_profile.activity_level_enc == 2  # "moderate" → 2

    def test_missing_name_raises(self):
        bad = {**VALID_PROFILE_DATA, "name": ""}
        with pytest.raises(ProfileValidationError):
            validate_and_build_profile(bad)

    def test_age_too_low_raises(self):
        bad = {**VALID_PROFILE_DATA, "age": 2}
        with pytest.raises(ProfileValidationError):
            validate_and_build_profile(bad)

    def test_age_too_high_raises(self):
        bad = {**VALID_PROFILE_DATA, "age": 200}
        with pytest.raises(ProfileValidationError):
            validate_and_build_profile(bad)

    def test_invalid_goal_raises(self):
        bad = {**VALID_PROFILE_DATA, "primary_goal": "Flying"}
        with pytest.raises(ProfileValidationError):
            validate_and_build_profile(bad)

    def test_invalid_activity_raises(self):
        bad = {**VALID_PROFILE_DATA, "activity_level": "hyperactive"}
        with pytest.raises(ProfileValidationError):
            validate_and_build_profile(bad)

    def test_invalid_dietary_raises(self):
        bad = {**VALID_PROFILE_DATA, "dietary_preference": "carnivore"}
        with pytest.raises(ProfileValidationError):
            validate_and_build_profile(bad)

    def test_flexibility_out_of_range_raises(self):
        bad = {**VALID_PROFILE_DATA, "flexibility_score": 6.0}
        with pytest.raises(ProfileValidationError):
            validate_and_build_profile(bad)

    def test_to_dict_has_bmi(self, valid_profile):
        d = valid_profile.to_dict()
        assert "bmi" in d
        assert d["bmi"] > 0

    def test_to_ml_features_keys(self, valid_profile):
        feats = valid_profile.to_ml_features()
        required_keys = [
            "age", "bmi", "activity_level", "yoga_experience",
            "flexibility_score", "strength_score", "sleep_hours",
            "session_duration_min", "primary_goal",
        ]
        for k in required_keys:
            assert k in feats, f"Missing key: {k}"


# ─────────────────────────────────────────────────────────────────────────────
# TDEE / macro estimation
# ─────────────────────────────────────────────────────────────────────────────

class TestTDEE:
    def test_tdee_greater_than_bmr(self, valid_profile):
        result = estimate_tdee(valid_profile)
        assert result["tdee_kcal"] > result["bmr_kcal"]

    def test_tdee_positive(self, valid_profile):
        result = estimate_tdee(valid_profile)
        assert result["tdee_kcal"] > 0

    def test_estimate_label_present(self, valid_profile):
        result = estimate_tdee(valid_profile)
        assert "ESTIMATE" in result["estimate_label"]
        assert "not a medical prescription" in result["estimate_label"].lower()

    def test_macro_targets_present(self, valid_profile):
        result = estimate_tdee(valid_profile)
        macros = result["macro_targets"]
        for key in ["protein_g", "carbs_g", "fat_g"]:
            assert key in macros
            assert macros[key] > 0

    def test_macro_pct_sum_to_one(self, valid_profile):
        result = estimate_tdee(valid_profile)
        m = result["macro_targets"]
        total = m["protein_pct"] + m["carbs_pct"] + m["fat_pct"]
        assert abs(total - 1.0) < 1e-6

    def test_female_bmr_less_than_male(self):
        male_data = {**VALID_PROFILE_DATA, "gender": "male"}
        female_data = {**VALID_PROFILE_DATA, "gender": "female"}
        male_p = validate_and_build_profile(male_data)
        female_p = validate_and_build_profile(female_data)
        assert estimate_tdee(male_p)["bmr_kcal"] > estimate_tdee(female_p)["bmr_kcal"]

    def test_active_higher_tdee_than_sedentary(self):
        active_data = {**VALID_PROFILE_DATA, "activity_level": "active"}
        sed_data    = {**VALID_PROFILE_DATA, "activity_level": "sedentary"}
        active_p = validate_and_build_profile(active_data)
        sed_p    = validate_and_build_profile(sed_data)
        assert estimate_tdee(active_p)["tdee_kcal"] > estimate_tdee(sed_p)["tdee_kcal"]

    def test_strength_goal_high_protein_pct(self):
        data = {**VALID_PROFILE_DATA, "primary_goal": "Strength"}
        p = validate_and_build_profile(data)
        result = estimate_tdee(p)
        assert result["macro_targets"]["protein_pct"] >= 0.30

    def test_weight_mgmt_goal_moderate_calorie_target(self):
        data = {**VALID_PROFILE_DATA, "primary_goal": "Weight Management"}
        p = validate_and_build_profile(data)
        result = estimate_tdee(p)
        # TDEE should be a sane positive value, not extreme
        assert 1000 < result["tdee_kcal"] < 6000


# ─────────────────────────────────────────────────────────────────────────────
# Yoga ranking (personalization)
# ─────────────────────────────────────────────────────────────────────────────

class TestYogaPersonalization:
    def test_returns_list(self, valid_profile):
        ml_feats = valid_profile.to_ml_features()
        recs = recommend_poses(ml_feats, top_n=5)
        assert isinstance(recs, list)
        assert len(recs) == 5

    def test_sorted_by_score(self, valid_profile):
        ml_feats = valid_profile.to_ml_features()
        recs = recommend_poses(ml_feats, top_n=10)
        scores = [r.suitability_score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_score_in_unit_interval(self, valid_profile):
        ml_feats = valid_profile.to_ml_features()
        for rec in recommend_poses(ml_feats, top_n=10):
            assert 0.0 <= rec.suitability_score <= 1.0

    def test_reason_not_empty(self, valid_profile):
        ml_feats = valid_profile.to_ml_features()
        for rec in recommend_poses(ml_feats, top_n=5):
            assert len(rec.reason) > 0

    def test_difficulty_filter_respected(self, valid_profile):
        ml_feats = valid_profile.to_ml_features()
        recs = recommend_poses(ml_feats, top_n=20, difficulty_filter="Beginner")
        assert all(r.difficulty == "Beginner" for r in recs)

    def test_beginner_scores_higher_for_novice(self):
        novice_data = {**VALID_PROFILE_DATA, "yoga_experience": "none", "flexibility_score": 1.0}
        expert_data = {**VALID_PROFILE_DATA, "yoga_experience": "advanced", "flexibility_score": 5.0}
        novice = validate_and_build_profile(novice_data)
        expert = validate_and_build_profile(expert_data)
        novice_recs = recommend_poses(novice.to_ml_features(), top_n=5)
        expert_recs = recommend_poses(expert.to_ml_features(), top_n=5)
        novice_diffs = {r.difficulty for r in novice_recs}
        expert_diffs = {r.difficulty for r in expert_recs}
        # Expert should get at least some Advanced poses; novice should not top-rank Advanced
        assert "Beginner" in novice_diffs
        assert "Advanced" not in novice_diffs or "Advanced" in expert_diffs


# ─────────────────────────────────────────────────────────────────────────────
# Ahar metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestAharMetadata:
    def test_catalog_not_empty(self):
        assert len(get_all_foods()) > 0

    def test_all_dietary_categories_valid(self):
        valid = {"vegan", "vegetarian", "egg", "non-vegetarian"}
        for f in get_all_foods():
            assert f.dietary_category in valid, f"Bad category: {f.name}"

    def test_calories_positive(self):
        for f in get_all_foods():
            assert f.calories_kcal >= 0, f"Negative calories: {f.name}"

    def test_macros_non_negative(self):
        for f in get_all_foods():
            assert f.protein_g >= 0
            assert f.carbs_g >= 0
            assert f.fat_g >= 0
            assert f.fibre_g >= 0

    def test_meal_suitability_not_empty(self):
        for f in get_all_foods():
            assert len(f.meal_suitability) > 0

    def test_goals_list_not_empty(self):
        for f in get_all_foods():
            assert len(f.goals) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Ahar filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestAharFiltering:
    def test_vegan_filter_excludes_non_veg(self):
        from ml.recommendation.ahar_metadata import FoodItem
        non_veg = FoodItem("Chicken", 150, 165, 31, 0, 3.6, 0, "non-vegetarian",
                           ["lunch"], [], ["Strength"], "Lean protein")
        assert not _passes_dietary_filter(non_veg, "vegan")

    def test_vegan_filter_allows_vegan(self):
        vegan_item = FoodItem("Dal", 200, 200, 10, 30, 2, 5, "vegan",
                              ["lunch"], [], ["General Fitness"], "Lentils")
        assert _passes_dietary_filter(vegan_item, "vegan")

    def test_vegetarian_allows_vegetarian(self):
        veg_item = FoodItem("Paneer", 150, 250, 15, 5, 18, 0, "vegetarian",
                            ["lunch"], ["dairy"], ["Strength"], "Paneer")
        assert _passes_dietary_filter(veg_item, "vegetarian")

    def test_vegetarian_excludes_non_veg(self):
        non_veg = FoodItem("Fish", 150, 177, 27, 0, 7, 0, "non-vegetarian",
                           ["lunch"], ["fish"], ["Strength"], "Fish")
        assert not _passes_dietary_filter(non_veg, "vegetarian")

    def test_none_diet_allows_all(self):
        for food in get_all_foods():
            assert _passes_dietary_filter(food, "none")

    def test_allergy_hard_filter_gluten(self):
        gluten_food = FoodItem("Toast", 60, 150, 5, 28, 2, 2, "vegetarian",
                               ["breakfast"], ["gluten"], ["General Fitness"], "Toast")
        assert not _passes_allergy_filter(gluten_food, ["gluten"])

    def test_allergy_hard_filter_dairy(self):
        dairy_food = FoodItem("Milk", 200, 120, 6, 10, 5, 0, "vegetarian",
                              ["breakfast"], ["dairy"], ["General Fitness"], "Milk")
        assert not _passes_allergy_filter(dairy_food, ["dairy"])

    def test_allergy_allows_non_allergen(self):
        for food in get_all_foods():
            if "gluten" not in food.allergens:
                assert _passes_allergy_filter(food, ["gluten"])
                break

    def test_allergy_case_insensitive(self):
        dairy_food = FoodItem("Cheese", 30, 110, 7, 1, 9, 0, "vegetarian",
                              ["snack"], ["dairy"], ["Strength"], "Cheese")
        assert not _passes_allergy_filter(dairy_food, ["DAIRY"])

    def test_no_allergies_passes_all(self):
        for food in get_all_foods():
            assert _passes_allergy_filter(food, [])


# ─────────────────────────────────────────────────────────────────────────────
# Ahar ranking
# ─────────────────────────────────────────────────────────────────────────────

class TestAharRanking:
    def test_returns_list(self):
        recs = recommend_ahar("General Fitness", top_n=5)
        assert isinstance(recs, list)
        assert len(recs) <= 5

    def test_sorted_by_score(self):
        recs = recommend_ahar("General Fitness", top_n=10)
        scores = [r.relevance_score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_score_bounded(self):
        for rec in recommend_ahar("Strength", top_n=10):
            assert 0.0 <= rec.relevance_score <= 1.0

    def test_vegan_filter_applied(self):
        recs = recommend_ahar("General Fitness", dietary_preference="vegan", top_n=20)
        for r in recs:
            assert r.dietary_category == "vegan", f"Non-vegan item in vegan results: {r.food_name}"

    def test_vegetarian_excludes_non_veg(self):
        recs = recommend_ahar("General Fitness", dietary_preference="vegetarian", top_n=20)
        for r in recs:
            assert r.dietary_category in {"vegan", "vegetarian"}

    def test_allergy_hard_filter_dairy(self):
        recs = recommend_ahar("General Fitness", allergies=["dairy"], top_n=30)
        for r in recs:
            food_name = r.food_name
            # Find the food item and check no dairy allergen
            from ml.recommendation.ahar_metadata import get_food
            food = get_food(food_name)
            if food:
                assert "dairy" not in [a.lower() for a in food.allergens], \
                    f"Dairy item in allergy-filtered results: {food_name}"

    def test_allergy_hard_filter_gluten(self):
        recs = recommend_ahar("Strength", allergies=["gluten"], top_n=30)
        for r in recs:
            from ml.recommendation.ahar_metadata import get_food
            food = get_food(r.food_name)
            if food:
                assert "gluten" not in [a.lower() for a in food.allergens]

    def test_meal_type_filter(self):
        recs = recommend_ahar("General Fitness", meal_type="breakfast", top_n=10)
        for r in recs:
            assert "breakfast" in r.meal_suitability

    def test_reason_not_empty(self):
        for rec in recommend_ahar("Strength", top_n=5):
            assert len(rec.reason) > 0

    def test_strength_goal_prefers_protein(self):
        recs = recommend_ahar("Strength", top_n=5)
        # Top 5 for Strength should include high-protein items
        total_protein = sum(r.protein_g for r in recs)
        assert total_protein > 20.0, "Strength goal should recommend protein-rich foods"

    def test_preference_boost(self):
        recs_with    = recommend_ahar("General Fitness", food_preferences=["Oats Porridge"], top_n=20)
        recs_without = recommend_ahar("General Fitness", food_preferences=[], top_n=20)
        # Oats should appear in the top results when preferred
        names_with = [r.food_name for r in recs_with]
        assert "Oats Porridge (plain)" in names_with


# ─────────────────────────────────────────────────────────────────────────────
# API: personalization endpoint
# ─────────────────────────────────────────────────────────────────────────────

from fastapi.testclient import TestClient
from app.main import app
from app.db.session import Base, engine

api_client = TestClient(app)

FULL_PROFILE_PAYLOAD = {
    "profile": {
        "name": "Ananya Sharma",
        "age": 28,
        "gender": "female",
        "height_cm": 162.0,
        "weight_kg": 58.0,
        "activity_level": "moderate",
        "primary_goal": "Flexibility",
        "yoga_experience": "beginner",
        "fitness_experience": "none",
        "flexibility_score": 2.5,
        "strength_score": 3.0,
        "balance_score": 3.0,
        "session_duration_min": 30.0,
        "days_per_week": 4,
        "dietary_preference": "vegetarian",
        "food_preferences": ["dal"],
        "allergies": ["gluten"],
        "sleep_hours": 7.0,
        "previous_performance_score": 5.0,
    },
    "top_n": 5,
}


@pytest.fixture(autouse=True)
def reset_db_personalization():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


class TestPersonalizationAPI:
    def test_endpoint_returns_200(self):
        res = api_client.post("/api/v1/personalization/recommend", json=FULL_PROFILE_PAYLOAD)
        assert res.status_code == 200, res.text

    def test_response_has_data_integrity(self):
        res = api_client.post("/api/v1/personalization/recommend", json=FULL_PROFILE_PAYLOAD)
        data = res.json()
        assert "data_integrity" in data
        assert "yoga_recommendation" in data["data_integrity"]
        assert "ahar_recommendation" in data["data_integrity"]

    def test_response_has_bmi(self):
        res = api_client.post("/api/v1/personalization/recommend", json=FULL_PROFILE_PAYLOAD)
        data = res.json()
        assert "bmi_info" in data
        assert data["bmi_info"]["bmi"] > 0

    def test_response_has_macro_estimate(self):
        res = api_client.post("/api/v1/personalization/recommend", json=FULL_PROFILE_PAYLOAD)
        data = res.json()
        assert "macro_estimate" in data
        assert "ESTIMATE" in data["macro_estimate"]["estimate_label"]

    def test_yoga_recommendations_returned(self):
        res = api_client.post("/api/v1/personalization/recommend", json=FULL_PROFILE_PAYLOAD)
        data = res.json()
        assert "yoga_recommendations" in data
        assert len(data["yoga_recommendations"]) > 0

    def test_ahar_recommendations_returned(self):
        res = api_client.post("/api/v1/personalization/recommend", json=FULL_PROFILE_PAYLOAD)
        data = res.json()
        assert "ahar_recommendations" in data
        assert len(data["ahar_recommendations"]) > 0

    def test_ahar_gluten_allergy_respected(self):
        """Allergy hard filter must hold end-to-end."""
        res = api_client.post("/api/v1/personalization/recommend", json=FULL_PROFILE_PAYLOAD)
        data = res.json()
        from ml.recommendation.ahar_metadata import get_food
        for item in data["ahar_recommendations"]:
            food = get_food(item["food_name"])
            if food:
                assert "gluten" not in [a.lower() for a in food.allergens], \
                    f"Gluten item passed allergy filter: {item['food_name']}"

    def test_invalid_goal_returns_422(self):
        bad = dict(FULL_PROFILE_PAYLOAD)
        bad["profile"] = {**bad["profile"], "primary_goal": "FlyYoga"}
        res = api_client.post("/api/v1/personalization/recommend", json=bad)
        assert res.status_code == 422

    def test_invalid_activity_returns_422(self):
        bad = dict(FULL_PROFILE_PAYLOAD)
        bad["profile"] = {**bad["profile"], "activity_level": "hyperactive"}
        res = api_client.post("/api/v1/personalization/recommend", json=bad)
        assert res.status_code == 422

    def test_goal_prediction_has_data_source(self):
        res = api_client.post("/api/v1/personalization/recommend", json=FULL_PROFILE_PAYLOAD)
        data = res.json()
        if data.get("goal_prediction"):
            assert "data_source" in data["goal_prediction"]
            # Must be labelled SYNTHETIC
            assert data["goal_prediction"]["data_source"] == "SYNTHETIC"

    def test_difficulty_prediction_has_data_source(self):
        res = api_client.post("/api/v1/personalization/recommend", json=FULL_PROFILE_PAYLOAD)
        data = res.json()
        if data.get("difficulty_prediction"):
            assert "data_source" in data["difficulty_prediction"]
            assert data["difficulty_prediction"]["data_source"] == "SYNTHETIC"


# ─────────────────────────────────────────────────────────────────────────────
# API: feedback endpoint
# ─────────────────────────────────────────────────────────────────────────────

LEGACY_PROFILE = {
    "name": "Test User",
    "age": 30,
    "gender": "male",
    "height_cm": 170.0,
    "weight_kg": 70.0,
    "activity_level": "moderate",
    "health_goal": "general_wellness",
    "dietary_preference": "none",
}


class TestFeedbackAPI:
    def _create_user(self):
        res = api_client.post("/api/v1/profile", json=LEGACY_PROFILE)
        assert res.status_code == 201
        return res.json()["id"]

    def test_submit_feedback_returns_201(self):
        user_id = self._create_user()
        fb = {
            "user_id": user_id,
            "category": "yoga",
            "item_name": "Downward Dog",
            "completed": 1,
            "satisfaction": 4.0,
        }
        res = api_client.post("/api/v1/feedback", json=fb)
        assert res.status_code == 201

    def test_feedback_response_has_correct_fields(self):
        user_id = self._create_user()
        fb = {
            "user_id": user_id,
            "category": "ahar",
            "item_name": "Dal + Brown Rice",
            "completed": 1,
            "perceived_difficulty": 2.0,
            "satisfaction": 5.0,
            "notes": "Loved it!",
        }
        res = api_client.post("/api/v1/feedback", json=fb)
        data = res.json()
        assert data["user_id"] == user_id
        assert data["item_name"] == "Dal + Brown Rice"
        assert data["satisfaction"] == 5.0

    def test_get_user_feedback(self):
        user_id = self._create_user()
        for i in range(3):
            api_client.post("/api/v1/feedback", json={
                "user_id": user_id, "category": "yoga",
                "item_name": f"Pose {i}", "completed": 1,
            })
        res = api_client.get(f"/api/v1/feedback/user/{user_id}")
        assert res.status_code == 200
        assert len(res.json()) == 3

    def test_feedback_invalid_category(self):
        user_id = self._create_user()
        fb = {"user_id": user_id, "category": "dance", "item_name": "Waltz"}
        res = api_client.post("/api/v1/feedback", json=fb)
        assert res.status_code == 422

    def test_feedback_nonexistent_user(self):
        fb = {"user_id": 9999, "category": "yoga", "item_name": "Tree Pose"}
        res = api_client.post("/api/v1/feedback", json=fb)
        assert res.status_code == 404

    def test_score_adjusted_on_feedback(self):
        """Feedback with completion + satisfaction should update feedback_adjusted_score."""
        user_id = self._create_user()
        # Get a recommendation ID
        recs_res = api_client.get(f"/api/v1/recommendations/{user_id}")
        all_recs = recs_res.json()
        yoga_recs = all_recs.get("yoga", [])
        if not yoga_recs:
            pytest.skip("No yoga recommendations available")
        rec_id = yoga_recs[0]["id"]
        original_score = yoga_recs[0]["score"]

        fb = {
            "user_id": user_id,
            "category": "yoga",
            "item_name": yoga_recs[0]["item_name"],
            "recommendation_id": rec_id,
            "completed": 1,
            "satisfaction": 5.0,
        }
        api_client.post("/api/v1/feedback", json=fb)

        # Re-fetch recommendations
        updated_res = api_client.get(f"/api/v1/recommendations/{user_id}")
        updated_yoga = updated_res.json().get("yoga", [])
        updated_rec = next((r for r in updated_yoga if r["id"] == rec_id), None)
        assert updated_rec is not None
        adj = updated_rec["feedback_adjusted_score"]
        assert adj is not None
        assert adj >= original_score  # positive feedback should not decrease score

    def test_not_completed_decreases_score(self):
        """Not-completed feedback should decrease feedback_adjusted_score."""
        user_id = self._create_user()
        recs_res = api_client.get(f"/api/v1/recommendations/{user_id}")
        yoga_recs = recs_res.json().get("yoga", [])
        if not yoga_recs:
            pytest.skip("No yoga recommendations available")
        rec_id = yoga_recs[0]["id"]
        original_score = yoga_recs[0]["score"]

        fb = {
            "user_id": user_id,
            "category": "yoga",
            "item_name": yoga_recs[0]["item_name"],
            "recommendation_id": rec_id,
            "completed": 0,
        }
        api_client.post("/api/v1/feedback", json=fb)
        updated_res = api_client.get(f"/api/v1/recommendations/{user_id}")
        updated_yoga = updated_res.json().get("yoga", [])
        updated_rec = next((r for r in updated_yoga if r["id"] == rec_id), None)
        assert updated_rec is not None
        adj = updated_rec["feedback_adjusted_score"]
        if adj is not None and original_score > 0.05:
            assert adj <= original_score  # not-completed should not increase score
