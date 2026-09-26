"""
backend/app/api/v1/personalization.py

Personalized recommendation endpoint — combines full user profile,
yoga feature-vector ranking, Ahar content-based ranking, and
SYNTHETIC goal/difficulty model predictions (clearly labelled).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from app.schemas.schemas import (
    FullProfileCreate,
    PersonalizedRecommendationRequest,
    PersonalizedRecommendationResponse,
    BMIInfo,
    MacroEstimate,
)

router = APIRouter()

DATA_INTEGRITY = {
    "yoga_recommendation":  "CONTENT-BASED RANKING — feature-vector suitability scoring; no real user-outcome data",
    "ahar_recommendation":  "CONTENT-BASED RANKING — curated food metadata; ICMR-NIN / USDA reference values; not a trained ML model",
    "goal_prediction":      "SYNTHETIC PROTOTYPE — trained on procedurally-generated data; not validated on real users",
    "difficulty_prediction":"SYNTHETIC PROTOTYPE — trained on procedurally-generated data; not validated on real users",
    "macro_estimate":       "ESTIMATE — Mifflin-St Jeor equation; population-level formula; not a medical prescription",
    "skeleton_classifier":  "REAL ML — trained on REAL BlazePose Yoga-82 dataset; test Top-1=0.8672",
    "image_classifier":     "REAL ML — EfficientNetB0 on REAL Yoga-82 images; training in progress",
}


@router.post("/recommend", response_model=PersonalizedRecommendationResponse, tags=["personalization"])
def get_personalized_recommendations(req: PersonalizedRecommendationRequest) -> Any:
    """
    Full personalized recommendation combining:
    - Yoga pose ranking (content-based, feature-vector)
    - Ahar/nutrition ranking (content-based, food metadata)
    - SYNTHETIC goal prediction (labelled)
    - SYNTHETIC difficulty prediction (labelled)
    - Calorie/macro ESTIMATE (Mifflin-St Jeor)

    Data provenance is explicitly labelled in the response.
    """
    from ml.personalization.profile import validate_and_build_profile, estimate_tdee
    from ml.recommendation.engine import recommend_poses
    from ml.recommendation.ahar_engine import recommend_ahar

    # ── Build + validate profile ───────────────────────────────────────────
    try:
        profile_data = req.profile.model_dump()
        profile = validate_and_build_profile(profile_data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ── Macro/calorie estimate ─────────────────────────────────────────────
    tdee_info = estimate_tdee(profile)

    # ── SYNTHETIC goal prediction (clearly labelled) ───────────────────────
    goal_prediction = None
    try:
        from ml.inference.predictor import predict_goal
        ml_features = profile.to_ml_features()
        goal_prediction = predict_goal(ml_features)
        # Already carries data_source: "SYNTHETIC"
    except FileNotFoundError:
        goal_prediction = {"data_source": "SYNTHETIC", "note": "model not yet trained"}
    except Exception as e:
        goal_prediction = {"data_source": "SYNTHETIC", "note": f"prediction failed: {str(e)[:80]}"}

    # ── SYNTHETIC difficulty prediction (clearly labelled) ────────────────
    difficulty_prediction = None
    try:
        from ml.inference.predictor import predict_pose_difficulty
        difficulty_prediction = predict_pose_difficulty(profile.to_ml_features())
    except FileNotFoundError:
        difficulty_prediction = {"data_source": "SYNTHETIC", "note": "model not yet trained"}
    except Exception as e:
        difficulty_prediction = {"data_source": "SYNTHETIC", "note": f"prediction failed: {str(e)[:80]}"}

    # ── Yoga recommendations ──────────────────────────────────────────────
    difficulty_filter = None
    if difficulty_prediction and "predicted_difficulty" in difficulty_prediction:
        difficulty_filter = difficulty_prediction["predicted_difficulty"]

    ml_features = profile.to_ml_features()
    yoga_recs_raw = recommend_poses(ml_features, top_n=req.top_n, difficulty_filter=difficulty_filter)
    yoga_recs = [r.to_dict() for r in yoga_recs_raw]

    # ── Ahar recommendations ──────────────────────────────────────────────
    macros = tdee_info["macro_targets"]
    ahar_recs_raw = recommend_ahar(
        goal=profile.primary_goal,
        dietary_preference=profile.dietary_preference,
        allergies=profile.allergies,
        food_preferences=profile.food_preferences,
        meal_type=req.meal_type,
        protein_target_g=macros["protein_g"],
        carbs_target_g=macros["carbs_g"],
        fat_target_g=macros["fat_g"],
        top_n=req.top_n,
    )
    ahar_recs = [r.to_dict() for r in ahar_recs_raw]

    # ── Response assembly ─────────────────────────────────────────────────
    bmi_info = BMIInfo(
        bmi=profile.bmi,
        category=profile.bmi_category,
    )
    macro_est = MacroEstimate(
        estimate_label=tdee_info["estimate_label"],
        bmr_kcal=tdee_info["bmr_kcal"],
        tdee_kcal=tdee_info["tdee_kcal"],
        activity_level=tdee_info["activity_level"],
        pal_used=tdee_info["pal_used"],
        goal=tdee_info["goal"],
        macro_targets=tdee_info["macro_targets"],
    )

    profile_summary = {
        "name": profile.name,
        "age": profile.age,
        "gender": profile.gender,
        "bmi": profile.bmi,
        "bmi_category": profile.bmi_category,
        "primary_goal": profile.primary_goal,
        "activity_level": profile.activity_level,
        "yoga_experience": profile.yoga_experience,
        "dietary_preference": profile.dietary_preference,
        "allergies": profile.allergies,
        "session_duration_min": profile.session_duration_min,
    }

    return PersonalizedRecommendationResponse(
        data_integrity=DATA_INTEGRITY,
        profile_summary=profile_summary,
        bmi_info=bmi_info,
        macro_estimate=macro_est,
        goal_prediction=goal_prediction,
        difficulty_prediction=difficulty_prediction,
        yoga_recommendations=yoga_recs,
        ahar_recommendations=ahar_recs,
    )
