"""
backend/app/api/v1/ml_endpoints.py

ML-powered endpoints that use the inference module.
"""
from __future__ import annotations

import sys
import tempfile
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Any, Optional

# Add project root to sys.path so ml/ is importable from the backend
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from ml.inference.predictor import (
    get_yoga_recommendations,
    predict_goal,
    predict_pose_difficulty,
    model_status,
)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class UserMLProfile(BaseModel):
    yoga_experience: int        = Field(1, ge=0, le=3, description="0=none, 1=beginner, 2=intermediate, 3=advanced")
    flexibility_score: float    = Field(3.0, ge=1.0, le=5.0)
    strength_score: float       = Field(3.0, ge=1.0, le=5.0)
    activity_level: int         = Field(1, ge=0, le=3, description="0=sedentary…3=active")
    session_duration_min: float = Field(30.0, gt=0)
    primary_goal: str           = Field("General Fitness")
    bmi: Optional[float]        = Field(None)
    sleep_hours: Optional[float]= Field(None)
    fitness_experience: Optional[int] = Field(None)
    balance_score: Optional[float]    = Field(None)
    fitness_level: Optional[int]      = Field(None)
    previous_performance_score: Optional[float] = Field(None)
    # Style preferences (0 or 1)
    prefers_standing: int    = Field(0, ge=0, le=1)
    prefers_balancing: int   = Field(0, ge=0, le=1)
    prefers_inversion: int   = Field(0, ge=0, le=1)
    prefers_restorative: int = Field(0, ge=0, le=1)


class RecommendationRequest(UserMLProfile):
    top_n: int = Field(10, ge=1, le=50)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/status", tags=["ml"])
def get_model_status():
    """Returns which ML models are currently available."""
    return model_status()


@router.post("/recommend", tags=["ml"])
def recommend(req: RecommendationRequest) -> dict[str, Any]:
    """
    Get top-N yoga pose recommendations using the feature-vector engine.
    Optionally uses trained difficulty predictor if available.
    """
    user = req.model_dump()
    recs = get_yoga_recommendations(user, top_n=req.top_n)
    return {
        "recommendations": recs,
        "count": len(recs),
        "difficulty_filter_applied": (Path(ROOT) / "models" / "difficulty_meta.json").exists(),
    }


@router.post("/predict/goal", tags=["ml"])
def predict_user_goal_endpoint(req: UserMLProfile) -> dict[str, Any]:
    """
    Predict the user's primary wellness goal.
    Requires trained user-goal classifier (python -m ml.training.train_user_goal).
    """
    try:
        result = predict_goal(req.model_dump())
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/predict/difficulty", tags=["ml"])
def predict_difficulty_endpoint(req: UserMLProfile) -> dict[str, Any]:
    """
    Predict the appropriate yoga difficulty level for this user.
    Requires trained difficulty predictor.
    """
    try:
        result = predict_pose_difficulty(req.model_dump())
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/predict/image", tags=["ml"])
async def predict_from_image(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Classify a yoga pose from an uploaded image file.
    Uses the EfficientNetB0 classifier trained on Yoga-82 (REAL ML).
    Returns predicted pose class, confidence, and top-5 predictions.

    Inference runs via .venv-tf subprocess (TF 2.15 / Python 3.9).
    This is a research/educational tool — NOT a medical assessment.
    """
    # Save upload to a temp file first (needed before spawning subprocess)
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        from ml.inference.predictor import classify_yoga_image
        result = classify_yoga_image(tmp_path, top_k=5)
        return result

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )
    except RuntimeError as exc:
        # Inference subprocess error — surface the message (no internal paths)
        msg = str(exc)
        # Strip any absolute filesystem paths from the detail
        import re
        msg = re.sub(r"/[^\s]+AarogyaAI[^\s]*", "<project_path>", msg)
        raise HTTPException(status_code=500, detail=msg[:400])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during pose classification: {type(exc).__name__}",
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
