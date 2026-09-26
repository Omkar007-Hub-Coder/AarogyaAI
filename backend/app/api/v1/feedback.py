"""
backend/app/api/v1/feedback.py

User feedback recording and retrieval.

When a user submits feedback on a recommendation:
1. The feedback is stored in user_feedback.
2. The recommendation's feedback_adjusted_score is updated using a
   simple weighted adjustment based on satisfaction and completion.

Adjustment formula (transparent, not RL):
  adjustment = (satisfaction / 5.0) * 0.1  — if completed
             = -0.05                        — if explicitly not completed
             = 0                            — if not reported

This is a simple heuristic re-ranking, NOT reinforcement learning.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.db.session import get_db
from app.models.feedback import UserFeedback
from app.models.recommendation import Recommendation
from app.models.user import UserProfile
from app.schemas.schemas import FeedbackCreate, FeedbackResponse

router = APIRouter()

# Maximum cumulative adjustment per recommendation (prevent unbounded drift)
MAX_ADJUSTMENT = 0.30


def _compute_adjustment(fb: FeedbackCreate) -> float:
    """
    Heuristic score adjustment based on feedback.
    Returns a signed float in [-0.05, +0.10].

    This is NOT reinforcement learning — it is a simple transparent heuristic.
    """
    if fb.completed == 0:
        return -0.05
    if fb.completed == 1 and fb.satisfaction is not None:
        # Scale satisfaction (1–5) to [0, +0.10]
        return (fb.satisfaction / 5.0) * 0.10
    return 0.0


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_feedback(payload: FeedbackCreate, db: Session = Depends(get_db)):
    """
    Record user feedback on a recommendation item.
    Applies a transparent heuristic score adjustment to the linked recommendation.
    """
    # Verify user exists
    user = db.query(UserProfile).filter(UserProfile.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    fb = UserFeedback(**payload.model_dump())
    db.add(fb)
    db.flush()  # get fb.id without committing

    # Update recommendation score if linked
    if payload.recommendation_id is not None:
        rec = db.query(Recommendation).filter(
            Recommendation.id == payload.recommendation_id,
            Recommendation.user_id == payload.user_id,
        ).first()
        if rec:
            current = rec.feedback_adjusted_score
            if current is None:
                current = rec.score
            adj = _compute_adjustment(payload)
            new_score = max(0.0, min(1.0, current + adj))
            rec.feedback_adjusted_score = round(new_score, 4)

    db.commit()
    db.refresh(fb)
    return fb


@router.get("/user/{user_id}", response_model=List[FeedbackResponse])
def get_user_feedback(user_id: int, db: Session = Depends(get_db)):
    """Retrieve all feedback submitted by a user."""
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return db.query(UserFeedback).filter(UserFeedback.user_id == user_id).all()
