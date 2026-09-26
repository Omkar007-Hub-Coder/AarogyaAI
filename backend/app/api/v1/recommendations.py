from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.recommendation import Recommendation
from app.models.user import UserProfile
from app.schemas.schemas import RecommendationsResponse, RecommendationItem

router = APIRouter()


@router.get("/{user_id}", response_model=RecommendationsResponse)
def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")

    recs = db.query(Recommendation).filter(Recommendation.user_id == user_id).all()

    def by_category(cat: str) -> list[RecommendationItem]:
        return [RecommendationItem.model_validate(r) for r in recs if r.category == cat]

    return RecommendationsResponse(
        user_id=user_id,
        yoga=by_category("yoga"),
        fitness=by_category("fitness"),
        ahar=by_category("ahar"),
    )
