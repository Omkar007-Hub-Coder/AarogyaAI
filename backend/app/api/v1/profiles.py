from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import UserProfile
from app.schemas.schemas import UserProfileCreate, UserProfileResponse
from app.services.recommender import generate_recommendations

router = APIRouter()


@router.post("", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
def create_profile(payload: UserProfileCreate, db: Session = Depends(get_db)):
    profile = UserProfile(**payload.model_dump())
    db.add(profile)
    db.commit()
    db.refresh(profile)
    generate_recommendations(profile, db)
    return profile


@router.get("/{profile_id}", response_model=UserProfileResponse)
def get_profile(profile_id: int, db: Session = Depends(get_db)):
    profile = db.query(UserProfile).filter(UserProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile
