from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Any
from datetime import datetime
import math


# ─────────────────────────────────────────────────────────────────────────────
# User Profile — minimal (legacy, kept for backward-compat)
# ─────────────────────────────────────────────────────────────────────────────

class UserProfileCreate(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=5, le=120)
    gender: str = Field(..., pattern="^(male|female|other)$")
    height_cm: float = Field(..., gt=50, lt=300)
    weight_kg: float = Field(..., gt=10, lt=500)
    activity_level: str = Field(..., pattern="^(sedentary|light|moderate|active)$")
    health_goal: str = Field(..., pattern="^(weight_loss|muscle_gain|flexibility|general_wellness)$")
    dietary_preference: Optional[str] = Field("none", pattern="^(vegetarian|vegan|non-vegetarian|none)$")


class UserProfileResponse(UserProfileCreate):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# User Profile — full (personalization)
# ─────────────────────────────────────────────────────────────────────────────

VALID_GOALS = {
    "Flexibility", "Strength", "Mobility",
    "Weight Management", "Stress Reduction", "General Fitness",
}
VALID_ACTIVITY = {"sedentary", "light", "moderate", "active"}
VALID_EXPERIENCE = {"none", "beginner", "intermediate", "advanced"}
VALID_DIETARY = {"vegetarian", "vegan", "non-vegetarian", "none"}
VALID_GENDERS = {"male", "female", "other"}


class FullProfileCreate(BaseModel):
    """
    Full user profile for the personalization + Ahar layer.
    Maps to ml.personalization.profile.UserProfileFull.
    """
    name: str = Field(..., min_length=1, max_length=120)
    age: int = Field(..., ge=5, le=120)
    gender: str = Field("other")
    height_cm: float = Field(..., gt=50.0, lt=300.0)
    weight_kg: float = Field(..., gt=10.0, lt=500.0)
    activity_level: str = Field(..., description="sedentary|light|moderate|active")
    primary_goal: str = Field(..., description="one of the 6 supported goals")

    # Optional enrichment
    yoga_experience: str = Field("none", description="none|beginner|intermediate|advanced")
    fitness_experience: str = Field("none")
    flexibility_score: float = Field(3.0, ge=1.0, le=5.0)
    strength_score: float = Field(3.0, ge=1.0, le=5.0)
    balance_score: float = Field(3.0, ge=1.0, le=5.0)
    session_duration_min: float = Field(30.0, ge=5.0, le=240.0)
    days_per_week: int = Field(3, ge=1, le=7)
    dietary_preference: str = Field("none")
    food_preferences: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list,
                                 description="HARD filter — e.g. ['gluten','dairy']")
    sleep_hours: float = Field(7.0, ge=1.0, le=24.0)
    previous_performance_score: float = Field(5.0, ge=0.0, le=10.0)

    @model_validator(mode="after")
    def validate_enum_fields(self) -> "FullProfileCreate":
        errs = []
        if self.gender not in VALID_GENDERS:
            errs.append(f"gender must be one of {sorted(VALID_GENDERS)}")
        if self.activity_level not in VALID_ACTIVITY:
            errs.append(f"activity_level must be one of {sorted(VALID_ACTIVITY)}")
        if self.primary_goal not in VALID_GOALS:
            errs.append(f"primary_goal must be one of {sorted(VALID_GOALS)}")
        if self.yoga_experience not in VALID_EXPERIENCE:
            errs.append(f"yoga_experience must be one of {sorted(VALID_EXPERIENCE)}")
        if self.fitness_experience not in VALID_EXPERIENCE:
            errs.append(f"fitness_experience must be one of {sorted(VALID_EXPERIENCE)}")
        if self.dietary_preference not in VALID_DIETARY:
            errs.append(f"dietary_preference must be one of {sorted(VALID_DIETARY)}")
        if errs:
            raise ValueError("; ".join(errs))
        return self


class BMIInfo(BaseModel):
    bmi: float
    category: str
    estimate_note: str = "WHO adult BMI classification — not a clinical diagnosis"


class MacroEstimate(BaseModel):
    estimate_label: str
    bmr_kcal: float
    tdee_kcal: float
    activity_level: str
    pal_used: float
    goal: str
    macro_targets: dict


class FullProfileResponse(BaseModel):
    profile: dict
    bmi_info: BMIInfo
    macro_estimate: MacroEstimate
    data_integrity: dict

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    id: int
    category: str
    item_name: str
    description: Optional[str] = None
    score: float
    feedback_adjusted_score: Optional[float] = None

    model_config = {"from_attributes": True}


class RecommendationsResponse(BaseModel):
    user_id: int
    yoga: list[RecommendationItem] = []
    fitness: list[RecommendationItem] = []
    ahar: list[RecommendationItem] = []


# ─────────────────────────────────────────────────────────────────────────────
# Personalized recommendation (ML-powered, full profile)
# ─────────────────────────────────────────────────────────────────────────────

class PersonalizedRecommendationRequest(BaseModel):
    profile: FullProfileCreate
    top_n: int = Field(10, ge=1, le=50)
    meal_type: Optional[str] = Field(
        None,
        description="Optional: breakfast|lunch|dinner|snack|pre-workout|post-workout"
    )


class YogaRecItem(BaseModel):
    pose_name: str
    suitability_score: float
    difficulty: str
    duration_min: float
    target_areas: List[str]
    category: str
    reason: str


class AharRecItem(BaseModel):
    food_name: str
    relevance_score: float
    dietary_category: str
    serving_size_g: float
    calories_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fibre_g: float
    meal_suitability: List[str]
    goal_alignment: List[str]
    reason: str


class PersonalizedRecommendationResponse(BaseModel):
    data_integrity: dict           # labels data provenance clearly
    profile_summary: dict
    bmi_info: BMIInfo
    macro_estimate: MacroEstimate
    goal_prediction: Optional[dict] = None   # SYNTHETIC label if used
    difficulty_prediction: Optional[dict] = None  # SYNTHETIC label if used
    yoga_recommendations: List[YogaRecItem]
    ahar_recommendations: List[AharRecItem]


# ─────────────────────────────────────────────────────────────────────────────
# Feedback
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    user_id: int = Field(..., ge=1)
    category: str = Field(..., description="yoga | fitness | ahar")
    item_name: str = Field(..., min_length=1)
    recommendation_id: Optional[int] = None
    completed: Optional[int] = Field(None, ge=0, le=1)
    perceived_difficulty: Optional[float] = Field(None, ge=1.0, le=5.0)
    comfort: Optional[float] = Field(None, ge=1.0, le=5.0)
    performance_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    satisfaction: Optional[float] = Field(None, ge=1.0, le=5.0)
    notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_category(self) -> "FeedbackCreate":
        if self.category not in {"yoga", "fitness", "ahar"}:
            raise ValueError("category must be one of: yoga, fitness, ahar")
        return self


class FeedbackResponse(BaseModel):
    id: int
    user_id: int
    category: str
    item_name: str
    recommendation_id: Optional[int]
    completed: Optional[int]
    perceived_difficulty: Optional[float]
    comfort: Optional[float]
    performance_score: Optional[float]
    satisfaction: Optional[float]
    notes: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}
