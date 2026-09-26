"""
Rule-based recommendation service used until ML models are trained.

Once models are saved to models/, this module will load and use them.
The interface stays the same — callers are decoupled from the model backend.
"""
from __future__ import annotations

import os
import joblib
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models.user import UserProfile
from app.models.recommendation import Recommendation

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


# ──────────────────────────────────────────────────────────────────────────────
# Static knowledge bases (seed data until real datasets are loaded)
# ──────────────────────────────────────────────────────────────────────────────

YOGA_CATALOG: list[dict[str, Any]] = [
    {"name": "Tadasana (Mountain Pose)", "goals": ["general_wellness", "flexibility"], "levels": ["sedentary", "light"]},
    {"name": "Surya Namaskar (Sun Salutation)", "goals": ["weight_loss", "general_wellness"], "levels": ["light", "moderate", "active"]},
    {"name": "Virabhadrasana I (Warrior I)", "goals": ["muscle_gain", "flexibility"], "levels": ["moderate", "active"]},
    {"name": "Balasana (Child's Pose)", "goals": ["general_wellness", "flexibility"], "levels": ["sedentary", "light", "moderate"]},
    {"name": "Trikonasana (Triangle Pose)", "goals": ["flexibility", "weight_loss"], "levels": ["light", "moderate"]},
    {"name": "Bhujangasana (Cobra Pose)", "goals": ["flexibility", "muscle_gain"], "levels": ["light", "moderate"]},
    {"name": "Setu Bandhasana (Bridge Pose)", "goals": ["muscle_gain", "general_wellness"], "levels": ["moderate", "active"]},
    {"name": "Shavasana (Corpse Pose)", "goals": ["general_wellness", "flexibility"], "levels": ["sedentary", "light", "moderate", "active"]},
]

FITNESS_CATALOG: list[dict[str, Any]] = [
    {"name": "Brisk Walking (30 min)", "goals": ["weight_loss", "general_wellness"], "levels": ["sedentary", "light"]},
    {"name": "Bodyweight Squats", "goals": ["muscle_gain", "weight_loss"], "levels": ["light", "moderate"]},
    {"name": "Push-ups", "goals": ["muscle_gain"], "levels": ["moderate", "active"]},
    {"name": "Jumping Jacks", "goals": ["weight_loss", "general_wellness"], "levels": ["light", "moderate"]},
    {"name": "Plank Hold (30s)", "goals": ["muscle_gain", "general_wellness"], "levels": ["moderate", "active"]},
    {"name": "Cycling (20 min)", "goals": ["weight_loss", "general_wellness"], "levels": ["light", "moderate", "active"]},
    {"name": "Dumbbell Rows", "goals": ["muscle_gain"], "levels": ["active"]},
    {"name": "Stretching Routine (15 min)", "goals": ["flexibility", "general_wellness"], "levels": ["sedentary", "light"]},
]

AHAR_CATALOG: list[dict[str, Any]] = [
    {"name": "Dal + Brown Rice", "goals": ["weight_loss", "general_wellness"], "diet": ["vegetarian", "vegan", "none"]},
    {"name": "Grilled Chicken + Salad", "goals": ["muscle_gain", "weight_loss"], "diet": ["non-vegetarian", "none"]},
    {"name": "Oats Porridge", "goals": ["weight_loss", "general_wellness"], "diet": ["vegetarian", "vegan", "none"]},
    {"name": "Paneer Tikka + Roti", "goals": ["muscle_gain"], "diet": ["vegetarian", "none"]},
    {"name": "Fruit Bowl + Nuts", "goals": ["general_wellness", "flexibility"], "diet": ["vegetarian", "vegan", "none"]},
    {"name": "Moong Dal Khichdi", "goals": ["weight_loss", "general_wellness"], "diet": ["vegetarian", "vegan", "none"]},
    {"name": "Egg Whites + Toast", "goals": ["muscle_gain"], "diet": ["non-vegetarian", "none"]},
    {"name": "Green Smoothie (Spinach + Banana)", "goals": ["general_wellness", "flexibility"], "diet": ["vegetarian", "vegan", "none"]},
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bmi(height_cm: float, weight_kg: float) -> float:
    return weight_kg / ((height_cm / 100) ** 2)


def _rule_based_yoga(profile: UserProfile) -> list[dict]:
    return [
        {"name": item["name"], "description": f"Suitable for {profile.health_goal} at {profile.activity_level} activity level", "score": 0.75}
        for item in YOGA_CATALOG
        if profile.health_goal in item["goals"] and profile.activity_level in item["levels"]
    ]


def _rule_based_fitness(profile: UserProfile) -> list[dict]:
    return [
        {"name": item["name"], "description": f"Recommended for {profile.health_goal}", "score": 0.70}
        for item in FITNESS_CATALOG
        if profile.health_goal in item["goals"] and profile.activity_level in item["levels"]
    ]


def _rule_based_ahar(profile: UserProfile) -> list[dict]:
    pref = profile.dietary_preference or "none"
    return [
        {"name": item["name"], "description": f"Good for {profile.health_goal}", "score": 0.72}
        for item in AHAR_CATALOG
        if profile.health_goal in item["goals"] and pref in item["diet"]
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────────

def generate_recommendations(profile: UserProfile, db: Session) -> None:
    """Generate recommendations for a user and persist them."""
    # Delete stale recommendations for this user
    db.query(Recommendation).filter(Recommendation.user_id == profile.id).delete()

    items: list[tuple[str, list[dict]]] = [
        ("yoga", _rule_based_yoga(profile)),
        ("fitness", _rule_based_fitness(profile)),
        ("ahar", _rule_based_ahar(profile)),
    ]

    for category, recs in items:
        for rec in recs:
            db.add(Recommendation(
                user_id=profile.id,
                category=category,
                item_name=rec["name"],
                description=rec["description"],
                score=rec["score"],
            ))
    db.commit()
