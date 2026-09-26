"""
seed_db.py — Populate the SQLite database with sample user profiles for testing.

Usage:
    cd AarogyaAI/backend
    python ../scripts/seed_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.db.session import SessionLocal, init_db  # noqa: E402
from app.models.user import UserProfile  # noqa: E402
from app.services.recommender import generate_recommendations  # noqa: E402

SAMPLE_PROFILES = [
    dict(name="Arjun", age=28, gender="male", height_cm=175, weight_kg=72,
         activity_level="moderate", health_goal="muscle_gain", dietary_preference="non-vegetarian"),
    dict(name="Priya", age=32, gender="female", height_cm=162, weight_kg=58,
         activity_level="light", health_goal="weight_loss", dietary_preference="vegetarian"),
    dict(name="Rahul", age=45, gender="male", height_cm=170, weight_kg=85,
         activity_level="sedentary", health_goal="general_wellness", dietary_preference="none"),
    dict(name="Sneha", age=25, gender="female", height_cm=158, weight_kg=52,
         activity_level="active", health_goal="flexibility", dietary_preference="vegan"),
]


def seed() -> None:
    init_db()
    db = SessionLocal()
    try:
        for p in SAMPLE_PROFILES:
            profile = UserProfile(**p)
            db.add(profile)
            db.commit()
            db.refresh(profile)
            generate_recommendations(profile, db)
            print(f"  Created profile id={profile.id} name={profile.name}")
        print("Seed complete.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
