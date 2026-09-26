"""
ml/personalization/profile.py

Structured user profile with validation, BMI calculation, and macro estimation.

This module is independent of the database layer — it works with plain dicts
or the UserProfileFull dataclass and can be used from both the ML pipeline
and the backend service layer.

Calorie / macro targets are ESTIMATES based on the Harris-Benedict equation
(revised Mifflin-St Jeor variant). They are clearly labelled as estimates
and must NOT be presented as medical prescriptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VALID_GOALS = {
    "Flexibility",
    "Strength",
    "Mobility",
    "Weight Management",
    "Stress Reduction",
    "General Fitness",
}

VALID_ACTIVITY_LEVELS = {
    "sedentary",    # 0  — little or no exercise
    "light",        # 1  — 1–3 days/week
    "moderate",     # 2  — 3–5 days/week
    "active",       # 3  — 6–7 days/week
}

ACTIVITY_LEVEL_ENC = {"sedentary": 0, "light": 1, "moderate": 2, "active": 3}
ACTIVITY_PAL = {
    "sedentary": 1.2,
    "light":     1.375,
    "moderate":  1.55,
    "active":    1.725,
}

VALID_GENDERS = {"male", "female", "other"}
VALID_DIETARY = {"vegetarian", "vegan", "non-vegetarian", "none"}
VALID_EXPERIENCES = {
    "none":         0,
    "beginner":     1,
    "intermediate": 2,
    "advanced":     3,
}
VALID_DIFFICULTIES = {"Beginner", "Intermediate", "Advanced"}


# ─────────────────────────────────────────────────────────────────────────────
# Validated profile dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserProfileFull:
    """
    Complete, validated user profile.

    Required fields
    ---------------
    name, age, height_cm, weight_kg, activity_level, primary_goal

    Optional fields (default shown)
    --------------------------------
    gender              : "other"
    yoga_experience     : "none"  (none|beginner|intermediate|advanced)
    fitness_experience  : "none"
    flexibility_score   : 3.0  (1–5)
    strength_score      : 3.0  (1–5)
    balance_score       : 3.0  (1–5)
    session_duration_min: 30.0
    days_per_week       : 3
    dietary_preference  : "none"
    food_preferences    : []  (list of food names or categories)
    allergies           : []  (list of allergen strings — HARD FILTER in Ahar)
    sleep_hours         : 7.0
    previous_performance_score: 5.0  (0–10, higher=better)
    """
    # Required
    name: str
    age: int
    height_cm: float
    weight_kg: float
    activity_level: str
    primary_goal: str

    # Optional
    gender: str = "other"
    yoga_experience: str = "none"
    fitness_experience: str = "none"
    flexibility_score: float = 3.0
    strength_score: float = 3.0
    balance_score: float = 3.0
    session_duration_min: float = 30.0
    days_per_week: int = 3
    dietary_preference: str = "none"
    food_preferences: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    sleep_hours: float = 7.0
    previous_performance_score: float = 5.0

    # Computed (set by __post_init__)
    bmi: float = field(init=False)
    bmi_category: str = field(init=False)
    yoga_experience_enc: int = field(init=False)
    fitness_experience_enc: int = field(init=False)
    activity_level_enc: int = field(init=False)

    def __post_init__(self):
        self.bmi = compute_bmi(self.height_cm, self.weight_kg)
        self.bmi_category = bmi_category(self.bmi)
        self.yoga_experience_enc = VALID_EXPERIENCES.get(self.yoga_experience, 0)
        self.fitness_experience_enc = VALID_EXPERIENCES.get(self.fitness_experience, 0)
        self.activity_level_enc = ACTIVITY_LEVEL_ENC.get(self.activity_level, 1)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Ensure lists are proper Python lists (not other iterables)
        d["food_preferences"] = list(d.get("food_preferences", []))
        d["allergies"] = list(d.get("allergies", []))
        return d

    def to_ml_features(self) -> dict[str, Any]:
        """
        Return the subset of fields expected by the ML models
        (user_goal_classifier and difficulty_predictor).
        """
        return {
            "age":                       float(self.age),
            "bmi":                       round(self.bmi, 2),
            "activity_level":            float(self.activity_level_enc),
            "yoga_experience":           float(self.yoga_experience_enc),
            "flexibility_score":         self.flexibility_score,
            "strength_score":            self.strength_score,
            "balance_score":             self.balance_score,
            "sleep_hours":               self.sleep_hours,
            "session_duration_min":      self.session_duration_min,
            "fitness_experience":        float(self.fitness_experience_enc),
            "fitness_level":             float(self.activity_level_enc),
            "previous_performance_score": self.previous_performance_score,
            # Recommendation engine keys
            "primary_goal":              self.primary_goal,
            "prefers_standing":          0,
            "prefers_balancing":         0,
            "prefers_inversion":         0,
            "prefers_restorative":       0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

class ProfileValidationError(ValueError):
    """Raised when a user profile field fails validation."""


def validate_and_build_profile(data: dict) -> UserProfileFull:
    """
    Validate raw input dict, raise ProfileValidationError on any issue,
    and return a validated UserProfileFull.
    """
    errors: list[str] = []

    # ── Required fields ───────────────────────────────────────────────────────
    name = str(data.get("name", "")).strip()
    if not name:
        errors.append("'name' is required and must be non-empty")

    age = data.get("age")
    if age is None:
        errors.append("'age' is required")
    elif not (5 <= int(age) <= 120):
        errors.append(f"'age' must be between 5 and 120 (got {age})")

    height_cm = data.get("height_cm")
    if height_cm is None:
        errors.append("'height_cm' is required")
    elif not (50.0 < float(height_cm) < 300.0):
        errors.append(f"'height_cm' must be between 50 and 300 (got {height_cm})")

    weight_kg = data.get("weight_kg")
    if weight_kg is None:
        errors.append("'weight_kg' is required")
    elif not (10.0 < float(weight_kg) < 500.0):
        errors.append(f"'weight_kg' must be between 10 and 500 (got {weight_kg})")

    activity_level = str(data.get("activity_level", "")).lower()
    if activity_level not in VALID_ACTIVITY_LEVELS:
        errors.append(f"'activity_level' must be one of {sorted(VALID_ACTIVITY_LEVELS)} (got '{activity_level}')")

    primary_goal = str(data.get("primary_goal", ""))
    if primary_goal not in VALID_GOALS:
        errors.append(f"'primary_goal' must be one of {sorted(VALID_GOALS)} (got '{primary_goal}')")

    # ── Optional with validation ───────────────────────────────────────────────
    gender = str(data.get("gender", "other")).lower()
    if gender not in VALID_GENDERS:
        errors.append(f"'gender' must be one of {sorted(VALID_GENDERS)} (got '{gender}')")

    yoga_exp = str(data.get("yoga_experience", "none")).lower()
    if yoga_exp not in VALID_EXPERIENCES:
        errors.append(f"'yoga_experience' must be one of {sorted(VALID_EXPERIENCES)} (got '{yoga_exp}')")

    fitness_exp = str(data.get("fitness_experience", "none")).lower()
    if fitness_exp not in VALID_EXPERIENCES:
        errors.append(f"'fitness_experience' must be one of {sorted(VALID_EXPERIENCES)} (got '{fitness_exp}')")

    dietary = str(data.get("dietary_preference", "none")).lower()
    if dietary not in VALID_DIETARY:
        errors.append(f"'dietary_preference' must be one of {sorted(VALID_DIETARY)} (got '{dietary}')")

    flex = float(data.get("flexibility_score", 3.0))
    if not (1.0 <= flex <= 5.0):
        errors.append(f"'flexibility_score' must be 1–5 (got {flex})")

    strength = float(data.get("strength_score", 3.0))
    if not (1.0 <= strength <= 5.0):
        errors.append(f"'strength_score' must be 1–5 (got {strength})")

    balance = float(data.get("balance_score", 3.0))
    if not (1.0 <= balance <= 5.0):
        errors.append(f"'balance_score' must be 1–5 (got {balance})")

    session_dur = float(data.get("session_duration_min", 30.0))
    if not (5.0 <= session_dur <= 240.0):
        errors.append(f"'session_duration_min' must be 5–240 (got {session_dur})")

    days_pw = int(data.get("days_per_week", 3))
    if not (1 <= days_pw <= 7):
        errors.append(f"'days_per_week' must be 1–7 (got {days_pw})")

    sleep = float(data.get("sleep_hours", 7.0))
    if not (1.0 <= sleep <= 24.0):
        errors.append(f"'sleep_hours' must be 1–24 (got {sleep})")

    perf = float(data.get("previous_performance_score", 5.0))
    if not (0.0 <= perf <= 10.0):
        errors.append(f"'previous_performance_score' must be 0–10 (got {perf})")

    allergies = list(data.get("allergies", []))
    food_prefs = list(data.get("food_preferences", []))

    if errors:
        raise ProfileValidationError("; ".join(errors))

    return UserProfileFull(
        name=name,
        age=int(age),
        height_cm=float(height_cm),
        weight_kg=float(weight_kg),
        activity_level=activity_level,
        primary_goal=primary_goal,
        gender=gender,
        yoga_experience=yoga_exp,
        fitness_experience=fitness_exp,
        flexibility_score=flex,
        strength_score=strength,
        balance_score=balance,
        session_duration_min=session_dur,
        days_per_week=days_pw,
        dietary_preference=dietary,
        food_preferences=food_prefs,
        allergies=[str(a).lower().strip() for a in allergies],
        sleep_hours=sleep,
        previous_performance_score=perf,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BMI helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_bmi(height_cm: float, weight_kg: float) -> float:
    """BMI = weight(kg) / height(m)^2"""
    if height_cm <= 0:
        raise ValueError(f"height_cm must be positive, got {height_cm}")
    return round(weight_kg / ((height_cm / 100.0) ** 2), 2)


def bmi_category(bmi: float) -> str:
    """
    WHO BMI classification (general adult population).
    Not a medical diagnosis — for informational use only.
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal weight"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"


# ─────────────────────────────────────────────────────────────────────────────
# Calorie / macro estimation  (ESTIMATE — not a medical prescription)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tdee(profile: UserProfileFull) -> dict[str, Any]:
    """
    Estimate Total Daily Energy Expenditure (TDEE) using the
    Mifflin-St Jeor equation (1990), widely used in nutrition research.

    FORMULA
    -------
    BMR (male)   = 10 × weight_kg + 6.25 × height_cm − 5 × age + 5
    BMR (female) = 10 × weight_kg + 6.25 × height_cm − 5 × age − 161
    BMR (other)  = average of male and female formulas
    TDEE = BMR × PAL (physical activity level multiplier)

    Macro split — goal-based general guidance (not a clinical diet plan):
      Weight Management : protein 30%, carbs 40%, fat 30%
      Strength          : protein 35%, carbs 45%, fat 20%
      Flexibility       : protein 25%, carbs 50%, fat 25%
      Mobility          : protein 25%, carbs 50%, fat 25%
      Stress Reduction  : protein 20%, carbs 55%, fat 25%
      General Fitness   : protein 25%, carbs 50%, fat 25%

    IMPORTANT: These are population-level estimates. Individual needs vary.
    Do not use as a medical prescription.
    """
    w, h, a = profile.weight_kg, profile.height_cm, float(profile.age)

    bmr_male   = 10 * w + 6.25 * h - 5 * a + 5.0
    bmr_female = 10 * w + 6.25 * h - 5 * a - 161.0

    if profile.gender == "male":
        bmr = bmr_male
    elif profile.gender == "female":
        bmr = bmr_female
    else:
        bmr = (bmr_male + bmr_female) / 2.0

    pal = ACTIVITY_PAL.get(profile.activity_level, 1.375)
    tdee = bmr * pal

    # Goal-based macro percentages
    macro_pct = {
        "Weight Management": {"protein_pct": 0.30, "carbs_pct": 0.40, "fat_pct": 0.30},
        "Strength":          {"protein_pct": 0.35, "carbs_pct": 0.45, "fat_pct": 0.20},
        "Flexibility":       {"protein_pct": 0.25, "carbs_pct": 0.50, "fat_pct": 0.25},
        "Mobility":          {"protein_pct": 0.25, "carbs_pct": 0.50, "fat_pct": 0.25},
        "Stress Reduction":  {"protein_pct": 0.20, "carbs_pct": 0.55, "fat_pct": 0.25},
        "General Fitness":   {"protein_pct": 0.25, "carbs_pct": 0.50, "fat_pct": 0.25},
    }
    pct = macro_pct.get(profile.primary_goal, macro_pct["General Fitness"])

    # Convert calorie percentages to grams
    # Protein/carbs: 4 kcal/g; Fat: 9 kcal/g
    protein_g = round((tdee * pct["protein_pct"]) / 4.0, 1)
    carbs_g   = round((tdee * pct["carbs_pct"])   / 4.0, 1)
    fat_g     = round((tdee * pct["fat_pct"])      / 9.0, 1)

    return {
        "estimate_label": "ESTIMATE — Mifflin-St Jeor equation, not a medical prescription",
        "bmr_kcal":        round(bmr, 0),
        "tdee_kcal":       round(tdee, 0),
        "activity_level":  profile.activity_level,
        "pal_used":        pal,
        "goal":            profile.primary_goal,
        "macro_targets": {
            "protein_g": protein_g,
            "carbs_g":   carbs_g,
            "fat_g":     fat_g,
            "protein_pct": pct["protein_pct"],
            "carbs_pct":   pct["carbs_pct"],
            "fat_pct":     pct["fat_pct"],
        },
    }
