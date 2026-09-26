"""
ml/recommendation/ahar_engine.py

Content-based Ahar (nutrition) recommendation engine.

METHOD: CONTENT-BASED RANKING — NOT ML
---------------------------------------
No nutrition dataset has been trained on; no ML model exists for nutrition.
Recommendations are produced by a transparent scoring function that combines:
  1. Dietary hard filters  — allergies and dietary_preference exclusions
  2. Goal alignment score  — how well the food's goals match the user's goal
  3. Macro fit score       — how well the food's macro profile matches the
                             user's estimated macro targets
  4. Meal suitability      — whether the food is appropriate for the requested
                             meal type (if specified)
  5. Preference boost      — user-stated food preferences increase score

All scoring is deterministic and transparent. No fabricated accuracy numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import math

from ml.recommendation.ahar_metadata import FoodItem, get_all_foods


# ─────────────────────────────────────────────────────────────────────────────
# Goal → scoring weight
# ─────────────────────────────────────────────────────────────────────────────

# How important is protein / carb / fat for each goal (0–1 weights)
MACRO_GOAL_WEIGHTS: dict[str, dict[str, float]] = {
    "Strength":          {"protein": 0.6, "carbs": 0.3, "fat": 0.1},
    "Weight Management": {"protein": 0.5, "carbs": 0.2, "fat": 0.3},
    "Flexibility":       {"protein": 0.3, "carbs": 0.4, "fat": 0.3},
    "Mobility":          {"protein": 0.3, "carbs": 0.4, "fat": 0.3},
    "Stress Reduction":  {"protein": 0.2, "carbs": 0.5, "fat": 0.3},
    "General Fitness":   {"protein": 0.4, "carbs": 0.4, "fat": 0.2},
}

# How important is calorie density for each goal (higher = more calories = better match)
# Weight Management benefits from lower calorie density; Strength benefits from higher
CALORIE_GOAL_DIRECTION: dict[str, float] = {
    "Strength":          +1.0,  # prefer calorie-dense
    "Weight Management": -1.0,  # prefer lower-calorie
    "Flexibility":        0.0,
    "Mobility":           0.0,
    "Stress Reduction":   0.0,
    "General Fitness":    0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AharRecommendation:
    food_name: str
    relevance_score: float
    dietary_category: str
    serving_size_g: float
    calories_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fibre_g: float
    meal_suitability: list[str]
    goal_alignment: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "food_name":        self.food_name,
            "relevance_score":  round(self.relevance_score, 4),
            "dietary_category": self.dietary_category,
            "serving_size_g":   self.serving_size_g,
            "calories_kcal":    self.calories_kcal,
            "protein_g":        self.protein_g,
            "carbs_g":          self.carbs_g,
            "fat_g":            self.fat_g,
            "fibre_g":          self.fibre_g,
            "meal_suitability": self.meal_suitability,
            "goal_alignment":   self.goal_alignment,
            "reason":           self.reason,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring components
# ─────────────────────────────────────────────────────────────────────────────

def _goal_alignment_score(food: FoodItem, goal: str) -> float:
    """1.0 if food supports this goal, 0.5 if food supports General Fitness, else 0."""
    if goal in food.goals:
        return 1.0
    if "General Fitness" in food.goals:
        return 0.3
    return 0.0


def _macro_fit_score(
    food: FoodItem,
    protein_target_g: float,
    carbs_target_g: float,
    fat_target_g: float,
    goal: str,
) -> float:
    """
    Score how well the food's macro profile contributes towards the daily target.
    Higher = closer to the goal-appropriate macro balance.

    Uses Gaussian similarity centred on a per-serving macro fraction:
      desired_fraction_protein = MACRO_GOAL_WEIGHTS[goal]["protein"]
      etc.

    Returns a value in [0, 1].
    """
    total_macros_g = food.protein_g + food.carbs_g + food.fat_g
    if total_macros_g < 0.5:
        return 0.3  # negligible macros (e.g. tea) — neutral score

    actual_prot_frac = food.protein_g / total_macros_g
    actual_carb_frac = food.carbs_g   / total_macros_g
    actual_fat_frac  = food.fat_g     / total_macros_g

    wts = MACRO_GOAL_WEIGHTS.get(goal, MACRO_GOAL_WEIGHTS["General Fitness"])
    # Compute total daily macro denominators (avoid /0)
    total_target = (protein_target_g or 1) + (carbs_target_g or 1) + (fat_target_g or 1)
    desired_prot = (protein_target_g or 1) / total_target
    desired_carb = (carbs_target_g or 1)   / total_target
    desired_fat  = (fat_target_g or 1)     / total_target

    # Weighted distance between actual and desired fractions
    dist = (
        wts["protein"] * (actual_prot_frac - desired_prot) ** 2
        + wts["carbs"]   * (actual_carb_frac - desired_carb) ** 2
        + wts["fat"]     * (actual_fat_frac  - desired_fat)  ** 2
    )
    return float(math.exp(-dist * 10))  # σ ≈ 0.32


def _calorie_score(food: FoodItem, goal: str) -> float:
    """
    For Weight Management: prefer <250 kcal/serving.
    For Strength: prefer >200 kcal/serving.
    Others: neutral (0.5).
    """
    direction = CALORIE_GOAL_DIRECTION.get(goal, 0.0)
    kcal = food.calories_kcal
    if direction < 0:   # Weight Management → lower is better
        return float(max(0.0, 1.0 - kcal / 500.0))
    elif direction > 0: # Strength → higher is better (up to ~600)
        return float(min(1.0, kcal / 400.0))
    return 0.5


def _preference_boost(food: FoodItem, food_preferences: list[str]) -> float:
    """Small score boost if the food name or category appears in preferences."""
    if not food_preferences:
        return 0.0
    food_lower = food.name.lower()
    for pref in food_preferences:
        if pref.lower() in food_lower or food_lower in pref.lower():
            return 0.15
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Hard filters
# ─────────────────────────────────────────────────────────────────────────────

# Which dietary_category values are permitted for each dietary_preference
_ALLOWED_CATEGORIES: dict[str, set[str]] = {
    "vegan":           {"vegan"},
    "vegetarian":      {"vegan", "vegetarian"},
    "non-vegetarian":  {"vegan", "vegetarian", "egg", "non-vegetarian"},
    "none":            {"vegan", "vegetarian", "egg", "non-vegetarian"},
}


def _passes_dietary_filter(food: FoodItem, dietary_preference: str) -> bool:
    allowed = _ALLOWED_CATEGORIES.get(dietary_preference, _ALLOWED_CATEGORIES["none"])
    return food.dietary_category in allowed


def _passes_allergy_filter(food: FoodItem, allergies: list[str]) -> bool:
    """Hard filter: exclude food if any user allergen matches food allergens."""
    if not allergies:
        return True
    food_allergens_lower = {a.lower() for a in food.allergens}
    return not any(a.lower() in food_allergens_lower for a in allergies)


# ─────────────────────────────────────────────────────────────────────────────
# Public recommendation function
# ─────────────────────────────────────────────────────────────────────────────

def recommend_ahar(
    goal: str,
    dietary_preference: str = "none",
    allergies: Optional[list[str]] = None,
    food_preferences: Optional[list[str]] = None,
    meal_type: Optional[str] = None,    # breakfast | lunch | dinner | snack | pre-workout | post-workout
    protein_target_g: float = 100.0,
    carbs_target_g: float   = 250.0,
    fat_target_g: float     = 70.0,
    top_n: int = 10,
) -> list[AharRecommendation]:
    """
    Recommend top-N food items using content-based scoring.

    Parameters
    ----------
    goal              : primary wellness goal
    dietary_preference: vegetarian | vegan | non-vegetarian | none
    allergies         : list of allergen strings — HARD FILTER (required)
    food_preferences  : list of preferred foods/categories — soft boost
    meal_type         : filter by meal type if provided
    protein_target_g  : estimated daily protein target (grams)
    carbs_target_g    : estimated daily carb target (grams)
    fat_target_g      : estimated daily fat target (grams)
    top_n             : max items to return

    Returns
    -------
    List of AharRecommendation sorted by relevance_score descending.
    """
    allergies        = [str(a).lower().strip() for a in (allergies or [])]
    food_preferences = list(food_preferences or [])

    results: list[AharRecommendation] = []

    for food in get_all_foods():
        # ── Hard filters ──────────────────────────────────────────────────────
        if not _passes_dietary_filter(food, dietary_preference):
            continue
        if not _passes_allergy_filter(food, allergies):
            continue
        if meal_type and meal_type not in food.meal_suitability:
            continue

        # ── Component scores ──────────────────────────────────────────────────
        goal_score    = _goal_alignment_score(food, goal)
        macro_score   = _macro_fit_score(food, protein_target_g, carbs_target_g, fat_target_g, goal)
        calorie_score = _calorie_score(food, goal)
        pref_boost    = _preference_boost(food, food_preferences)

        # Weighted composite (weights sum to ~1 before pref_boost)
        score = (
            0.40 * goal_score
            + 0.35 * macro_score
            + 0.25 * calorie_score
            + pref_boost
        )
        score = min(score, 1.0)

        # ── Build reason string ───────────────────────────────────────────────
        reason_parts = []
        if goal in food.goals:
            reason_parts.append(f"supports your goal ({goal})")
        if food.protein_g >= 8.0:
            reason_parts.append(f"protein-rich ({food.protein_g}g/serving)")
        if food.fibre_g >= 3.0:
            reason_parts.append(f"high fibre ({food.fibre_g}g)")
        if food.calories_kcal < 150 and goal == "Weight Management":
            reason_parts.append("low calorie")
        if pref_boost > 0:
            reason_parts.append("matches your food preferences")
        if not reason_parts:
            reason_parts.append(food.benefits[:60])
        reason = "; ".join(reason_parts).capitalize() + "."

        results.append(AharRecommendation(
            food_name=food.name,
            relevance_score=score,
            dietary_category=food.dietary_category,
            serving_size_g=food.serving_size_g,
            calories_kcal=food.calories_kcal,
            protein_g=food.protein_g,
            carbs_g=food.carbs_g,
            fat_g=food.fat_g,
            fibre_g=food.fibre_g,
            meal_suitability=food.meal_suitability,
            goal_alignment=[g for g in food.goals if g == goal],
            reason=reason,
        ))

    results.sort(key=lambda r: r.relevance_score, reverse=True)
    return results[:top_n]
