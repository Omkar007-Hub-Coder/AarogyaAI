"""
ml/recommendation/ahar_metadata.py

Structured food / meal metadata for the content-based Ahar recommender.

DATA SOURCE & PROVENANCE
------------------------
This catalog is curated domain knowledge drawn from widely available
nutritional reference sources (ICMR-NIN Indian Food Composition Tables 2017,
USDA FoodData Central, standard dietetics textbooks).

It is NOT a trained ML model and NOT derived from a proprietary dataset.
The nutritional values are approximate serving-level figures; they should
not be used for clinical dietetic planning.

Fields per FoodItem
-------------------
name             : item name (str)
serving_size_g   : reference serving size in grams (float)
calories_kcal    : energy per serving (float)
protein_g        : protein per serving (float)
carbs_g          : carbohydrates per serving (float)
fat_g            : fat per serving (float)
fibre_g          : dietary fibre per serving (float)
dietary_category : vegetarian | vegan | non-vegetarian | egg
meal_suitability : list of breakfast | lunch | dinner | snack | pre-workout | post-workout
allergens        : list of allergen strings (gluten, dairy, nuts, eggs, soy, shellfish, fish)
goals            : list of goals this food primarily supports
benefits         : brief plain-text summary
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class FoodItem:
    name: str
    serving_size_g: float
    calories_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fibre_g: float
    dietary_category: str           # vegetarian | vegan | non-vegetarian | egg
    meal_suitability: list[str]
    allergens: list[str]            # HARD filter by allergy
    goals: list[str]                # from GOAL_CLASSES in engine.py
    benefits: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Food catalog
# ─────────────────────────────────────────────────────────────────────────────

FOOD_CATALOG: list[FoodItem] = [

    # ── Breakfast ─────────────────────────────────────────────────────────────
    FoodItem("Oats Porridge (plain)",     250, 150, 5.0,  27.0, 2.5, 4.0,
             "vegan",       ["breakfast","snack"],
             ["gluten"],
             ["Weight Management","General Fitness","Stress Reduction"],
             "High fibre, slow-release carbs; reduces cholesterol"),

    FoodItem("Poha (flattened rice)",      200, 244, 4.4,  46.0, 4.2, 2.1,
             "vegan",       ["breakfast"],
             [],
             ["General Fitness","Weight Management"],
             "Light on digestion, good pre-workout carb source"),

    FoodItem("Idli (2 pieces + sambar)",   200, 130, 4.2,  25.0, 0.8, 3.1,
             "vegan",       ["breakfast","dinner"],
             [],
             ["Weight Management","General Fitness"],
             "Fermented food; good for gut health; low fat"),

    FoodItem("Egg White Omelette (3 whites)", 120, 51, 10.8, 0.5, 0.2, 0.0,
             "egg",         ["breakfast","post-workout"],
             ["eggs"],
             ["Strength","Weight Management"],
             "High protein, very low fat; excellent for muscle recovery"),

    FoodItem("Banana",                       120, 89,  1.1, 23.0, 0.3, 2.6,
             "vegan",       ["breakfast","snack","pre-workout"],
             [],
             ["Weight Management","General Fitness","Flexibility"],
             "Quick energy, potassium-rich; good pre-workout fuel"),

    FoodItem("Greek Yoghurt (low fat)",      150, 88, 12.5,  6.0, 0.7, 0.0,
             "vegetarian",  ["breakfast","snack","post-workout"],
             ["dairy"],
             ["Strength","Weight Management","General Fitness"],
             "High protein, probiotic; supports muscle recovery"),

    FoodItem("Moong Dal Chilla (2 pieces)",  120, 148,  8.8, 22.0, 2.4, 3.2,
             "vegan",       ["breakfast","snack"],
             [],
             ["Weight Management","Strength","General Fitness"],
             "Protein-rich lentil pancake; high fibre; gluten-free"),

    FoodItem("Whole Wheat Toast + Peanut Butter", 90, 210, 8.5, 26.0, 9.0, 3.5,
             "vegan",       ["breakfast","snack"],
             ["gluten","nuts"],
             ["Strength","General Fitness"],
             "Balanced carb + protein; peanut butter adds healthy fats"),

    # ── Lunch / Dinner ────────────────────────────────────────────────────────
    FoodItem("Dal + Brown Rice (1 bowl each)", 350, 360, 14.0, 67.0, 3.5, 6.5,
             "vegan",       ["lunch","dinner"],
             [],
             ["Weight Management","General Fitness","Strength"],
             "Complete protein combination; high fibre; traditional Indian staple"),

    FoodItem("Paneer Bhurji + 2 Rotis",     300, 390, 20.0, 42.0, 14.0, 4.0,
             "vegetarian",  ["lunch","dinner"],
             ["dairy","gluten"],
             ["Strength","General Fitness"],
             "High protein from paneer; rotis provide complex carbs"),

    FoodItem("Grilled Chicken Breast (150g)", 150, 165, 31.0,  0.0, 3.6, 0.0,
             "non-vegetarian",["lunch","dinner","post-workout"],
             [],
             ["Strength","Weight Management"],
             "Lean protein; very low fat; ideal for muscle building"),

    FoodItem("Grilled Fish (Rohu, 150g)",    150, 177, 27.5,  0.0, 7.0, 0.0,
             "non-vegetarian",["lunch","dinner"],
             ["fish"],
             ["Strength","Weight Management","General Fitness"],
             "Lean protein + omega-3 fatty acids; heart-healthy"),

    FoodItem("Rajma (kidney bean) Curry",    200, 218,  8.7, 36.0, 4.2, 8.4,
             "vegan",       ["lunch","dinner"],
             [],
             ["Weight Management","Strength","General Fitness"],
             "High protein and fibre; slow-digesting carbs"),

    FoodItem("Sambar + 2 Idli",             300, 195,  7.5, 37.0, 2.0, 5.2,
             "vegan",       ["lunch","dinner"],
             [],
             ["Weight Management","General Fitness"],
             "Lentil-vegetable broth; excellent micronutrient profile"),

    FoodItem("Chicken Tikka + Salad",        250, 220, 28.0,  6.0, 9.0, 2.0,
             "non-vegetarian",["lunch","dinner"],
             [],
             ["Strength","Weight Management"],
             "High protein; moderate fat; grilled preparation reduces calories"),

    FoodItem("Tofu Stir-Fry + Quinoa",       300, 280, 18.5, 30.0, 9.0, 5.5,
             "vegan",       ["lunch","dinner"],
             ["soy"],
             ["Strength","Weight Management","General Fitness"],
             "Complete vegan protein; quinoa is a complete amino acid source"),

    FoodItem("Palak Paneer + 1 Roti",        250, 280, 13.5, 22.0, 14.0, 4.0,
             "vegetarian",  ["lunch","dinner"],
             ["dairy","gluten"],
             ["Strength","General Fitness","Flexibility"],
             "Iron-rich (spinach) + calcium (paneer); micronutrient dense"),

    FoodItem("Moong Dal Khichdi",            300, 258,  9.5, 44.0, 4.5, 4.5,
             "vegan",       ["lunch","dinner"],
             [],
             ["Weight Management","Stress Reduction","General Fitness"],
             "Easily digestible; gentle on gut; sattvic food for stress"),

    FoodItem("Curd Rice (1 cup)",            200, 168,  5.0, 30.0, 3.2, 1.0,
             "vegetarian",  ["lunch","dinner"],
             ["dairy"],
             ["Stress Reduction","General Fitness"],
             "Probiotic; cooling; light on digestion; good for recovery days"),

    # ── Snacks ────────────────────────────────────────────────────────────────
    FoodItem("Handful of Mixed Nuts (30g)",   30, 185,  5.0,  6.0, 17.0, 2.5,
             "vegan",       ["snack"],
             ["nuts"],
             ["General Fitness","Strength","Stress Reduction"],
             "Healthy fats, protein, and micronutrients; satiating"),

    FoodItem("Fruit Bowl (Seasonal, 200g)",  200,  90,  1.5, 22.0, 0.5, 3.0,
             "vegan",       ["snack","breakfast"],
             [],
             ["General Fitness","Weight Management","Stress Reduction"],
             "Vitamins, minerals, antioxidants; low calorie, high volume"),

    FoodItem("Sprout Salad (100g)",          100,  62,  4.4, 11.0, 0.4, 3.2,
             "vegan",       ["snack","lunch"],
             [],
             ["Weight Management","General Fitness","Flexibility"],
             "Enzyme-rich living food; very high micronutrient density"),

    FoodItem("Peanut Chaat (30g unsalted)",   30, 167,  7.5,  6.0, 14.0, 2.2,
             "vegan",       ["snack","pre-workout"],
             ["nuts"],
             ["Strength","Weight Management"],
             "Protein and healthy fat; good sustained energy"),

    FoodItem("Cucumber + Hummus (2 tbsp)",   120,  78,  3.5,  8.5, 3.5, 2.0,
             "vegan",       ["snack"],
             ["sesame"],
             ["Weight Management","Stress Reduction"],
             "Low calorie; fibre + protein; anti-inflammatory"),

    # ── Pre / Post Workout ────────────────────────────────────────────────────
    FoodItem("Whey Protein Shake (1 scoop)", 250,  120, 24.0,  4.0, 1.5, 0.0,
             "vegetarian",  ["post-workout","snack"],
             ["dairy"],
             ["Strength","Weight Management"],
             "Fast-absorbing protein; ideal for muscle repair post-workout"),

    FoodItem("Banana + Nut Butter (1 tbsp)", 150,  193, 4.5, 29.0, 8.0, 3.5,
             "vegan",       ["pre-workout","snack"],
             ["nuts"],
             ["General Fitness","Strength","Weight Management"],
             "Quick carbs + fat for sustained energy before exercise"),

    FoodItem("Coconut Water (250ml)",        250,   45, 0.5, 10.5, 0.5, 1.0,
             "vegan",       ["pre-workout","post-workout","snack"],
             [],
             ["General Fitness","Stress Reduction"],
             "Natural electrolyte replenishment; hydrating; low calorie"),

    FoodItem("Boiled Egg (2 whole)",         120,  156, 12.6,  1.2, 10.4, 0.0,
             "egg",         ["breakfast","post-workout","snack"],
             ["eggs"],
             ["Strength","Weight Management"],
             "Complete protein; good fats; portable and cheap"),

    # ── Hydration / Drinks ───────────────────────────────────────────────────
    FoodItem("Green Tea (unsweetened, 200ml)", 200,  3, 0.2,  0.5, 0.0, 0.0,
             "vegan",       ["breakfast","snack"],
             [],
             ["Weight Management","Stress Reduction","General Fitness"],
             "Antioxidants, mild caffeine; supports metabolism and focus"),

    FoodItem("Turmeric Milk (haldi doodh)",  200,  120, 4.8, 12.0, 4.5, 0.0,
             "vegetarian",  ["dinner","snack"],
             ["dairy"],
             ["Stress Reduction","Flexibility","Mobility"],
             "Anti-inflammatory (curcumin); aids sleep and recovery"),
]

# Lookup by name
FOOD_BY_NAME: dict[str, FoodItem] = {f.name: f for f in FOOD_CATALOG}


def get_all_foods() -> list[FoodItem]:
    return FOOD_CATALOG


def get_food(name: str) -> FoodItem | None:
    return FOOD_BY_NAME.get(name)


def foods_to_list() -> list[dict[str, Any]]:
    return [f.to_dict() for f in FOOD_CATALOG]
