"""
ml/recommendation/engine.py

Feature-vector yoga recommendation engine.

Each yoga pose and user profile are represented as numeric feature vectors.
A weighted suitability score is computed using:

  score = Σ w_i · compatibility_i(user_feature_i, pose_requirement_i)

No if/else goal-routing — the entire computation is numeric.

The engine works without trained ML models (no dataset required).
When user-goal and difficulty models are trained, their predictions are
used to weight the final ranking.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any

from ml.recommendation.yoga_metadata import YogaPose, get_all_poses


# ─────────────────────────────────────────────────────────────────────────────
# Goal → numeric index mapping
# ─────────────────────────────────────────────────────────────────────────────

GOAL_INDEX = {
    "Flexibility":        0,
    "Strength":           1,
    "Mobility":           2,
    "Weight Management":  3,
    "Stress Reduction":   4,
    "General Fitness":    5,
}
N_GOALS = len(GOAL_INDEX)


# ─────────────────────────────────────────────────────────────────────────────
# Pose feature vector
# ─────────────────────────────────────────────────────────────────────────────

def pose_to_vector(pose: YogaPose) -> np.ndarray:
    """
    Convert a YogaPose to a numeric feature vector (length = 9 + N_GOALS = 15).

    [0] difficulty_num  (1–3)
    [1] flexibility_req (1–5)
    [2] strength_req    (1–5)
    [3] experience_req  (0–3)
    [4] duration_min
    [5] is_standing     (0/1)
    [6] is_balancing    (0/1)
    [7] is_inversion    (0/1)
    [8] is_restorative  (0/1 — supine/prone/restorative categories)
    [9..14] goal_flags  (one-hot over GOAL_INDEX)
    """
    goal_vec = np.zeros(N_GOALS, dtype=np.float32)
    for g in pose.goals:
        if g in GOAL_INDEX:
            goal_vec[GOAL_INDEX[g]] = 1.0

    cat = pose.category.lower()
    return np.array([
        float(pose.difficulty_num),
        float(pose.flexibility_req),
        float(pose.strength_req),
        float(pose.experience_req),
        float(pose.duration_min),
        float(cat == "standing"),
        float(cat == "balancing"),
        float(cat == "inversion"),
        float(cat in ("supine", "prone", "restorative")),
        *goal_vec,
    ], dtype=np.float32)


POSE_VECTOR_DIM = 9 + N_GOALS  # 15


# ─────────────────────────────────────────────────────────────────────────────
# User feature vector
# ─────────────────────────────────────────────────────────────────────────────

def user_to_vector(user: dict) -> np.ndarray:
    """
    Convert user profile dict to a numeric compatibility vector (same dim as pose vector).

    Expected keys (all optional, defaults to mid-range):
      yoga_experience (0–3), flexibility_score (1–5), strength_score (1–5),
      activity_level (0–3), session_duration_min (minutes),
      prefers_standing (0/1), prefers_balancing (0/1),
      prefers_inversion (0/1), prefers_restorative (0/1),
      primary_goal (str from GOAL_INDEX)

    The user vector represents what the user *can handle / prefers*, so the
    suitability score is high when pose requirements ≤ user capability.
    """
    goal_vec = np.zeros(N_GOALS, dtype=np.float32)
    goal = user.get("primary_goal", "General Fitness")
    if goal in GOAL_INDEX:
        goal_vec[GOAL_INDEX[goal]] = 1.0

    # Map activity_level (0-3) and yoga_experience (0-3) to difficulty headroom
    yoga_exp   = float(user.get("yoga_experience", 1))
    activity   = float(user.get("activity_level", 1))
    difficulty_headroom = 1.0 + yoga_exp * 0.67   # range [1.0, 3.0]

    return np.array([
        difficulty_headroom,
        float(user.get("flexibility_score", 3)),
        float(user.get("strength_score", 3)),
        yoga_exp,
        float(user.get("session_duration_min", 30)),
        float(user.get("prefers_standing", 0)),
        float(user.get("prefers_balancing", 0)),
        float(user.get("prefers_inversion", 0)),
        float(user.get("prefers_restorative", 0)),
        *goal_vec,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Suitability scoring
# ─────────────────────────────────────────────────────────────────────────────

# Dimension weights — tunable hyperparameters
WEIGHTS = np.array([
    2.0,   # difficulty match
    1.5,   # flexibility capability ≥ requirement
    1.5,   # strength capability ≥ requirement
    1.5,   # experience ≥ requirement
    0.5,   # duration preference
    0.3,   # style: standing
    0.3,   # style: balancing
    0.3,   # style: inversion
    0.3,   # style: restorative
    *([2.5] * N_GOALS),  # goal alignment (highest weight)
], dtype=np.float32)


def _dim_score(user_val: float, pose_val: float, dim_type: str) -> float:
    """
    Compute per-dimension compatibility score in [0, 1].

    dim_type:
      'capability' — user value should be ≥ pose requirement
      'goal_flag'  — both 1 gives full score; partial credit for overlap
      'duration'   — Gaussian similarity
      'style'      — both 1 gives full score
    """
    if dim_type == "capability":
        if pose_val == 0:
            return 1.0
        return float(np.clip(user_val / pose_val, 0.0, 1.0))
    elif dim_type == "goal_flag":
        return float(min(user_val * pose_val, 1.0))
    elif dim_type == "duration":
        diff = abs(user_val - pose_val)
        return float(np.exp(-(diff ** 2) / (2 * 15 ** 2)))  # σ=15 min
    elif dim_type == "style":
        return float(min(user_val * pose_val, 1.0)) if pose_val > 0 else 0.5
    return 0.5


DIM_TYPES = (
    ["capability"] * 4  # difficulty, flexibility, strength, experience
    + ["duration"]       # duration
    + ["style"] * 4      # standing, balancing, inversion, restorative
    + ["goal_flag"] * N_GOALS
)


def compute_suitability(user_vec: np.ndarray, pose_vec: np.ndarray) -> float:
    """Weighted suitability score in [0, 1]."""
    scores = np.array([
        _dim_score(float(user_vec[i]), float(pose_vec[i]), DIM_TYPES[i])
        for i in range(len(DIM_TYPES))
    ], dtype=np.float32)
    weighted = float(np.dot(WEIGHTS, scores) / np.sum(WEIGHTS))
    return weighted


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecommendationResult:
    pose_name: str
    suitability_score: float
    difficulty: str
    duration_min: float
    target_areas: list[str]
    category: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pose_name":        self.pose_name,
            "suitability_score": round(self.suitability_score, 4),
            "difficulty":       self.difficulty,
            "duration_min":     self.duration_min,
            "target_areas":     self.target_areas,
            "category":         self.category,
            "reason":           self.reason,
        }


def _build_reason(user: dict, pose: YogaPose, score: float) -> str:
    parts = []
    goal = user.get("primary_goal", "General Fitness")
    if goal in pose.goals:
        parts.append(f"aligns with your goal ({goal})")
    flex = user.get("flexibility_score", 3)
    if flex >= pose.flexibility_req:
        parts.append("suits your flexibility level")
    strength = user.get("strength_score", 3)
    if strength >= pose.strength_req:
        parts.append("matches your strength level")
    if not parts:
        parts.append("broadly suitable for your profile")
    return "; ".join(parts).capitalize() + "."


def recommend_poses(
    user: dict,
    top_n: int = 10,
    difficulty_filter: str | None = None,
) -> list[RecommendationResult]:
    """
    Recommend top-N yoga poses for a user profile.

    Parameters
    ----------
    user : dict
        User profile (see user_to_vector for expected keys).
    top_n : int
        Number of poses to return.
    difficulty_filter : str | None
        Optionally restrict to 'Beginner' | 'Intermediate' | 'Advanced'.

    Returns
    -------
    List of RecommendationResult, sorted by suitability_score descending.
    """
    user_vec = user_to_vector(user)
    all_poses = get_all_poses()

    if difficulty_filter:
        all_poses = [p for p in all_poses if p.difficulty == difficulty_filter]

    results = []
    for pose in all_poses:
        pose_vec = pose_to_vector(pose)
        score = compute_suitability(user_vec, pose_vec)
        results.append(RecommendationResult(
            pose_name=pose.name,
            suitability_score=score,
            difficulty=pose.difficulty,
            duration_min=pose.duration_min,
            target_areas=pose.target_areas,
            category=pose.category,
            reason=_build_reason(user, pose, score),
        ))

    results.sort(key=lambda r: r.suitability_score, reverse=True)
    return results[:top_n]
