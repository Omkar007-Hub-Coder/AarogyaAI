"""
ml/inference/predictor.py

Unified inference API for all AarogyaAI models.

Loads models once on first call and caches them in memory.
Raises clear errors when models are not yet trained.

NOTE — Image classifier special case
─────────────────────────────────────
TensorFlow 2.15 only runs under Python 3.9 (`.venv-tf`).  The FastAPI
backend uses Python 3.14 (`.venv`) where no TF wheel is available.

`classify_yoga_image` therefore spawns a short-lived subprocess under the
`.venv-tf` Python interpreter, calling `scripts/infer_yoga_image.py`.
The subprocess prints one JSON object to stdout which is parsed here.
All other models (skeleton RF, goal, difficulty) run directly in the
backend process as they are pure scikit-learn / joblib.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"

# Path to the TF-capable Python interpreter
_TF_PYTHON = PROJECT_ROOT / ".venv-tf" / "bin" / "python3.9"

# Path to the standalone inference script
_INFER_SCRIPT = PROJECT_ROOT / "scripts" / "infer_yoga_image.py"

# Module-level caches (non-TF models only)
_skeleton_model   = None
_skeleton_le      = None
_goal_model       = None
_goal_le          = None
_goal_meta        = None
_difficulty_model = None
_difficulty_le    = None
_difficulty_meta  = None


def _require_model(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Train the {name} model first:\n"
            f"  python -m ml.training.train_{name.replace(' ', '_').lower()}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Yoga image classifier  (subprocess → .venv-tf)
# ─────────────────────────────────────────────────────────────────────────────

def classify_yoga_image(image_path: str, top_k: int = 5) -> dict[str, Any]:
    """
    Classify a yoga pose from an image file.

    Inference is delegated to a subprocess running under .venv-tf (Python 3.9
    + TensorFlow 2.15) because TF has no Python 3.14-compatible wheel.

    Raises
    ------
    FileNotFoundError  — model checkpoint or inference script not found
    RuntimeError       — subprocess returned non-zero exit or unparseable JSON
    """
    # Check model exists before spawning a process
    model_candidates = [
        MODELS_DIR / "yoga_classifier_best.keras",
        MODELS_DIR / "yoga_classifier.keras",
    ]
    model_path = next((p for p in model_candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            "No trained yoga image classifier found in models/. "
            "Complete EfficientNetB0 training first."
        )

    # Deployment mode: if the local TensorFlow environment is unavailable,
    # run TensorFlow directly in the backend process.
    if not _TF_PYTHON.exists():
        try:
            import tensorflow as tf
            import numpy as np
            from ml.preprocessing.image_utils import preprocess_for_model

            meta_path = MODELS_DIR / "yoga_classifier_meta.json"

            if not meta_path.exists():
                raise FileNotFoundError(
                    f"Yoga classifier metadata not found: {meta_path}"
                )

            with open(meta_path, "r") as f:
                meta = json.load(f)

            class_names = meta["class_names"]

            model = tf.keras.models.load_model(str(model_path))

            arr = preprocess_for_model(image_path)

            if arr is None:
                raise RuntimeError(
                    f"Could not load image: {image_path}"
                )

            batch = arr[np.newaxis, ...]
            probs = model.predict(batch, verbose=0)[0]

            top_indices = np.argsort(probs)[::-1][:top_k]

            result = {
                "predicted_pose": class_names[int(top_indices[0])],
                "confidence": float(probs[top_indices[0]]),
                "top_k": [
                    {
                        "pose": class_names[int(i)],
                        "confidence": float(probs[i]),
                    }
                    for i in top_indices
                ],
            }

            logger.info(
                "Yoga image classified directly with TensorFlow: %s (conf=%.3f)",
                result["predicted_pose"],
                result["confidence"],
            )

            return result

        except Exception as exc:
            logger.exception("Direct TensorFlow image inference failed.")
            raise RuntimeError(
                f"Yoga image inference failed: {exc}"
            ) from exc

    if not _INFER_SCRIPT.exists():
        raise FileNotFoundError(f"Inference script not found: {_INFER_SCRIPT}")

    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "PYTHONUNBUFFERED": "1"}

    try:
        proc = subprocess.run(
            [
                str(_TF_PYTHON),
                str(_INFER_SCRIPT),
                "--image",      str(image_path),
                "--top-k",      str(top_k),
                "--models-dir", str(MODELS_DIR),
            ],
            capture_output=True,
            text=True,
            timeout=120,   # model load + single-image inference ≤ 2 min
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Image inference subprocess timed out after 120 s.")

    # Parse JSON from stdout (ignore stderr which contains TF/Metal noise)
    stdout = proc.stdout.strip()
    if not stdout:
        stderr_snippet = proc.stderr[-500:] if proc.stderr else "(no stderr)"
        raise RuntimeError(
            f"Inference subprocess produced no output (exit {proc.returncode}).\n"
            f"stderr tail: {stderr_snippet}"
        )

    try:
        result: dict[str, Any] = json.loads(stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Inference subprocess returned non-JSON output: {stdout[:200]}"
        )

    if "error" in result:
        raise RuntimeError(f"Inference error: {result['error']}")

    logger.info(
        "Yoga image classified via subprocess: %s (conf=%.3f)",
        result.get("predicted_pose"),
        result.get("confidence", 0.0),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify_skeleton(landmarks_array, top_k: int = 5) -> dict[str, Any]:
    """
    Classify a yoga pose from a (33, 3+) skeleton array.
    landmarks_array: numpy array shape (33, 3) or (33, 4)
    """
    global _skeleton_model, _skeleton_le
    if _skeleton_model is None:
        _require_model(MODELS_DIR / "skeleton_classes.json", "skeleton")
        from ml.models.skeleton_classifier import load_skeleton_classifier
        _skeleton_model, _skeleton_le = load_skeleton_classifier(MODELS_DIR)
        logger.info("Skeleton classifier loaded.")

    from ml.models.skeleton_classifier import predict_pose_from_skeleton
    return predict_pose_from_skeleton(landmarks_array, _skeleton_model, _skeleton_le, top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
# User goal predictor
# ─────────────────────────────────────────────────────────────────────────────

def predict_goal(profile: dict, top_k: int = 3) -> dict[str, Any]:
    """
    Predict wellness goal for a user profile dict.
    See ml.models.user_goal_classifier.FEATURE_COLUMNS for expected keys.
    """
    global _goal_model, _goal_le, _goal_meta
    if _goal_model is None:
        _require_model(MODELS_DIR / "goal_classifier_meta.json", "user_goal")
        from ml.models.user_goal_classifier import load_goal_classifier
        _goal_model, _goal_le, _goal_meta = load_goal_classifier(MODELS_DIR)
        logger.info("Goal classifier loaded.")

    from ml.models.user_goal_classifier import predict_user_goal
    return predict_user_goal(profile, _goal_model, _goal_le, top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty predictor
# ─────────────────────────────────────────────────────────────────────────────

def predict_pose_difficulty(profile: dict) -> dict[str, Any]:
    """
    Predict yoga difficulty level for a user.
    See ml.models.difficulty_predictor.FEATURE_COLUMNS for expected keys.
    """
    global _difficulty_model, _difficulty_le, _difficulty_meta
    if _difficulty_model is None:
        _require_model(MODELS_DIR / "difficulty_meta.json", "difficulty")
        from ml.models.difficulty_predictor import load_difficulty_predictor
        _difficulty_model, _difficulty_le, _difficulty_meta = load_difficulty_predictor(MODELS_DIR)
        logger.info("Difficulty predictor loaded.")

    from ml.models.difficulty_predictor import predict_difficulty
    return predict_difficulty(profile, _difficulty_model, _difficulty_le)


# ─────────────────────────────────────────────────────────────────────────────
# Yoga recommendation
# ─────────────────────────────────────────────────────────────────────────────

def get_yoga_recommendations(user_profile: dict, top_n: int = 10) -> list[dict[str, Any]]:
    """
    Get ranked yoga pose recommendations for a user.

    Optionally uses trained difficulty predictor to set difficulty_filter.
    Falls back gracefully if models are not yet trained.

    Parameters
    ----------
    user_profile : dict
        Must include: primary_goal, yoga_experience, flexibility_score,
        strength_score, activity_level, session_duration_min.
        Optional: prefers_standing, prefers_balancing, prefers_inversion,
                  prefers_restorative.

    Returns
    -------
    List of recommendation dicts.
    """
    difficulty_filter = None

    # Try to predict difficulty from user profile
    if (MODELS_DIR / "difficulty_meta.json").exists():
        try:
            result = predict_pose_difficulty(user_profile)
            difficulty_filter = result.get("predicted_difficulty")
        except Exception as e:
            logger.warning(f"Difficulty prediction unavailable: {e}")

    from ml.recommendation.engine import recommend_poses
    recs = recommend_poses(user_profile, top_n=top_n, difficulty_filter=difficulty_filter)
    return [r.to_dict() for r in recs]


def model_status() -> dict[str, bool]:
    """Return availability status of each model."""
    # Image classifier: accept either the best checkpoint or the final saved model
    _yoga_model_present = (
        (MODELS_DIR / "yoga_classifier_best.keras").exists()
        or (MODELS_DIR / "yoga_classifier.keras").exists()
    )
    return {
        "yoga_image_classifier": _yoga_model_present,
        "skeleton_classifier":   (MODELS_DIR / "skeleton_classes.json").exists(),
        "user_goal_classifier":  (MODELS_DIR / "goal_classifier_meta.json").exists(),
        "difficulty_predictor":  (MODELS_DIR / "difficulty_meta.json").exists(),
    }
