"""
ml/preprocessing/skeleton_utils.py

Feature engineering for BlazePose 33-keypoint skeleton data.

BlazePose landmark indices (MediaPipe convention):
  0  = nose
  11 = left shoulder,  12 = right shoulder
  13 = left elbow,     14 = right elbow
  15 = left wrist,     16 = right wrist
  23 = left hip,       24 = right hip
  25 = left knee,      26 = right knee
  27 = left ankle,     28 = right ankle
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional


# 33 landmarks × (x, y, z) = 99 raw coordinate features
# visibility columns are dropped (model-level artefact, not geometric)
N_LANDMARKS = 33
COORD_DIMS = 3  # x, y, z

# Key joint pairs for angle computation (index_a, vertex, index_b)
ANGLE_TRIPLETS = [
    # Arms
    (11, 13, 15),   # left shoulder-elbow-wrist
    (12, 14, 16),   # right shoulder-elbow-wrist
    (13, 11, 23),   # left elbow-shoulder-hip
    (14, 12, 24),   # right elbow-shoulder-hip
    # Legs
    (23, 25, 27),   # left hip-knee-ankle
    (24, 26, 28),   # right hip-knee-ankle
    (11, 23, 25),   # left shoulder-hip-knee
    (12, 24, 26),   # right shoulder-hip-knee
    # Torso
    (11, 12, 24),   # left shoulder-right shoulder-right hip
    (23, 24, 26),   # left hip-right hip-right knee
]


def _angle_between(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Angle at `vertex` between vectors vertex→a and vertex→b, in degrees."""
    v1 = a - vertex
    v2 = b - vertex
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def extract_skeleton_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Given a (33, 3) or (33, 4) landmark array (x, y, z[, visibility]):
    1. Keep only x, y, z → (33, 3)
    2. Normalize coordinates relative to hip midpoint and torso height
    3. Compute joint angles (10 angles)
    Returns a flat feature vector of length 33*3 + 10 = 109.
    """
    coords = landmarks[:, :3].copy().astype(np.float32)

    # Normalize: center on hip midpoint, scale by torso height
    left_hip  = coords[23]
    right_hip = coords[24]
    left_shoulder  = coords[11]
    right_shoulder = coords[12]

    hip_mid = (left_hip + right_hip) / 2.0
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    torso_height = np.linalg.norm(shoulder_mid - hip_mid) + 1e-8

    coords = (coords - hip_mid) / torso_height

    flat_coords = coords.flatten()  # 99 features

    # Joint angles
    angles = np.array([
        _angle_between(coords[a], coords[v], coords[b])
        for a, v, b in ANGLE_TRIPLETS
    ], dtype=np.float32)  # 10 features

    return np.concatenate([flat_coords, angles])  # 109 features


SKELETON_FEATURE_DIM = N_LANDMARKS * COORD_DIMS + len(ANGLE_TRIPLETS)  # 109


def load_skeleton_npy_dir(
    split_dir: str,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load BlazePose skeleton data from a directory with the structure:
        split_dir/
            ClassName1/
                sample_image_1.npy   # shape (33, 3)
                sample_image_2.npy
                ...
            ClassName2/
                ...

    Each .npy file contains a (33, 3) float64 array (x, y, z) — already
    normalized by the dataset provider.  We re-apply our own
    hip-centered / torso-scaled normalization on top (via
    extract_skeleton_features) to get a consistent 109-dim feature vector.

    Returns
    -------
    X : np.ndarray, shape (N, SKELETON_FEATURE_DIM), dtype float32
    y : np.ndarray, shape (N,), dtype str  — class-folder names as labels
    """
    split_path = Path(split_dir)
    if not split_path.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    class_dirs = sorted(p for p in split_path.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {split_dir}")

    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    skipped = 0

    for class_dir in class_dirs:
        label = class_dir.name
        npy_files = sorted(class_dir.glob("*.npy"))
        for fpath in npy_files:
            try:
                arr = np.load(fpath)
                if arr.shape[0] != N_LANDMARKS or arr.ndim != 2 or arr.shape[1] < COORD_DIMS:
                    if verbose:
                        print(f"[skeleton_utils] Unexpected shape {arr.shape} in {fpath}, skipping")
                    skipped += 1
                    continue
                if not np.isfinite(arr).all():
                    if verbose:
                        print(f"[skeleton_utils] Non-finite values in {fpath}, skipping")
                    skipped += 1
                    continue
                feat = extract_skeleton_features(arr)
                X_list.append(feat)
                y_list.append(label)
            except Exception as e:
                if verbose:
                    print(f"[skeleton_utils] Failed to load {fpath}: {e}")
                skipped += 1

    if skipped > 0:
        print(f"[skeleton_utils] Skipped {skipped} files due to errors/bad shape")

    if not X_list:
        raise ValueError(f"No valid samples loaded from {split_dir}")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=object)
    return X, y


def load_skeleton_csv(csv_path: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Load a BlazePose keypoints CSV.
    Expected columns: landmark_0_x, landmark_0_y, landmark_0_z[, landmark_0_vis], ..., label

    Returns (X: float32 array shape (N, 109), y: str array shape (N,)) or None if loading fails.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[skeleton_utils] Failed to load {csv_path}: {e}")
        return None

    if "label" not in df.columns:
        print(f"[skeleton_utils] 'label' column not found in {csv_path}")
        return None

    labels = df["label"].values

    # Detect coordinate columns
    coord_cols = [c for c in df.columns if c.startswith("landmark_") and
                  (c.endswith("_x") or c.endswith("_y") or c.endswith("_z"))]

    if len(coord_cols) == 0:
        # Try alternative: columns named x_0, y_0, z_0 ...
        coord_cols = [c for c in df.columns if c != "label"]

    raw = df[coord_cols].values.astype(np.float32)

    # Reshape: we need (N, 33, 3) or (N, 33, 4)
    n_samples = len(raw)
    n_per_landmark = len(coord_cols) // N_LANDMARKS
    if n_per_landmark < 3:
        print(f"[skeleton_utils] Not enough columns per landmark ({n_per_landmark}), expected ≥3")
        return None

    landmarks_batch = raw.reshape(n_samples, N_LANDMARKS, n_per_landmark)

    X = np.vstack([
        extract_skeleton_features(landmarks_batch[i]).reshape(1, -1)
        for i in range(n_samples)
    ]).astype(np.float32)

    return X, labels
