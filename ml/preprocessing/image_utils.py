"""
ml/preprocessing/image_utils.py

Image loading, validation, resizing, normalization utilities.
Used by both the preprocessing pipeline and the inference engine.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# EfficientNetB0 / MobileNetV2 standard input
IMAGE_SIZE = (224, 224)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(path: str | Path, size: tuple[int, int] = IMAGE_SIZE) -> Optional[np.ndarray]:
    """
    Load a single image, resize to `size`, return float32 array in [0,1].
    Returns None if the image is unreadable or corrupted.
    """
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr
    except Exception as e:
        logger.warning(f"Could not load image {path}: {e}")
        return None


def normalize_image(arr: np.ndarray) -> np.ndarray:
    """Apply ImageNet mean/std normalization to a float32 [0,1] HWC array."""
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_for_model(path: str | Path, size: tuple[int, int] = IMAGE_SIZE) -> Optional[np.ndarray]:
    """
    Full pipeline: load → resize → normalize.
    Returns shape (H, W, 3) float32, ready for model input (after adding batch dim).
    """
    arr = load_image(path, size)
    if arr is None:
        return None
    return normalize_image(arr)


def image_hash(path: str | Path) -> Optional[str]:
    """MD5 hash of raw image bytes — used for duplicate detection."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def is_valid_image(path: str | Path) -> bool:
    """Quick check: can PIL open this file?"""
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def scan_image_directory(root: str | Path) -> dict:
    """
    Walk a directory tree, collect image paths, detect corrupted files.
    Returns:
        {
          "valid":    [Path, ...],
          "corrupted":[Path, ...],
          "total":    int,
        }
    """
    root = Path(root)
    valid, corrupted = [], []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in extensions:
            if is_valid_image(p):
                valid.append(p)
            else:
                corrupted.append(p)
    return {"valid": valid, "corrupted": corrupted, "total": len(valid) + len(corrupted)}
