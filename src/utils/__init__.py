"""Shared utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def normalize_confidence(score: float) -> float:
    """Clamp and normalize a raw engine score to [0, 1]."""
    if score < 0:
        return 0.0
    if score > 1.0:
        return min(score / 100.0, 1.0) if score <= 100.0 else 1.0
    return float(score)


def mean_box_confidence(boxes: list) -> float:
    """Mean confidence over box-like objects with a ``confidence`` attribute."""
    if not boxes:
        return 0.0
    return sum(normalize_confidence(b.confidence) for b in boxes) / len(boxes)


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to BGR uint8 (OpenCV convention)."""
    rgb = np.array(pil_image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def quad_to_aabb(points: list | np.ndarray) -> tuple[int, int, int, int]:
    """Convert polygon/quadrilateral points to axis-aligned bounding box."""
    arr = np.asarray(points, dtype=float)
    xs, ys = arr[:, 0], arr[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def save_bgr(path: Path, image_bgr: np.ndarray) -> None:
    """Write a BGR image, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)


def bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Intersection-over-union for two axis-aligned boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / max(area_a + area_b - inter, 1)
