"""Shared page-level image metrics (edge density, saturation)."""

from __future__ import annotations

import cv2
import numpy as np


def page_edge_density(image_bgr: np.ndarray) -> float:
    """Fraction of Canny edge pixels on a full BGR page image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges) / max(edges.size, 1))


def page_mean_saturation(image_bgr: np.ndarray) -> float:
    """Mean HSV saturation channel on a full BGR page image."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())
