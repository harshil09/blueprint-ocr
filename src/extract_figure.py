"""
Engineering-figure extraction (OpenCV morphology pipeline).

Single module for the full flow: OCR text masking, binarization, connected
components, candidate merge/validation, bbox refinement, and final crop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from src import config
from src.logger import get_logger
from src.ocr.ocr_engine import TextBox
from src.profile_config import CropProfile, ProfileConfig, resolve_profile_config
from src.utils import bbox_iou, save_bgr

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentCandidate:
    box: tuple[int, int, int, int]  # x, y, width, height
    component_score: float


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    gate_score: float
    edge_density: float
    drawing_coverage: float
    text_overlap: float
    border_touch_count: int
    variance: float
    area_ratio: float
    aspect_ratio: float


@dataclass(frozen=True)
class FigureSelectionResult:
    box: tuple[int, int, int, int]
    component_score: float
    gate_score: float
    quality: QualityResult


# ---------------------------------------------------------------------------
# Box geometry
# ---------------------------------------------------------------------------


def _box_xywh_to_xyxy(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return x, y, x + w, y + h


def _box_xyxy_to_xywh(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def _axis_gap(a_min: int, a_max: int, b_min: int, b_max: int) -> int:
    if a_max < b_min:
        return b_min - a_max
    if b_max < a_min:
        return a_min - b_max
    return 0


def _pair_gaps(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int]:
    ax1, ay1, ax2, ay2 = _box_xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _box_xywh_to_xyxy(b)
    return (
        _axis_gap(ax1, ax2, bx1, bx2),
        _axis_gap(ay1, ay2, by1, by2),
    )


def _union_xywh(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x1 = min(_box_xywh_to_xyxy(b)[0] for b in boxes)
    y1 = min(_box_xywh_to_xyxy(b)[1] for b in boxes)
    x2 = max(_box_xywh_to_xyxy(b)[2] for b in boxes)
    y2 = max(_box_xywh_to_xyxy(b)[3] for b in boxes)
    return _box_xyxy_to_xywh((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# OCR text masking (shared by morphology + layout extraction)
# ---------------------------------------------------------------------------


def _clip_padded_bbox(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    pad: int,
    clip_w: int,
    clip_h: int,
) -> tuple[int, int, int, int]:
    """Clip axis-aligned box (x0,y0,x1,y1) with padding to clip_w × clip_h."""
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(clip_w, x1 + pad),
        min(clip_h, y1 + pad),
    )


def build_layout_annotation_mask(
    shape: tuple[int, int],
    profile_config: ProfileConfig | None = None,
) -> np.ndarray:
    """Mask typical drawing-sheet annotation zones (notes + title block)."""
    pcfg = profile_config or resolve_profile_config(None)
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    notes_h = int(h * pcfg.notes_block_height_ratio)
    notes_w = int(w * pcfg.notes_block_width_ratio)
    mask[:notes_h, :notes_w] = 255

    band_h = int(h * pcfg.bottom_annotation_band_ratio)
    mask[h - band_h :, :] = 255

    title_h = int(h * config.TITLE_BLOCK_HEIGHT_RATIO)
    title_w = int(w * pcfg.title_block_width_ratio)
    mask[h - title_h :, w - title_w :] = 255
    return mask


def build_text_mask(
    shape: tuple[int, int],
    boxes: list[TextBox],
    padding: int | None = None,
    *,
    include_layout_zones: bool = True,
    profile_config: ProfileConfig | None = None,
) -> np.ndarray:
    """Uint8 mask with OCR text regions and layout annotation zones set to 255."""
    pad = padding if padding is not None else config.TEXT_MASK_PADDING_PX
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x0, y0, x1, y1 = box.bbox
        px0, py0, px1, py1 = _clip_padded_bbox(x0, y0, x1, y1, pad, w, h)
        mask[py0:py1, px0:px1] = 255
    if include_layout_zones:
        mask = cv2.bitwise_or(
            mask, build_layout_annotation_mask(shape, profile_config=profile_config)
        )
    return mask


def apply_text_mask_to_bgr(
    image_bgr: np.ndarray,
    boxes: list[TextBox],
    padding: int | None = None,
    fill_value: int = 255,
) -> np.ndarray:
    """Paint OCR text regions to white (paper) on a copy of the page."""
    masked = image_bgr.copy()
    mask = build_text_mask(image_bgr.shape[:2], boxes, padding=padding)
    masked[mask > 0] = fill_value
    return masked


def estimate_text_coverage_in_bbox(
    bbox: tuple[int, int, int, int],
    boxes: list[TextBox],
    padding: int | None = None,
) -> float:
    """Fraction of bbox area covered by OCR boxes (0–1)."""
    pad = padding if padding is not None else config.TEXT_MASK_PADDING_PX
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if w <= 1 or h <= 1:
        return 1.0

    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        bx0, by0, bx1, by1 = box.bbox
        ix0, iy0, ix1, iy1 = _clip_padded_bbox(bx0, by0, bx1, by1, pad, x1, y1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        mask[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = 255

    return float(mask.mean() / 255.0)


def masked_non_text_edges(
    image_bgr: np.ndarray,
    boxes: list[TextBox],
) -> np.ndarray:
    """Canny edges on structural (non-text) regions — layout figure detection."""
    h, w = image_bgr.shape[:2]
    non_text = cv2.bitwise_not(build_text_mask((h, w), boxes))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_and(edges, non_text)
    return cv2.dilate(combined, np.ones((5, 5), np.uint8), iterations=2)


# ---------------------------------------------------------------------------
# Binary preprocessing & component discovery
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


def merge_candidate_regions(
    candidates: list[ComponentCandidate],
    page_shape: tuple[int, int],
) -> list[ComponentCandidate]:
    if len(candidates) <= 1:
        return list(candidates)

    page_h, page_w = page_shape
    max_h_gap = int(page_w * config.MERGE_HORIZONTAL_GAP_RATIO)
    max_v_gap = int(page_h * config.MERGE_VERTICAL_GAP_RATIO)
    n = len(candidates)
    uf = _UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            h_gap, v_gap = _pair_gaps(candidates[i].box, candidates[j].box)
            if h_gap < max_h_gap and v_gap < max_v_gap:
                uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(uf.find(idx), []).append(idx)

    merged: list[ComponentCandidate] = []
    for indices in groups.values():
        boxes = [candidates[i].box for i in indices]
        scores = [candidates[i].component_score for i in indices]
        merged.append(
            ComponentCandidate(box=_union_xywh(boxes), component_score=max(scores))
        )
    merged.sort(key=lambda c: c.component_score, reverse=True)
    return merged


def remove_page_border(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    margin_x = int(w * config.BORDER_MARGIN_RATIO)
    margin_y = int(h * config.BORDER_MARGIN_RATIO)
    binary = binary.copy()
    binary[:margin_y, :] = 0
    binary[h - margin_y :, :] = 0
    binary[:, :margin_x] = 0
    binary[:, w - margin_x :] = 0
    return binary


def remove_title_block(binary: np.ndarray) -> np.ndarray:
    """Clear the full-width bottom annotation band (title block, LOFT, notes)."""
    h, _w = binary.shape
    band_h = int(h * config.BOTTOM_ANNOTATION_BAND_RATIO)
    binary = binary.copy()
    binary[h - band_h :, :] = 0
    return binary


def remove_notes_block(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    notes_h = int(h * config.NOTES_BLOCK_HEIGHT_RATIO)
    notes_w = int(w * config.NOTES_BLOCK_WIDTH_RATIO)
    binary = binary.copy()
    binary[:notes_h, :notes_w] = 0
    return binary


def _morphology_analysis_scale(page_h: int, page_w: int) -> float:
    max_side = max(page_h, page_w)
    limit = config.MORPHOLOGY_ANALYSIS_MAX_SIDE
    if max_side <= limit:
        return 1.0
    return limit / max_side


def _scale_text_boxes(
    boxes: list[TextBox],
    scale: float,
) -> list[TextBox]:
    if scale == 1.0 or not boxes:
        return boxes
    scaled: list[TextBox] = []
    for box in boxes:
        x0, y0, x1, y1 = box.bbox
        scaled.append(
            TextBox(
                text=box.text,
                confidence=box.confidence,
                bbox=(
                    int(x0 * scale),
                    int(y0 * scale),
                    int(x1 * scale),
                    int(y1 * scale),
                ),
            )
        )
    return scaled


def _scale_box_xywh(
    box: tuple[int, int, int, int],
    inv_scale: float,
) -> tuple[int, int, int, int]:
    if inv_scale == 1.0:
        return box
    x, y, w, h = box
    return (
        int(round(x * inv_scale)),
        int(round(y * inv_scale)),
        int(round(w * inv_scale)),
        int(round(h * inv_scale)),
    )


def _box_area_ratio(box: tuple[int, int, int, int], page_shape: tuple[int, int]) -> float:
    ph, pw = page_shape
    _, _, w, h = box
    return (w * h) / max(ph * pw, 1)


def apply_morphology_close(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_KERNEL)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )


def score_component(
    stats_row: np.ndarray,
    centroid: np.ndarray,
    page_shape: tuple[int, int],
    binary_roi: np.ndarray,
) -> float:
    x, y, w, h, area = stats_row
    page_h, page_w = page_shape
    page_area = page_h * page_w
    component_area_ratio = area / page_area

    if component_area_ratio < config.MIN_COMPONENT_AREA_RATIO:
        return -1.0
    if component_area_ratio > config.MAX_COMPONENT_AREA_RATIO:
        return -1.0

    cx, cy = centroid
    center_x, center_y = page_w / 2, page_h / 2
    dist = np.hypot(cx - center_x, cy - center_y)
    max_dist = np.hypot(center_x, center_y)
    centrality_score = 1.0 - (dist / max_dist) if max_dist > 0 else 0.0

    drawing_bottom = page_h * config.DRAWING_ZONE_MAX_BOTTOM_RATIO
    if cy > drawing_bottom:
        return -1.0
    ideal_cy = page_h * 0.38
    vertical_score = 1.0 - min(1.0, abs(cy - ideal_cy) / max(ideal_cy, 1.0))

    density = float(np.count_nonzero(binary_roi) / max(w * h, 1))
    aspect_ratio = w / max(h, 1)
    aspect_score = 1.0 if 0.3 <= aspect_ratio <= 3.5 else 0.3

    return (
        config.AREA_WEIGHT * component_area_ratio
        + config.CENTRALITY_WEIGHT * centrality_score
        + config.DENSITY_WEIGHT * density
        + config.ASPECT_WEIGHT * aspect_score
        + config.VERTICAL_POSITION_WEIGHT * vertical_score
    )


def find_top_component_candidates(
    binary: np.ndarray,
    top_k: int | None = None,
) -> list[ComponentCandidate]:
    k = top_k if top_k is not None else config.TOP_K_CANDIDATES
    h, w = binary.shape
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    candidates: list[ComponentCandidate] = []
    for i in range(1, num_labels):
        x, y, bw, bh, _area = stats[i]
        roi = binary[y : y + bh, x : x + bw]
        score = score_component(stats[i], centroids[i], (h, w), roi)
        if score > 0:
            candidates.append(
                ComponentCandidate(
                    box=(int(x), int(y), int(bw), int(bh)),
                    component_score=float(score),
                )
            )

    candidates.sort(key=lambda c: c.component_score, reverse=True)
    return candidates[:k]


def suppress_duplicate_candidates(
    candidates: list[ComponentCandidate],
    iou_threshold: float | None = None,
) -> list[ComponentCandidate]:
    thresh = (
        iou_threshold if iou_threshold is not None else config.IOU_DUPLICATE_THRESHOLD
    )
    kept: list[ComponentCandidate] = []
    for candidate in candidates:
        aabb = _box_xywh_to_xyxy(candidate.box)
        if any(bbox_iou(aabb, _box_xywh_to_xyxy(k.box)) > thresh for k in kept):
            continue
        kept.append(candidate)
    return kept


# ---------------------------------------------------------------------------
# Quality gate & bbox refinement
# ---------------------------------------------------------------------------


def _padded_crop_bounds(
    box: tuple[int, int, int, int],
    page_shape: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    ph, pw = page_shape
    x, y, w, h = box
    return (
        max(0, x - padding),
        max(0, y - padding),
        min(pw, x + w + padding),
        min(ph, y + h + padding),
    )


def _border_touch_count(
    box: tuple[int, int, int, int],
    page_shape: tuple[int, int],
) -> int:
    ph, pw = page_shape
    x, y, w, h = box
    margin_x = int(pw * config.BORDER_MARGIN_RATIO)
    margin_y = int(ph * config.BORDER_MARGIN_RATIO)
    touches = 0
    if x <= margin_x:
        touches += 1
    if y <= margin_y:
        touches += 1
    if x + w >= pw - margin_x:
        touches += 1
    if y + h >= ph - margin_y:
        touches += 1
    return touches


def _edge_density(gray_crop: np.ndarray) -> float:
    if gray_crop.size == 0:
        return 0.0
    edges = cv2.Canny(gray_crop, 50, 150)
    return float(np.count_nonzero(edges) / edges.size)


def _drawing_coverage(gray_crop: np.ndarray, crop_area: int) -> float:
    if gray_crop.size == 0 or crop_area <= 0:
        return 0.0
    edges = cv2.Canny(gray_crop, 50, 150)
    return float(np.count_nonzero(edges) / crop_area)


def validate_crop_candidate(
    image_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    text_boxes: list[TextBox] | None,
) -> QualityResult:
    ph, pw = image_bgr.shape[:2]
    page_area = ph * pw
    x, y, w, h = box
    crop_area = w * h
    area_ratio = crop_area / max(page_area, 1)
    aspect_ratio = w / max(h, 1)

    x0, y0, x1, y1 = _padded_crop_bounds(box, (ph, pw), config.FIGURE_PADDING)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return QualityResult(
            passed=False,
            gate_score=0.0,
            edge_density=0.0,
            drawing_coverage=0.0,
            text_overlap=1.0,
            border_touch_count=4,
            variance=0.0,
            area_ratio=area_ratio,
            aspect_ratio=aspect_ratio,
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    variance = float(np.std(gray))
    edge_density = _edge_density(gray)
    drawing_coverage = _drawing_coverage(gray, crop_area)
    text_overlap = (
        estimate_text_coverage_in_bbox(_box_xywh_to_xyxy(box), text_boxes or [])
        if text_boxes
        else 0.0
    )
    border_touch_count = _border_touch_count(box, (ph, pw))

    edge_score = min(1.0, edge_density / max(config.MIN_EDGE_DENSITY, 1e-6))
    coverage_score = min(
        1.0, drawing_coverage / max(config.MIN_DRAWING_COVERAGE, 1e-6)
    )
    if text_overlap <= config.MAX_TEXT_OVERLAP:
        text_score = 1.0 - (text_overlap / max(config.MAX_TEXT_OVERLAP, 1e-6))
    else:
        text_score = 0.0
    border_score = 1.0 - (border_touch_count / 4.0)
    variance_score = min(1.0, variance / 12.0)
    aspect_score = 1.0 if 0.2 <= aspect_ratio <= 5.0 else 0.0
    area_score = (
        1.0
        if config.MIN_COMPONENT_AREA_RATIO
        <= area_ratio
        <= config.MAX_COMPONENT_AREA_RATIO
        else 0.0
    )

    gate_score = (
        edge_score
        + coverage_score
        + text_score
        + border_score
        + variance_score
        + aspect_score
        + area_score
    ) / 7.0

    passed = (
        gate_score >= config.QUALITY_GATE_THRESHOLD
        and drawing_coverage >= config.MIN_DRAWING_COVERAGE
    )
    return QualityResult(
        passed=passed,
        gate_score=round(gate_score, 4),
        edge_density=round(edge_density, 4),
        drawing_coverage=round(drawing_coverage, 6),
        text_overlap=round(text_overlap, 4),
        border_touch_count=border_touch_count,
        variance=round(variance, 2),
        area_ratio=round(area_ratio, 4),
        aspect_ratio=round(aspect_ratio, 3),
    )


def refine_selected_bbox(
    image_bgr: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x, y, bw, bh = box
    if bw <= 0 or bh <= 0:
        return box

    page_h, page_w = image_bgr.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(page_w, int(x) + int(bw))
    y2 = min(page_h, int(y) + int(bh))
    roi_w, roi_h = x2 - x1, y2 - y1
    if roi_w <= 0 or roi_h <= 0:
        return box

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return box

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return box

    crop_area = roi_h * roi_w
    min_contour_area = crop_area * config.REFINEMENT_MIN_CONTOUR_AREA_RATIO
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    if not valid_contours:
        return box

    all_points = np.vstack([c.reshape(-1, 2) for c in valid_contours])
    rx, ry, rw, rh = cv2.boundingRect(all_points)
    pad_x = int(rw * config.REFINEMENT_PADDING_RATIO)
    pad_y = int(rh * config.REFINEMENT_PADDING_RATIO)
    rx1 = max(0, rx - pad_x)
    ry1 = max(0, ry - pad_y)
    rx2 = min(roi_w, rx + rw + pad_x)
    ry2 = min(roi_h, ry + rh + pad_y)
    refined_w, refined_h = rx2 - rx1, ry2 - ry1
    if refined_w <= 0 or refined_h <= 0:
        return box

    if refined_w * refined_h < bw * bh * config.REFINEMENT_MIN_RETAINED_AREA_RATIO:
        return box

    refined_page = (x1 + rx1, y1 + ry1, refined_w, refined_h)
    if not config.REFINEMENT_USE_UNION:
        return refined_page

    ox2, oy2 = x + bw, y + bh
    rx2p, ry2p = x1 + rx1 + refined_w, y1 + ry1 + refined_h
    ux1 = min(x, x1 + rx1)
    uy1 = min(y, y1 + ry1)
    ux2 = max(ox2, rx2p)
    uy2 = max(oy2, ry2p)
    return (ux1, uy1, ux2 - ux1, uy2 - uy1)


# ---------------------------------------------------------------------------
# Projection-based drawing bounds + crop validation
# ---------------------------------------------------------------------------


def find_main_drawing_bbox_via_projection(
    image_bgr: np.ndarray,
    text_boxes: list[TextBox] | None = None,
) -> tuple[int, int, int, int] | None:
    """
    Locate the main line-art region using row/column edge projections.

    Works well for hollow CAD drawings where connected components fragment.
    """
    h, w = image_bgr.shape[:2]
    margin_x = int(w * config.BORDER_MARGIN_RATIO)
    margin_y = int(h * config.BORDER_MARGIN_RATIO)
    max_bottom = int(h * config.DRAWING_ZONE_MAX_BOTTOM_RATIO)
    right_limit = int(w * (1.0 - config.PAGE_MARGIN_RIGHT_RATIO))

    working = image_bgr
    if text_boxes:
        working = apply_text_mask_to_bgr(image_bgr, text_boxes)

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges[:margin_y, :] = 0
    edges[max_bottom:, :] = 0
    edges[:, :margin_x] = 0
    edges[:, right_limit:] = 0

    row_profile = edges.sum(axis=1).astype(np.float32)
    col_profile = edges.sum(axis=0).astype(np.float32)
    if row_profile.max() < config.PROJECTION_MIN_PROFILE_SUM:
        return None

    row_thresh = max(
        row_profile.max() * config.PROJECTION_EDGE_ROW_THRESH,
        config.PROJECTION_MIN_PROFILE_SUM,
    )
    col_thresh = max(
        col_profile.max() * config.PROJECTION_EDGE_COL_THRESH,
        config.PROJECTION_MIN_PROFILE_SUM,
    )

    rows = np.where(row_profile >= row_thresh)[0]
    cols = np.where(col_profile >= col_thresh)[0]
    if len(rows) < 8 or len(cols) < 8:
        return None

    y1, y2 = int(rows[0]), int(rows[-1])
    x1, x2 = int(cols[0]), int(cols[-1])
    pad_x = max(int((x2 - x1) * config.CROP_CONTENT_MARGIN_RATIO), config.CROP_MIN_MARGIN_PX)
    pad_y = max(int((y2 - y1) * config.CROP_CONTENT_MARGIN_RATIO), config.CROP_MIN_MARGIN_PX)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 20:
        return None
    return (x1, y1, bw, bh)


def _profile_key(profile: str | object | None) -> str | None:
    if profile is None:
        return None
    if hasattr(profile, "value"):
        return str(profile.value)
    return str(profile)


def _active_profile_config(
    profile: str | object | None,
    profile_config: ProfileConfig | None = None,
) -> ProfileConfig:
    if profile_config is not None:
        return profile_config
    return resolve_profile_config(profile)


def _crop_limit_values(
    profile: str | object | None,
    profile_config: ProfileConfig | None = None,
) -> tuple[float, float, float, float]:
    """Return max_aspect, min_height_ratio, min_width_ratio, min_area_ratio."""
    pcfg = _active_profile_config(profile, profile_config)
    return (
        pcfg.max_crop_aspect_ratio,
        pcfg.min_crop_height_ratio,
        pcfg.min_crop_width_ratio,
        pcfg.min_crop_area_ratio,
    )


def line_art_density_in_bbox(
    image_bgr: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    text_boxes: list[TextBox] | None = None,
    *,
    profile_config: ProfileConfig | None = None,
) -> float:
    """Fraction of bbox pixels that are line-art ink (edges/thin strokes, not text)."""
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    mask = build_line_art_mask(
        image_bgr, text_boxes, profile_config=profile_config
    )
    roi = mask[y0:y1, x0:x1]
    return float(np.count_nonzero(roi)) / max(roi.size, 1)


def is_valid_figure_crop(
    box: tuple[int, int, int, int],
    page_shape: tuple[int, int],
    profile: str | object | None = None,
    *,
    line_art_density: float | None = None,
    profile_config: ProfileConfig | None = None,
) -> bool:
    """Reject edge slivers, tiny fragments, and extreme aspect ratios."""
    pcfg = _active_profile_config(profile, profile_config)
    ph, pw = page_shape
    x, y, bw, bh = box
    if bw <= 0 or bh <= 0:
        return False

    max_aspect, min_h_ratio, min_w_ratio, min_area = _crop_limit_values(
        profile, profile_config=pcfg
    )
    area_ratio = (bw * bh) / max(ph * pw, 1)
    aspect = bw / max(bh, 1)
    height_ratio = bh / max(ph, 1)

    if area_ratio < min_area:
        return False
    if area_ratio > pcfg.max_figure_output_area_ratio:
        return False
    if aspect < pcfg.min_crop_aspect_ratio or aspect > max_aspect:
        return False
    if bw < pw * min_w_ratio:
        return False
    if height_ratio < min_h_ratio:
        return False

    # Wide shallow crops must contain enough line art (blocks empty dimension strips).
    if aspect > 4.0 and height_ratio < 0.22:
        density = line_art_density if line_art_density is not None else 0.0
        if density < pcfg.min_line_art_density:
            return False
    return True


def _bbox_quality_score(
    box: tuple[int, int, int, int],
    page_shape: tuple[int, int],
    *,
    source: str,
) -> float:
    ph, pw = page_shape
    x, y, bw, bh = box
    area_ratio = (bw * bh) / max(ph * pw, 1)
    aspect = bw / max(bh, 1)
    cy = (y + bh / 2) / max(ph, 1)

    ideal_area = 0.30
    area_score = 1.0 - min(1.0, abs(area_ratio - ideal_area) / ideal_area)

    if cy > config.DRAWING_ZONE_MAX_BOTTOM_RATIO:
        vertical_score = 0.0
    else:
        ideal_cy = 0.36
        vertical_score = 1.0 - min(1.0, abs(cy - ideal_cy) / 0.35)

    aspect_score = 1.0 if 0.6 <= aspect <= 2.2 else 0.5
    source_bonus = 0.05 if source == "projection" else 0.0

    return area_score * 0.35 + vertical_score * 0.40 + aspect_score * 0.20 + source_bonus


def _should_prefer_projection_over_morphology(
    morph_box: tuple[int, int, int, int],
    proj_box: tuple[int, int, int, int],
    page_shape: tuple[int, int],
) -> bool:
    morph_area = morph_box[2] * morph_box[3]
    proj_area = proj_box[2] * proj_box[3]
    if proj_area <= 0:
        return False
    if morph_area < proj_area * config.MORPHOLOGY_PROJECTION_MIN_AREA_FRAC:
        return True
    morph_xyxy = _box_xywh_to_xyxy(morph_box)
    proj_xyxy = _box_xywh_to_xyxy(proj_box)
    return bbox_iou(morph_xyxy, proj_xyxy) < config.MORPHOLOGY_PROJECTION_MIN_IOU


def select_best_drawing_box(
    image_bgr: np.ndarray,
    morphology_selection: FigureSelectionResult | None,
    text_boxes: list[TextBox] | None,
) -> tuple[tuple[int, int, int, int] | None, str]:
    """Pick the best bbox between morphology connected-components and edge projection."""
    page_shape = image_bgr.shape[:2]
    candidates: list[tuple[tuple[int, int, int, int], str, float]] = []

    proj_box = find_main_drawing_bbox_via_projection(image_bgr, text_boxes)
    if proj_box is not None and is_valid_figure_crop(proj_box, page_shape):
        candidates.append(
            (proj_box, "projection", _bbox_quality_score(proj_box, page_shape, source="projection"))
        )

    if morphology_selection is not None:
        morph_box = morphology_selection.box
        if is_valid_figure_crop(morph_box, page_shape):
            if proj_box is not None and _should_prefer_projection_over_morphology(
                morph_box, proj_box, page_shape
            ):
                log.info(
                    "Using projection bbox (morphology fragment: area %.1f%% vs projection %.1f%%)",
                    100 * morph_box[2] * morph_box[3] / (page_shape[0] * page_shape[1]),
                    100 * proj_box[2] * proj_box[3] / (page_shape[0] * page_shape[1]),
                )
            else:
                score = _bbox_quality_score(morph_box, page_shape, source="morphology")
                score *= morphology_selection.gate_score
                candidates.append((morph_box, "morphology", score))
        elif proj_box is None:
            log.debug("Morphology bbox failed validation and no projection fallback")

    if not candidates and proj_box is not None:
        log.debug("Using projection bbox without strict size validation")
        return proj_box, "projection"

    if not candidates:
        return None, "none"

    best = max(candidates, key=lambda c: c[2])
    return best[0], best[1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _selection_rank(
    entry: FigureSelectionResult,
    page_shape: tuple[int, int],
) -> float:
    area_ratio = _box_area_ratio(entry.box, page_shape)
    max_ratio = config.MAX_FIGURE_OUTPUT_AREA_RATIO
    if area_ratio > max_ratio:
        size_penalty = max_ratio / area_ratio
    else:
        size_penalty = 1.0 + (max_ratio - area_ratio) * 0.15
    return entry.component_score * entry.gate_score * size_penalty


def select_best_validated_candidate(
    image_bgr: np.ndarray,
    candidates: list[ComponentCandidate],
    text_boxes: list[TextBox] | None,
) -> FigureSelectionResult | None:
    if not candidates:
        return None

    page_shape = image_bgr.shape[:2]
    max_ratio = config.MAX_FIGURE_OUTPUT_AREA_RATIO
    validated: list[FigureSelectionResult] = []
    fallbacks: list[FigureSelectionResult] = []

    for candidate in candidates:
        quality = validate_crop_candidate(image_bgr, candidate.box, text_boxes)
        entry = FigureSelectionResult(
            box=candidate.box,
            component_score=candidate.component_score,
            gate_score=quality.gate_score,
            quality=quality,
        )
        if quality.passed:
            validated.append(entry)
        else:
            fallbacks.append(entry)

    def _prefer_sized(pool: list[FigureSelectionResult]) -> list[FigureSelectionResult]:
        compact = [e for e in pool if _box_area_ratio(e.box, page_shape) <= max_ratio]
        return compact if compact else pool

    pool = _prefer_sized(validated) if validated else _prefer_sized(fallbacks)
    return max(pool, key=lambda e: _selection_rank(e, page_shape))


def _content_mask_in_region(gray: np.ndarray) -> np.ndarray:
    """Binary mask of ink and edges (includes thin centerlines)."""
    ink = (gray < config.CROP_INK_GRAY_MAX).astype(np.uint8) * 255
    edges = cv2.Canny(gray, 25, 90)
    return cv2.bitwise_or(ink, edges)


def _detect_dense_text_block_mask(gray: np.ndarray) -> np.ndarray:
    """Mask filled text/table blocks (title block paragraphs, notes)."""
    h, w = gray.shape
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h, iterations=1)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_v, iterations=1)

    page_area = h * w
    min_area = int(page_area * config.DENSE_TEXT_BLOCK_MIN_AREA_RATIO)
    mask = np.zeros((h, w), dtype=np.uint8)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_block_area = int(page_area * config.DENSE_TEXT_BLOCK_MAX_AREA_RATIO)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        block_area = cw * ch
        if block_area < min_area or block_area > max_block_area:
            continue
        if block_area > page_area * 0.22:
            continue
        roi = closed[y : y + ch, x : x + cw]
        fill = float(np.count_nonzero(roi)) / max(roi.size, 1)
        if fill >= config.DENSE_TEXT_BLOCK_MIN_FILL:
            mask[y : y + ch, x : x + cw] = 255
    return mask


def build_line_art_mask(
    image_bgr: np.ndarray,
    text_boxes: list[TextBox] | None = None,
    *,
    profile_config: ProfileConfig | None = None,
) -> np.ndarray:
    """Edges and thin strokes with OCR + dense text regions removed."""
    pcfg = _active_profile_config(None, profile_config)
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        gray,
        pcfg.line_art_canny_low,
        pcfg.line_art_canny_high,
    )
    thin = cv2.erode(
        (gray < config.CROP_INK_GRAY_MAX).astype(np.uint8) * 255,
        np.ones((2, 2), np.uint8),
        iterations=1,
    )
    line_art = cv2.bitwise_or(edges, thin)

    exclude = build_text_mask(
        (h, w),
        text_boxes or [],
        include_layout_zones=False,
        profile_config=pcfg,
    )
    exclude = cv2.bitwise_or(exclude, _detect_dense_text_block_mask(gray))
    line_art[exclude > 0] = 0

    speckle_kernel = np.ones((3, 3), np.uint8)
    line_art = cv2.morphologyEx(line_art, cv2.MORPH_OPEN, speckle_kernel, iterations=1)

    margin_x = int(w * config.BORDER_MARGIN_RATIO)
    margin_y = int(h * config.BORDER_MARGIN_RATIO)
    line_art[:margin_y, :] = 0
    line_art[h - margin_y :, :] = 0
    line_art[:, :margin_x] = 0
    line_art[:, w - margin_x :] = 0
    return line_art


def _line_art_strip_density(mask: np.ndarray, edge: str, strip_px: int) -> float:
    mh, mw = mask.shape[:2]
    s = max(2, strip_px)
    if edge == "top":
        roi = mask[: min(s, mh), :]
    elif edge == "bottom":
        roi = mask[max(0, mh - s) :, :]
    elif edge == "left":
        roi = mask[:, : min(s, mw)]
    else:
        roi = mask[:, max(0, mw - s) :]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / roi.size


def tighten_bbox_to_line_art(
    image_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text_boxes: list[TextBox] | None = None,
    *,
    profile_config: ProfileConfig | None = None,
) -> tuple[int, int, int, int]:
    """
    Tighten crop to line-art ink: trim empty margins; expand vertically only
    for wide shallow seeds (side views like fuselage profiles).
    """
    h, w = image_bgr.shape[:2]
    if x2 <= x1 or y2 <= y1:
        return x1, y1, x2, y2

    pcfg = _active_profile_config(None, profile_config)
    mask = build_line_art_mask(image_bgr, text_boxes, profile_config=pcfg)
    bw, bh = x2 - x1, y2 - y1
    aspect = bw / max(bh, 1)
    seed_area = bw * bh
    seed = (x1, y1, x2, y2)
    max_area = int(h * w * pcfg.max_figure_output_area_ratio)

    seed_mask = mask[y1:y2, x1:x2]
    if seed_mask.size == 0 or np.count_nonzero(seed_mask) < 24:
        return seed

    is_cad_wide = pcfg.crop_profile == CropProfile.CAD_WIDE
    wide_aspect_thresh = 2.0 if is_cad_wide else 3.0

    # Wide shallow seed: snap vertical bounds to line-art rows (fuselage, side views).
    if aspect > wide_aspect_thresh:
        pad_y = max(
            int(bh * pcfg.line_art_tighten_search_ratio),
            int(h * pcfg.line_art_tighten_wide_vertical_search_ratio),
        )
        sy1 = max(0, y1 - pad_y)
        sy2 = min(h, y2 + pad_y)
        row_sum = mask[sy1:sy2, x1:x2].sum(axis=1).astype(np.float32)
        if row_sum.max() > 4:
            row_thresh = max(row_sum.max() * pcfg.line_art_row_col_thresh_frac, 4.0)
            rows = np.where(row_sum >= row_thresh)[0]
            if len(rows) >= 2:
                y1 = sy1 + int(rows[0])
                y2 = sy1 + int(rows[-1]) + 1

    # Shrink empty margins on all four sides.
    strip_px = max(8, int(min(x2 - x1, y2 - y1) * (0.05 if is_cad_wide else 0.04)))
    edge_thresh = 0.005 if is_cad_wide else 0.006
    min_seed_frac = 0.28 if is_cad_wide else 0.35
    for _ in range(28 if is_cad_wide else 24):
        roi = mask[y1:y2, x1:x2]
        if roi.size == 0:
            break
        changed = False
        if y2 - y1 > 40 and _line_art_strip_density(roi, "top", strip_px) < edge_thresh:
            y1 += max(2, strip_px // 2)
            changed = True
        if y2 - y1 > 40 and _line_art_strip_density(roi, "bottom", strip_px) < edge_thresh:
            y2 -= max(2, strip_px // 2)
            changed = True
        if x2 - x1 > 40 and _line_art_strip_density(roi, "left", strip_px) < edge_thresh:
            x1 += max(2, strip_px // 2)
            changed = True
        if x2 - x1 > 40 and _line_art_strip_density(roi, "right", strip_px) < edge_thresh:
            x2 -= max(2, strip_px // 2)
            changed = True
        if not changed:
            break
        if (x2 - x1) * (y2 - y1) < seed_area * min_seed_frac:
            break

    margin_x = max(
        int((x2 - x1) * pcfg.line_art_tighten_margin_ratio),
        pcfg.line_art_tighten_min_margin_px,
    )
    margin_y = max(
        int((y2 - y1) * pcfg.line_art_tighten_margin_ratio),
        pcfg.line_art_tighten_min_margin_px,
    )
    nx1 = max(0, x1 - margin_x)
    ny1 = max(0, y1 - margin_y)
    nx2 = min(w, x2 + margin_x)
    ny2 = min(h, y2 + margin_y)

    tightened = (nx1, ny1, nx2, ny2)
    if (nx2 - nx1) * (ny2 - ny1) > max_area:
        tightened = (
            max(0, x1 - margin_x // 2),
            max(0, y1 - margin_y // 2),
            min(w, x2 + margin_x // 2),
            min(h, y2 + margin_y // 2),
        )
    if tightened[2] <= tightened[0] + 5 or tightened[3] <= tightened[1] + 5:
        return seed

    tight_area = (tightened[2] - tightened[0]) * (tightened[3] - tightened[1])
    if tight_area < seed_area * pcfg.tighten_min_retained_area_frac:
        return seed
    return tightened


def _text_overlap_on_edge_strip(
    bbox_xyxy: tuple[int, int, int, int],
    text_boxes: list[TextBox],
    edge: str,
    strip_px: int,
) -> float:
    x0, y0, x1, y1 = bbox_xyxy
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return 0.0

    if edge == "top":
        strip = (x0, y0, x1, min(y1, y0 + strip_px))
    elif edge == "bottom":
        strip = (x0, max(y0, y1 - strip_px), x1, y1)
    elif edge == "left":
        strip = (x0, y0, min(x1, x0 + strip_px), y1)
    else:
        strip = (max(x0, x1 - strip_px), y0, x1, y1)
    return estimate_text_coverage_in_bbox(strip, text_boxes)


def shrink_bbox_from_text_overlap(
    image_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text_boxes: list[TextBox] | None,
    *,
    profile_config: ProfileConfig | None = None,
) -> tuple[int, int, int, int]:
    """Trim crop edges that contain OCR text until overlap is below target."""
    if not text_boxes:
        return x1, y1, x2, y2

    pcfg = _active_profile_config(None, profile_config)
    h, w = image_bgr.shape[:2]
    orig_area = max((x2 - x1) * (y2 - y1), 1)
    bbox = (x1, y1, x2, y2)

    for _ in range(pcfg.text_shrink_max_iterations):
        overlap = estimate_text_coverage_in_bbox(bbox, text_boxes)
        if overlap <= pcfg.text_shrink_target_overlap:
            break

        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        step_x = max(2, int(bw * pcfg.text_shrink_step_ratio))
        step_y = max(2, int(bh * pcfg.text_shrink_step_ratio))
        strip_px = max(12, int(min(bw, bh) * 0.06))

        edges = ("top", "bottom", "left", "right")
        edge_scores = {
            e: _text_overlap_on_edge_strip(bbox, text_boxes, e, strip_px) for e in edges
        }
        edge = max(edges, key=lambda e: edge_scores[e])
        if edge_scores[edge] < 0.01:
            break

        bx0, by0, bx1, by1 = bbox
        if edge == "top":
            by0 = min(by1 - 10, by0 + step_y)
        elif edge == "bottom":
            by1 = max(by0 + 10, by1 - step_y)
        elif edge == "left":
            bx0 = min(bx1 - 10, bx0 + step_x)
        else:
            bx1 = max(bx0 + 10, bx1 - step_x)

        new_area = max((bx1 - bx0) * (by1 - by0), 1)
        if new_area < orig_area * pcfg.text_shrink_min_remaining_ratio:
            break
        if (
            line_art_density_in_bbox(image_bgr, (bx0, by0, bx1, by1), text_boxes)
            < pcfg.min_line_art_density * 0.5
        ):
            break
        bbox = (bx0, by0, bx1, by1)

    return bbox


def refine_figure_crop_bounds(
    image_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text_boxes: list[TextBox] | None = None,
    profile: str | object | None = None,
    *,
    profile_config: ProfileConfig | None = None,
) -> tuple[int, int, int, int]:
    """Line-art tighten then post-crop text shrink."""
    pcfg = _active_profile_config(profile, profile_config)
    x1, y1, x2, y2 = tighten_bbox_to_line_art(
        image_bgr, x1, y1, x2, y2, text_boxes, profile_config=pcfg
    )
    x1, y1, x2, y2 = shrink_bbox_from_text_overlap(
        image_bgr, x1, y1, x2, y2, text_boxes, profile_config=pcfg
    )
    return x1, y1, x2, y2


def _validate_and_crop(
    image_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text_boxes: list[TextBox] | None,
    profile: str | object | None,
    *,
    profile_config: ProfileConfig | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    pcfg = _active_profile_config(profile, profile_config)
    if x2 <= x1 or y2 <= y1:
        return None
    final_box = (x1, y1, x2 - x1, y2 - y1)
    density = line_art_density_in_bbox(
        image_bgr, (x1, y1, x2, y2), text_boxes, profile_config=pcfg
    )
    if not is_valid_figure_crop(
        final_box,
        image_bgr.shape[:2],
        profile,
        line_art_density=density,
        profile_config=pcfg,
    ):
        return None
    return image_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def _pad_detection_bbox_xyxy(
    box_xywh: tuple[int, int, int, int],
    page_shape: tuple[int, int],
    *,
    pad_ratio: float | None = None,
) -> tuple[int, int, int, int]:
    """Light padding on a detector bbox without full-page finalize expansion."""
    ph, pw = page_shape
    x, y, bw, bh = box_xywh
    ratio = pad_ratio if pad_ratio is not None else config.FINAL_EXPANSION_RATIO
    pad_x = max(int(bw * ratio), config.LINE_ART_TIGHTEN_MIN_MARGIN_PX)
    pad_y = max(int(bh * ratio), config.LINE_ART_TIGHTEN_MIN_MARGIN_PX)
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(pw, x + bw + pad_x),
        min(ph, y + bh + pad_y),
    )


def prepare_primary_crop(
    image_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text_boxes: list[TextBox] | None,
    profile: str | object | None = None,
    *,
    profile_config: ProfileConfig | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """
    Refine bbox, validate with profile limits, return crop and xyxy bbox.
    """
    pcfg = _active_profile_config(profile, profile_config)
    if x2 <= x1 or y2 <= y1:
        return None

    seed = (x1, y1, x2, y2)
    rx1, ry1, rx2, ry2 = refine_figure_crop_bounds(
        image_bgr,
        x1,
        y1,
        x2,
        y2,
        text_boxes,
        profile=profile,
        profile_config=pcfg,
    )
    result = _validate_and_crop(
        image_bgr, rx1, ry1, rx2, ry2, text_boxes, profile, profile_config=pcfg
    )
    if result is not None:
        return result

    # Fallback: text-shrink seed only (avoid failed tighten on photo-heavy PDFs).
    sx1, sy1, sx2, sy2 = shrink_bbox_from_text_overlap(
        image_bgr,
        seed[0],
        seed[1],
        seed[2],
        seed[3],
        text_boxes,
        profile_config=pcfg,
    )
    return _validate_and_crop(
        image_bgr, sx1, sy1, sx2, sy2, text_boxes, profile, profile_config=pcfg
    )


def finalize_crop_bounds(
    image_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text_boxes: list[TextBox] | None = None,
) -> tuple[int, int, int, int]:
    """
    Expand crop to include all nearby line work with uniform margins.

    Prevents clipped wheels, centerlines, dimension lines, and photo edges.
    """
    h, w = image_bgr.shape[:2]
    if x2 <= x1 or y2 <= y1:
        return x1, y1, x2, y2

    bw, bh = x2 - x1, y2 - y1
    search_x = int(max(bw * config.CROP_SEARCH_EXPAND_RATIO, config.CROP_MIN_MARGIN_PX))
    search_y = int(max(bh * config.CROP_SEARCH_EXPAND_RATIO, config.CROP_MIN_MARGIN_PX))

    sx1 = max(0, x1 - search_x)
    sy1 = max(0, y1 - search_y)
    sx2 = min(w, x2 + search_x)
    sy2 = min(h, y2 + search_y)

    patch = image_bgr[sy1:sy2, sx1:sx2]
    if patch.size == 0:
        return x1, y1, x2, y2

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    content = _content_mask_in_region(gray)

    row_sum = content.sum(axis=1).astype(np.float32)
    col_sum = content.sum(axis=0).astype(np.float32)
    if row_sum.max() < 8 or col_sum.max() < 8:
        pad = max(config.CROP_MIN_MARGIN_PX, int(min(bw, bh) * config.CROP_CONTENT_MARGIN_RATIO))
        return (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(w, x2 + pad),
            min(h, y2 + pad),
        )

    row_thresh = max(row_sum.max() * 0.015, 6.0)
    col_thresh = max(col_sum.max() * 0.015, 6.0)
    rows = np.where(row_sum >= row_thresh)[0]
    cols = np.where(col_sum >= col_thresh)[0]
    if len(rows) < 3 or len(cols) < 3:
        pad = max(config.CROP_MIN_MARGIN_PX, int(min(bw, bh) * config.CROP_CONTENT_MARGIN_RATIO))
        return (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(w, x2 + pad),
            min(h, y2 + pad),
        )

    cy1 = sy1 + int(rows[0])
    cy2 = sy1 + int(rows[-1]) + 1
    cx1 = sx1 + int(cols[0])
    cx2 = sx1 + int(cols[-1]) + 1

    # Union with seed box so we never shrink below the detector's region.
    cx1 = min(cx1, x1)
    cy1 = min(cy1, y1)
    cx2 = max(cx2, x2)
    cy2 = max(cy2, y2)

    cbw, cbh = cx2 - cx1, cy2 - cy1
    pad_x = max(int(cbw * config.CROP_CONTENT_MARGIN_RATIO), config.CROP_MIN_MARGIN_PX)
    pad_y = max(int(cbh * config.CROP_CONTENT_MARGIN_RATIO), config.CROP_MIN_MARGIN_PX)

    page_margin_x = int(w * config.BORDER_MARGIN_RATIO)
    page_margin_y = int(h * config.BORDER_MARGIN_RATIO)
    max_bottom = int(h * config.DRAWING_ZONE_MAX_BOTTOM_RATIO)

    nx1 = max(page_margin_x, cx1 - pad_x)
    ny1 = max(page_margin_y, cy1 - pad_y)
    nx2 = min(w - page_margin_x, cx2 + pad_x)
    ny2 = min(h - page_margin_y, cy2 + pad_y)

    # Only cap bottom when the extension is mostly empty (title band), not for full drawings.
    if ny2 > max_bottom:
        band = content[max(0, max_bottom - sy1) : min(sy2 - sy1, h - sy1), :]
        if band.size > 0 and float(np.count_nonzero(band)) / max(band.size, 1) < 0.004:
            ny2 = max_bottom

    if nx2 <= nx1 + 5 or ny2 <= ny1 + 5:
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2


def _expand_crop_toward_drawing_top(
    image_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> int:
    """Extend crop upward to include disconnected nose/cone line work above the selection."""
    margin_y = int(image_bgr.shape[0] * config.BORDER_MARGIN_RATIO)
    if y1 <= margin_y + 5:
        return y1

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    band = gray[margin_y:y1, x1:x2]
    if band.size == 0:
        return y1

    edges = cv2.Canny(band, 50, 150)
    row_density = edges.mean(axis=1) / 255.0
    threshold = max(config.MIN_EDGE_DENSITY * 0.6, 0.012)
    dense_rows = np.where(row_density >= threshold)[0]
    if len(dense_rows) == 0:
        return y1
    return margin_y + int(dense_rows[0])


def _fit_crop_to_max_area(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    page_h: int,
    page_w: int,
) -> tuple[int, int, int, int]:
    """Shrink crop symmetrically if it exceeds MAX_FIGURE_OUTPUT_AREA_RATIO."""
    max_area = int(page_h * page_w * config.MAX_FIGURE_OUTPUT_AREA_RATIO)
    width = max(x2 - x1, 1)
    height = y2 - y1
    if width * height <= max_area:
        return x1, y1, x2, y2

    scale = (max_area / (width * height)) ** 0.5
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half_w = int(width * scale / 2)
    half_h = int(height * scale / 2)
    margin_x = int(page_w * config.BORDER_MARGIN_RATIO)
    margin_y = int(page_h * config.BORDER_MARGIN_RATIO)
    nx1 = max(margin_x, int(cx - half_w))
    ny1 = max(margin_y, int(cy - half_h))
    nx2 = min(page_w - margin_x, int(cx + half_w))
    ny2 = min(page_h - margin_y, int(cy + half_h))
    if nx2 <= nx1 + 5 or ny2 <= ny1 + 5:
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2


def _compute_final_crop_bounds(
    box: tuple[int, int, int, int],
    page_h: int,
    page_w: int,
) -> tuple[int, int, int, int]:
    x, y, bw, bh = box
    padding_x = int(bw * config.FINAL_EXPANSION_RATIO)
    padding_y = int(bh * config.FINAL_EXPANSION_RATIO)
    return (
        max(0, x - padding_x),
        max(0, y - padding_y),
        min(page_w, x + bw + padding_x),
        min(page_h, y + bh + padding_y),
    )


def _prepare_binary_and_candidates(
    image: np.ndarray,
    text_boxes: list[TextBox] | None,
    *,
    apply_text_mask: bool,
) -> tuple[int, int, list[ComponentCandidate], FigureSelectionResult | None]:
    page_h, page_w = image.shape[:2]
    scale = _morphology_analysis_scale(page_h, page_w)
    inv_scale = 1.0 / scale

    if scale < 1.0:
        analysis = cv2.resize(
            image,
            (int(page_w * scale), int(page_h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        analysis = image

    boxes = text_boxes or []
    analysis_boxes = _scale_text_boxes(boxes, scale)

    working = analysis
    if apply_text_mask:
        working = apply_text_mask_to_bgr(
            analysis,
            analysis_boxes,
            padding=max(4, int(config.TEXT_MASK_PADDING_PX * scale)),
        )

    gray = (
        cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        if len(working.shape) == 3
        else working.copy()
    )
    h, w = gray.shape
    binary = adaptive_binarize(gray)
    binary = remove_page_border(binary)
    binary = remove_notes_block(binary)
    binary = remove_title_block(binary)
    binary = apply_morphology_close(binary)

    raw = find_top_component_candidates(binary)
    merged = merge_candidate_regions(raw, page_shape=(h, w))
    candidates = suppress_duplicate_candidates(merged)

    if inv_scale != 1.0:
        candidates = [
            ComponentCandidate(
                box=_scale_box_xywh(c.box, inv_scale),
                component_score=c.component_score,
            )
            for c in candidates
        ]

    selection = select_best_validated_candidate(image, candidates, text_boxes)
    if selection is not None and inv_scale != 1.0:
        selection = replace(
            selection,
            box=_scale_box_xywh(selection.box, inv_scale),
        )
    return page_h, page_w, candidates, selection


def compute_engineering_figure_crop(
    image: np.ndarray,
    text_boxes: list[TextBox] | None = None,
    *,
    apply_text_mask: bool = True,
) -> tuple[np.ndarray, tuple[int, int, int, int], FigureSelectionResult] | None:
    """
    Run the morphology/projection pipeline and return the crop in memory.

    Returns (crop_bgr, bbox_xyxy, selection) or None when no valid crop exists.
    """
    if image is None:
        raise ValueError("Input image is None")

    h, w, _candidates, selection = _prepare_binary_and_candidates(
        image, text_boxes, apply_text_mask=apply_text_mask
    )

    best_box, _source = select_best_drawing_box(image, selection, text_boxes)
    if best_box is None:
        return None

    if selection is None or best_box != selection.box:
        quality = validate_crop_candidate(image, best_box, text_boxes)
        selection = FigureSelectionResult(
            box=best_box,
            component_score=selection.component_score if selection else 0.0,
            gate_score=quality.gate_score,
            quality=quality,
        )
    elif selection is not None:
        selection = replace(selection, box=best_box)

    refined_box = refine_selected_bbox(image, selection.box)
    pre_area = selection.box[2] * selection.box[3]
    post_area = refined_box[2] * refined_box[3]
    pre_cy = selection.box[1] + selection.box[3] / 2
    post_cy = refined_box[1] + refined_box[3] / 2
    if post_area >= pre_area * 0.45 and post_cy <= pre_cy + h * 0.08:
        selection = replace(selection, box=refined_box)

    x1, y1, x2, y2 = _compute_final_crop_bounds(selection.box, h, w)
    y1 = _expand_crop_toward_drawing_top(image, x1, y1, x2, y2)
    x1, y1, x2, y2 = finalize_crop_bounds(image, x1, y1, x2, y2, text_boxes)
    x1, y1, x2, y2 = _fit_crop_to_max_area(x1, y1, x2, y2, h, w)
    if y2 <= y1 + 10:
        return None

    prepared = prepare_primary_crop(
        image, x1, y1, x2, y2, text_boxes, profile="engineering_sheet"
    )
    crop_area_ratio = (x2 - x1) * (y2 - y1) / max(h * w, 1)
    if prepared is None or crop_area_ratio > config.MAX_FIGURE_OUTPUT_AREA_RATIO * 0.88:
        proj_box = find_main_drawing_bbox_via_projection(image, text_boxes)
        if proj_box is not None:
            px1, py1, px2, py2 = _pad_detection_bbox_xyxy(proj_box, (h, w))
            alt = prepare_primary_crop(
                image, px1, py1, px2, py2, text_boxes, profile="engineering_sheet"
            )
            if alt is not None:
                alt_area = (alt[1][2] - alt[1][0]) * (alt[1][3] - alt[1][1]) / max(h * w, 1)
                if prepared is None or alt_area < crop_area_ratio * 0.85:
                    prepared = alt
                    log.info(
                        "Using tighter projection crop (%.1f%% vs morphology %.1f%% of page)",
                        100 * alt_area,
                        100 * crop_area_ratio,
                    )

    if prepared is None:
        final_box = (x1, y1, x2 - x1, y2 - y1)
        log.info(
            "Engineering figure crop rejected (invalid geometry %dx%d, %.1f%% of page)",
            final_box[2],
            final_box[3],
            100.0 * final_box[2] * final_box[3] / (h * w),
        )
        return None

    cropped, (x1, y1, x2, y2) = prepared
    crop_area = (x2 - x1) * (y2 - y1)
    if crop_area / max(h * w, 1) > config.MAX_FIGURE_OUTPUT_AREA_RATIO:
        log.info(
            "Engineering figure crop too large (%.1f%% of page); skipping",
            100.0 * crop_area / (h * w),
        )
        return None

    return cropped, (x1, y1, x2, y2), selection


def extract_engineering_figure(
    image: np.ndarray,
    output_path: str | Path,
    text_boxes: list[TextBox] | None = None,
    *,
    apply_text_mask: bool = True,
) -> Path | None:
    path, _ = extract_engineering_figure_with_metadata(
        image, output_path, text_boxes, apply_text_mask=apply_text_mask
    )
    return path


def extract_engineering_figure_with_metadata(
    image: np.ndarray,
    output_path: str | Path,
    text_boxes: list[TextBox] | None = None,
    *,
    apply_text_mask: bool = True,
) -> tuple[Path | None, FigureSelectionResult | None]:
    result = compute_engineering_figure_crop(
        image, text_boxes, apply_text_mask=apply_text_mask
    )
    if result is None:
        return None, None

    cropped, _bbox, selection = result
    output_path = Path(output_path)
    save_bgr(output_path, cropped)
    return output_path, selection
