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
from src.ocr.ocr_engine import TextBox
from src.utils import bbox_iou, save_bgr


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


def build_text_mask(
    shape: tuple[int, int],
    boxes: list[TextBox],
    padding: int | None = None,
) -> np.ndarray:
    """Uint8 mask with OCR text regions set to 255."""
    pad = padding if padding is not None else config.TEXT_MASK_PADDING_PX
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x0, y0, x1, y1 = box.bbox
        px0, py0, px1, py1 = _clip_padded_bbox(x0, y0, x1, y1, pad, w, h)
        mask[py0:py1, px0:px1] = 255
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
    h, _ = binary.shape
    title_h = int(h * config.TITLE_BLOCK_HEIGHT_RATIO)
    binary = binary.copy()
    binary[h - title_h :, :] = 0
    return binary


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

    density = float(np.count_nonzero(binary_roi) / max(w * h, 1))
    aspect_ratio = w / max(h, 1)
    aspect_score = 1.0 if 0.3 <= aspect_ratio <= 3.5 else 0.3

    return (
        config.AREA_WEIGHT * component_area_ratio
        + config.CENTRALITY_WEIGHT * centrality_score
        + config.DENSITY_WEIGHT * density
        + config.ASPECT_WEIGHT * aspect_score
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

    return (x1 + rx1, y1 + ry1, refined_w, refined_h)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_best_validated_candidate(
    image_bgr: np.ndarray,
    candidates: list[ComponentCandidate],
    text_boxes: list[TextBox] | None,
) -> FigureSelectionResult | None:
    if not candidates:
        return None

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

    pool = validated if validated else fallbacks
    return max(pool, key=lambda e: e.component_score * e.gate_score)


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
    working = image
    if apply_text_mask and text_boxes:
        working = apply_text_mask_to_bgr(image, text_boxes)

    gray = (
        cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        if len(working.shape) == 3
        else working.copy()
    )
    h, w = gray.shape
    binary = adaptive_binarize(gray)
    binary = remove_page_border(binary)
    binary = remove_title_block(binary)
    binary = apply_morphology_close(binary)

    raw = find_top_component_candidates(binary)
    merged = merge_candidate_regions(raw, page_shape=(h, w))
    candidates = suppress_duplicate_candidates(merged)
    selection = select_best_validated_candidate(image, candidates, text_boxes)
    return h, w, candidates, selection


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
    if image is None:
        raise ValueError("Input image is None")

    h, w, _candidates, selection = _prepare_binary_and_candidates(
        image, text_boxes, apply_text_mask=apply_text_mask
    )
    if selection is None:
        return None, None

    refined_box = refine_selected_bbox(image, selection.box)
    selection = replace(selection, box=refined_box)
    x1, y1, x2, y2 = _compute_final_crop_bounds(selection.box, h, w)
    cropped = image[y1:y2, x1:x2]

    output_path = Path(output_path)
    save_bgr(output_path, cropped)
    return output_path, selection
