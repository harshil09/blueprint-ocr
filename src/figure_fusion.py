"""
Page profiling, candidate scoring, and primary-figure fusion.

Selects at most one ranked figure per page (one per document page for PNG /
single-frame TIFF; one per PDF/TIFF frame for multi-page inputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from src import config
from src.extract_figure import estimate_text_coverage_in_bbox
from src.logger import get_logger
from src.ocr.ocr_engine import OcrPageResult
from src.profile_config import (
    CropProfile,
    ProfileConfig,
    default_crop_profile,
    get_profile_config,
    resolve_crop_profile,
    resolve_profile_config,
)
from src.utils import bbox_iou

log = get_logger(__name__)


class PageProfile(str, Enum):
    ENGINEERING_SHEET = "engineering_sheet"
    PHOTO_DATASHEET = "photo_datasheet"
    SIMPLE_IMAGE = "simple_image"
    DIGITAL_PDF = "digital_pdf"
    TEXT_HEAVY = "text_heavy"
    MIXED = "mixed"


class FigureType(str, Enum):
    LINE_ART = "line_art"
    PHOTO = "photo"
    EMBEDDED = "embedded"
    FULL_PAGE = "full_page"
    NONE = "none"


@dataclass(frozen=True)
class FigureCandidate:
    page_index: int
    method: str
    figure_type: FigureType
    bbox: tuple[int, int, int, int] | None  # x0, y0, x1, y1 page coords
    area_ratio: float
    gate_score: float
    text_overlap: float
    completeness: float
    crop: np.ndarray | None = None
    embedded_bytes: bytes | None = None
    embedded_ext: str = "png"
    component_score: float = 0.0
    quality_passed: bool = False

    @property
    def composite_score(self) -> float:
        """Backward-compatible score using the mixed-datasheet profile defaults."""
        return candidate_composite_score(
            self, get_profile_config(CropProfile.MIXED_DATASHEET)
        )


def candidate_composite_score(candidate: FigureCandidate, pcfg: ProfileConfig) -> float:
    max_text = max(pcfg.primary_figure_max_text_overlap, 1e-6)
    text_penalty = max(0.0, 1.0 - candidate.text_overlap / max_text)
    if candidate.text_overlap > pcfg.primary_figure_max_text_overlap:
        text_penalty *= 0.4
    completeness_factor = 0.5 + 0.5 * min(1.0, max(0.0, candidate.completeness))
    quality_factor = 1.0 if candidate.quality_passed else 0.88
    base = (
        candidate.gate_score * 0.45
        + candidate.component_score * 0.15
        + text_penalty * 0.25
    )
    return base * completeness_factor * quality_factor


def _page_text_coverage(ocr: OcrPageResult, page_shape: tuple[int, int]) -> float:
    if not ocr.boxes:
        return 0.0
    ph, pw = page_shape
    mask = np.zeros((ph, pw), dtype=np.uint8)
    for box in ocr.boxes:
        x0, y0, x1, y1 = box.bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(pw, x1), min(ph, y1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return float(mask.mean() / 255.0)


def _edge_density(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges) / max(edges.size, 1))


def _mean_saturation(image_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


def classify_page(
    image_bgr: np.ndarray,
    ocr: OcrPageResult,
    *,
    page_count: int,
    embedded_on_page: int = 0,
    is_pdf: bool = False,
) -> PageProfile:
    ph, pw = image_bgr.shape[:2]
    text_heavy_pcfg = get_profile_config(CropProfile.TEXT_HEAVY)
    eng_pcfg = get_profile_config(CropProfile.BLUEPRINT_LARGE)

    text_cov = _page_text_coverage(ocr, (ph, pw))
    if text_cov >= text_heavy_pcfg.page_text_heavy_coverage_ratio and _edge_density(image_bgr) < 0.015:
        return PageProfile.TEXT_HEAVY

    edge_density = _edge_density(image_bgr)
    mean_sat = _mean_saturation(image_bgr)

    # Scanned engineering PDFs often embed the full-page raster — still crop via line art.
    if embedded_on_page > 0 and is_pdf:
        if mean_sat < 15 and edge_density >= 0.006:
            return PageProfile.ENGINEERING_SHEET
        return PageProfile.DIGITAL_PDF

    # Full CAD sheets with title block / notes (landscape, many OCR boxes).
    page_aspect = pw / max(ph, 1)
    ocr_box_count = len(ocr.boxes)
    if (
        page_count == 1
        and page_aspect >= 1.15
        and ocr_box_count >= 15
        and edge_density >= 0.012
    ):
        return PageProfile.ENGINEERING_SHEET

    # Isolated line-art PNG (few labels, no dense title block) → simple raster.
    if (
        page_count == 1
        and edge_density >= eng_pcfg.engineering_edge_density_min
        and ocr_box_count < 12
    ):
        return PageProfile.SIMPLE_IMAGE

    # Grayscale technical drawings at high DPI have lower edge density per pixel.
    if mean_sat < 12 and edge_density >= 0.008:
        return PageProfile.ENGINEERING_SHEET

    if mean_sat < config.PHOTO_SATURATION_MAX + 10 and edge_density >= eng_pcfg.engineering_edge_density_min:
        return PageProfile.ENGINEERING_SHEET

    if mean_sat < config.PHOTO_SATURATION_MAX + 15 and edge_density >= 0.01:
        return PageProfile.ENGINEERING_SHEET

    if page_count == 1 and mean_sat < config.PHOTO_SATURATION_MAX + 20:
        return PageProfile.SIMPLE_IMAGE

    if mean_sat < config.PHOTO_SATURATION_MAX + 20:
        return PageProfile.PHOTO_DATASHEET

    return PageProfile.MIXED


def _edge_strip_completeness(outer: np.ndarray, inner: np.ndarray) -> float:
    """Compare ink in a strip just outside the crop vs along the crop edge."""
    if outer.size == 0:
        return 1.0
    outer_den = float(np.count_nonzero(outer)) / outer.size
    inner_den = float(np.count_nonzero(inner)) / inner.size if inner.size else 0.0
    if outer_den < 0.006:
        return 1.0
    if outer_den > inner_den * 1.35 and outer_den > 0.015:
        return max(0.35, 1.0 - (outer_den - inner_den) * 4.0)
    return 1.0


def crop_completeness_score(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    """
    Per-edge check: ink in strips outside the bbox vs along the bbox edge.

    Large CAD crops intentionally include nearby line work; this avoids
    penalizing every engineering sheet to zero.
    """
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return 0.35

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ink = gray < config.CROP_INK_GRAY_MAX
    band = max(10, int(min(x1 - x0, y1 - y0) * config.COMPLETENESS_RING_EXPAND_RATIO))

    scores: list[float] = []

    if y0 >= band:
        scores.append(
            _edge_strip_completeness(ink[y0 - band : y0, x0:x1], ink[y0 : y0 + band, x0:x1])
        )
    else:
        scores.append(1.0)

    if y1 + band <= h:
        scores.append(
            _edge_strip_completeness(ink[y1 : y1 + band, x0:x1], ink[y1 - band : y1, x0:x1])
        )
    else:
        scores.append(1.0)

    if x0 >= band:
        scores.append(
            _edge_strip_completeness(ink[y0:y1, x0 - band : x0], ink[y0:y1, x0 : x0 + band])
        )
    else:
        scores.append(1.0)

    if x1 + band <= w:
        scores.append(
            _edge_strip_completeness(ink[y0:y1, x1 : x1 + band], ink[y0:y1, x1 - band : x1])
        )
    else:
        scores.append(1.0)

    return max(0.35, sum(scores) / len(scores))


def _method_prior(method: str, pcfg: ProfileConfig) -> float:
    return pcfg.method_priors.get(method, 0.6)


def score_candidate(
    candidate: FigureCandidate,
    profile: PageProfile,
    *,
    pcfg: ProfileConfig | None = None,
) -> float:
    active = pcfg or resolve_profile_config(default_crop_profile(profile))
    prior = _method_prior(candidate.method, active)
    score = candidate_composite_score(candidate, active) * prior
    if candidate.area_ratio > active.area_penalty_threshold:
        score *= max(0.55, active.area_penalty_threshold / candidate.area_ratio)
    if (
        candidate.method == "embedded"
        and candidate.area_ratio > active.embedded_fusion_max_area_ratio
    ):
        score *= 0.35
    return score


def _dedupe_candidates(candidates: list[FigureCandidate]) -> list[FigureCandidate]:
    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda c: candidate_composite_score(
            c, get_profile_config(CropProfile.MIXED_DATASHEET)
        ),
        reverse=True,
    )
    kept: list[FigureCandidate] = []
    for cand in ranked:
        if cand.bbox is None:
            kept.append(cand)
            continue
        if any(
            other.bbox is not None and bbox_iou(cand.bbox, other.bbox) > config.IOU_DUPLICATE_THRESHOLD
            for other in kept
        ):
            continue
        kept.append(cand)
    return kept


def select_primary_candidate(
    candidates: list[FigureCandidate],
    profile: PageProfile,
    *,
    pcfg: ProfileConfig | None = None,
    crop_profile: CropProfile | None = None,
) -> FigureCandidate | None:
    active = pcfg or resolve_profile_config(crop_profile or default_crop_profile(profile))

    if profile == PageProfile.TEXT_HEAVY or active.crop_profile == CropProfile.TEXT_HEAVY:
        log.debug("Page profile text_heavy — skipping figure extraction")
        return None

    pool = _dedupe_candidates(candidates)
    if not pool:
        return None

    scored = [(c, score_candidate(c, profile, pcfg=active)) for c in pool]
    best, best_score = max(scored, key=lambda item: item[1])

    if best.text_overlap > active.primary_figure_max_text_overlap and best_score < 0.55:
        log.debug(
            "Best candidate rejected (text_overlap=%.3f score=%.3f)",
            best.text_overlap,
            best_score,
        )
        alternatives = [
            c for c, s in scored
            if c is not best and s >= active.primary_figure_min_confidence
        ]
        if alternatives:
            best = max(alternatives, key=lambda c: score_candidate(c, profile, pcfg=active))
            best_score = score_candidate(best, profile, pcfg=active)
        else:
            return None

    if best.completeness < active.primary_figure_min_completeness and best_score < 0.5:
        better = [
            c for c, s in scored
            if c.completeness >= active.primary_figure_min_completeness
            and s >= active.primary_figure_min_confidence
        ]
        if better:
            best = max(better, key=lambda c: score_candidate(c, profile, pcfg=active))
            best_score = score_candidate(best, profile, pcfg=active)

    if best_score < active.primary_figure_min_confidence:
        fallback_pool = [
            c for c in pool
            if (c.crop is not None or c.embedded_bytes)
            and c.gate_score >= config.QUALITY_GATE_THRESHOLD
            and c.text_overlap <= config.MAX_TEXT_OVERLAP
        ]
        if fallback_pool:
            best = max(fallback_pool, key=lambda c: c.gate_score)
            best_score = score_candidate(best, profile, pcfg=active)
            log.info(
                "Primary figure fallback: method=%s gate=%.3f score=%.3f",
                best.method,
                best.gate_score,
                best_score,
            )
        else:
            log.info(
                "No primary figure (best %s score=%.3f below threshold %.3f)",
                best.method,
                best_score,
                active.primary_figure_min_confidence,
            )
            return None

    log.info(
        "Primary figure selected: method=%s type=%s score=%.3f gate=%.3f "
        "text_overlap=%.3f completeness=%.3f profile=%s crop_profile=%s",
        best.method,
        best.figure_type.value,
        best_score,
        best.gate_score,
        best.text_overlap,
        best.completeness,
        profile.value,
        active.crop_profile.value,
    )
    return best


def build_candidate_from_crop(
    page_index: int,
    method: str,
    figure_type: FigureType,
    image_bgr: np.ndarray,
    crop: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    text_boxes: list,
    *,
    gate_score: float = 0.5,
    component_score: float = 0.0,
    quality_passed: bool = False,
) -> FigureCandidate:
    ph, pw = image_bgr.shape[:2]
    area_ratio = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1]) / max(ph * pw, 1)
    text_overlap = estimate_text_coverage_in_bbox(bbox_xyxy, text_boxes)
    completeness = crop_completeness_score(image_bgr, bbox_xyxy)
    return FigureCandidate(
        page_index=page_index,
        method=method,
        figure_type=figure_type,
        bbox=bbox_xyxy,
        area_ratio=area_ratio,
        gate_score=gate_score,
        text_overlap=text_overlap,
        completeness=completeness,
        crop=crop,
        component_score=component_score,
        quality_passed=quality_passed,
    )
