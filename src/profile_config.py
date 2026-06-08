"""
Per-document crop profile configuration tables.

Replaces a single global threshold set with profile-specific bundles keyed by
``CropProfile``.  ``PageProfile`` (from page classification) maps to a default
crop profile; ``resolve_crop_profile`` may refine that choice from page metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from src import config
from src.logger import get_logger
from src.ocr.ocr_engine import OcrPageResult

log = get_logger(__name__)
_PROFILE_FILE_HANDLER_ATTACHED = False


class CropProfile(str, Enum):
    """Crop-strategy profile — finer-grained than page classification."""

    CAD_COMPACT = "cad_compact"
    CAD_WIDE = "cad_wide"
    BLUEPRINT_LARGE = "blueprint_large"
    SCANNED_PDF = "scanned_pdf"
    MIXED_DATASHEET = "mixed_datasheet"
    PHOTO_SHEET = "photo_sheet"
    SIMPLE_RASTER = "simple_raster"
    DIGITAL_PDF = "digital_pdf"
    TEXT_HEAVY = "text_heavy"


@dataclass(frozen=True)
class ProfileConfig:
    """Threshold bundle used by crop validation, refinement, and fusion."""

    crop_profile: CropProfile

    # --- Crop validation ---
    min_crop_width_ratio: float
    min_crop_height_ratio: float
    min_crop_area_ratio: float
    max_crop_aspect_ratio: float
    min_crop_aspect_ratio: float
    max_figure_output_area_ratio: float
    min_line_art_density: float

    # --- Fusion / selection ---
    primary_figure_min_confidence: float
    primary_figure_max_text_overlap: float
    primary_figure_min_completeness: float
    method_priors: dict[str, float] = field(default_factory=dict)
    area_penalty_threshold: float = 0.48

    # --- Embedded PDF images ---
    embedded_fusion_max_area_ratio: float = 0.14
    embedded_fusion_min_page_dimension_ratio: float = 0.42

    # --- Line-art tightener ---
    line_art_canny_low: int = 30
    line_art_canny_high: int = 100
    line_art_tighten_search_ratio: float = 0.10
    line_art_tighten_wide_vertical_search_ratio: float = 0.28
    line_art_tighten_margin_ratio: float = 0.03
    line_art_tighten_min_margin_px: int = 12
    line_art_row_col_thresh_frac: float = 0.018
    tighten_min_retained_area_frac: float = 0.52

    # --- Post-crop text shrink ---
    text_shrink_target_overlap: float = 0.10
    text_shrink_step_ratio: float = 0.025
    text_shrink_max_iterations: int = 16
    text_shrink_min_remaining_ratio: float = 0.45

    # --- Page classification helpers ---
    page_text_heavy_coverage_ratio: float = 0.42
    engineering_edge_density_min: float = 0.02

    # --- Fixed layout zones (notes / title block) ---
    notes_block_width_ratio: float = 0.42
    notes_block_height_ratio: float = 0.28
    title_block_width_ratio: float = 0.48
    bottom_annotation_band_ratio: float = 0.42
    drawing_zone_max_bottom_ratio: float = 0.78

    @property
    def uses_engineering_crop_limits(self) -> bool:
        return self.crop_profile not in (
            CropProfile.PHOTO_SHEET,
            CropProfile.DIGITAL_PDF,
            CropProfile.TEXT_HEAVY,
        )


def _priors(
    morphology: float,
    projection: float,
    embedded: float,
    layout: float,
    photo: float,
) -> dict[str, float]:
    return {
        "morphology": morphology,
        "projection": projection,
        "embedded": embedded,
        "layout": layout,
        "photo": photo,
    }


# ---------------------------------------------------------------------------
# Profile tables — each row is a complete threshold bundle.
# ---------------------------------------------------------------------------

PROFILE_TABLE: dict[CropProfile, ProfileConfig] = {
    CropProfile.CAD_COMPACT: ProfileConfig(
        crop_profile=CropProfile.CAD_COMPACT,
        min_crop_width_ratio=0.32,
        min_crop_height_ratio=0.20,
        min_crop_area_ratio=0.12,
        max_crop_aspect_ratio=3.8,
        min_crop_aspect_ratio=0.28,
        max_figure_output_area_ratio=0.55,
        min_line_art_density=0.012,
        primary_figure_min_confidence=0.38,
        primary_figure_max_text_overlap=0.22,
        primary_figure_min_completeness=0.24,
        method_priors=_priors(1.0, 0.95, 0.70, 0.72, 0.50),
        area_penalty_threshold=0.45,
        embedded_fusion_max_area_ratio=0.12,
        text_shrink_target_overlap=0.08,
    ),
    CropProfile.CAD_WIDE: ProfileConfig(
        crop_profile=CropProfile.CAD_WIDE,
        min_crop_width_ratio=0.28,
        min_crop_height_ratio=0.10,
        min_crop_area_ratio=0.08,
        max_crop_aspect_ratio=8.0,
        min_crop_aspect_ratio=0.18,
        max_figure_output_area_ratio=0.48,
        min_line_art_density=0.008,
        primary_figure_min_confidence=0.33,
        primary_figure_max_text_overlap=0.18,
        primary_figure_min_completeness=0.18,
        method_priors=_priors(1.0, 0.96, 0.72, 0.68, 0.48),
        area_penalty_threshold=0.42,
        line_art_tighten_wide_vertical_search_ratio=0.22,
        line_art_tighten_search_ratio=0.08,
        tighten_min_retained_area_frac=0.32,
        text_shrink_target_overlap=0.07,
        text_shrink_max_iterations=20,
        title_block_width_ratio=0.55,
        bottom_annotation_band_ratio=0.40,
        notes_block_width_ratio=0.36,
        notes_block_height_ratio=0.24,
    ),
    CropProfile.BLUEPRINT_LARGE: ProfileConfig(
        crop_profile=CropProfile.BLUEPRINT_LARGE,
        min_crop_width_ratio=0.28,
        min_crop_height_ratio=0.14,
        min_crop_area_ratio=0.10,
        max_crop_aspect_ratio=6.5,
        min_crop_aspect_ratio=0.25,
        max_figure_output_area_ratio=0.62,
        min_line_art_density=0.009,
        primary_figure_min_confidence=0.34,
        primary_figure_max_text_overlap=0.22,
        primary_figure_min_completeness=0.20,
        method_priors=_priors(1.0, 0.94, 0.80, 0.74, 0.52),
        area_penalty_threshold=0.50,
        embedded_fusion_max_area_ratio=0.14,
        bottom_annotation_band_ratio=0.44,
        title_block_width_ratio=0.55,
        drawing_zone_max_bottom_ratio=0.76,
        text_shrink_target_overlap=0.09,
        text_shrink_max_iterations=18,
    ),
    CropProfile.SCANNED_PDF: ProfileConfig(
        crop_profile=CropProfile.SCANNED_PDF,
        min_crop_width_ratio=0.30,
        min_crop_height_ratio=0.16,
        min_crop_area_ratio=0.11,
        max_crop_aspect_ratio=5.5,
        min_crop_aspect_ratio=0.26,
        max_figure_output_area_ratio=0.60,
        min_line_art_density=0.008,
        primary_figure_min_confidence=0.33,
        primary_figure_max_text_overlap=0.30,
        primary_figure_min_completeness=0.18,
        method_priors=_priors(0.95, 0.98, 0.55, 0.76, 0.50),
        area_penalty_threshold=0.50,
        embedded_fusion_max_area_ratio=0.10,
        embedded_fusion_min_page_dimension_ratio=0.38,
        line_art_canny_low=25,
        line_art_canny_high=90,
        text_shrink_target_overlap=0.14,
        text_shrink_max_iterations=20,
    ),
    CropProfile.MIXED_DATASHEET: ProfileConfig(
        crop_profile=CropProfile.MIXED_DATASHEET,
        min_crop_width_ratio=0.30,
        min_crop_height_ratio=0.16,
        min_crop_area_ratio=0.11,
        max_crop_aspect_ratio=5.0,
        min_crop_aspect_ratio=0.26,
        max_figure_output_area_ratio=0.58,
        min_line_art_density=0.010,
        primary_figure_min_confidence=0.36,
        primary_figure_max_text_overlap=0.26,
        primary_figure_min_completeness=0.22,
        method_priors=_priors(0.96, 0.92, 0.65, 0.80, 0.58),
        text_shrink_target_overlap=0.12,
        text_shrink_max_iterations=18,
        notes_block_width_ratio=0.38,
        bottom_annotation_band_ratio=0.38,
    ),
    CropProfile.PHOTO_SHEET: ProfileConfig(
        crop_profile=CropProfile.PHOTO_SHEET,
        min_crop_width_ratio=0.28,
        min_crop_height_ratio=0.18,
        min_crop_area_ratio=0.10,
        max_crop_aspect_ratio=4.2,
        min_crop_aspect_ratio=0.30,
        max_figure_output_area_ratio=0.55,
        min_line_art_density=0.006,
        primary_figure_min_confidence=0.36,
        primary_figure_max_text_overlap=0.20,
        primary_figure_min_completeness=0.22,
        method_priors=_priors(0.60, 0.65, 0.88, 0.78, 1.0),
        area_penalty_threshold=0.50,
        embedded_fusion_max_area_ratio=0.18,
    ),
    CropProfile.SIMPLE_RASTER: ProfileConfig(
        crop_profile=CropProfile.SIMPLE_RASTER,
        min_crop_width_ratio=0.26,
        min_crop_height_ratio=0.14,
        min_crop_area_ratio=0.08,
        max_crop_aspect_ratio=7.5,
        min_crop_aspect_ratio=0.20,
        max_figure_output_area_ratio=0.70,
        min_line_art_density=0.008,
        primary_figure_min_confidence=0.33,
        primary_figure_max_text_overlap=0.28,
        primary_figure_min_completeness=0.18,
        method_priors=_priors(1.0, 0.93, 0.72, 0.68, 0.55),
        area_penalty_threshold=0.58,
        line_art_tighten_wide_vertical_search_ratio=0.30,
    ),
    CropProfile.DIGITAL_PDF: ProfileConfig(
        crop_profile=CropProfile.DIGITAL_PDF,
        min_crop_width_ratio=0.32,
        min_crop_height_ratio=0.20,
        min_crop_area_ratio=0.12,
        max_crop_aspect_ratio=3.8,
        min_crop_aspect_ratio=0.28,
        max_figure_output_area_ratio=0.50,
        min_line_art_density=0.012,
        primary_figure_min_confidence=0.40,
        primary_figure_max_text_overlap=0.18,
        primary_figure_min_completeness=0.24,
        method_priors=_priors(0.70, 0.72, 0.95, 0.80, 0.85),
        embedded_fusion_max_area_ratio=0.20,
        embedded_fusion_min_page_dimension_ratio=0.50,
    ),
    CropProfile.TEXT_HEAVY: ProfileConfig(
        crop_profile=CropProfile.TEXT_HEAVY,
        min_crop_width_ratio=0.40,
        min_crop_height_ratio=0.30,
        min_crop_area_ratio=0.20,
        max_crop_aspect_ratio=3.0,
        min_crop_aspect_ratio=0.35,
        max_figure_output_area_ratio=0.40,
        min_line_art_density=0.015,
        primary_figure_min_confidence=0.55,
        primary_figure_max_text_overlap=0.12,
        primary_figure_min_completeness=0.30,
        method_priors=_priors(0.50, 0.50, 0.40, 0.45, 0.40),
        page_text_heavy_coverage_ratio=0.38,
    ),
}


_PAGE_TO_CROP: dict[str, CropProfile] = {
    "engineering_sheet": CropProfile.BLUEPRINT_LARGE,
    "simple_image": CropProfile.SIMPLE_RASTER,
    "photo_datasheet": CropProfile.PHOTO_SHEET,
    "digital_pdf": CropProfile.DIGITAL_PDF,
    "text_heavy": CropProfile.TEXT_HEAVY,
    "mixed": CropProfile.MIXED_DATASHEET,
}


def _setup_profile_config_file_log() -> None:
    """Attach a file handler so profile assignments are easy to grep per image."""
    global _PROFILE_FILE_HANDLER_ATTACHED
    if _PROFILE_FILE_HANDLER_ATTACHED or not config.PROFILE_CONFIG_LOG_ENABLED:
        return

    log_path = Path(config.PROFILE_CONFIG_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    log.addHandler(handler)
    _PROFILE_FILE_HANDLER_ATTACHED = True
    log.info("Profile config file logging enabled: %s", log_path.resolve())


def profile_config_summary(pcfg: ProfileConfig) -> dict[str, float | int | str]:
    """Key thresholds from a profile bundle (for logging / debugging)."""
    return {
        "crop_profile": pcfg.crop_profile.value,
        "max_crop_aspect_ratio": pcfg.max_crop_aspect_ratio,
        "min_crop_height_ratio": pcfg.min_crop_height_ratio,
        "max_figure_output_area_ratio": pcfg.max_figure_output_area_ratio,
        "min_line_art_density": pcfg.min_line_art_density,
        "primary_figure_min_confidence": pcfg.primary_figure_min_confidence,
        "primary_figure_max_text_overlap": pcfg.primary_figure_max_text_overlap,
        "embedded_fusion_max_area_ratio": pcfg.embedded_fusion_max_area_ratio,
        "text_shrink_target_overlap": pcfg.text_shrink_target_overlap,
    }


def log_page_profile_assignment(
    *,
    source_path: Path | str,
    page_index: int,
    page_profile: str,
    crop_profile: CropProfile,
    pcfg: ProfileConfig,
    page_size: tuple[int, int] | None = None,
    seed_bbox_xyxy: tuple[int, int, int, int] | None = None,
    embedded_count: int = 0,
    figure_method: str | None = None,
    figure_emitted: bool = True,
) -> None:
    """
    Log which image/page uses which ProfileConfig bundle.

    Writes to the console (via root logger) and to ``logs/profile_config.log``.
    """
    _setup_profile_config_file_log()

    source = Path(source_path)
    seed_aspect = None
    if seed_bbox_xyxy is not None:
        x0, y0, x1, y1 = seed_bbox_xyxy
        bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
        seed_aspect = round(bw / bh, 3)

    size_str = f"{page_size[0]}x{page_size[1]}" if page_size else "unknown"
    summary = profile_config_summary(pcfg)

    log.info(
        "PROFILE_CONFIG | file=%s | page=%d | size=%s | page_profile=%s | "
        "crop_profile=%s | embedded=%d | seed_aspect=%s | figure_method=%s | "
        "figure_emitted=%s | max_aspect=%.2f | max_area=%.2f | max_text_overlap=%.2f | "
        "min_confidence=%.2f",
        source.name,
        page_index,
        size_str,
        page_profile,
        crop_profile.value,
        embedded_count,
        seed_aspect if seed_aspect is not None else "n/a",
        figure_method or "none",
        figure_emitted,
        pcfg.max_crop_aspect_ratio,
        pcfg.max_figure_output_area_ratio,
        pcfg.primary_figure_max_text_overlap,
        pcfg.primary_figure_min_confidence,
    )
    log.debug("PROFILE_CONFIG detail | file=%s | page=%d | %s", source.name, page_index, summary)


def get_profile_config(crop_profile: CropProfile) -> ProfileConfig:
    """Return the config bundle for a crop profile."""
    return PROFILE_TABLE[crop_profile]


def default_crop_profile(page_profile: str | object) -> CropProfile:
    """Map coarse page classification to a default crop profile."""
    key = page_profile.value if hasattr(page_profile, "value") else str(page_profile)
    return _PAGE_TO_CROP.get(key, CropProfile.MIXED_DATASHEET)


def _profile_key(profile: CropProfile | str | object | None) -> str | None:
    if profile is None:
        return None
    if isinstance(profile, CropProfile):
        return profile.value
    if hasattr(profile, "value"):
        return str(profile.value)
    return str(profile)


def resolve_profile_config(
    profile: CropProfile | str | object | None,
) -> ProfileConfig:
    """
    Resolve any profile identifier passed through the pipeline to a config bundle.
    """
    if isinstance(profile, CropProfile):
        return get_profile_config(profile)
    if isinstance(profile, ProfileConfig):
        return profile
    key = _profile_key(profile)
    if key in {p.value for p in CropProfile}:
        return get_profile_config(CropProfile(key))
    if key in _PAGE_TO_CROP:
        return get_profile_config(_PAGE_TO_CROP[key])
    return get_profile_config(CropProfile.MIXED_DATASHEET)


def _edge_density(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges) / max(edges.size, 1))


def _mean_saturation(image_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


def resolve_crop_profile(
    page_profile: str | object,
    image_bgr: np.ndarray,
    ocr: OcrPageResult | None = None,
    *,
    seed_bbox_xyxy: tuple[int, int, int, int] | None = None,
    is_pdf: bool = False,
    embedded_on_page: int = 0,
    page_count: int = 1,
    source_suffix: str = "",
) -> CropProfile:
    """
    Refine page profile into a crop profile using page metrics and optional seed bbox.
    """
    page_key = page_profile.value if hasattr(page_profile, "value") else str(page_profile)
    base = default_crop_profile(page_profile)

    if page_key == "text_heavy":
        return CropProfile.TEXT_HEAVY

    suffix = source_suffix.lower()
    ph, pw = image_bgr.shape[:2]
    page_aspect = pw / max(ph, 1)
    ocr_box_count = len(ocr.boxes) if ocr is not None else 0

    if page_count == 1 and suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        if page_key == "simple_image" and ocr_box_count < 12:
            base = CropProfile.SIMPLE_RASTER
        elif page_key == "engineering_sheet":
            base = CropProfile.BLUEPRINT_LARGE

    if seed_bbox_xyxy is not None:
        x0, y0, x1, y1 = seed_bbox_xyxy
        bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
        seed_aspect = bw / bh
        height_ratio = bh / max(ph, 1)

        # Wide shallow side views (fuselage, long profiles) on landscape sheets.
        if (
            page_aspect >= 1.2
            and seed_aspect >= 2.2
            and base
            in (
                CropProfile.BLUEPRINT_LARGE,
                CropProfile.SIMPLE_RASTER,
                CropProfile.MIXED_DATASHEET,
            )
        ):
            return CropProfile.CAD_WIDE
        # Moderately wide content with limited vertical ink extent.
        if seed_aspect >= 2.15 and height_ratio < 0.58 and base in (
            CropProfile.BLUEPRINT_LARGE,
            CropProfile.SIMPLE_RASTER,
            CropProfile.MIXED_DATASHEET,
        ):
            return CropProfile.CAD_WIDE
        if seed_aspect >= 3.8 and base in (
            CropProfile.BLUEPRINT_LARGE,
            CropProfile.SIMPLE_RASTER,
            CropProfile.MIXED_DATASHEET,
        ):
            return CropProfile.CAD_WIDE
        if seed_aspect <= 0.55 and height_ratio >= 0.35:
            return CropProfile.CAD_COMPACT

    if is_pdf and embedded_on_page > 0:
        if _mean_saturation(image_bgr) < 15 and _edge_density(image_bgr) >= 0.006:
            return CropProfile.SCANNED_PDF
        return CropProfile.DIGITAL_PDF

    if base == CropProfile.BLUEPRINT_LARGE and page_aspect >= 1.25:
        return CropProfile.BLUEPRINT_LARGE

    if base == CropProfile.SIMPLE_RASTER and page_aspect >= 1.4:
        return CropProfile.CAD_WIDE

    return base
