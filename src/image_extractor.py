"""
Figure extraction: layout figures, manufacturing photos, embedded PDF images,
and primary-figure fusion (one ranked output per page).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from src import config
from src.extract_figure import (
    FigureSelectionResult,
    estimate_text_coverage_in_bbox,
    finalize_crop_bounds,
    find_main_drawing_bbox_via_projection,
    is_valid_figure_crop,
    line_art_density_in_bbox,
    masked_non_text_edges,
    prepare_primary_crop,
    _pad_detection_bbox_xyxy,
)
from src.figure_fusion import (
    FigureCandidate,
    FigureType,
    PageProfile,
    build_candidate_from_crop,
    classify_page,
    crop_completeness_score,
    score_candidate,
    select_primary_candidate,
)
from src.profile_config import (
    CropProfile,
    ProfileConfig,
    effective_max_figure_output_area,
    get_profile_config,
    log_page_profile_assignment,
    resolve_crop_profile,
    resolve_effective_profile_config,
)
from src.logger import get_logger
from src.ocr.ocr_engine import OcrPageResult, TextBox
from src.utils import bbox_iou, save_bgr
from src.utils.image_metrics import page_edge_density, page_mean_saturation

log = get_logger(__name__)


@dataclass(frozen=True)
class ExtractedFigure:
    source_path: Path
    page_index: int
    figure_index: int
    image_path: Path
    method: str
    bbox: tuple[int, int, int, int] | None
    area_ratio: float
    figure_type: str = "line_art"
    confidence: float = 0.0
    page_profile: str = ""
    crop_profile: str = ""
    text_overlap: float = 0.0
    completeness: float = 0.0
    is_primary: bool = True


@dataclass(frozen=True)
class MorphologyPageCandidate:
    crop: np.ndarray
    bbox: tuple[int, int, int, int]
    selection: FigureSelectionResult


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


def primary_figure_path(
    output_dir: Path,
    page_index: int,
    page_count: int,
    *,
    source_stem: str | None = None,
) -> Path:
    """One file per page; single-page documents use a stable name."""
    img_dir = output_dir / "images"
    ext = config.OUTPUT_IMAGE_FORMAT
    if page_count == 1:
        return img_dir / f"primary_figure.{ext}"
    return img_dir / f"primary_figure_p{page_index:03d}.{ext}"


# ---------------------------------------------------------------------------
# Embedded PDF images
# ---------------------------------------------------------------------------


def _skip_embedded_for_fusion(
    img_width: int,
    img_height: int,
    page_width: int,
    page_height: int,
    area_ratio: float,
    *,
    pcfg: ProfileConfig | None = None,
) -> bool:
    """Omit full-page embedded rasters so raster line-art crops are preferred."""
    active = pcfg or get_profile_config(CropProfile.SCANNED_PDF)
    if area_ratio > active.embedded_fusion_max_area_ratio:
        return True
    if page_width > 0 and page_height > 0:
        dim_ratio = active.embedded_fusion_min_page_dimension_ratio
        if (
            img_width >= page_width * dim_ratio
            or img_height >= page_height * dim_ratio
        ) and area_ratio > 0.07:
            return True
    return False


def embedded_image_counts_by_page(
    pdf_path: Path,
    *,
    page_sizes: list[tuple[int, int]] | None = None,
    min_bytes: int | None = None,
    full_page_only: bool = True,
) -> dict[int, int]:
    """
    Count embedded raster images per PDF page (above size threshold).

    When ``full_page_only`` is True (default), only full-page scan embeds are
    counted — small logos or partial figures do not trigger scanned-PDF routing.

    Pass the per-page count as ``embedded_on_page`` to :func:`classify_page` and
    :func:`resolve_crop_profile` (see :func:`is_full_page_embedded_pdf`).
    """
    threshold = min_bytes if min_bytes is not None else config.MIN_EMBEDDED_IMAGE_BYTES
    counts: dict[int, int] = {}
    doc = fitz.open(pdf_path)
    try:
        for page_i, page in enumerate(doc):
            pw, ph = (0, 0)
            if page_sizes and page_i < len(page_sizes):
                pw, ph = page_sizes[page_i]
            n = 0
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base = doc.extract_image(xref)
                if not base or len(base.get("image", b"")) < threshold:
                    continue
                iw, ih = base.get("width", 0), base.get("height", 0)
                if full_page_only and pw and ph:
                    if not _is_full_page_embedded(iw, ih, pw, ph):
                        continue
                n += 1
            if n:
                counts[page_i] = n
    finally:
        doc.close()
    return counts


def _is_full_page_embedded(
    img_width: int,
    img_height: int,
    page_width: int,
    page_height: int,
) -> bool:
    if page_width <= 0 or page_height <= 0:
        return False
    ratio = config.FULL_PAGE_EMBEDDED_RATIO
    if img_width >= page_width * ratio:
        return True
    page_area = page_width * page_height
    return (img_width * img_height) >= page_area * 0.35


def _embedded_candidates_by_page(
    pdf_path: Path,
    page_sizes: list[tuple[int, int]],
    *,
    pcfg_by_page: dict[int, ProfileConfig] | None = None,
) -> dict[int, list[FigureCandidate]]:
    threshold = config.MIN_EMBEDDED_IMAGE_BYTES
    skip_full = config.SKIP_FULL_PAGE_EMBEDDED
    by_page: dict[int, list[FigureCandidate]] = {}

    doc = fitz.open(pdf_path)
    try:
        for page_i, page in enumerate(doc):
            pw, ph = (0, 0)
            if page_i < len(page_sizes):
                pw, ph = page_sizes[page_i]
            page_cands: list[FigureCandidate] = []
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base = doc.extract_image(xref)
                if not base or len(base.get("image", b"")) < threshold:
                    continue
                iw, ih = base.get("width", 0), base.get("height", 0)
                if skip_full and _is_full_page_embedded(iw, ih, pw, ph):
                    continue
                page_area = max(pw * ph, 1)
                area_ratio = (iw * ih) / page_area if pw and ph else 0.3
                page_pcfg = (pcfg_by_page or {}).get(page_i)
                if _skip_embedded_for_fusion(
                    iw, ih, pw, ph, area_ratio, pcfg=page_pcfg
                ):
                    log.debug(
                        "Skipping embedded scan for fusion page %d (%dx%d, %.1f%% of page)",
                        page_i,
                        iw,
                        ih,
                        100 * area_ratio,
                    )
                    continue
                page_cands.append(
                    FigureCandidate(
                        page_index=page_i,
                        method="embedded",
                        figure_type=FigureType.EMBEDDED,
                        bbox=None,
                        area_ratio=area_ratio,
                        gate_score=0.72,
                        text_overlap=0.0,
                        completeness=1.0,
                        embedded_bytes=base["image"],
                        embedded_ext=base.get("ext", "png"),
                        quality_passed=True,
                    )
                )
            if page_cands:
                by_page[page_i] = page_cands
    finally:
        doc.close()
    return by_page


# ---------------------------------------------------------------------------
# Layout / photo / projection heuristics
# ---------------------------------------------------------------------------


def _is_valid_photo_bbox(
    bbox: tuple[int, int, int, int],
    page_shape: tuple[int, int],
    profile: PageProfile | None = None,
    *,
    image_bgr: np.ndarray | None = None,
    text_boxes: list[TextBox] | None = None,
    profile_config: ProfileConfig | None = None,
) -> bool:
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    density = None
    if image_bgr is not None:
        density = line_art_density_in_bbox(
            image_bgr, bbox, text_boxes, profile_config=profile_config
        )
    return is_valid_figure_crop(
        (x0, y0, bw, bh),
        page_shape,
        profile,
        line_art_density=density,
        profile_config=profile_config,
    )


def _is_engineering_line_drawing(image_bgr: np.ndarray) -> bool:
    if page_mean_saturation(image_bgr) > config.PHOTO_SATURATION_MAX + 15:
        return False
    return page_edge_density(image_bgr) >= config.ENGINEERING_EDGE_DENSITY_MIN


def _find_layout_figures(
    image_bgr: np.ndarray,
    text_boxes: list[TextBox],
    min_area_ratio: float,
) -> list[tuple[tuple[int, int, int, int], float]]:
    h, w = image_bgr.shape[:2]
    page_area = h * w
    min_area = int(page_area * min_area_ratio)
    combined = masked_non_text_edges(image_bgr, text_boxes)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    candidates: list[tuple[tuple[int, int, int, int], float]] = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < min_area:
            continue
        if area / page_area > config.MAX_FIGURE_OUTPUT_AREA_RATIO + 0.05:
            continue
        aspect = cw / max(ch, 1)
        if aspect < 0.15 or aspect > 8.0:
            continue
        roi = gray[y : y + ch, x : x + cw]
        if roi.size == 0 or float(np.std(roi)) < 8.0:
            continue
        candidates.append(((x, y, x + cw, y + ch), area / page_area))

    candidates.sort(key=lambda c: c[1], reverse=True)
    merged: list[tuple[tuple[int, int, int, int], float]] = []
    for bbox, ratio in candidates:
        if any(bbox_iou(bbox, kept[0]) > 0.5 for kept in merged):
            continue
        merged.append((bbox, ratio))
    return merged


def _refine_bbox_to_photo_frame(
    image_bgr: np.ndarray,
    coarse_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = coarse_bbox
    roi = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    rh, rw = roi.shape[:2]
    if rh < 80 or rw < 80:
        return coarse_bbox

    def peak_in_band(profile: np.ndarray, band: tuple[float, float]) -> int:
        n = len(profile)
        a, b = int(n * band[0]), int(n * band[1])
        if b <= a + 5:
            return a
        return a + int(np.argmax(profile[a:b]))

    cx0, cx1 = int(rw * 0.12), int(rw * 0.88)
    strip = roi[:, cx0:cx1]
    row_mean = strip.mean(axis=1)
    row_std = strip.std(axis=1)

    top = 0
    for y in range(int(rh * 0.08), int(rh * 0.40)):
        if row_mean[y] < 175 and row_std[y] > 32:
            top = y
            break

    bottom = rh - 1
    for y in range(int(rh * 0.42), int(rh * 0.92)):
        if row_mean[y] > 185 and row_std[y] < 45:
            bottom = y
            break

    if bottom <= top + 50:
        return coarse_bbox

    band_gray = roi[top:bottom, :]
    edges = cv2.Canny(band_gray, 35, 110)
    col_e = cv2.GaussianBlur(edges.mean(axis=0).reshape(1, -1), (1, 31), 0).flatten()
    left = peak_in_band(col_e, config.PHOTO_FRAME_EDGE_LEFT_BAND)
    right = peak_in_band(col_e, config.PHOTO_FRAME_EDGE_RIGHT_BAND)

    inset = config.PHOTO_FRAME_INSET_PX
    left = min(max(left + inset, 0), rw - 2)
    top = min(max(top + inset, 0), rh - 2)
    right = max(min(right - inset, rw), left + 1)
    bottom = max(min(bottom - inset, rh), top + 1)

    refined = (x0 + left, y0 + top, x0 + right, y0 + bottom)
    coarse_area = max((x1 - x0) * (y1 - y0), 1)
    refined_area = (refined[2] - refined[0]) * (refined[3] - refined[1])
    if refined_area < coarse_area * config.PHOTO_FRAME_MIN_AREA_VS_COARSE:
        return coarse_bbox
    rw_r = refined[2] - refined[0]
    rh_r = refined[3] - refined[1]
    aspect = rw_r / max(rh_r, 1)
    if aspect < 0.9 or aspect > 2.8:
        return coarse_bbox
    return refined


def _find_manufacturing_photo_bbox(
    image_bgr: np.ndarray,
) -> tuple[tuple[int, int, int, int], float] | None:
    h, w = image_bgr.shape[:2]
    page_area = h * w
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    local_contrast = cv2.absdiff(gray, blur)

    photo_mask = (
        (sat < config.PHOTO_SATURATION_MAX)
        & (local_contrast > 6)
        & (gray > 25)
        & (gray < 250)
    ).astype(np.uint8) * 255

    y0 = int(h * config.PHOTO_SEARCH_TOP)
    y1 = int(h * config.PHOTO_SEARCH_BOTTOM)
    x0 = int(w * config.PHOTO_SEARCH_LEFT)
    x1 = int(w * config.PHOTO_SEARCH_RIGHT)
    band = np.zeros_like(photo_mask)
    band[y0:y1, x0:x1] = 255
    photo_mask = cv2.bitwise_and(photo_mask, band)
    photo_mask = cv2.morphologyEx(
        photo_mask, cv2.MORPH_CLOSE, np.ones((21, 11), np.uint8), iterations=2
    )
    photo_mask = cv2.morphologyEx(
        photo_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
    )

    contours, _ = cv2.findContours(
        photo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(contour)
    area_ratio = (cw * ch) / page_area
    if area_ratio < config.PHOTO_MIN_AREA_RATIO or area_ratio > config.PHOTO_MAX_AREA_RATIO:
        return None

    roi = photo_mask[y : y + ch, x : x + cw]
    thresh = config.PHOTO_PROJECTION_THRESHOLD
    row_den = roi.mean(axis=1) / 255.0
    col_den = roi.mean(axis=0) / 255.0
    rows = np.where(row_den > thresh)[0]
    cols = np.where(col_den > thresh)[0]
    if len(rows) and len(cols):
        ty0, ty1 = y + int(rows[0]), y + int(rows[-1])
        tx0, tx1 = x + int(cols[0]), x + int(cols[-1])
    else:
        ty0, ty1, tx0, tx1 = y, y + ch, x, x + cw

    pad = config.PHOTO_BBOX_PADDING_PX
    bbox = (
        max(0, tx0 - pad),
        max(0, ty0 - pad),
        min(w, tx1 + pad),
        min(h, ty1 + pad),
    )
    if config.PHOTO_FRAME_TRIM_ENABLED:
        bbox = _refine_bbox_to_photo_frame(image_bgr, bbox)
    tight_ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / page_area
    return bbox, tight_ratio


def _collect_projection_candidate(
    page_index: int,
    image_bgr: np.ndarray,
    text_boxes: list[TextBox],
    profile: PageProfile,
    *,
    profile_config: ProfileConfig,
) -> FigureCandidate | None:
    box = find_main_drawing_bbox_via_projection(image_bgr, text_boxes)
    if box is None:
        return None

    x1, y1, x2, y2 = _pad_detection_bbox_xyxy(box, image_bgr.shape[:2])
    prepared = prepare_primary_crop(
        image_bgr,
        x1,
        y1,
        x2,
        y2,
        text_boxes,
        profile=profile,
        profile_config=profile_config,
    )
    if prepared is None:
        return None

    crop, bbox_xyxy = prepared
    return build_candidate_from_crop(
        page_index,
        "projection",
        FigureType.LINE_ART,
        image_bgr,
        crop,
        bbox_xyxy,
        text_boxes,
        gate_score=0.58,
        quality_passed=True,
    )


def _collect_photo_candidate(
    page_index: int,
    image_bgr: np.ndarray,
    text_boxes: list[TextBox],
    profile: PageProfile,
    *,
    profile_config: ProfileConfig,
) -> FigureCandidate | None:
    if _is_engineering_line_drawing(image_bgr):
        return None

    found = _find_manufacturing_photo_bbox(image_bgr)
    if not found:
        return None

    bbox, _area_ratio = found
    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = finalize_crop_bounds(image_bgr, x0, y0, x1, y1, text_boxes)
    prepared = prepare_primary_crop(
        image_bgr,
        x0,
        y0,
        x1,
        y1,
        text_boxes,
        profile=profile,
        profile_config=profile_config,
    )
    if prepared is None:
        return None

    crop, final_bbox = prepared
    cov = estimate_text_coverage_in_bbox(final_bbox, text_boxes)
    if cov > config.PHOTO_MAX_TEXT_COVERAGE:
        return None

    return build_candidate_from_crop(
        page_index,
        "photo",
        FigureType.PHOTO,
        image_bgr,
        crop,
        final_bbox,
        text_boxes,
        gate_score=0.62,
        quality_passed=True,
    )


def _collect_layout_candidate(
    page_index: int,
    image_bgr: np.ndarray,
    text_boxes: list[TextBox],
    profile: PageProfile,
    *,
    profile_config: ProfileConfig,
) -> FigureCandidate | None:
    regions = _find_layout_figures(image_bgr, text_boxes, config.MIN_FIGURE_AREA_RATIO)
    if not regions:
        return None

    bbox, _ratio = regions[0]
    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = finalize_crop_bounds(image_bgr, x0, y0, x1, y1, text_boxes)
    prepared = prepare_primary_crop(
        image_bgr,
        x0,
        y0,
        x1,
        y1,
        text_boxes,
        profile=profile,
        profile_config=profile_config,
    )
    if prepared is None:
        return None

    crop, final_bbox = prepared
    return build_candidate_from_crop(
        page_index,
        "layout",
        FigureType.LINE_ART,
        image_bgr,
        crop,
        final_bbox,
        text_boxes,
        gate_score=0.5,
        quality_passed=False,
    )


def _collect_morphology_candidate(
    page_index: int,
    image_bgr: np.ndarray,
    text_boxes: list[TextBox],
    morph: MorphologyPageCandidate | None,
    profile: PageProfile,
    *,
    profile_config: ProfileConfig,
) -> FigureCandidate | None:
    if morph is None:
        return None

    x1, y1, x2, y2 = morph.bbox
    page_area = image_bgr.shape[0] * image_bgr.shape[1]
    morph_area = (x2 - x1) * (y2 - y1) / max(page_area, 1)
    max_area = effective_max_figure_output_area(profile_config)
    if morph_area > max_area * 0.88:
        log.debug(
            "Morphology area %.1f%% too large on page %d — deferring to other extractors",
            100 * morph_area,
            page_index,
        )
        return None

    prepared = prepare_primary_crop(
        image_bgr,
        x1,
        y1,
        x2,
        y2,
        text_boxes,
        profile=profile,
        profile_config=profile_config,
    )
    if prepared is None:
        return None

    crop, bbox_xyxy = prepared
    sel = morph.selection
    return build_candidate_from_crop(
        page_index,
        "morphology",
        FigureType.LINE_ART,
        image_bgr,
        crop,
        bbox_xyxy,
        text_boxes,
        gate_score=sel.gate_score,
        component_score=sel.component_score,
        quality_passed=sel.quality.passed,
    )


def _collect_page_candidates(
    page_index: int,
    image_bgr: np.ndarray,
    ocr: OcrPageResult,
    profile: PageProfile,
    morph: MorphologyPageCandidate | None,
    embedded: list[FigureCandidate],
    *,
    profile_config: ProfileConfig,
) -> list[FigureCandidate]:
    text_boxes = ocr.boxes
    candidates: list[FigureCandidate] = []

    morph_cand = _collect_morphology_candidate(
        page_index,
        image_bgr,
        text_boxes,
        morph,
        profile,
        profile_config=profile_config,
    )
    if morph_cand is not None:
        candidates.append(morph_cand)

    proj = _collect_projection_candidate(
        page_index,
        image_bgr,
        text_boxes,
        profile,
        profile_config=profile_config,
    )
    if proj is not None:
        candidates.append(proj)

    photo = _collect_photo_candidate(
        page_index,
        image_bgr,
        text_boxes,
        profile,
        profile_config=profile_config,
    )
    if photo is not None:
        candidates.append(photo)

    layout = _collect_layout_candidate(
        page_index,
        image_bgr,
        text_boxes,
        profile,
        profile_config=profile_config,
    )
    if layout is not None:
        candidates.append(layout)

    candidates.extend(embedded)
    return candidates


def _save_primary_candidate(
    source_path: Path,
    candidate: FigureCandidate,
    out_path: Path,
    profile: PageProfile,
    *,
    crop_profile: CropProfile,
    profile_config: ProfileConfig,
) -> ExtractedFigure:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if candidate.embedded_bytes:
        ext = candidate.embedded_ext.lstrip(".") or config.OUTPUT_IMAGE_FORMAT
        if ext != out_path.suffix.lstrip("."):
            out_path = out_path.with_suffix(f".{ext}")
        out_path.write_bytes(candidate.embedded_bytes)
    elif candidate.crop is not None:
        save_bgr(out_path, candidate.crop)
    else:
        raise ValueError(f"Candidate {candidate.method} has no image data")

    confidence = score_candidate(candidate, profile, pcfg=profile_config)
    return ExtractedFigure(
        source_path=source_path,
        page_index=candidate.page_index,
        figure_index=0,
        image_path=out_path,
        method=candidate.method,
        bbox=candidate.bbox,
        area_ratio=candidate.area_ratio,
        figure_type=candidate.figure_type.value,
        confidence=round(confidence, 4),
        page_profile=profile.value,
        crop_profile=crop_profile.value,
        text_overlap=round(candidate.text_overlap, 4),
        completeness=round(candidate.completeness, 4),
        is_primary=True,
    )


def extract_primary_figures(
    source_path: Path,
    pages: list,
    ocr_results: list[OcrPageResult],
    output_dir: Path,
    *,
    morphology_by_page: dict[int, MorphologyPageCandidate] | None = None,
) -> list[ExtractedFigure]:
    """
    Fuse all extractors per page and emit exactly one primary figure per page
    (when confidence thresholds pass). Single-page PNG/TIFF → one file;
    multi-page PDF/TIFF → one file per page.
    """
    morphology_by_page = morphology_by_page or {}
    page_count = len(pages)
    page_sizes = [(p.width, p.height) for p in pages]
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    is_pdf = source_path.suffix.lower() == ".pdf"
    embedded_counts = (
        embedded_image_counts_by_page(source_path, page_sizes=page_sizes)
        if is_pdf
        else {}
    )
    page_profiles: dict[int, PageProfile] = {}
    crop_profiles: dict[int, CropProfile] = {}
    pcfg_by_page: dict[int, ProfileConfig] = {}
    scale_by_page: dict[int, dict] = {}

    for page, ocr in zip(pages, ocr_results):
        embedded_on_page = embedded_counts.get(page.page_index, 0)
        profile = classify_page(
            page.image,
            ocr,
            page_count=page_count,
            embedded_on_page=embedded_on_page,
            is_pdf=is_pdf,
        )
        page_profiles[page.page_index] = profile
        morph = morphology_by_page.get(page.page_index)
        seed_bbox = morph.bbox if morph is not None else None
        crop_profile = resolve_crop_profile(
            profile,
            page.image,
            ocr,
            seed_bbox_xyxy=seed_bbox,
            is_pdf=is_pdf,
            embedded_on_page=embedded_on_page,
            page_count=page_count,
            source_suffix=source_path.suffix,
        )
        crop_profiles[page.page_index] = crop_profile
        pcfg, _metrics, scale_summary = resolve_effective_profile_config(
            crop_profile,
            page.image,
            ocr,
            seed_bbox_xyxy=seed_bbox,
        )
        pcfg_by_page[page.page_index] = pcfg
        scale_by_page[page.page_index] = scale_summary

    embedded_by_page: dict[int, list[FigureCandidate]] = {}
    if is_pdf:
        embedded_by_page = _embedded_candidates_by_page(
            source_path, page_sizes, pcfg_by_page=pcfg_by_page
        )

    primary_figures: list[ExtractedFigure] = []

    for page, ocr in zip(pages, ocr_results):
        profile = page_profiles[page.page_index]
        crop_profile = crop_profiles[page.page_index]
        pcfg = pcfg_by_page[page.page_index]
        embedded_count = len(embedded_by_page.get(page.page_index, []))
        morph = morphology_by_page.get(page.page_index)
        seed_bbox = morph.bbox if morph is not None else None

        candidates = _collect_page_candidates(
            page.page_index,
            page.image,
            ocr,
            profile,
            morphology_by_page.get(page.page_index),
            embedded_by_page.get(page.page_index, []),
            profile_config=pcfg,
        )

        winner = select_primary_candidate(
            candidates, profile, pcfg=pcfg, crop_profile=crop_profile
        )

        log_page_profile_assignment(
            source_path=source_path,
            page_index=page.page_index,
            page_profile=profile.value,
            crop_profile=crop_profile,
            pcfg=pcfg,
            page_size=(page.width, page.height),
            seed_bbox_xyxy=seed_bbox,
            embedded_count=embedded_count,
            figure_method=winner.method if winner else None,
            figure_emitted=winner is not None,
            scale_summary=scale_by_page.get(page.page_index),
        )

        if winner is None:
            log.info("Page %d: no primary figure emitted", page.page_index)
            continue

        out_path = primary_figure_path(output_dir, page.page_index, page_count)
        primary_figures.append(
            _save_primary_candidate(
                source_path,
                winner,
                out_path,
                profile,
                crop_profile=crop_profile,
                profile_config=pcfg,
            )
        )

    log.info(
        "Primary figure extraction: %d/%d page(s) with output",
        len(primary_figures),
        page_count,
    )
    return primary_figures


# Backward-compatible alias used by older imports.
def extract_all_figures(
    source_path: Path,
    pages: list,
    ocr_results: list,
    output_dir: Path,
    *,
    morphology_by_page: dict[int, Path | str] | None = None,
) -> list[ExtractedFigure]:
    """Deprecated: use extract_primary_figures with MorphologyPageCandidate."""
    log.warning(
        "extract_all_figures called with legacy morphology paths — "
        "run extract_primary_figures from the pipeline instead"
    )
    return extract_primary_figures(
        source_path,
        pages,
        ocr_results,
        output_dir,
        morphology_by_page={},
    )
