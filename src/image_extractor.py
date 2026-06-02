"""
Figure extraction: layout figures, manufacturing photos, embedded PDF images,
and region-grid helpers for large-format OCR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from src import config
from src.extract_figure import (
    estimate_text_coverage_in_bbox,
    masked_non_text_edges,
)
from src.ocr.ocr_engine import TextBox
from src.utils import bbox_iou, save_bgr


@dataclass(frozen=True)
class ExtractedFigure:
    source_path: Path
    page_index: int
    figure_index: int
    image_path: Path
    method: str  # embedded | layout | photo | morphology
    bbox: tuple[int, int, int, int] | None
    area_ratio: float


# ---------------------------------------------------------------------------
# Embedded PDF images
# ---------------------------------------------------------------------------


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


def extract_embedded_pdf_images(
    pdf_path: Path,
    output_dir: Path,
    page_sizes: list[tuple[int, int]] | None = None,
    min_bytes: int | None = None,
) -> list[ExtractedFigure]:
    threshold = min_bytes or config.MIN_EMBEDDED_IMAGE_BYTES
    skip_full = config.SKIP_FULL_PAGE_EMBEDDED
    figures: list[ExtractedFigure] = []
    doc = fitz.open(pdf_path)
    try:
        fig_idx = 0
        for page_i, page in enumerate(doc):
            pw, ph = (0, 0)
            if page_sizes and page_i < len(page_sizes):
                pw, ph = page_sizes[page_i]
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base = doc.extract_image(xref)
                if not base or len(base.get("image", b"")) < threshold:
                    continue
                iw, ih = base.get("width", 0), base.get("height", 0)
                if skip_full and _is_full_page_embedded(iw, ih, pw, ph):
                    continue
                ext = base.get("ext", "png")
                out_path = (
                    output_dir / f"{pdf_path.stem}_p{page_i:03d}_embedded_{fig_idx:03d}.{ext}"
                )
                out_path.write_bytes(base["image"])
                figures.append(
                    ExtractedFigure(
                        source_path=pdf_path,
                        page_index=page_i,
                        figure_index=fig_idx,
                        image_path=out_path,
                        method="embedded",
                        bbox=None,
                        area_ratio=0.0,
                    )
                )
                fig_idx += 1
    finally:
        doc.close()
    return figures


# ---------------------------------------------------------------------------
# Layout / photo heuristics
# ---------------------------------------------------------------------------


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


def extract_manufacturing_photo(
    source_path: Path,
    page_index: int,
    image_bgr: np.ndarray,
    output_dir: Path,
) -> list[ExtractedFigure]:
    found = _find_manufacturing_photo_bbox(image_bgr)
    if not found:
        return []
    bbox, area_ratio = found
    x0, y0, x1, y1 = bbox
    crop = image_bgr[y0:y1, x0:x1]
    out_path = (
        output_dir
        / f"{source_path.stem}_p{page_index:03d}_manufacturing_photo.{config.OUTPUT_IMAGE_FORMAT}"
    )
    save_bgr(out_path, crop)
    return [
        ExtractedFigure(
            source_path=source_path,
            page_index=page_index,
            figure_index=0,
            image_path=out_path,
            method="photo",
            bbox=bbox,
            area_ratio=area_ratio,
        )
    ]


def extract_layout_figures(
    source_path: Path,
    page_index: int,
    image_bgr: np.ndarray,
    text_boxes: list[TextBox],
    output_dir: Path,
    min_area_ratio: float | None = None,
) -> list[ExtractedFigure]:
    ratio = min_area_ratio or config.MIN_FIGURE_AREA_RATIO
    regions = _find_layout_figures(image_bgr, text_boxes, ratio)
    figures: list[ExtractedFigure] = []
    for idx, (bbox, area_ratio) in enumerate(regions):
        x0, y0, x1, y1 = bbox
        crop = image_bgr[y0:y1, x0:x1]
        out_path = (
            output_dir
            / f"{source_path.stem}_p{page_index:03d}_figure_{idx:03d}.{config.OUTPUT_IMAGE_FORMAT}"
        )
        save_bgr(out_path, crop)
        figures.append(
            ExtractedFigure(
                source_path=source_path,
                page_index=page_index,
                figure_index=idx,
                image_path=out_path,
                method="layout",
                bbox=bbox,
                area_ratio=area_ratio,
            )
        )
    return figures


def extract_all_figures(
    source_path: Path,
    pages: list,
    ocr_results: list,
    output_dir: Path,
    *,
    morphology_by_page: dict[int, Path | str] | None = None,
) -> list[ExtractedFigure]:
    """
    Orchestrate figure extraction: embedded PDF, photo heuristics, layout fallback.

    When morphology succeeded for a page, skip photo/layout to avoid duplicate crops.
    """
    figures: list[ExtractedFigure] = []
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    page_sizes = [(p.width, p.height) for p in pages]
    morphology_by_page = morphology_by_page or {}

    for page_index, morph_path in morphology_by_page.items():
        figures.append(
            ExtractedFigure(
                source_path=source_path,
                page_index=page_index,
                figure_index=0,
                image_path=Path(morph_path),
                method="morphology",
                bbox=None,
                area_ratio=0.0,
            )
        )

    if source_path.suffix.lower() == ".pdf":
        figures.extend(
            extract_embedded_pdf_images(source_path, img_dir, page_sizes=page_sizes)
        )

    for page, ocr in zip(pages, ocr_results):
        if page.page_index in morphology_by_page:
            continue

        photos = extract_manufacturing_photo(
            page.source_path, page.page_index, page.image, img_dir
        )
        if photos:
            bbox = photos[0].bbox
            if bbox is not None:
                cov = estimate_text_coverage_in_bbox(bbox, ocr.boxes)
                if cov > config.PHOTO_MAX_TEXT_COVERAGE:
                    photos = []
        figures.extend(photos)

        if not photos:
            figures.extend(
                extract_layout_figures(
                    page.source_path,
                    page.page_index,
                    page.image,
                    ocr.boxes,
                    img_dir,
                )
            )

    return figures
