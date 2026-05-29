"""Extract manufacturing figures / aircraft images from document pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import pymupdf as fitz
import numpy as np

import config
from src.ocr_engine import TextBox #Imports OCR text box structure.contains text, confidence, and bounding box.


@dataclass(frozen=True)
class ExtractedFigure:
    source_path: Path
    page_index: int
    figure_index: int
    image_path: Path
    method: str  # "embedded" | "layout" | "photo"
    bbox: tuple[int, int, int, int] | None
    area_ratio: float


def _save_bgr(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)


#Detects if the embedded image is actually entire scanned page.
def _is_full_page_embedded(
    img_width: int,
    img_height: int,
    page_width: int,
    page_height: int,
) -> bool:
    """True when the embedded raster is the full scanned page (image-only PDF)."""
    if page_width <= 0 or page_height <= 0:
        return False
    ratio = config.FULL_PAGE_EMBEDDED_RATIO
    # Scanned pages often match rendered width but not height (letter vs image aspect).
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
    """Pull raster images embedded in PDF object streams (not full-page scans)."""
    threshold = min_bytes or config.MIN_EMBEDDED_IMAGE_BYTES
    skip_full = config.SKIP_FULL_PAGE_EMBEDDED
    figures: list[ExtractedFigure] = []
    #opens the pdf file using pymupdf.
    doc = fitz.open(pdf_path)
    try:
        fig_idx = 0
        #loops through each page in the pdf.
        for page_i, page in enumerate(doc):
            pw, ph = (0, 0)
            if page_sizes and page_i < len(page_sizes):
                pw, ph = page_sizes[page_i]
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                # extracts the image from the pdf.
                base = doc.extract_image(xref)
                if not base or len(base.get("image", b"")) < threshold:
                    continue
                    #if the image is too small, it is skipped.
                iw, ih = base.get("width", 0), base.get("height", 0)
                if skip_full and _is_full_page_embedded(iw, ih, pw, ph):
                    #if the image is the full page, it is skipped.
                    continue
                ext = base.get("ext", "png")
                #saves the image to the output directory.
                out_path = output_dir / f"{pdf_path.stem}_p{page_i:03d}_embedded_{fig_idx:03d}.{ext}"
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

#creates a mask of the text regions.
def _mask_text_regions(
    shape: tuple[int, int],
    boxes: list[TextBox],
    padding: int,
) -> np.ndarray:
    h, w = shape
    # Creates black image
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x0, y0, x1, y1 = box.bbox
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(w, x1 + padding)
        y1 = min(h, y1 + padding)
        # we need to white out the text regions.
        mask[y0:y1, x0:x1] = 255
    return mask


def _estimate_text_coverage_in_bbox(
    bbox: tuple[int, int, int, int],
    boxes: list[TextBox],
    padding: int,
) -> float:
    """
    Estimate how much of a bbox is covered by OCR text boxes.

    Returns a fraction in [0, 1] (1 = almost everything is text).
    """
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    if w <= 1 or h <= 1:
        return 1.0

    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        bx0, by0, bx1, by1 = box.bbox
        ix0 = max(x0, bx0 - padding)
        iy0 = max(y0, by0 - padding)
        ix1 = min(x1, bx1 + padding)
        iy1 = min(y1, by1 + padding)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        mask[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = 255

    return float(mask.mean() / 255.0)

#detect diagrams and photos.
def _find_layout_figures(
    image_bgr: np.ndarray,
    text_boxes: list[TextBox],
    min_area_ratio: float,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Detect large non-text regions likely to be diagrams or photos."""
    #get page dimensions.
    h, w = image_bgr.shape[:2]
    #calculate page area.
    page_area = h * w
    min_area = int(page_area * min_area_ratio)

    text_mask = _mask_text_regions(
        (h, w), text_boxes, config.TEXT_MASK_PADDING_PX
    )
    #non text regions are white. and text regions are black. inverse
    non_text = cv2.bitwise_not(text_mask)
    #edge detection Convert to grayscale.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Emphasize structural content (lines, shading) outside text blocks
    edges = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_and(edges, non_text)
    combined = cv2.dilate(combined, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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
        if roi.size == 0:
            continue
        # Skip near-uniform margins (blank bands)
        if float(np.std(roi)) < 8.0:
            continue
        candidates.append(((x, y, x + cw, y + ch), area / page_area))

    # Merge overlapping boxes, keep largest
    candidates.sort(key=lambda c: c[1], reverse=True)
    merged: list[tuple[tuple[int, int, int, int], float]] = []
    for bbox, ratio in candidates:
        if any(_iou(bbox, kept[0]) > 0.5 for kept in merged):
            continue
        merged.append((bbox, ratio))
    return merged


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
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


def _refine_bbox_to_photo_frame(
    image_bgr: np.ndarray,
    coarse_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    Trim a coarse photo bbox to the visible photograph only.

    - Top/bottom: brightness transition between the photo and paper/text below.
    - Left/right: strongest vertical frame edges inside the photo band.
    """
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

    # Vertical bounds: photo vs white paper / tables (center strip avoids side notes)
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

    # Horizontal bounds: frame edges within the photo band
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
    """
    Locate the central aircraft / hangar photograph on mixed text+photo pages.

    Uses low-saturation + local texture, then merges split regions with morphology
    so the full photograph is captured (not a partial crop).
    """
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
    #Only search center area. skips header and footer.
    band[y0:y1, x0:x1] = 255
    photo_mask = cv2.bitwise_and(photo_mask, band)

    # Close vertically first to bridge thin gaps inside the photograph
    photo_mask = cv2.morphologyEx(
        photo_mask,
        cv2.MORPH_CLOSE,
        np.ones((21, 11), np.uint8),
        iterations=2,
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

    # Light trim using projection (low threshold keeps full photo height)
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
    tx0 = max(0, tx0 - pad)
    ty0 = max(0, ty0 - pad)
    tx1 = min(w, tx1 + pad)
    ty1 = min(h, ty1 + pad)
    bbox = (tx0, ty0, tx1, ty1)

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
    """Crop the central manufacturing photograph (aircraft / hangar image)."""
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
    _save_bgr(out_path, crop)
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
    """Crop figure regions from a rendered page using OCR text masks."""
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
        _save_bgr(out_path, crop)
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
) -> list[ExtractedFigure]:
    """Manufacturing photo crops, optional layout regions, and non-full-page embedded images."""
    figures: list[ExtractedFigure] = []
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    page_sizes = [(p.width, p.height) for p in pages]

    if source_path.suffix.lower() == ".pdf":
        figures.extend(
            extract_embedded_pdf_images(
                source_path, img_dir, page_sizes=page_sizes
            )
        )

    for page, ocr in zip(pages, ocr_results):
        photos = extract_manufacturing_photo(
            page.source_path,
            page.page_index,
            page.image,
            img_dir,
        )

        # If the "photo" bbox overlaps too much OCR text, it's likely not a
        # real photo (e.g., an engineering drawing can match photo heuristics).
        if photos:
            bbox = photos[0].bbox
            if bbox is not None:
                cov = _estimate_text_coverage_in_bbox(
                    bbox=bbox,
                    boxes=ocr.boxes,
                    padding=config.TEXT_MASK_PADDING_PX,
                )
                if cov > config.PHOTO_MAX_TEXT_COVERAGE:
                    photos = []

        figures.extend(photos)

        # Layout fallback only when accepted photo detection fails
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
