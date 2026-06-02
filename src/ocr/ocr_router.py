"""OCR router: regional grid, Rapid/Paddle fallback, and ensemble confidence."""

from __future__ import annotations

import re

import numpy as np

from src import config
from src.ocr.ocr_engine import OcrEngine, OcrPageResult, TextBox
from src.ocr.paddle_engine import PaddleOcrEngine
from src.utils import bbox_iou, mean_box_confidence, normalize_confidence


# ---------------------------------------------------------------------------
# Region grid (large blueprint pages)
# ---------------------------------------------------------------------------

"""Break large pages into smaller chunks for OCR."""
def split_page_regions(
    
    width: int,
    height: int,
    *,
    rows: int,
    cols: int,
    overlap_ratio: float,
) -> list[tuple[int, int, int, int]]:
    """Split page into overlapping axis-aligned regions for regional OCR."""
    regions: list[tuple[int, int, int, int]] = []
    overlap_x = int(width * overlap_ratio / max(cols, 1))
    overlap_y = int(height * overlap_ratio / max(rows, 1))
    cell_w = width // cols
    cell_h = height // rows

    for row in range(rows):
        for col in range(cols):
            x0 = max(0, col * cell_w - overlap_x)
            y0 = max(0, row * cell_h - overlap_y)
            x1 = min(width, (col + 1) * cell_w + overlap_x)
            y1 = min(height, (row + 1) * cell_h + overlap_y)
            if x1 - x0 > 32 and y1 - y0 > 32:
                regions.append((x0, y0, x1, y1))
    return regions


def merge_region_ocr_results(
    partials: list[OcrPageResult],
    engine: str = "rapid",
) -> OcrPageResult:
    """Merge regional OCR boxes, dropping heavy overlaps (keep higher confidence)."""
    merged_boxes: list[TextBox] = []
    for partial in partials:
        for box in partial.boxes:
            duplicate = False
            for kept in list(merged_boxes):
                if bbox_iou(box.bbox, kept.bbox) > 0.55:
                    if box.confidence > kept.confidence:
                        merged_boxes.remove(kept)
                        merged_boxes.append(box)
                    duplicate = True
                    break
            if not duplicate:
                merged_boxes.append(box)

    lines = [b.text for b in merged_boxes]
    return OcrPageResult(full_text="\n".join(lines), boxes=merged_boxes, engine=engine)


# ---------------------------------------------------------------------------
# Ensemble agreement
# ---------------------------------------------------------------------------


def _normalize_text_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def agreement_ratio(primary: OcrPageResult, secondary: OcrPageResult) -> float:
    """Fraction of primary lines that appear in the secondary OCR output."""
    if not primary.boxes:
        return 0.0
    secondary_lines = {_normalize_text_line(b.text) for b in secondary.boxes}
    matches = sum(
        1 for b in primary.boxes if _normalize_text_line(b.text) in secondary_lines
    )
    return matches / len(primary.boxes)


def boost_confidence_on_agreement(
    result: OcrPageResult,
    agreement: float,
    boost: float | None = None,
    min_agreement: float | None = None,
) -> OcrPageResult:
    """Return a copy with boosted per-box confidence when engines agree."""
    boost_val = boost if boost is not None else config.OCR_ENSEMBLE_AGREEMENT_BOOST
    min_agree = (
        min_agreement
        if min_agreement is not None
        else config.OCR_ENSEMBLE_MIN_AGREEMENT_RATIO
    )
    if agreement < min_agree or not result.boxes:
        return result

    boosted: list[TextBox] = []
    for box in result.boxes:
        conf = min(1.0, normalize_confidence(box.confidence) + boost_val * agreement)
        boosted.append(TextBox(text=box.text, confidence=conf, bbox=box.bbox))
    return OcrPageResult(full_text=result.full_text, boxes=boosted, engine=result.engine)


def mean_page_confidence(result: OcrPageResult) -> float:
    return mean_box_confidence(result.boxes)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class OcrRouter:
    def __init__(
        self,
        primary=None,
        fallback=None,
        enable_fallback: bool = True,
        enable_regions: bool | None = None,
    ):
        self.primary = primary or OcrEngine()
        self.fallback = fallback or PaddleOcrEngine()
        self.enable_fallback = enable_fallback
        self.enable_regions = (
            enable_regions
            if enable_regions is not None
            else config.REGION_OCR_ENABLED
        )

    def _ocr_with_rapid(self, image_bgr: np.ndarray) -> OcrPageResult:
        h, w = image_bgr.shape[:2]
        big_page = max(h, w) >= config.REGION_OCR_MIN_SIDE_PX

        if not (self.enable_regions and big_page):
            return self.primary.recognize_page(image_bgr)

        regions = split_page_regions(
            w,
            h,
            rows=config.REGION_OCR_GRID_ROWS,
            cols=config.REGION_OCR_GRID_COLS,
            overlap_ratio=config.REGION_OCR_OVERLAP_RATIO,
        )
        parts = [self.primary.recognize_region(image_bgr, box) for box in regions]
        return merge_region_ocr_results(parts, engine="rapid")

    def _should_use_paddle(self, rapid_result: OcrPageResult) -> bool:
        if not self.enable_fallback:
            return False
        if rapid_result.box_count == 0:
            return True
        return mean_page_confidence(rapid_result) < config.OCR_ROUTER_FALLBACK_MEAN_CONF

    def recognize_page(self, image_bgr: np.ndarray) -> OcrPageResult:
        rapid = self._ocr_with_rapid(image_bgr)

        if not self._should_use_paddle(rapid):
            return rapid

        try:
            paddle = self.fallback.recognize_page(image_bgr)
        except Exception:
            return rapid

        rapid_score = mean_page_confidence(rapid)
        paddle_score = mean_page_confidence(paddle)
        if paddle_score > rapid_score + 0.02:
            best = paddle
            other = rapid
        else:
            best = rapid
            other = paddle

        agree = agreement_ratio(best, other)
        return boost_confidence_on_agreement(
            best,
            agreement=agree,
            boost=config.OCR_ENSEMBLE_AGREEMENT_BOOST,
            min_agreement=config.OCR_ENSEMBLE_MIN_AGREEMENT_RATIO,
        )
