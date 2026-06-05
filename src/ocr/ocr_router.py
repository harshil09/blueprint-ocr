"""
OCR Router:
- RapidOCR primary engine (with optional region splitting)
- PaddleOCR fallback
- Confidence-based selection + ensemble agreement boost
"""

from __future__ import annotations

import re
import numpy as np

from src import config
from src.logger import get_logger

from src.ocr.ocr_engine import OcrEngine, OcrPageResult, TextBox
from src.ocr.paddle_engine import PaddleOcrEngine

from src.utils import bbox_iou, mean_box_confidence, normalize_confidence


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Region Utilities
# ---------------------------------------------------------------------------

def split_page_regions(
    width: int,
    height: int,
    *,
    rows: int,
    cols: int,
    overlap_ratio: float,
) -> list[tuple[int, int, int, int]]:
    """Split image into overlapping grid regions."""
    regions: list[tuple[int, int, int, int]] = []

    overlap_x = int(width * overlap_ratio / max(cols, 1))
    overlap_y = int(height * overlap_ratio / max(rows, 1))

    cell_w = width // cols
    cell_h = height // rows

    for r in range(rows):
        for c in range(cols):
            x0 = max(0, c * cell_w - overlap_x)
            y0 = max(0, r * cell_h - overlap_y)
            x1 = min(width, (c + 1) * cell_w + overlap_x)
            y1 = min(height, (r + 1) * cell_h + overlap_y)

            if (x1 - x0) > 32 and (y1 - y0) > 32:
                regions.append((x0, y0, x1, y1))

    return regions


def merge_region_ocr_results(
    results: list[OcrPageResult],
    engine: str = "rapid",
) -> OcrPageResult:
    """Merge OCR outputs from overlapping regions."""

    merged: list[TextBox] = []

    for result in results:
        for box in result.boxes:
            replaced = False

            for existing in merged[:]:
                if bbox_iou(box.bbox, existing.bbox) > 0.55:
                    replaced = True

                    if box.confidence > existing.confidence:
                        merged.remove(existing)
                        merged.append(box)
                    break

            if not replaced:
                merged.append(box)

    return OcrPageResult(
        full_text="\n".join(b.text for b in merged),
        boxes=merged,
        engine=engine,
    )


# ---------------------------------------------------------------------------
# Ensemble Helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()) if text else ""


def agreement_ratio(primary: OcrPageResult, secondary: OcrPageResult) -> float:
    """How many primary lines appear in secondary output."""
    if not primary.boxes:
        return 0.0

    secondary_set = {_normalize_text(b.text) for b in secondary.boxes}

    matches = sum(
        1 for b in primary.boxes
        if _normalize_text(b.text) in secondary_set
    )

    return matches / len(primary.boxes)


def boost_confidence_on_agreement(
    result: OcrPageResult,
    agreement: float,
    boost: float | None = None,
    min_agreement: float | None = None,
) -> OcrPageResult:
    """Boost confidence when both engines agree."""
    if not result.boxes:
        return result

    boost_val = boost if boost is not None else config.OCR_ENSEMBLE_AGREEMENT_BOOST
    min_agree = min_agreement if min_agreement is not None else config.OCR_ENSEMBLE_MIN_AGREEMENT_RATIO

    if agreement < min_agree:
        return result

    boosted = [
        TextBox(
            text=b.text,
            confidence=min(
                1.0,
                normalize_confidence(b.confidence) + boost_val * agreement
            ),
            bbox=b.bbox,
        )
        for b in result.boxes
    ]

    return OcrPageResult(result.full_text, boosted, result.engine)


def mean_page_confidence(result: OcrPageResult) -> float:
    return mean_box_confidence(result.boxes)


def ocr_confidence_stats(result: OcrPageResult) -> dict[str, float | int]:
    """Compute OCR confidence statistics for routing decisions."""
    if not result.boxes:
        return {
            "box_count": 0,
            "char_count": len(result.full_text),
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "median_confidence": 0.0,
            "low_confidence_boxes": 0,
            "low_confidence_ratio": 0.0,
        }

    confs = sorted(normalize_confidence(b.confidence) for b in result.boxes)

    n = len(confs)
    mid = n // 2
    median = confs[mid] if n % 2 else (confs[mid - 1] + confs[mid]) / 2

    threshold = config.OCR_ROUTER_FALLBACK_MEAN_CONF
    low_count = sum(c < threshold for c in confs)

    return {
        "box_count": n,
        "char_count": len(result.full_text),
        "mean_confidence": round(sum(confs) / n, 4),
        "min_confidence": round(confs[0], 4),
        "max_confidence": round(confs[-1], 4),
        "median_confidence": round(median, 4),
        "low_confidence_boxes": low_count,
        "low_confidence_ratio": round(low_count / n, 4),
    }


# ---------------------------------------------------------------------------
# OCR Router
# ---------------------------------------------------------------------------

class OcrRouter:
    def __init__(
        self,
        primary: OcrEngine | None = None,
        fallback: PaddleOcrEngine | None = None,
        enable_fallback: bool = True,
        enable_regions: bool | None = None,
    ):
        self.primary = primary or OcrEngine()
        self.fallback = fallback or PaddleOcrEngine()

        self.enable_fallback = enable_fallback
        self.enable_regions = (
            config.REGION_OCR_ENABLED if enable_regions is None else enable_regions
        )

    # -----------------------------
    # Rapid OCR pipeline
    # -----------------------------
    def _run_rapid(self, image: np.ndarray) -> OcrPageResult:
        h, w = image.shape[:2]

        big_page = max(h, w) >= config.REGION_OCR_MIN_SIDE_PX

        if not (self.enable_regions and big_page):
            return self.primary.recognize_page(image)

        regions = split_page_regions(
            w,
            h,
            rows=config.REGION_OCR_GRID_ROWS,
            cols=config.REGION_OCR_GRID_COLS,
            overlap_ratio=config.REGION_OCR_OVERLAP_RATIO,
        )

        results = [
            self.primary.recognize_region(image, r)
            for r in regions
        ]

        return merge_region_ocr_results(results)

    # -----------------------------
    # Fallback decision
    # -----------------------------
    def _should_use_paddle(self, result: OcrPageResult) -> tuple[bool, str]:
        if not self.enable_fallback:
            return False, "fallback disabled"

        if result.box_count == 0:
            return True, "no OCR output"

        mean_conf = mean_page_confidence(result)

        if mean_conf < config.OCR_ROUTER_FALLBACK_MEAN_CONF:
            return True, f"low confidence ({mean_conf:.3f})"

        return False, "confidence OK"

    # -----------------------------
    # Main entry
    # -----------------------------
    def recognize_page(self, image: np.ndarray) -> OcrPageResult:
        rapid = self._run_rapid(image)

        log.info(
            "RapidOCR boxes=%d mean_conf=%.3f",
            rapid.box_count,
            mean_page_confidence(rapid),
        )

        if not self._should_use_paddle(rapid)[0]:
            return rapid

        log.info("PaddleOCR fallback triggered")

        try:
            paddle = self.fallback.recognize_page(image)
        except Exception as e:
            log.warning("PaddleOCR failed: %s", e)
            return rapid

        rapid_score = mean_page_confidence(rapid)
        paddle_score = mean_page_confidence(paddle)

        best = paddle if paddle_score > rapid_score + 0.02 else rapid
        other = rapid if best is paddle else paddle

        agreement = agreement_ratio(best, other)

        log.info(
            "Selected=%s rapid=%.3f paddle=%.3f agreement=%.2f",
            "paddle" if best is paddle else "rapid",
            rapid_score,
            paddle_score,
            agreement,
        )

        return boost_confidence_on_agreement(
            best,
            agreement=agreement,
        )