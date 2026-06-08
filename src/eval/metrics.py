"""Crop metrics and failure categorization."""

from __future__ import annotations

from dataclasses import dataclass

from src.eval.categories import FailureCategory
from src.eval.labels import ExpectedCrop, LabeledPage
from src.utils import bbox_iou


@dataclass(frozen=True)
class PageEvalResult:
    """Evaluation outcome for one labeled page."""

    sample_id: str
    source_path: str
    page_index: int
    passed: bool
    failure_category: FailureCategory
    iou: float | None
    predicted_bbox: tuple[int, int, int, int] | None
    predicted_area_ratio: float | None
    predicted_text_overlap: float | None
    predicted_confidence: float | None
    predicted_method: str | None
    page_profile: str | None
    crop_profile: str | None
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "source_path": self.source_path,
            "page_index": self.page_index,
            "passed": self.passed,
            "failure_category": self.failure_category.value,
            "iou": self.iou,
            "predicted_bbox": self.predicted_bbox,
            "predicted_area_ratio": self.predicted_area_ratio,
            "predicted_text_overlap": self.predicted_text_overlap,
            "predicted_confidence": self.predicted_confidence,
            "predicted_method": self.predicted_method,
            "page_profile": self.page_profile,
            "crop_profile": self.crop_profile,
            "details": self.details,
        }


def _area_ratio(
    bbox: tuple[int, int, int, int] | None,
    page_shape: tuple[int, int],
) -> float | None:
    if bbox is None:
        return None
    ph, pw = page_shape
    x0, y0, x1, y1 = bbox
    return (x1 - x0) * (y1 - y0) / max(ph * pw, 1)


def categorize_failure(
    label: LabeledPage,
    *,
    predicted_bbox: tuple[int, int, int, int] | None,
    iou: float | None,
    text_overlap: float | None,
    area_ratio: float | None,
    confidence: float | None,
    crop_profile: str | None,
    min_confidence: float = 0.35,
) -> tuple[bool, FailureCategory, str]:
    """Return (passed, category, details) for one page."""
    exp = label.expected

    if not exp.expect_figure:
        if predicted_bbox is not None:
            return (
                False,
                FailureCategory.UNEXPECTED_FIGURE,
                "Figure extracted but none expected",
            )
        return True, FailureCategory.GOOD, "Correctly skipped figure extraction"

    if predicted_bbox is None:
        return False, FailureCategory.NO_FIGURE, "No primary figure emitted"

    if confidence is not None and confidence < min_confidence:
        return (
            False,
            FailureCategory.LOW_CONFIDENCE,
            f"Confidence {confidence:.3f} below {min_confidence:.3f}",
        )

    if (
        exp.expected_crop_profile
        and crop_profile
        and crop_profile != exp.expected_crop_profile
    ):
        return (
            False,
            FailureCategory.PROFILE_MISMATCH,
            f"Expected crop_profile={exp.expected_crop_profile}, got {crop_profile}",
        )

    if text_overlap is not None and text_overlap > exp.max_text_overlap:
        return (
            False,
            FailureCategory.TEXT_IN_CROP,
            f"Text overlap {text_overlap:.3f} > {exp.max_text_overlap:.3f}",
        )

    if exp.max_area_ratio is not None and area_ratio is not None:
        if area_ratio > exp.max_area_ratio:
            return (
                False,
                FailureCategory.FULL_PAGE_WIN,
                f"Area ratio {area_ratio:.3f} > {exp.max_area_ratio:.3f}",
            )

    if exp.min_area_ratio is not None and area_ratio is not None:
        if area_ratio < exp.min_area_ratio:
            return (
                False,
                FailureCategory.TOO_TIGHT,
                f"Area ratio {area_ratio:.3f} < {exp.min_area_ratio:.3f}",
            )

    if exp.bbox_xyxy is not None and iou is not None:
        if iou < exp.min_iou:
            pred_area = area_ratio or 0.0
            exp_x0, exp_y0, exp_x1, exp_y1 = exp.bbox_xyxy
            exp_area = (exp_x1 - exp_x0) * (exp_y1 - exp_y0)
            pred_x0, pred_y0, pred_x1, pred_y1 = predicted_bbox
            pred_box_area = (pred_x1 - pred_x0) * (pred_y1 - pred_y0)
            if pred_box_area < exp_area * 0.65:
                return (
                    False,
                    FailureCategory.TOO_TIGHT,
                    f"IoU {iou:.3f} < {exp.min_iou:.3f} (crop too small)",
                )
            if pred_box_area > exp_area * 1.45:
                return (
                    False,
                    FailureCategory.TOO_LOOSE,
                    f"IoU {iou:.3f} < {exp.min_iou:.3f} (crop too large)",
                )
            return (
                False,
                FailureCategory.WRONG_VIEW,
                f"IoU {iou:.3f} < {exp.min_iou:.3f} (wrong region)",
            )

    return True, FailureCategory.GOOD, "All checks passed"


def evaluate_page_crop(
    label: LabeledPage,
    *,
    predicted_bbox: tuple[int, int, int, int] | None,
    page_shape: tuple[int, int],
    text_overlap: float | None = None,
    confidence: float | None = None,
    method: str | None = None,
    page_profile: str | None = None,
    crop_profile: str | None = None,
    min_confidence: float = 0.35,
) -> PageEvalResult:
    """Score one labeled page against pipeline output."""
    iou = None
    if label.expected.bbox_xyxy is not None and predicted_bbox is not None:
        iou = bbox_iou(label.expected.bbox_xyxy, predicted_bbox)

    area_ratio = _area_ratio(predicted_bbox, page_shape)
    passed, category, details = categorize_failure(
        label,
        predicted_bbox=predicted_bbox,
        iou=iou,
        text_overlap=text_overlap,
        area_ratio=area_ratio,
        confidence=confidence,
        crop_profile=crop_profile,
        min_confidence=min_confidence,
    )

    return PageEvalResult(
        sample_id=label.id,
        source_path=str(label.source_path),
        page_index=label.page_index,
        passed=passed,
        failure_category=category,
        iou=round(iou, 4) if iou is not None else None,
        predicted_bbox=predicted_bbox,
        predicted_area_ratio=round(area_ratio, 4) if area_ratio is not None else None,
        predicted_text_overlap=text_overlap,
        predicted_confidence=confidence,
        predicted_method=method,
        page_profile=page_profile,
        crop_profile=crop_profile,
        details=details,
    )
