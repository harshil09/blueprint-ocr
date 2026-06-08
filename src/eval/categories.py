"""Failure categories for figure-crop evaluation."""

from __future__ import annotations

from enum import Enum


class FailureCategory(str, Enum):
    """Standardized failure bucket for a labeled page evaluation."""

    GOOD = "good"
    FULL_PAGE_WIN = "full_page_win"
    TEXT_IN_CROP = "text_in_crop"
    TOO_TIGHT = "too_tight"
    TOO_LOOSE = "too_loose"
    WRONG_VIEW = "wrong_view"
    NO_FIGURE = "no_figure"
    UNEXPECTED_FIGURE = "unexpected_figure"
    LOW_CONFIDENCE = "low_confidence"
    PROFILE_MISMATCH = "profile_mismatch"
    SKIPPED = "skipped"
