"""Ground-truth OCR accuracy metrics (CER, WER, character/word accuracy)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


def normalize_for_comparison(text: str) -> str:
    """Lowercase, collapse whitespace, strip for fair OCR vs reference comparison."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text.strip())
    return text


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


@dataclass(frozen=True)
class OcrAccuracyMetrics:
    """Comparison of OCR output against reference (ground-truth) text."""

    reference_source: str  # ground_truth_file | pdf_text_layer | none
    reference_chars: int
    reference_words: int
    hypothesis_chars: int
    hypothesis_words: int
    char_error_rate: float  # CER: 0 = perfect, 1 = all wrong
    word_error_rate: float  # WER: 0 = perfect, 1 = all wrong
    char_accuracy: float  # 1 - CER, clamped to [0, 1]
    word_accuracy: float  # 1 - WER, clamped to [0, 1]
    measured: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _error_rate(distance: int, reference_len: int) -> float:
    if reference_len <= 0:
        return 0.0 if distance == 0 else 1.0
    return min(1.0, distance / reference_len)


def _accuracy(error_rate: float) -> float:
    return max(0.0, 1.0 - error_rate)


def compare_ocr_to_reference(
    hypothesis: str,
    reference: str,
    *,
    reference_source: str,
) -> OcrAccuracyMetrics:
    """
    Compute CER/WER between OCR hypothesis and reference text.

    Character accuracy = 1 - CER; word accuracy = 1 - WER.
    """
    ref_norm = normalize_for_comparison(reference)
    hyp_norm = normalize_for_comparison(hypothesis)

    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))
    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []

    char_dist = _levenshtein(ref_chars, hyp_chars)
    word_dist = _levenshtein(ref_words, hyp_words)

    cer = _error_rate(char_dist, len(ref_chars))
    wer = _error_rate(word_dist, len(ref_words))

    return OcrAccuracyMetrics(
        reference_source=reference_source,
        reference_chars=len(ref_chars),
        reference_words=len(ref_words),
        hypothesis_chars=len(hyp_chars),
        hypothesis_words=len(hyp_words),
        char_error_rate=round(cer, 4),
        word_error_rate=round(wer, 4),
        char_accuracy=round(_accuracy(cer), 4),
        word_accuracy=round(_accuracy(wer), 4),
    )


def aggregate_accuracy(metrics: list[OcrAccuracyMetrics]) -> dict[str, float | int | str | None]:
    """Document-level averages over pages that were measured."""
    measured = [m for m in metrics if m.measured and m.reference_chars > 0]
    if not measured:
        return {
            "pages_measured": 0,
            "pages_total": len(metrics),
            "mean_char_accuracy": None,
            "mean_word_accuracy": None,
            "mean_char_error_rate": None,
            "mean_word_error_rate": None,
        }

    return {
        "pages_measured": len(measured),
        "pages_total": len(metrics),
        "mean_char_accuracy": round(
            sum(m.char_accuracy for m in measured) / len(measured), 4
        ),
        "mean_word_accuracy": round(
            sum(m.word_accuracy for m in measured) / len(measured), 4
        ),
        "mean_char_error_rate": round(
            sum(m.char_error_rate for m in measured) / len(measured), 4
        ),
        "mean_word_error_rate": round(
            sum(m.word_error_rate for m in measured) / len(measured), 4
        ),
    }


def build_confidence_proxy(
    ocr_results: list,
    *,
    page_indices: list[int] | None = None,
) -> dict:
    """
    Confidence-based quality proxy when no ground-truth reference exists.

    This is NOT accuracy — it reflects how confident the OCR engine was.
    """
    from src.ocr.ocr_router import ocr_confidence_stats

    per_page: list[dict] = []
    for i, ocr in enumerate(ocr_results):
        stats = ocr_confidence_stats(ocr)
        page_index = page_indices[i] if page_indices else i
        per_page.append(
            {
                "page_index": page_index,
                "engine": ocr.engine,
                **stats,
            }
        )

    confidences = [p["mean_confidence"] for p in per_page if p["box_count"] > 0]
    summary: dict[str, float | int | None] = {
        "pages_total": len(per_page),
        "pages_with_text": len(confidences),
        "mean_confidence": None,
        "min_confidence": None,
        "max_confidence": None,
    }
    if confidences:
        summary["mean_confidence"] = round(sum(confidences) / len(confidences), 4)
        summary["min_confidence"] = round(min(confidences), 4)
        summary["max_confidence"] = round(max(confidences), 4)

    return {
        "note": (
            "Engine-reported confidence only — not ground-truth accuracy. "
            "Add a ground-truth file to measure real char/word accuracy."
        ),
        "summary": summary,
        "per_page": per_page,
    }
