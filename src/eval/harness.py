"""Run labeled figure-crop evaluation against the extraction pipeline."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.document_loader import load_document
from src.eval.categories import FailureCategory
from src.eval.labels import EvalManifest, LabeledPage, load_manifest
from src.eval.metrics import PageEvalResult, evaluate_page_crop
from src.extract_figure import compute_engineering_figure_crop
from src.figure_fusion import classify_page
from src.image_extractor import (
    MorphologyPageCandidate,
    embedded_image_counts_by_page,
    extract_primary_figures,
)
from src.profile_config import (
    apply_dynamic_layout_zones,
    get_profile_config,
    resolve_crop_profile,
)
from src.logger import get_logger
from src.ocr.ocr_router import OcrRouter

log = get_logger(__name__)


@dataclass
class EvalReport:
    """Aggregated evaluation report."""

    manifest_path: str
    total_samples: int
    evaluated: int
    skipped_missing: int
    passed: int
    failed: int
    pass_rate: float
    failure_counts: dict[str, int] = field(default_factory=dict)
    pages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _morphology_by_page(
    pages,
    ocr_results,
    *,
    source_path: Path,
) -> dict[int, MorphologyPageCandidate]:
    by_page: dict[int, MorphologyPageCandidate] = {}
    is_pdf = source_path.suffix.lower() == ".pdf"
    page_count = len(pages)
    embedded_counts = (
        embedded_image_counts_by_page(
            source_path,
            page_sizes=[(p.width, p.height) for p in pages],
        )
        if is_pdf
        else {}
    )
    for page, ocr in zip(pages, ocr_results):
        embedded_on_page = embedded_counts.get(page.page_index, 0)
        page_profile = classify_page(
            page.image,
            ocr,
            page_count=page_count,
            embedded_on_page=embedded_on_page,
            is_pdf=is_pdf,
        )
        crop_profile = resolve_crop_profile(
            page_profile,
            page.image,
            ocr,
            is_pdf=is_pdf,
            embedded_on_page=embedded_on_page,
            page_count=page_count,
            source_suffix=source_path.suffix,
        )
        morph_pcfg = apply_dynamic_layout_zones(
            get_profile_config(crop_profile),
            ocr,
            page.image.shape[:2],
        )
        result = compute_engineering_figure_crop(
            page.image,
            ocr.boxes,
            profile=page_profile,
            profile_config=morph_pcfg,
        )
        if result is None:
            continue
        crop, bbox_xyxy, selection = result
        by_page[page.page_index] = MorphologyPageCandidate(
            crop=crop,
            bbox=bbox_xyxy,
            selection=selection,
        )
    return by_page


def _figure_for_page(figures, page_index: int):
    for fig in figures:
        if fig.page_index == page_index:
            return fig
    return None


def evaluate_labeled_page(
    label: LabeledPage,
    *,
    ocr_router: OcrRouter | None = None,
    output_dir: Path | None = None,
) -> PageEvalResult:
    """Run extraction for one labeled page and score the crop."""
    if not label.source_path.is_file():
        return PageEvalResult(
            sample_id=label.id,
            source_path=str(label.source_path),
            page_index=label.page_index,
            passed=False,
            failure_category=FailureCategory.SKIPPED,
            iou=None,
            predicted_bbox=None,
            predicted_area_ratio=None,
            predicted_text_overlap=None,
            predicted_confidence=None,
            predicted_method=None,
            page_profile=None,
            crop_profile=None,
            details="Source file not found",
        )

    router = ocr_router or OcrRouter()
    pages = load_document(label.source_path)
    if label.page_index >= len(pages):
        return PageEvalResult(
            sample_id=label.id,
            source_path=str(label.source_path),
            page_index=label.page_index,
            passed=False,
            failure_category=FailureCategory.SKIPPED,
            iou=None,
            predicted_bbox=None,
            predicted_area_ratio=None,
            predicted_text_overlap=None,
            predicted_confidence=None,
            predicted_method=None,
            page_profile=None,
            crop_profile=None,
            details=f"Page index {label.page_index} out of range ({len(pages)} pages)",
        )

    target_page = pages[label.page_index]
    ocr_results = [router.recognize_page(p.image) for p in pages]
    morph_map = _morphology_by_page(
        pages, ocr_results, source_path=label.source_path
    )

    eval_out = output_dir or Path("/tmp/figure_eval") / label.id
    eval_out.mkdir(parents=True, exist_ok=True)

    figures = extract_primary_figures(
        label.source_path,
        pages,
        ocr_results,
        eval_out,
        morphology_by_page=morph_map,
    )
    figure = _figure_for_page(figures, label.page_index)

    return evaluate_page_crop(
        label,
        predicted_bbox=figure.bbox if figure else None,
        page_shape=target_page.image.shape[:2],
        text_overlap=figure.text_overlap if figure else None,
        confidence=figure.confidence if figure else None,
        method=figure.method if figure else None,
        page_profile=figure.page_profile if figure else None,
        crop_profile=figure.crop_profile if figure else None,
    )


def run_figure_eval(
    manifest_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    write_report: bool = True,
) -> EvalReport:
    """
    Evaluate all labeled samples in a manifest.

    Returns an :class:`EvalReport` and optionally writes ``eval_report.json``.
    """
    manifest = load_manifest(manifest_path)
    report_dir = Path(output_dir) if output_dir else manifest.manifest_path.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    router = OcrRouter()
    page_results: list[PageEvalResult] = []
    skipped = 0

    for label in manifest.samples:
        if not label.source_path.is_file():
            skipped += 1
            page_results.append(
                PageEvalResult(
                    sample_id=label.id,
                    source_path=str(label.source_path),
                    page_index=label.page_index,
                    passed=False,
                    failure_category=FailureCategory.SKIPPED,
                    iou=None,
                    predicted_bbox=None,
                    predicted_area_ratio=None,
                    predicted_text_overlap=None,
                    predicted_confidence=None,
                    predicted_method=None,
                    page_profile=None,
                    crop_profile=None,
                    details="Source file not found",
                )
            )
            continue

        log.info("Evaluating %s (page %d)", label.id, label.page_index)
        page_results.append(
            evaluate_labeled_page(
                label,
                ocr_router=router,
                output_dir=report_dir / label.id,
            )
        )

    evaluated = [r for r in page_results if r.failure_category != FailureCategory.SKIPPED]
    passed = sum(1 for r in evaluated if r.passed)
    failed = len(evaluated) - passed
    pass_rate = passed / max(len(evaluated), 1)

    failure_counter = Counter(
        r.failure_category.value for r in evaluated if not r.passed
    )

    report = EvalReport(
        manifest_path=str(manifest.manifest_path),
        total_samples=len(manifest.samples),
        evaluated=len(evaluated),
        skipped_missing=skipped,
        passed=passed,
        failed=failed,
        pass_rate=round(pass_rate, 4),
        failure_counts=dict(failure_counter),
        pages=[r.to_dict() for r in page_results],
    )

    if write_report:
        out_path = report_dir / "eval_report.json"
        out_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Wrote evaluation report: %s", out_path)

    return report
