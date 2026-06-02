"""Run OCR, figure extraction, and keyword extraction on a document."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from src import config
from src.document_loader import PageImage, load_document
from src.extract_figure import extract_engineering_figure_with_metadata
from src.image_extractor import ExtractedFigure, extract_all_figures
from src.keyword_extractor import extract_keywords, keywords_as_strings
from src.logger import get_logger
from src.ocr.accuracy import (
    OcrAccuracyMetrics,
    aggregate_accuracy,
    build_confidence_proxy,
    compare_ocr_to_reference,
)
from src.ocr.ocr_engine import OcrEngine, OcrPageResult
from src.ocr.ocr_router import OcrRouter, mean_page_confidence, ocr_confidence_stats
from src.ocr.reference_text import ground_truth_hints, load_reference_pages, page_reference_usable

log = get_logger(__name__)


@dataclass
class PageExtraction:
    page_index: int
    ocr_text: str
    ocr_engine: str
    ocr_mean_confidence: float
    keywords: list[dict]
    keyword_list: list[str]
    engineering_figure_path: str | None = None
    morphology_component_score: float | None = None
    morphology_gate_score: float | None = None
    morphology_quality_passed: bool | None = None
    rapid_ocr_accuracy: dict | None = None
    pipeline_ocr_accuracy: dict | None = None


@dataclass
class ExtractionResult:
    source_file: str
    pages: list[PageExtraction] = field(default_factory=list)
    all_keywords: list[str] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)
    output_dir: str = ""
    ocr_accuracy_summary: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _run_ocr(ocr_backend, image) -> OcrPageResult:
    """Run OCR and make sure the result has an engine name."""
    result = ocr_backend.recognize_page(image)
    if result.engine:
        return result
    name = type(ocr_backend).__name__.lower()
    engine = "paddle" if "paddle" in name else "rapid"
    return OcrPageResult(
        full_text=result.full_text,
        boxes=result.boxes,
        engine=engine,
    )


def _figure_to_dict(figure: ExtractedFigure) -> dict:
    return {
        "page_index": figure.page_index,
        "figure_index": figure.figure_index,
        "method": figure.method,
        "path": str(figure.image_path),
        "bbox": figure.bbox,
        "area_ratio": figure.area_ratio,
    }


def _write_json_files(
    doc_out: Path,
    result: ExtractionResult,
    full_text: str,
    accuracy_report: dict | None = None,
) -> None:
    (doc_out / "ocr_full_text.txt").write_text(full_text, encoding="utf-8")

    keywords_data = {
        "document": result.source_file,
        "all_keywords": result.all_keywords,
        "per_page": [
            {
                "page_index": p.page_index,
                "ocr_engine": p.ocr_engine,
                "ocr_mean_confidence": p.ocr_mean_confidence,
                "morphology_component_score": p.morphology_component_score,
                "morphology_gate_score": p.morphology_gate_score,
                "morphology_quality_passed": p.morphology_quality_passed,
                "rapid_ocr_accuracy": p.rapid_ocr_accuracy,
                "pipeline_ocr_accuracy": p.pipeline_ocr_accuracy,
                "keywords": p.keywords,
            }
            for p in result.pages
        ],
    }
    (doc_out / "keywords.json").write_text(
        json.dumps(keywords_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (doc_out / "extraction_report.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if accuracy_report is not None:
        (doc_out / "ocr_accuracy.json").write_text(
            json.dumps(accuracy_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Wrote OCR accuracy report: %s", doc_out / "ocr_accuracy.json")


def _evaluate_ocr_accuracy(
    hypothesis: str,
    reference: str,
    reference_source: str,
) -> OcrAccuracyMetrics | None:
    if not page_reference_usable(reference):
        return None
    return compare_ocr_to_reference(
        hypothesis,
        reference,
        reference_source=reference_source,
    )


class ExtractionPipeline:
    """Load a file, OCR each page, extract figures and keywords, save JSON + images."""

    def __init__(
        self,
        ocr_engine=None,
        output_dir: Path | str | None = None,
        *,
        accuracy_enabled: bool | None = None,
    ):
        if ocr_engine is None:
            self.ocr = OcrRouter()
        else:
            self.ocr = ocr_engine
        self.output_dir = Path(output_dir or config.DEFAULT_OUTPUT_DIR)
        self.accuracy_enabled = (
            accuracy_enabled
            if accuracy_enabled is not None
            else config.OCR_ACCURACY_ENABLED
        )
        self._rapid_engine: OcrEngine | None = None

    def _get_rapid_engine(self) -> OcrEngine:
        if self._rapid_engine is None:
            self._rapid_engine = OcrEngine()
        return self._rapid_engine

    def process(
        self,
        document_path: Path | str,
        *,
        original_filename: str | None = None,
    ) -> ExtractionResult:
        path = Path(document_path).resolve()
        doc_out = self.output_dir / path.stem
        doc_out.mkdir(parents=True, exist_ok=True)
        (doc_out / "images").mkdir(parents=True, exist_ok=True)

        log.info("=== Extraction started: %s ===", path.name)
        if original_filename:
            log.info("Original upload filename: %s", original_filename)
        log.info("Output directory: %s", doc_out)

        pages = load_document(path)
        ocr_results: list[OcrPageResult] = []
        page_results: list[PageExtraction] = []
        all_page_text: list[str] = []
        morphology_paths: dict[int, Path] = {}

        reference_pages: list[str] = []
        reference_source = "none"
        rapid_accuracy_rows: list[OcrAccuracyMetrics] = []
        pipeline_accuracy_rows: list[OcrAccuracyMetrics] = []

        if self.accuracy_enabled:
            reference_pages, reference_source = load_reference_pages(
                path,
                len(pages),
                original_filename=original_filename,
            )
            if reference_source == "none":
                log.warning(
                    "OCR ground-truth accuracy skipped for %s — no reference text found. "
                    "Scanned PDFs need a ground-truth file. Try one of:\n  %s",
                    original_filename or path.name,
                    "\n  ".join(ground_truth_hints(path, original_filename)),
                )
            else:
                log.info(
                    "OCR accuracy reference: source=%s pages=%d",
                    reference_source,
                    len(reference_pages),
                )
        else:
            log.info("OCR accuracy measurement disabled (OCR_ACCURACY_ENABLED=False)")

        for page in tqdm(pages, desc=f"OCR {path.name}", unit="page"):
            log.info(
                "--- Page %d/%d (%dx%d) ---",
                page.page_index + 1,
                len(pages),
                page.width,
                page.height,
            )
            ocr = _run_ocr(self.ocr, page.image)
            ocr_results.append(ocr)
            all_page_text.append(ocr.full_text)

            stats = ocr_confidence_stats(ocr)
            log.info(
                "OCR result: engine=%s boxes=%d chars=%d mean_conf=%.3f "
                "(engine-reported confidence, not ground-truth accuracy)",
                ocr.engine,
                stats["box_count"],
                stats["char_count"],
                stats["mean_confidence"],
            )

            rapid_accuracy_dict = None
            pipeline_accuracy_dict = None
            if self.accuracy_enabled and reference_source != "none":
                ref_text = (
                    reference_pages[page.page_index]
                    if page.page_index < len(reference_pages)
                    else ""
                )
                if config.OCR_ACCURACY_MEASURE_RAPID:
                    rapid_result = self._get_rapid_engine().recognize_page(page.image)
                    rapid_metrics = _evaluate_ocr_accuracy(
                        rapid_result.full_text, ref_text, reference_source
                    )
                    if rapid_metrics is not None:
                        rapid_accuracy_dict = rapid_metrics.to_dict()
                        rapid_accuracy_rows.append(rapid_metrics)
                        log.info(
                            "RapidOCR accuracy page %d: char=%.1f%% word=%.1f%% "
                            "(CER=%.3f WER=%.3f ref_chars=%d)",
                            page.page_index,
                            rapid_metrics.char_accuracy * 100,
                            rapid_metrics.word_accuracy * 100,
                            rapid_metrics.char_error_rate,
                            rapid_metrics.word_error_rate,
                            rapid_metrics.reference_chars,
                        )
                if config.OCR_ACCURACY_MEASURE_PIPELINE:
                    pipeline_metrics = _evaluate_ocr_accuracy(
                        ocr.full_text, ref_text, reference_source
                    )
                    if pipeline_metrics is not None:
                        pipeline_accuracy_dict = pipeline_metrics.to_dict()
                        pipeline_accuracy_rows.append(pipeline_metrics)
                        if ocr.engine != "rapid":
                            log.info(
                                "Pipeline OCR (%s) accuracy page %d: char=%.1f%% word=%.1f%%",
                                ocr.engine,
                                page.page_index,
                                pipeline_metrics.char_accuracy * 100,
                                pipeline_metrics.word_accuracy * 100,
                            )

            figure_path = (
                doc_out
                / f"engineering_figure_{page.page_index}.{config.OUTPUT_IMAGE_FORMAT}"
            )
            eng_path, selection = extract_engineering_figure_with_metadata(
                page.image,
                figure_path,
                text_boxes=ocr.boxes,
                apply_text_mask=True,
            )
            if eng_path is not None:
                morphology_paths[page.page_index] = eng_path
                log.info(
                    "Engineering figure extracted: %s (quality_passed=%s "
                    "component_score=%.3f gate_score=%.3f)",
                    eng_path.name,
                    selection.quality.passed if selection else None,
                    selection.component_score if selection else 0.0,
                    selection.gate_score if selection else 0.0,
                )
            else:
                log.info("No engineering figure from morphology on page %d", page.page_index)

            keywords = extract_keywords(ocr.full_text)
            log.info("Keywords extracted: %d on page %d", len(keywords), page.page_index)
            comp_score = gate_score = None
            quality_ok = None
            if selection is not None:
                comp_score = round(selection.component_score, 4)
                gate_score = round(selection.gate_score, 4)
                quality_ok = selection.quality.passed

            page_results.append(
                PageExtraction(
                    page_index=page.page_index,
                    ocr_text=ocr.full_text,
                    ocr_engine=ocr.engine,
                    ocr_mean_confidence=round(mean_page_confidence(ocr), 4),
                    keywords=keywords,
                    keyword_list=[k["keyword"] for k in keywords],
                    engineering_figure_path=str(eng_path) if eng_path else None,
                    morphology_component_score=comp_score,
                    morphology_gate_score=gate_score,
                    morphology_quality_passed=quality_ok,
                    rapid_ocr_accuracy=rapid_accuracy_dict,
                    pipeline_ocr_accuracy=pipeline_accuracy_dict,
                )
            )

        full_text = "\n\n".join(all_page_text)
        all_keywords = keywords_as_strings(full_text)

        log.info("Running figure extraction (embedded / photo / layout)")
        figures = extract_all_figures(
            path,
            pages,
            ocr_results,
            doc_out,
            morphology_by_page=morphology_paths,
        )

        engine_counts: dict[str, int] = {}
        confidences: list[float] = []
        for p in page_results:
            engine_counts[p.ocr_engine] = engine_counts.get(p.ocr_engine, 0) + 1
            confidences.append(p.ocr_mean_confidence)

        log.info(
            "=== Extraction complete: %s | pages=%d figures=%d keywords=%d ===",
            path.name,
            len(page_results),
            len(figures),
            len(all_keywords),
        )
        log.info("OCR engines used: %s", engine_counts)
        if confidences:
            doc_mean_conf = sum(confidences) / len(confidences)
            log.info(
                "Document OCR quality (engine-reported): mean_conf=%.3f min=%.3f max=%.3f",
                doc_mean_conf,
                min(confidences),
                max(confidences),
            )

        ocr_accuracy_summary = None
        accuracy_report: dict | None = None

        if self.accuracy_enabled:
            confidence_proxy = build_confidence_proxy(
                ocr_results,
                page_indices=[p.page_index for p in pages],
            )

            if reference_source != "none":
                rapid_summary = aggregate_accuracy(rapid_accuracy_rows)
                pipeline_summary = aggregate_accuracy(pipeline_accuracy_rows)
                ocr_accuracy_summary = {
                    "status": "measured",
                    "reference_source": reference_source,
                    "rapid_ocr": rapid_summary,
                    "pipeline_ocr": pipeline_summary,
                    "confidence_proxy": confidence_proxy,
                }
                accuracy_report = {
                    "status": "measured",
                    "document": str(path),
                    "original_filename": original_filename,
                    "reference_source": reference_source,
                    "ground_truth_accuracy": {
                        "rapid_ocr": {
                            "summary": rapid_summary,
                            "per_page": [p.rapid_ocr_accuracy for p in page_results],
                        },
                        "pipeline_ocr": {
                            "summary": pipeline_summary,
                            "per_page": [p.pipeline_ocr_accuracy for p in page_results],
                        },
                    },
                    "confidence_proxy": confidence_proxy,
                }
                if rapid_summary.get("pages_measured", 0) > 0:
                    log.info(
                        "RapidOCR ground-truth accuracy: char=%.1f%% word=%.1f%% "
                        "(%d/%d pages measured)",
                        (rapid_summary["mean_char_accuracy"] or 0) * 100,
                        (rapid_summary["mean_word_accuracy"] or 0) * 100,
                        rapid_summary["pages_measured"],
                        rapid_summary["pages_total"],
                    )
                else:
                    log.warning(
                        "Reference loaded (%s) but no pages had enough text to measure "
                        "(min %d chars per page)",
                        reference_source,
                        config.OCR_ACCURACY_MIN_REFERENCE_CHARS,
                    )
            else:
                ocr_accuracy_summary = {
                    "status": "no_reference",
                    "reference_source": "none",
                    "ground_truth_accuracy": None,
                    "confidence_proxy": confidence_proxy,
                    "how_to_enable_accuracy": ground_truth_hints(path, original_filename),
                }
                accuracy_report = {
                    "status": "no_reference",
                    "message": (
                        "No ground-truth reference found. Real char/word accuracy cannot "
                        "be computed for scanned PDFs without a reference file."
                    ),
                    "document": str(path),
                    "original_filename": original_filename,
                    "reference_source": "none",
                    "ground_truth_accuracy": None,
                    "confidence_proxy": confidence_proxy,
                    "how_to_enable_accuracy": ground_truth_hints(path, original_filename),
                }

            proxy = confidence_proxy["summary"]
            if proxy.get("mean_confidence") is not None:
                log.info(
                    "OCR confidence proxy (not ground-truth accuracy): mean=%.3f "
                    "min=%.3f max=%.3f (%d/%d pages with text)",
                    proxy["mean_confidence"],
                    proxy["min_confidence"],
                    proxy["max_confidence"],
                    proxy["pages_with_text"],
                    proxy["pages_total"],
                )
            log.info("OCR accuracy report -> %s", doc_out / "ocr_accuracy.json")

        result = ExtractionResult(
            source_file=str(path),
            pages=page_results,
            all_keywords=all_keywords,
            figures=[_figure_to_dict(f) for f in figures],
            output_dir=str(doc_out),
            ocr_accuracy_summary=ocr_accuracy_summary,
        )
        _write_json_files(
            doc_out,
            result,
            full_text,
            accuracy_report if self.accuracy_enabled else None,
        )
        return result
