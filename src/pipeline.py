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
from src.ocr.ocr_engine import OcrPageResult
from src.ocr.ocr_router import OcrRouter, mean_page_confidence


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


@dataclass
class ExtractionResult:
    source_file: str
    pages: list[PageExtraction] = field(default_factory=list)
    all_keywords: list[str] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)
    output_dir: str = ""

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


def _write_json_files(doc_out: Path, result: ExtractionResult, full_text: str) -> None:
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


class ExtractionPipeline:
    """Load a file, OCR each page, extract figures and keywords, save JSON + images."""

    def __init__(self, ocr_engine=None, output_dir: Path | str | None = None):
        if ocr_engine is None:
            self.ocr = OcrRouter()
        else:
            self.ocr = ocr_engine
        self.output_dir = Path(output_dir or config.DEFAULT_OUTPUT_DIR)

    def process(self, document_path: Path | str) -> ExtractionResult:
        path = Path(document_path).resolve()
        doc_out = self.output_dir / path.stem
        doc_out.mkdir(parents=True, exist_ok=True)
        (doc_out / "images").mkdir(parents=True, exist_ok=True)

        pages = load_document(path)
        ocr_results: list[OcrPageResult] = []
        page_results: list[PageExtraction] = []
        all_page_text: list[str] = []
        morphology_paths: dict[int, Path] = {}

        for page in tqdm(pages, desc=f"OCR {path.name}", unit="page"):
            ocr = _run_ocr(self.ocr, page.image)
            ocr_results.append(ocr)
            all_page_text.append(ocr.full_text)

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

            keywords = extract_keywords(ocr.full_text)
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
                )
            )

        full_text = "\n\n".join(all_page_text)
        all_keywords = keywords_as_strings(full_text)

        figures = extract_all_figures(
            path,
            pages,
            ocr_results,
            doc_out,
            morphology_by_page=morphology_paths,
        )

        result = ExtractionResult(
            source_file=str(path),
            pages=page_results,
            all_keywords=all_keywords,
            figures=[_figure_to_dict(f) for f in figures],
            output_dir=str(doc_out),
        )
        _write_json_files(doc_out, result, full_text)
        return result
