"""End-to-end extraction pipeline."""

from __future__ import annotations
#if doesnt work remove it```python id="2u4kry"
from src.extract_figure import extract_engineering_figure

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

import config
from src.document_loader import load_document
from src.image_extractor import extract_all_figures
from src.keyword_extractor import extract_keywords, keywords_as_strings
from src.ocr_engine import OcrEngine, OcrPageResult


@dataclass
class PageExtraction:
    page_index: int
    ocr_text: str
    keywords: list[dict]
    keyword_list: list[str]


@dataclass
class ExtractionResult:
    source_file: str
    pages: list[PageExtraction] = field(default_factory=list)
    all_keywords: list[str] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)
    output_dir: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ExtractionPipeline:
    def __init__(
        self,
        ocr_engine=None,
        output_dir: Path | str | None = None,
    ):
        self.ocr = ocr_engine or OcrEngine()
        self.output_dir = Path(output_dir or config.DEFAULT_OUTPUT_DIR)

    def process(self, document_path: Path | str) -> ExtractionResult:
        path = Path(document_path).resolve()
        doc_out = self.output_dir / path.stem
        doc_out.mkdir(parents=True, exist_ok=True)

        pages = load_document(path)
        
        for idx, page in enumerate(pages):

            figure_output = (
                doc_out / f"engineering_figure_{idx}.png"
            )

            extract_engineering_figure(
                page.image,
                figure_output,
            )
    

        ocr_results: list[OcrPageResult] = []
        page_extractions: list[PageExtraction] = []
        combined_text_parts: list[str] = []

        for page in tqdm(pages, desc=f"OCR {path.name}", unit="page"):
            ocr = self.ocr.recognize_page(page.image)
            ocr_results.append(ocr)
            combined_text_parts.append(ocr.full_text)

            kw_ranked = extract_keywords(ocr.full_text)
            kw_strings = [k["keyword"] for k in kw_ranked]
            page_extractions.append(
                PageExtraction(
                    page_index=page.page_index,
                    ocr_text=ocr.full_text,
                    keywords=kw_ranked,
                    keyword_list=kw_strings,
                )
            )

        full_text = "\n\n".join(combined_text_parts)
        all_keywords = keywords_as_strings(full_text)

        figures = extract_all_figures(path, pages, ocr_results, doc_out)
        figure_records = [
            {
                "page_index": f.page_index,
                "figure_index": f.figure_index,
                "method": f.method,
                "path": str(f.image_path),
                "bbox": f.bbox,
                "area_ratio": f.area_ratio,
            }
            for f in figures
        ]

        result = ExtractionResult(
            source_file=str(path),
            pages=page_extractions,
            all_keywords=all_keywords,
            figures=figure_records,
            output_dir=str(doc_out),
        )

        self._write_outputs(doc_out, result, full_text)
        return result

    @staticmethod
    def _write_outputs(doc_out: Path, result: ExtractionResult, full_text: str) -> None:
        (doc_out / "ocr_full_text.txt").write_text(full_text, encoding="utf-8")
        (doc_out / "keywords.json").write_text(
            json.dumps(
                {
                    "document": result.source_file,
                    "all_keywords": result.all_keywords,
                    "per_page": [
                        {
                            "page_index": p.page_index,
                            "keywords": p.keywords,
                        }
                        for p in result.pages
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (doc_out / "extraction_report.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
