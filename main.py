"""FastAPI entry point for engineering document extraction."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

from src.ocr.ocr_router import OcrRouter
from src.pipeline import ExtractionPipeline

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Engineering Document Extraction API")


@app.post("/extract")
async def extract_file(file: UploadFile = File(...)) -> dict:
    """Upload a PDF/TIFF/image and run the full extraction pipeline."""
    suffix = Path(file.filename or "document.pdf").suffix
    upload_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pipeline = ExtractionPipeline(ocr_engine=OcrRouter(), output_dir=OUTPUT_DIR)
    result = pipeline.process(upload_path)

    return {
        "status": "success",
        "keywords": result.all_keywords,
        "figures_count": len(result.figures),
        "figures": result.figures,
        "output_dir": str(result.output_dir),
        "pages": [
            {
                "page_index": p.page_index,
                "ocr_engine": p.ocr_engine,
                "ocr_mean_confidence": p.ocr_mean_confidence,
                "engineering_figure_path": p.engineering_figure_path,
                "keyword_count": len(p.keyword_list),
            }
            for p in result.pages
        ],
    }
