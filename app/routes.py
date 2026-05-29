from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File

from src.pipeline import ExtractionPipeline
from src.paddle_ocr_engine import PaddleOcrEngine

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@router.post("/extract")
async def extract_file(
    file: UploadFile = File(...)
):

    suffix = Path(file.filename).suffix

    temp_name = f"{uuid.uuid4()}{suffix}"

    upload_path = UPLOAD_DIR / temp_name

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ocr_engine = PaddleOcrEngine()

    pipeline = ExtractionPipeline(
        ocr_engine=ocr_engine,
        output_dir=OUTPUT_DIR,
    )

    result = pipeline.process(upload_path)

    """return {
        "status": "success",
        "result": result,
    }"""

    return {
    "status": "success",
    "keywords": result.all_keywords,
    "figures_count": len(result.figures),
    "output_dir": str(result.output_dir),
    }