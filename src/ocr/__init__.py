"""OCR engines and router."""

from src.ocr.ocr_engine import OcrEngine, OcrPageResult, RapidOcrEngine, TextBox
from src.ocr.paddle_engine import PaddleOcrEngine
from src.ocr.ocr_router import OcrRouter, mean_page_confidence
