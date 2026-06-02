"""OCR engines and router."""

from src.ocr.ocr_engine import OcrEngine, OcrPageResult, RapidOcrEngine, TextBox
from src.ocr.paddle_engine import PaddleOcrEngine
from src.ocr.accuracy import OcrAccuracyMetrics, aggregate_accuracy, compare_ocr_to_reference
from src.ocr.ocr_router import OcrRouter, mean_page_confidence, ocr_confidence_stats
from src.ocr.reference_text import load_reference_pages, page_reference_usable
