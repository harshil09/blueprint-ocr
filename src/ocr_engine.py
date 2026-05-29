"""RapidOCR (ONNX) text recognition — no Tesseract, no PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rapidocr_onnxruntime import RapidOCR

import config


@dataclass(frozen=True)
class TextBox:
    text: str #stores ocr text eg. "boeing 70"
    confidence: float 
    # axis-aligned bounding box: (x_min, y_min, x_max, y_max)
    bbox: tuple[int, int, int, int] #defines where text exists (100, 50, 300, 90) imp for masking and extraction of image


@dataclass(frozen=True)
class OcrPageResult: #container for full page ocr output
    full_text: str
    boxes: list[TextBox]


class OcrEngine:
    """Lazy-loaded RapidOCR reader."""

    _reader: RapidOCR | None = None

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool | None = None,
        min_confidence: float | None = None,
    ):
        self.languages = languages or config.OCR_LANGUAGES
        self.gpu = gpu if gpu is not None else config.OCR_GPU
        self.min_confidence = (
            min_confidence if min_confidence is not None else config.OCR_MIN_CONFIDENCE
        )

    def _get_reader(self) -> RapidOCR:
        if OcrEngine._reader is None:
            OcrEngine._reader = RapidOCR()
        return OcrEngine._reader

    @staticmethod
    # imp because OCR engines return rotated polygons:
    def _quad_to_aabb(points: list | np.ndarray) -> tuple[int, int, int, int]:
        arr = np.asarray(points)
        xs, ys = arr[:, 0], arr[:, 1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()) # creates rectangle Axis-aligned bounding box

    def recognize_page(self, image_bgr: np.ndarray) -> OcrPageResult:
        reader = self._get_reader()
        raw, _ = reader(image_bgr)

        boxes: list[TextBox] = []
        lines: list[str] = []
        if not raw:
            return OcrPageResult(full_text="", boxes=[])

        for item in raw:
            quad, text, conf = item
            try:
                score = float(conf)
            except (TypeError, ValueError):
                continue
            if score < self.min_confidence:
                continue
            text = str(text).strip()
            if not text:
                continue
            aabb = self._quad_to_aabb(quad)
            boxes.append(TextBox(text=text, confidence=score, bbox=aabb))
            lines.append(text)

        return OcrPageResult(full_text="\n".join(lines), boxes=boxes)


# Backward-compatible alias
EasyOcrEngine = OcrEngine
