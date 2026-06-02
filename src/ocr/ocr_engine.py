"""RapidOCR (ONNX) — primary OCR engine and shared OCR types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rapidocr_onnxruntime import RapidOCR

from src import config
from src.utils import normalize_confidence, quad_to_aabb


@dataclass(frozen=True)
class TextBox:
    """Single OCR detection with axis-aligned bounding box."""

    text: str
    confidence: float
    # (x_min, y_min, x_max, y_max)
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class OcrPageResult:
    """Full-page (or region-merged) OCR output."""

    full_text: str
    boxes: list[TextBox]
    engine: str = "rapid"

    @property
    def box_count(self) -> int:
        return len(self.boxes)


class OcrEngine:
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

    def recognize_page(self, image_bgr: np.ndarray) -> OcrPageResult:
        reader = self._get_reader()
        raw, _ = reader(image_bgr)

        boxes: list[TextBox] = []
        lines: list[str] = []
        if not raw:
            return OcrPageResult(full_text="", boxes=[], engine="rapid")

        for item in raw:
            quad, text, conf = item
            try:
                score = normalize_confidence(float(conf))
            except (TypeError, ValueError):
                continue
            if score < self.min_confidence:
                continue
            text = str(text).strip()
            if not text:
                continue
            aabb = quad_to_aabb(quad)
            boxes.append(TextBox(text=text, confidence=score, bbox=aabb))
            lines.append(text)

        return OcrPageResult(full_text="\n".join(lines), boxes=boxes, engine="rapid")

    def recognize_region(
        self,
        image_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> OcrPageResult:
        """OCR a crop; bbox coordinates are offset back to page space."""
        x0, y0, x1, y1 = bbox
        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return OcrPageResult(full_text="", boxes=[], engine="rapid")
        result = self.recognize_page(crop)
        shifted: list[TextBox] = []
        for box in result.boxes:
            bx0, by0, bx1, by1 = box.bbox
            shifted.append(
                TextBox(
                    text=box.text,
                    confidence=box.confidence,
                    bbox=(bx0 + x0, by0 + y0, bx1 + x0, by1 + y0),
                )
            )
        return OcrPageResult(
            full_text=result.full_text,
            boxes=shifted,
            engine="rapid",
        )


RapidOcrEngine = OcrEngine
