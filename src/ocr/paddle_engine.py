"""PaddleOCR — fallback OCR for low-confidence RapidOCR results."""

from __future__ import annotations

import numpy as np

from src import config
from src.ocr.ocr_engine import OcrPageResult, TextBox
from src.utils import normalize_confidence, quad_to_aabb


class PaddleOcrEngine:
    """
    PaddleOCR 3.x engine (requires paddlepaddle in the same venv).

    Imported lazily so Rapid-only deployments work without Paddle installed.
    """

    _reader = None

    def __init__(
        self,
        lang: str | None = None,
        use_angle_cls: bool = True,
        min_confidence: float | None = None,
    ):
        self.lang = lang or (config.OCR_LANGUAGES[0] if config.OCR_LANGUAGES else "en")
        self.use_angle_cls = use_angle_cls
        self.min_confidence = (
            min_confidence if min_confidence is not None else config.OCR_MIN_CONFIDENCE
        )

    def _get_reader(self):
        if PaddleOcrEngine._reader is not None:
            return PaddleOcrEngine._reader

        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "PaddleOCR is not installed. Install with:\n"
                "  pip install paddlepaddle paddleocr\n"
                "Use Python 3.10 or 3.11."
            ) from e

        try:
            PaddleOcrEngine._reader = PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=bool(self.use_angle_cls),
                enable_mkldnn=False,
                text_det_limit_side_len=1500,
                text_det_limit_type="max",
                cpu_threads=4,
            )
        except RuntimeError as e:
            msg = str(e)
            if "paddle_static" in msg and "paddlepaddle" in msg:
                raise RuntimeError(
                    "PaddleOCR needs `paddlepaddle` in this venv. "
                    "Install: pip install paddlepaddle paddleocr"
                ) from e
            raise

        return PaddleOcrEngine._reader

    @staticmethod
    def _parse_v3_result(page_result) -> tuple[list[str], list, list[float]]:
        if page_result is None:
            return [], [], []
        get = getattr(page_result, "get", None)
        if get is None:
            return [], [], []
        texts = list(get("rec_texts") or [])
        polys = list(get("rec_polys") or get("dt_polys") or [])
        scores = list(get("rec_scores") or [])
        if scores and len(scores) == len(texts):
            return texts, polys, [float(s) for s in scores]
        return texts, polys, [1.0] * len(texts)

    @staticmethod
    def _parse_legacy_result(page) -> tuple[list[str], list, list[float]]:
        texts: list[str] = []
        polys: list = []
        scores: list[float] = []
        if not page:
            return texts, polys, scores
        for item in page:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            box, text_conf = item[0], item[1]
            if not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
                continue
            text = str(text_conf[0]).strip()
            if not text:
                continue
            try:
                score = float(text_conf[1])
            except (TypeError, ValueError):
                continue
            texts.append(text)
            polys.append(box)
            scores.append(score)
        return texts, polys, scores

    def recognize_page(self, image_bgr: np.ndarray) -> OcrPageResult:
        reader = self._get_reader()
        image_rgb = image_bgr[:, :, ::-1]
        result = reader.predict(image_rgb)
        if not result:
            return OcrPageResult(full_text="", boxes=[], engine="paddle")

        page = result[0]
        texts, polys, scores = self._parse_v3_result(page)
        if not texts:
            texts, polys, scores = self._parse_legacy_result(page)

        boxes: list[TextBox] = []
        lines: list[str] = []
        for text, poly, score in zip(texts, polys, scores):
            text = str(text).strip()
            conf = normalize_confidence(float(score))
            if not text or conf < self.min_confidence:
                continue
            aabb = quad_to_aabb(poly)
            boxes.append(TextBox(text=text, confidence=conf, bbox=aabb))
            lines.append(text)

        return OcrPageResult(full_text="\n".join(lines), boxes=boxes, engine="paddle")

    def recognize_region(
        self,
        image_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> OcrPageResult:
        x0, y0, x1, y1 = bbox
        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return OcrPageResult(full_text="", boxes=[], engine="paddle")
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
            engine="paddle",
        )
