"""PaddleOCR-based text recognition (no Tesseract)."""

from __future__ import annotations

import numpy as np

import config
from src.ocr_engine import OcrPageResult, TextBox


class PaddleOcrEngine:
    """
    OCR engine using PaddleOCR 3.x (requires paddlepaddle in the same venv).

    PaddleOCR is imported lazily so `--ocr rapid` works without Paddle installed.
    """

    _reader = None

    def __init__(
        self,
        lang: str | None = None,
        use_angle_cls: bool = True,
        min_confidence: float | None = None,
    ):
        self.lang = lang or "en"
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
                "Use Python 3.10 or 3.11 (not 3.14)."
            ) from e

        try:
            # PaddlePaddle 3.3.x: MKLDNN/PIR path crashes on CPU — disable it.
            PaddleOcrEngine._reader = PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=False, #Disables page rotation classifier. saves memory
                use_doc_unwarping=False,#Disables curved-page correction.used for books and camera photos
                use_textline_orientation=bool(self.use_angle_cls),#Detects rotated text lines.
                #use_textline_orientation=False,
                enable_mkldnn=False,

                # IMPORTANT MEMORY SETTINGS
                text_det_limit_side_len=1500,
                text_det_limit_type="max",

                # CPU optimization
                cpu_threads=4,

            )
        except RuntimeError as e:
            msg = str(e)
            if "paddle_static" in msg and "paddlepaddle" in msg:
                raise RuntimeError(
                    "PaddleOCR needs `paddlepaddle` in this venv. "
                    "Install: pip install paddlepaddle paddleocr\n"
                    "Or use: python main.py ... --ocr rapid"
                ) from e
            raise

        return PaddleOcrEngine._reader

    @staticmethod
    def _quad_to_aabb(points) -> tuple[int, int, int, int]:
        arr = np.asarray(points, dtype=float)
        xs, ys = arr[:, 0], arr[:, 1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _parse_v3_result(page_result) -> tuple[list[str], list[tuple], list[float]]:
        """Parse PaddleOCR 3.x OCRResult (dict-like)."""
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
    def _parse_legacy_result(page) -> tuple[list[str], list[tuple], list[float]]:
        """Parse PaddleOCR 2.x style: [[box, (text, conf)], ...]."""
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

        # PaddleOCR 3.x: use predict() (ocr() is deprecated; cls= is invalid)
        result = reader.predict(image_rgb)
        if not result:
            return OcrPageResult(full_text="", boxes=[])

        page = result[0]

        texts, polys, scores = self._parse_v3_result(page)
        if not texts:
            texts, polys, scores = self._parse_legacy_result(page)

        boxes: list[TextBox] = []
        lines: list[str] = []

        for text, poly, score in zip(texts, polys, scores):
            text = str(text).strip()
            if not text or score < self.min_confidence:
                continue
            aabb = self._quad_to_aabb(poly)
            boxes.append(TextBox(text=text, confidence=float(score), bbox=aabb))
            lines.append(text)

        return OcrPageResult(full_text="\n".join(lines), boxes=boxes)
