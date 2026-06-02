"""Load PDF, TIFF, and image files as page images (BGR arrays for OpenCV)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz
import numpy as np
from PIL import Image

from src import config
from src.utils import pil_to_bgr


@dataclass(frozen=True)
class PageImage:
    source_path: Path
    page_index: int
    image: np.ndarray
    width: int
    height: int


def _make_page(path: Path, page_index: int, bgr: np.ndarray) -> PageImage:
    h, w = bgr.shape[:2]
    return PageImage(path, page_index, bgr, w, h)


def _load_pdf(path: Path, dpi: int) -> list[PageImage]:
    pages = []
    doc = fitz.open(path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pil = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(_make_page(path, i, pil_to_bgr(pil)))
    finally:
        doc.close()
    return pages


def _load_tiff(path: Path, dpi: int) -> list[PageImage]:
    pages = []
    with Image.open(path) as img:
        n = getattr(img, "n_frames", 1)
        for i in range(n):
            img.seek(i)
            pages.append(_make_page(path, i, pil_to_bgr(img.convert("RGB"))))
    return pages


def _load_single_image(path: Path) -> list[PageImage]:
    with Image.open(path) as img:
        bgr = pil_to_bgr(img.convert("RGB"))
    return [_make_page(path, 0, bgr)]


def load_document(path: Path | str, dpi: int | None = None) -> list[PageImage]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    dpi = dpi or config.OUTPUT_DPI
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _load_pdf(path, dpi)
    if ext in {".tif", ".tiff"}:
        return _load_tiff(path, dpi)
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return _load_single_image(path)

    raise ValueError(
        f"Unsupported format: {ext}. Use .pdf, .tif, .tiff, or common image formats."
    )
