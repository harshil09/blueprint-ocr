"""Load PDF and TIFF documents into page images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz
import numpy as np
from PIL import Image

import config


@dataclass(frozen=True)
class PageImage:
    source_path: Path
    page_index: int
    image: np.ndarray  # BGR uint8 (OpenCV convention)
    width: int
    height: int


def _pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def _load_pdf_pages(path: Path, dpi: int) -> list[PageImage]:
    doc = fitz.open(path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[PageImage] = []
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pil = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            bgr = _pil_to_bgr(pil)
            pages.append(
                PageImage(
                    source_path=path,
                    page_index=i,
                    image=bgr,
                    width=bgr.shape[1],
                    height=bgr.shape[0],
                )
            )
    finally:
        doc.close()
    return pages


def _load_tiff_pages(path: Path, dpi: int) -> list[PageImage]:
    pages: list[PageImage] = []
    with Image.open(path) as img:
        frame_count = getattr(img, "n_frames", 1)
        for i in range(frame_count):
            img.seek(i)
            frame = img.convert("RGB")
            bgr = _pil_to_bgr(frame)
            pages.append(
                PageImage(
                    source_path=path,
                    page_index=i,
                    image=bgr,
                    width=bgr.shape[1],
                    height=bgr.shape[0],
                )
            )
    return pages


def _load_image_pages(path: Path) -> list[PageImage]:
    """Load a single raster image (png/jpg/jpeg/bmp/webp) as one page."""
    pages: list[PageImage] = []
    with Image.open(path) as img:
        frame = img.convert("RGB")
        bgr = _pil_to_bgr(frame)
        pages.append(
            PageImage(
                source_path=path,
                page_index=0,
                image=bgr,
                width=bgr.shape[1],
                height=bgr.shape[0],
            )
        )
    return pages


def load_document(path: Path | str, dpi: int | None = None) -> list[PageImage]:
    """Load all pages from a PDF or TIFF file as BGR images."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    render_dpi = dpi or config.OUTPUT_DPI
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf_pages(path, render_dpi)
    if suffix in {".tif", ".tiff"}:
        return _load_tiff_pages(path, render_dpi)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return _load_image_pages(path)

    raise ValueError(
        f"Unsupported format: {suffix}. Use .pdf, .tif, .tiff, or common image formats."
    )
