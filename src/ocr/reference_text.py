"""Load per-page reference text for OCR accuracy evaluation."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf as fitz

from src import config
from src.logger import get_logger

log = get_logger(__name__)


def _split_multipage_ground_truth(raw: str, page_count: int) -> list[str]:
    """Split a single ground-truth file into per-page strings."""
    if "\f" in raw:
        parts = raw.split("\f")
    elif re.search(r"^---\s*page\s+\d+\s*---", raw, flags=re.MULTILINE | re.IGNORECASE):
        parts = re.split(
            r"^---\s*page\s+\d+\s*---\s*$",
            raw,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        parts = [p for p in parts if p.strip()]
    else:
        parts = [raw]

    if len(parts) == 1 and page_count > 1:
        parts = [raw] * page_count
    while len(parts) < page_count:
        parts.append("")
    return [p.strip() for p in parts[:page_count]]


def _stem_variants(document_path: Path, original_filename: str | None) -> list[str]:
    """Candidate name stems for ground-truth lookup (API uploads use UUID filenames)."""
    seen: set[str] = set()
    stems: list[str] = []

    def add(value: str) -> None:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            stems.append(value)

    add(document_path.stem)
    if document_path.stem.startswith("streamlit_"):
        add(document_path.stem[len("streamlit_") :])
    if original_filename:
        add(Path(original_filename).stem)

    return stems


def _ground_truth_file_candidates(
    document_path: Path,
    original_filename: str | None,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            paths.append(path)

    for stem in _stem_variants(document_path, original_filename):
        add(document_path.parent / f"{stem}{config.OCR_GROUND_TRUTH_SUFFIX}")
        add(config.OCR_GROUND_TRUTH_DIR / f"{stem}{config.OCR_GROUND_TRUTH_SUFFIX}")

    return paths


def _load_ground_truth_file(path: Path, page_count: int) -> list[str] | None:
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8")
    pages = _split_multipage_ground_truth(raw, page_count)
    log.info("Loaded ground truth from %s (%d page(s))", path, len(pages))
    return pages


def _load_per_page_ground_truth(
    document_path: Path,
    page_count: int,
    *,
    original_filename: str | None = None,
) -> list[str] | None:
    pages: list[str] = []
    found_any = False

    for i in range(page_count):
        text = ""
        for stem in _stem_variants(document_path, original_filename):
            candidates = [
                document_path.parent / f"{stem}_page_{i}_ground_truth.txt",
                document_path.parent / f"{stem}_page_{i:03d}_ground_truth.txt",
                document_path.parent / "ground_truth" / stem / f"page_{i}.txt",
                document_path.parent / "ground_truth" / stem / f"page_{i:03d}.txt",
                config.OCR_GROUND_TRUTH_DIR / stem / f"page_{i}.txt",
                config.OCR_GROUND_TRUTH_DIR / stem / f"page_{i:03d}.txt",
                config.OCR_GROUND_TRUTH_DIR / f"{stem}_page_{i}_ground_truth.txt",
                config.OCR_GROUND_TRUTH_DIR / f"{stem}_page_{i:03d}_ground_truth.txt",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    text = candidate.read_text(encoding="utf-8").strip()
                    found_any = True
                    break
            if text:
                break
        pages.append(text)

    if not found_any:
        return None
    log.info("Loaded per-page ground truth (%d non-empty page(s))", sum(1 for p in pages if p))
    return pages


def extract_pdf_text_layer(pdf_path: Path, page_count: int) -> list[str] | None:
    """Extract embedded PDF text per page (not available for pure scans)."""
    if pdf_path.suffix.lower() != ".pdf":
        return None

    pages: list[str] = []
    doc = fitz.open(pdf_path)
    try:
        for i in range(min(len(doc), page_count)):
            pages.append(doc[i].get_text("text").strip())
    finally:
        doc.close()

    usable = sum(
        1 for p in pages if len(p) >= config.OCR_ACCURACY_MIN_REFERENCE_CHARS
    )
    if usable == 0:
        log.info(
            "PDF text layer empty or too short for accuracy (%s) — likely scanned PDF",
            pdf_path.name,
        )
        return None

    log.info(
        "PDF text layer: %d/%d page(s) have enough reference text",
        usable,
        len(pages),
    )
    while len(pages) < page_count:
        pages.append("")
    return pages[:page_count]


def ground_truth_hints(
    document_path: Path,
    original_filename: str | None = None,
) -> list[str]:
    """Human-readable paths the user can create to enable accuracy measurement."""
    stems = _stem_variants(document_path, original_filename)
    primary = stems[-1] if original_filename else stems[0]
    return [
        f"{config.OCR_GROUND_TRUTH_DIR / f'{primary}{config.OCR_GROUND_TRUTH_SUFFIX}'}",
        f"{config.OCR_GROUND_TRUTH_DIR / primary / 'page_0.txt'} (per-page)",
        f"{document_path.parent / f'{primary}{config.OCR_GROUND_TRUTH_SUFFIX}'}",
        "Use a PDF with an embedded text layer (not a pure scan)",
    ]


def load_reference_pages(
    document_path: Path,
    page_count: int,
    *,
    mode: str | None = None,
    original_filename: str | None = None,
) -> tuple[list[str], str]:
    """
    Return (reference_text_per_page, source_label).

    source_label is one of: ground_truth_file, pdf_text_layer, none.
    """
    mode = (mode or config.OCR_ACCURACY_REFERENCE_MODE).lower()
    path = Path(document_path)

    if mode == "none":
        return [""] * page_count, "none"

    if mode in {"auto", "ground_truth"}:
        for gt_path in _ground_truth_file_candidates(path, original_filename):
            gt_pages = _load_ground_truth_file(gt_path, page_count)
            if gt_pages is not None:
                return gt_pages, "ground_truth_file"

        gt_pages = _load_per_page_ground_truth(
            path, page_count, original_filename=original_filename
        )
        if gt_pages is not None:
            return gt_pages, "ground_truth_file"

        if mode == "ground_truth":
            log.warning("Ground truth mode enabled but no reference files found for %s", path.name)
            return [""] * page_count, "none"

    if mode in {"auto", "pdf_text"}:
        pdf_pages = extract_pdf_text_layer(path, page_count)
        if pdf_pages is not None:
            return pdf_pages, "pdf_text_layer"

    return [""] * page_count, "none"


def page_reference_usable(reference: str) -> bool:
    return len(reference.strip()) >= config.OCR_ACCURACY_MIN_REFERENCE_CHARS
