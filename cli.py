#!/usr/bin/env python3
"""CLI for aircraft document image and keyword extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Allow running as `python main.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import ExtractionPipeline


@click.command()
@click.argument(
    "inputs",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Recursively process supported files from this folder.",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default="/home/dell/IMAGE_EXTRACTION_OUTPUT",
    help="Directory for extracted images and JSON reports.",
)
@click.option(
    "--gpu",
    is_flag=True,
    help="Reserved for future GPU backends (RapidOCR ONNX uses CPU by default).",
)
@click.option(
    "--ocr",
    type=click.Choice(["rapid", "paddle"], case_sensitive=False),
    default="rapid",
    show_default=True,
    help="OCR engine to use (no Tesseract).",
)
@click.option(
    "--max-keywords",
    default=50,
    show_default=True,
    help="Maximum keywords to extract per document.",
)
def main(
    inputs: tuple[Path, ...],
    input_dir: Path | None,
    output_dir: Path,
    gpu: bool,
    ocr: str,
    max_keywords: int,
) -> None:
    """Extract keywords and manufacturing figures from PDF/TIFF/images."""
    supported_exts = {
        ".pdf",
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
    }

    docs: list[Path] = list(inputs)
    if input_dir is not None:
        docs.extend(
            [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported_exts]
        )

    # De-duplicate
    docs = sorted({p.resolve() for p in docs})

    if not docs:
        click.echo(
            "Provide at least one input file OR use --input-dir (supported: .pdf/.tif/.tiff/.png/.jpg/.jpeg/.bmp/.webp).",
            err=True,
        )
        raise SystemExit(1)

    import config

    config.OCR_GPU = gpu
    config.MAX_KEYWORDS = max_keywords

    # Select OCR engine
    if ocr.lower() == "rapid":
        from src.ocr_engine import OcrEngine as RapidOcrEngine

        engine = RapidOcrEngine()
    else:
        from src.paddle_ocr_engine import PaddleOcrEngine

        engine = PaddleOcrEngine()

    pipeline = ExtractionPipeline(ocr_engine=engine, output_dir=output_dir)
    for doc in docs:
        click.echo(f"Processing: {doc}")
        result = pipeline.process(doc)
        click.echo(f"  Keywords: {len(result.all_keywords)}")
        click.echo(f"  Figures:  {len(result.figures)}")
        click.echo(f"  Output:   {result.output_dir}")


if __name__ == "__main__":
    main()
