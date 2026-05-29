# Aircraft Document Extraction

Extract **keywords** from unstructured manufacturing text and **aircraft / figure images** from `.pdf`, `.tiff`, and single-page image files like `.png`.

- **OCR:** RapidOCR by default (ONNX; **not** Tesseract; no PyTorch), with an optional PaddleOCR mode
- **Keywords:** [YAKE](https://github.com/LIAAD/yake) (unsupervised keyphrase extraction)
- **Images:** Central manufacturing photograph crop (plus separate embedded images when the PDF contains real figures, not a full-page scan)

## Requirements

- Python 3.10+
- Linux/macOS/Windows

## Install

```bash
cd /home/dell/IMAGE_EXTRACTION
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On first run, RapidOCR downloads ONNX models (~50MB).

## Usage

```bash
python main.py document.pdf
python main.py scan.tiff -o /home/dell/EXTRACTED_RESULTS --max-keywords 80
python main.py doc1.pdf doc2.tiff --gpu
python main.py your_image.png --ocr paddle
python main.py --input-dir /home/dell/INPUT_DOCS -o /home/dell/IMAGE_EXTRACTION_OUTPUT
```

## Output layout

For each input file `X.pdf` / `X.tiff` / `X.png`, results go to:
- `$OUTPUT_BASE/X/` (where `$OUTPUT_BASE` is what you pass with `-o`, or the default)

| File | Description |
|------|-------------|
| `ocr_full_text.txt` | Full OCR text (all pages) |
| `keywords.json` | Ranked keywords (document + per page) |
| `extraction_report.json` | Complete structured report |
| `images/` | `*_manufacturing_photo.png` or `*_figure_*.png` (cropped images) |

## How it works

1. **Load** — PDF pages rendered at 200 DPI; multi-page TIFF supported.
2. **OCR** — RapidOCR detects text and bounding boxes on each page (or use `--ocr paddle`).
3. **Keywords** — YAKE extracts n-gram keyphrases from unstructured OCR text.
4. **Figures** — The central photograph is located, then trimmed to its visible frame border (margin handwriting removed). Full-page embedded rasters from scanned PDFs are skipped.

## Programmatic API

```python
from pathlib import Path
from src.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline(output_dir="/home/dell/IMAGE_EXTRACTION_OUTPUT")
result = pipeline.process(Path("aircraft_manual.pdf"))

print(result.all_keywords[:10])
for fig in result.figures:
    print(fig["path"])
```

## Tuning

Edit `config.py`:

- `OCR_MIN_CONFIDENCE` — filter low-confidence OCR lines
- `MAX_KEYWORDS` / `KEYWORD_NGRAM_MAX` — keyword count and phrase length
- `MIN_FIGURE_AREA_RATIO` — minimum figure size vs page area
- `MIN_EMBEDDED_IMAGE_BYTES` — skip tiny PDF icons

## Notes

- Scanned documents work best at 200+ DPI source quality.
- Layout detection assumes figures occupy sizable areas outside dense text blocks.
- OCR runs on CPU via ONNX Runtime; installs without the large PyTorch wheel.
