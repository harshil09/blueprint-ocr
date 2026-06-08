"""
Streamlit UI for engineering document extraction.

Uses the FastAPI backend by default, or runs the pipeline in-process when
"Run locally" is enabled in the sidebar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

# Project root on path when launched as: streamlit run app/streamlit_app.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config

SUPPORTED_TYPES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _run_local_pipeline(uploaded_file) -> dict:
    from src.logger import setup_logging
    from src.ocr.ocr_router import OcrRouter
    from src.pipeline import ExtractionPipeline

    setup_logging()

    upload_dir = _ROOT / "uploads"
    output_dir = _ROOT / "outputs"
    upload_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    suffix = Path(uploaded_file.name).suffix or ".pdf"
    upload_path = upload_dir / f"streamlit_{uploaded_file.name}"
    upload_path.write_bytes(uploaded_file.getvalue())

    pipeline = ExtractionPipeline(ocr_engine=OcrRouter(), output_dir=output_dir)
    result = pipeline.process(upload_path, original_filename=uploaded_file.name)

    return {
        "status": "success",
        "keywords": result.all_keywords,
        "figures_count": len(result.figures),
        "output_dir": str(result.output_dir),
        "figures": result.figures,
        "ocr_accuracy": result.ocr_accuracy_summary,
        "pages": [
            {
                "page_index": p.page_index,
                "ocr_engine": p.ocr_engine,
                "ocr_mean_confidence": p.ocr_mean_confidence,
                "engineering_figure_path": p.engineering_figure_path,
                "primary_figure_path": p.primary_figure_path,
                "figure_method": p.figure_method,
                "figure_type": p.figure_type,
                "figure_confidence": p.figure_confidence,
                "page_profile": p.page_profile,
                "keyword_count": len(p.keyword_list),
                "rapid_ocr_accuracy": p.rapid_ocr_accuracy,
                "pipeline_ocr_accuracy": p.pipeline_ocr_accuracy,
            }
            for p in result.pages
        ],
        "full_report": result.to_dict(),
    }


def _call_api(api_base: str, uploaded_file, timeout: int) -> dict:
    url = f"{api_base.rstrip('/')}/extract"
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    response = requests.post(url, files=files, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _list_preview_images(output_dir: str) -> list[Path]:
    root = Path(output_dir)
    if not root.is_dir():
        return []
    paths: list[Path] = []
    primary = sorted(root.glob("images/primary_figure*"))
    if primary:
        return [p for p in primary if p.is_file()]
    legacy = sorted(root.glob("engineering_figure_*.png"))
    if legacy:
        return [p for p in legacy if p.is_file()]
    return [
        p
        for p in sorted(root.glob("images/*"))
        if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
    ]


def _render_results(data: dict) -> None:
    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    keywords = data.get("keywords") or []
    keyword_count = len(keywords)
    c1.metric("Keywords (count)", keyword_count)
    figures_count = data.get("figures_count", 0)
    c2.metric("Primary figures", figures_count)
    c3.metric("Pages", len(data.get("pages") or []))
    if figures_count == 0:
        st.error(
            "No figure was extracted for this document. "
            "Check logs or try a clearer scan. Output is under the directory below."
        )
    if config.MAX_KEYWORDS is not None and keyword_count == config.MAX_KEYWORDS:
        c1.caption(f"Capped at MAX_KEYWORDS={config.MAX_KEYWORDS}")

    if data.get("output_dir"):
        st.caption(f"Output directory: `{data['output_dir']}`")

    if keywords:
        st.subheader("Keywords")
        st.write(", ".join(f"`{kw}`" for kw in keywords))
        #if len(keywords) > 40:
         #   st.caption(f"+ {len(keywords) - 40} more")

    pages = data.get("pages") or []
    if pages:
        st.subheader("Per-page OCR")
        st.dataframe(pages, use_container_width=True, hide_index=True)

    ocr_accuracy = data.get("ocr_accuracy")
    if ocr_accuracy:
        st.subheader("OCR accuracy / quality")
        status = ocr_accuracy.get("status", "unknown")
        if status == "measured":
            rapid = ocr_accuracy.get("rapid_ocr") or {}
            st.success(
                f"Ground-truth accuracy measured ({ocr_accuracy.get('reference_source')}). "
                f"Char: **{(rapid.get('mean_char_accuracy') or 0) * 100:.1f}%** · "
                f"Word: **{(rapid.get('mean_word_accuracy') or 0) * 100:.1f}%**"
            )
        else:
            st.warning(
                "Ground-truth accuracy not available (scanned PDF or no reference file). "
                "See `ocr_accuracy.json` in the output folder for confidence proxy metrics."
            )
            proxy = ocr_accuracy.get("confidence_proxy") or {}
            summary = proxy.get("summary") or {}
            if summary.get("mean_confidence") is not None:
                st.metric(
                    "OCR confidence proxy (not accuracy)",
                    f"{summary['mean_confidence'] * 100:.1f}%",
                    help="Engine-reported confidence. Add a ground-truth file for real accuracy.",
                )
            hints = ocr_accuracy.get("how_to_enable_accuracy") or []
            if hints:
                st.caption("To enable real accuracy, add ground truth at:")
                for hint in hints:
                    st.code(hint, language=None)

    output_dir = data.get("output_dir")
    if output_dir:
        accuracy_path = Path(output_dir) / "ocr_accuracy.json"
        if accuracy_path.is_file():
            st.caption(f"Full report: `{accuracy_path}`")

    figures = data.get("figures")
    if figures:
        st.subheader("Primary extracted figures")
        st.dataframe(figures, use_container_width=True, hide_index=True)
    elif pages:
        st.caption("Per-page primary figure paths are in the table above (`primary_figure_path`).")

    output_dir = data.get("output_dir")
    if output_dir:
        previews = _list_preview_images(output_dir)
        if previews:
            st.subheader("Figure previews")
            cols = st.columns(min(3, len(previews)))
            for i, img_path in enumerate(previews[:9]):
                with cols[i % len(cols)]:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
            if len(previews) > 9:
                st.caption(f"Showing 9 of {len(previews)} images under `{output_dir}`")

    with st.expander("Full JSON response"):
        st.json(data.get("full_report") or data)


st.set_page_config(
    page_title="Engineering Document Extraction",
    page_icon="📐",
    layout="wide",
)

st.title("Engineering Document Extraction")
st.markdown(
    "Upload aircraft drawings, blueprints, or technical PDFs/TIFFs. "
    "OpenCV morphology + ensemble OCR (RapidOCR → PaddleOCR fallback)."
)

with st.sidebar:
    st.header("Settings")
    run_local = st.checkbox(
        "Run pipeline locally",
        value=False,
        help="Process in this Python process (no FastAPI server required).",
    )
    api_base = st.text_input(
        "API base URL",
        value="http://127.0.0.1:8000",
        disabled=run_local,
    )
    timeout = st.number_input("API timeout (seconds)", min_value=30, max_value=600, value=300)
    st.divider()
    st.caption(
        "Start API: `uvicorn main:app --reload`  \n"
        "CLI: `python cli.py your_file.pdf`"
    )

uploaded_file = st.file_uploader(
    "Upload document",
    type=SUPPORTED_TYPES,
    help="PDF, TIFF, or raster image",
)

status = st.empty()

if st.button("Process", type="primary", disabled=uploaded_file is None):
    if not uploaded_file:
        st.warning("Upload a file first.")
    else:
        status.info("Processing… this may take a minute for large drawings.")
        try:
            if run_local:
                data = _run_local_pipeline(uploaded_file)
            else:
                data = _call_api(api_base, uploaded_file, int(timeout))
            status.success("Done.")
            _render_results(data)
        except requests.exceptions.ConnectionError:
            status.error(
                "Could not reach the API. Start it with "
                "`uvicorn main:app --reload` or enable **Run pipeline locally**."
            )
        except requests.exceptions.HTTPError as e:
            status.error(f"API error ({e.response.status_code}): {e.response.text}")
        except Exception as e:
            status.error(f"Processing failed: {e}")

elif uploaded_file is None:
    st.info("Choose a file, then click **Process**.")
