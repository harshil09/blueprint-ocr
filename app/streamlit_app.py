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

SUPPORTED_TYPES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _run_local_pipeline(uploaded_file) -> dict:
    from src import config
    from src.ocr.ocr_router import OcrRouter
    from src.pipeline import ExtractionPipeline

    upload_dir = _ROOT / "uploads"
    output_dir = _ROOT / "outputs"
    upload_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    suffix = Path(uploaded_file.name).suffix or ".pdf"
    upload_path = upload_dir / f"streamlit_{uploaded_file.name}"
    upload_path.write_bytes(uploaded_file.getvalue())

    pipeline = ExtractionPipeline(ocr_engine=OcrRouter(), output_dir=output_dir)
    result = pipeline.process(upload_path)

    return {
        "status": "success",
        "keywords": result.all_keywords,
        "figures_count": len(result.figures),
        "output_dir": str(result.output_dir),
        "figures": result.figures,
        "pages": [
            {
                "page_index": p.page_index,
                "ocr_engine": p.ocr_engine,
                "ocr_mean_confidence": p.ocr_mean_confidence,
                "engineering_figure_path": p.engineering_figure_path,
                "keyword_count": len(p.keyword_list),
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
    for pattern in ("engineering_figure_*.png", "images/*"):
        paths.extend(sorted(root.glob(pattern)))
    return [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()]


def _render_results(data: dict) -> None:
    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Keywords", len(data.get("keywords") or []))
    c2.metric("Figures", data.get("figures_count", 0))
    c3.metric("Pages", len(data.get("pages") or []))

    if data.get("output_dir"):
        st.caption(f"Output directory: `{data['output_dir']}`")

    keywords = data.get("keywords") or []
    if keywords:
        st.subheader("Keywords")
        st.write(", ".join(f"`{kw}`" for kw in keywords[:40]))
        if len(keywords) > 40:
            st.caption(f"+ {len(keywords) - 40} more")

    pages = data.get("pages") or []
    if pages:
        st.subheader("Per-page OCR")
        st.dataframe(pages, use_container_width=True, hide_index=True)

    figures = data.get("figures")
    if figures:
        st.subheader("Extracted figures")
        st.dataframe(figures, use_container_width=True, hide_index=True)

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
