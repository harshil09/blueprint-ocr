"""Phase 1 crop reliability tests (stdlib runner — no pytest required)."""

from __future__ import annotations

import cv2
import numpy as np

from src.extract_figure import (
    _shrink_xyxy_to_max_area,
    compute_drawing_zone_bbox,
    figure_crop_reject_reason,
    prepare_primary_crop,
)
from src.profile_config import (
    CropProfile,
    apply_dynamic_layout_zones,
    compute_page_metrics,
    detect_scanned_raster_page,
    effective_max_figure_output_area,
    get_profile_config,
    is_full_page_embedded_pdf,
    scale_profile_config,
)
from src.utils.image_metrics import page_edge_density, page_mean_saturation
from src import config
from src.ocr.ocr_engine import OcrPageResult, TextBox


def _blank_page(w: int = 800, h: int = 600) -> np.ndarray:
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def test_shrink_xyxy_reduces_oversize_bbox():
    x1, y1, x2, y2 = _shrink_xyxy_to_max_area(50, 50, 750, 550, (600, 800), 0.30)
    area = (x2 - x1) * (y2 - y1)
    assert area <= 800 * 600 * 0.30 + 1


def test_figure_crop_reject_reason_area_too_large():
    pcfg = get_profile_config(CropProfile.BLUEPRINT_LARGE)
    reason = figure_crop_reject_reason(
        (0, 0, 700, 500),
        (600, 800),
        profile_config=pcfg,
    )
    assert reason == "area_too_large"


def test_apply_dynamic_layout_zones_expands_title_block():
    pcfg = get_profile_config(CropProfile.BLUEPRINT_LARGE)
    boxes = [
        TextBox(text="A", confidence=0.9, bbox=(500, 520, 560, 540)),
        TextBox(text="B", confidence=0.9, bbox=(600, 530, 660, 550)),
        TextBox(text="C", confidence=0.9, bbox=(520, 560, 580, 580)),
        TextBox(text="D", confidence=0.9, bbox=(610, 565, 670, 585)),
        TextBox(text="E", confidence=0.9, bbox=(540, 540, 600, 560)),
        TextBox(text="F", confidence=0.9, bbox=(630, 550, 690, 570)),
    ]
    ocr = OcrPageResult(full_text="x", boxes=boxes, engine="rapid")
    updated = apply_dynamic_layout_zones(pcfg, ocr, (600, 800))
    assert updated.title_block_width_ratio >= pcfg.title_block_width_ratio


def test_scale_profile_disabled_by_default():
    pcfg = get_profile_config(CropProfile.CAD_WIDE)
    metrics = compute_page_metrics(_blank_page(), None)
    scaled, summary = scale_profile_config(pcfg, metrics)
    assert summary["scaling"] == "disabled"
    assert scaled.max_figure_output_area_ratio == pcfg.max_figure_output_area_ratio


def test_drawing_zone_merger_trims_oversize_projection():
    img = _blank_page()
    cv2.line(img, (80, 280), (720, 280), (0, 0, 0), 2)
    cv2.line(img, (80, 320), (720, 320), (0, 0, 0), 2)
    cv2.rectangle(img, (100, 200), (200, 260), (0, 0, 0), 2)

    morph = (100, 210, 120, 60)
    proj = (60, 180, 700, 200)
    pcfg = get_profile_config(CropProfile.BLUEPRINT_LARGE)
    box, source = compute_drawing_zone_bbox(
        img, morph, proj, [], profile_config=pcfg
    )
    assert box is not None
    assert source in ("drawing_zone", "projection_trimmed", "morphology", "projection")
    page_area = img.shape[0] * img.shape[1]
    assert (box[2] * box[3]) / page_area < 0.55


def test_effective_max_figure_output_area_caps_cad_wide():
    pcfg = get_profile_config(CropProfile.CAD_WIDE)
    assert pcfg.max_figure_output_area_ratio == 0.48
    assert effective_max_figure_output_area(pcfg) == config.CAD_WIDE_MAX_OUTPUT_AREA_RATIO


def test_detect_scanned_raster_page_high_res_grayscale():
    img = np.ones((3200, 2400, 3), dtype=np.uint8) * 245
    for y in range(400, 2800, 80):
        cv2.line(img, (200, y), (2200, y), (30, 30, 30), 2)
    for x in range(300, 2100, 120):
        cv2.line(img, (x, 500), (x, 2500), (30, 30, 30), 1)
    assert detect_scanned_raster_page(img, embedded_on_page=0)


def test_prepare_primary_crop_cad_wide_respects_area_cap():
    img = _blank_page(1200, 800)
    for x in range(100, 1100, 35):
        cv2.line(img, (x, 320), (x, 480), (0, 0, 0), 1)
    pcfg = get_profile_config(CropProfile.CAD_WIDE)
    cap = effective_max_figure_output_area(pcfg)
    result = prepare_primary_crop(
        img, 40, 280, 1160, 520, [], profile_config=pcfg
    )
    assert result is not None
    _crop, bbox = result
    area_ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (img.shape[0] * img.shape[1])
    assert area_ratio <= cap + 0.02


def test_page_image_metrics_shared():
    img = _blank_page()
    cv2.line(img, (100, 300), (700, 300), (0, 0, 0), 2)
    assert page_edge_density(img) > 0.0
    assert page_mean_saturation(img) >= 0.0


def test_is_full_page_embedded_pdf_semantics():
    assert is_full_page_embedded_pdf(embedded_on_page=1, is_pdf=True)
    assert not is_full_page_embedded_pdf(embedded_on_page=1, is_pdf=False)
    assert not is_full_page_embedded_pdf(embedded_on_page=0, is_pdf=True)


def test_prepare_primary_crop_recovers_from_oversize_seed():
    img = _blank_page()
    for x in range(120, 680, 40):
        cv2.line(img, (x, 250), (x, 350), (0, 0, 0), 1)
    pcfg = get_profile_config(CropProfile.BLUEPRINT_LARGE)
    result = prepare_primary_crop(
        img, 50, 200, 750, 400, [], profile_config=pcfg
    )
    assert result is not None
    _crop, bbox = result
    area_ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (img.shape[0] * img.shape[1])
    assert area_ratio <= pcfg.max_figure_output_area_ratio + 0.02
