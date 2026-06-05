"""Central configuration for the engineering document extraction pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE: Path | None = None  # e.g. Path("logs/extraction.log")

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
OCR_LANGUAGES = ["en"]
OCR_GPU = False
OCR_MIN_CONFIDENCE = 0.25
# Mean page confidence below this triggers PaddleOCR fallback in the router.
OCR_ROUTER_FALLBACK_MEAN_CONF = 0.45
# Boost applied when RapidOCR and PaddleOCR agree on normalized text.
OCR_ENSEMBLE_AGREEMENT_BOOST = 0.08
# Minimum fraction of boxes that must agree to apply ensemble boost.
OCR_ENSEMBLE_MIN_AGREEMENT_RATIO = 0.35

# Ground-truth OCR accuracy (CER/WER vs reference text).
# Requires reference from ground-truth files and/or PDF text layer (auto mode).
OCR_ACCURACY_ENABLED = True
# auto | ground_truth | pdf_text | none
OCR_ACCURACY_REFERENCE_MODE = "auto"
OCR_ACCURACY_MIN_REFERENCE_CHARS = 20
OCR_GROUND_TRUTH_SUFFIX = "_ground_truth.txt"
# Project folder for ground-truth files (works with API UUID uploads).
OCR_GROUND_TRUTH_DIR = Path(__file__).resolve().parent.parent / "ground_truth"
# Measure RapidOCR specifically (always) plus pipeline router result when different.
OCR_ACCURACY_MEASURE_RAPID = True
OCR_ACCURACY_MEASURE_PIPELINE = True

# ---------------------------------------------------------------------------
# Keyword extraction (YAKE)
# ---------------------------------------------------------------------------
# Set to None to disable capping and return as many keywords as YAKE can rank.
MAX_KEYWORDS: int | None = None
KEYWORD_NGRAM_MAX = 3
KEYWORD_DEDUP_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Figure / layout extraction
# ---------------------------------------------------------------------------
MIN_FIGURE_AREA_RATIO = 0.05
# Reject morphology/layout crops larger than this fraction of the page.
# (MAX_FIGURE_OUTPUT_AREA_RATIO defined with engineering figure settings below)
TEXT_MASK_PADDING_PX = 12
# Fixed layout zones masked even when OCR misses text (notes / title block).
NOTES_BLOCK_WIDTH_RATIO = 0.42
NOTES_BLOCK_HEIGHT_RATIO = 0.28
TITLE_BLOCK_WIDTH_RATIO = 0.48
# Full-width bottom strip (notes, LOFT, title block) excluded from figure detection.
BOTTOM_ANNOTATION_BAND_RATIO = 0.45
# Final crop cannot extend below this fraction of page height.
# Exclude right-edge sidebar text / scan artifacts from drawing detection.
PAGE_MARGIN_RIGHT_RATIO = 0.08
# Edge-projection fallback (finds full line-art region on hollow CAD drawings).
PROJECTION_MIN_PROFILE_SUM = 40
# Valid engineering-figure crop (reject slivers and tiny fragments).
MIN_CROP_WIDTH_RATIO = 0.32
MIN_CROP_HEIGHT_RATIO = 0.20
MIN_CROP_AREA_RATIO = 0.12
# Wide side views (fuselage, wing profiles) exceed 3.5:1; align with layout figure limit.
MAX_CROP_ASPECT_RATIO = 3.8
MIN_CROP_ASPECT_RATIO = 0.28
# Prefer projection bbox when morphology covers less than this fraction of it.
MORPHOLOGY_PROJECTION_MIN_IOU = 0.38
MORPHOLOGY_PROJECTION_MIN_AREA_FRAC = 0.62
MIN_EMBEDDED_IMAGE_BYTES = 8_000
SKIP_FULL_PAGE_EMBEDDED = True
FULL_PAGE_EMBEDDED_RATIO = 0.85

# Manufacturing photo (central photograph) detection
PHOTO_SEARCH_TOP = 0.12
PHOTO_SEARCH_BOTTOM = 0.80
PHOTO_SEARCH_LEFT = 0.06
PHOTO_SEARCH_RIGHT = 0.94
PHOTO_MIN_AREA_RATIO = 0.08
PHOTO_MAX_AREA_RATIO = 0.55
PHOTO_SATURATION_MAX = 50
PHOTO_BBOX_PADDING_PX = 10
PHOTO_PROJECTION_THRESHOLD = 0.12
PHOTO_FRAME_TRIM_ENABLED = True
PHOTO_FRAME_INSET_PX = 5
PHOTO_FRAME_EDGE_LEFT_BAND = (0.05, 0.22)
PHOTO_FRAME_EDGE_RIGHT_BAND = (0.78, 0.95)
PHOTO_FRAME_MIN_AREA_VS_COARSE = 0.25
PHOTO_MAX_TEXT_COVERAGE = 0.09

OUTPUT_IMAGE_FORMAT = "png"
OUTPUT_DPI = 300

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path("/home/dell/IMAGE_EXTRACTION_OUTPUT")

# ---------------------------------------------------------------------------
# Engineering figure extraction (OpenCV morphology pipeline)
# ---------------------------------------------------------------------------
MIN_COMPONENT_AREA_RATIO = 0.01
MAX_COMPONENT_AREA_RATIO = 0.70
BORDER_MARGIN_RATIO = 0.03
TITLE_BLOCK_HEIGHT_RATIO = 0.22
# Run connected-component analysis at this max side (px) for scale-stable crops.
MORPHOLOGY_ANALYSIS_MAX_SIDE = 2048
MORPH_KERNEL = (3, 3)
CENTRALITY_WEIGHT = 0.25
AREA_WEIGHT = 0.20
DENSITY_WEIGHT = 0.20
ASPECT_WEIGHT = 0.15
VERTICAL_POSITION_WEIGHT = 0.20
FIGURE_PADDING = 25

# Figure region expansion (merge fragmented connected components)
MERGE_HORIZONTAL_GAP_RATIO = 0.05
MERGE_VERTICAL_GAP_RATIO = 0.05

# Multi-candidate selection + quality gate
TOP_K_CANDIDATES = 5
QUALITY_GATE_THRESHOLD = 0.45
IOU_DUPLICATE_THRESHOLD = 0.70
MIN_EDGE_DENSITY = 0.03
MIN_DRAWING_COVERAGE = 0.015
MAX_TEXT_OVERLAP = 0.35

# Final crop expansion after validated selection (fraction of box width/height)
FINAL_EXPANSION_RATIO = 0.06
# Content-aware crop: search band around candidate box and uniform margin on ink bounds.
CROP_SEARCH_EXPAND_RATIO = 0.22
#0.06 was
CROP_CONTENT_MARGIN_RATIO = 0.06
CROP_MIN_MARGIN_PX = 28
CROP_INK_GRAY_MAX = 248
# Softer projection thresholds to include centerlines and dimension extensions.
PROJECTION_EDGE_ROW_THRESH = 0.03
PROJECTION_EDGE_COL_THRESH = 0.03
# Allow larger crops when the drawing fills the sheet (actuator, engine side views).
MAX_FIGURE_OUTPUT_AREA_RATIO = 0.65 
DRAWING_ZONE_MAX_BOTTOM_RATIO = 0.78

# Bounding-box refinement: union with original box so thin extensions are not clipped.
REFINEMENT_USE_UNION = True
REFINEMENT_PADDING_RATIO = 0.04
REFINEMENT_MIN_CONTOUR_AREA_RATIO = 0.001
REFINEMENT_MIN_RETAINED_AREA_RATIO = 0.02

# ---------------------------------------------------------------------------
# Region-based OCR (large blueprint pages)
# ---------------------------------------------------------------------------
REGION_OCR_ENABLED = True
REGION_OCR_MIN_SIDE_PX = 1200
REGION_OCR_GRID_ROWS = 2
REGION_OCR_GRID_COLS = 2
REGION_OCR_OVERLAP_RATIO = 0.08

