"""Pipeline configuration."""

from pathlib import Path

# OCR
OCR_LANGUAGES = ["en"]
OCR_GPU = False
OCR_MIN_CONFIDENCE = 0.25

# Keyword extraction (YAKE)
MAX_KEYWORDS = 50
# Maximum n-gram length for keyword extraction.
KEYWORD_NGRAM_MAX = 3
#Removes duplicate keywords.
KEYWORD_DEDUP_THRESHOLD = 0.85

# Figure / aircraft image extraction
MIN_FIGURE_AREA_RATIO = 0.05  # min fraction of page area
TEXT_MASK_PADDING_PX = 12
#Ignore embedded images smaller than 8,000 bytes (~8 KB).
MIN_EMBEDDED_IMAGE_BYTES = 8_000
# Skip embedded rasters that are the full scanned page (image-only PDFs)
SKIP_FULL_PAGE_EMBEDDED = True
FULL_PAGE_EMBEDDED_RATIO = 0.85
# Manufacturing photo (central photograph) detection
PHOTO_SEARCH_TOP = 0.12
PHOTO_SEARCH_BOTTOM = 0.68
PHOTO_SEARCH_LEFT = 0.06
PHOTO_SEARCH_RIGHT = 0.94
PHOTO_MIN_AREA_RATIO = 0.08
PHOTO_MAX_AREA_RATIO = 0.55
PHOTO_SATURATION_MAX = 50
PHOTO_BBOX_PADDING_PX = 10
PHOTO_PROJECTION_THRESHOLD = 0.12
# Tight trim to visible photo frame (removes margin handwriting)
PHOTO_FRAME_TRIM_ENABLED = True
PHOTO_FRAME_INSET_PX = 5
PHOTO_FRAME_EDGE_LEFT_BAND = (0.05, 0.22)
PHOTO_FRAME_EDGE_RIGHT_BAND = (0.78, 0.95)
PHOTO_FRAME_MIN_AREA_VS_COARSE = 0.25

# If the detected "photo" bbox overlaps too much OCR text, it's likely not
# a real photo (e.g., engineering drawings match the same visual heuristics).
# In that case, we fall back to layout-based diagram cropping.
PHOTO_MAX_TEXT_COVERAGE = 0.09
OUTPUT_IMAGE_FORMAT = "png"
#Controls rendering resolution. Higher DPI = sharper images.
OUTPUT_DPI = 200

# Paths
# Default to a system folder so extracted images/JSON are not written into the repo.
DEFAULT_OUTPUT_DIR = Path("/home/dell/IMAGE_EXTRACTION_OUTPUT")

# ENGINEERING FIGURE EXTRACTION

MIN_COMPONENT_AREA_RATIO = 0.01
MAX_COMPONENT_AREA_RATIO = 0.70

BORDER_MARGIN_RATIO = 0.03
TITLE_BLOCK_HEIGHT_RATIO = 0.22

MORPH_KERNEL = (3, 3)

CENTRALITY_WEIGHT = 0.35
AREA_WEIGHT = 0.25
DENSITY_WEIGHT = 0.25
ASPECT_WEIGHT = 0.15

FIGURE_PADDING = 25
