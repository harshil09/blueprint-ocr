from pathlib import Path

import cv2
import numpy as np

import config


# =========================================================
# REMOVE PAGE BORDER
# =========================================================

def remove_page_border(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape

    margin_x = int(w * config.BORDER_MARGIN_RATIO)
    margin_y = int(h * config.BORDER_MARGIN_RATIO)

    binary[:margin_y, :] = 0
    binary[h - margin_y:, :] = 0

    binary[:, :margin_x] = 0
    binary[:, w - margin_x:] = 0

    return binary


# =========================================================
# REMOVE TITLE BLOCK
# =========================================================

def remove_title_block(binary: np.ndarray) -> np.ndarray:
    h, _ = binary.shape

    title_h = int(h * config.TITLE_BLOCK_HEIGHT_RATIO)

    binary[h - title_h:, :] = 0

    return binary


# =========================================================
# COMPONENT SCORING
# =========================================================

def component_score(
    stats,
    centroid,
    page_shape,
    binary_roi,
):
    x, y, w, h, area = stats

    page_h, page_w = page_shape

    page_area = page_h * page_w

    component_area_ratio = area / page_area

    # Reject tiny regions
    if component_area_ratio < config.MIN_COMPONENT_AREA_RATIO:
        return -1

    # Reject huge/full-page regions
    if component_area_ratio > config.MAX_COMPONENT_AREA_RATIO:
        return -1

    # =====================================================
    # CENTRALITY SCORE
    # =====================================================

    cx, cy = centroid

    center_x = page_w / 2
    center_y = page_h / 2

    dist = np.sqrt(
        (cx - center_x) ** 2 +
        (cy - center_y) ** 2
    )

    max_dist = np.sqrt(center_x**2 + center_y**2)

    centrality_score = 1.0 - (dist / max_dist)

    # =====================================================
    # DENSITY SCORE
    # =====================================================

    density = np.count_nonzero(binary_roi) / max(w * h, 1)

    # =====================================================
    # ASPECT SCORE
    # =====================================================

    aspect_ratio = w / max(h, 1)

    if 0.3 <= aspect_ratio <= 3.5:
        aspect_score = 1.0
    else:
        aspect_score = 0.3

    # =====================================================
    # FINAL SCORE
    # =====================================================

    final_score = (
        config.AREA_WEIGHT * component_area_ratio +
        config.CENTRALITY_WEIGHT * centrality_score +
        config.DENSITY_WEIGHT * density +
        config.ASPECT_WEIGHT * aspect_score
    )

    return final_score


# =========================================================
# MAIN EXTRACTION
# =========================================================

def extract_engineering_figure(
    image: np.ndarray,
    output_path: str | Path,
):
    """
    Extract main engineering figure from TIFF/PDF page image.
    """

    if image is None:
        raise ValueError("Input image is None")

    # =====================================================
    # CONVERT TO GRAYSCALE
    # =====================================================

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # =====================================================
    # ADAPTIVE THRESHOLD
    # =====================================================

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    # =====================================================
    # REMOVE PAGE LAYOUT ARTIFACTS
    # =====================================================

    binary = remove_page_border(binary)

    binary = remove_title_block(binary)

    # =====================================================
    # MORPHOLOGY
    # =====================================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        config.MORPH_KERNEL,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    # =====================================================
    # CONNECTED COMPONENT ANALYSIS
    # =====================================================

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    best_score = -1
    best_box = None

    for i in range(1, num_labels):

        x, y, bw, bh, area = stats[i]

        roi = binary[y:y + bh, x:x + bw]

        score = component_score(
            stats=stats[i],
            centroid=centroids[i],
            page_shape=(h, w),
            binary_roi=roi,
        )

        if score > best_score:
            best_score = score
            best_box = (x, y, bw, bh)

    # =====================================================
    # NO COMPONENT FOUND
    # =====================================================

    if best_box is None:
        print("No engineering figure found")
        return None

    # =====================================================
    # CROP FINAL REGION
    # =====================================================

    x, y, bw, bh = best_box

    padding = config.FIGURE_PADDING

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)

    cropped = image[y1:y2, x1:x2]

    # =====================================================
    # SAVE OUTPUT
    # =====================================================

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(str(output_path), cropped)

    print(f"Saved engineering figure: {output_path}")

    return output_path
