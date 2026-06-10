# Crop profile configuration tables

Crop thresholds are defined in code at `src/profile_config.py` in the `PROFILE_TABLE` dict.
Each `CropProfile` has a complete `ProfileConfig` bundle (aspect limits, fusion thresholds,
line-art settings, method priors, etc.).

## Profiles

| CropProfile | Use case |
|-------------|----------|
| `cad_compact` | Standard CAD parts, moderate aspect |
| `cad_wide` | Fuselage, wing profiles, wide side views |
| `blueprint_large` | Full engineering sheets, large assemblies |
| `scanned_pdf` | Scanned PDF pages with embedded raster |
| `mixed_datasheet` | Text + figures on same page |
| `photo_sheet` | Manufacturing photo datasheets |
| `simple_raster` | Single PNG/TIFF figures |
| `digital_pdf` | Born-digital PDF with vector/embedded images |
| `text_heavy` | Spec pages — skip figure extraction |

## Flow

1. `classify_page()` → `PageProfile` (coarse page type)
2. `resolve_crop_profile()` → `CropProfile` (refined using seed bbox / PDF hints)
3. `apply_dynamic_layout_zones()` → OCR-driven title-block expansion
4. `resolve_effective_profile_config()` → scaled `ProfileConfig` (when `PAGE_DERIVED_SCALING_ENABLED`)
5. Crop path uses `prepare_primary_crop()` with tighten-on-reject and drawing-zone merger

To tune behavior, edit the corresponding row in `PROFILE_TABLE` rather than global `config.py` knobs.
