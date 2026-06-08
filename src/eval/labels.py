"""Labeled test-set schema and manifest loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExpectedCrop:
    """Ground-truth expectations for one page."""

    bbox_xyxy: tuple[int, int, int, int] | None
    min_iou: float = 0.50
    max_text_overlap: float = 0.20
    max_area_ratio: float | None = None
    min_area_ratio: float | None = None
    expected_crop_profile: str | None = None
    expect_figure: bool = True


@dataclass(frozen=True)
class LabeledPage:
    """One labeled page in the evaluation set."""

    id: str
    source_path: Path
    page_index: int
    document_tags: tuple[str, ...] = ()
    notes: str = ""
    expected: ExpectedCrop = field(
        default_factory=lambda: ExpectedCrop(bbox_xyxy=None)
    )

    @property
    def expect_figure(self) -> bool:
        return self.expected.expect_figure


@dataclass(frozen=True)
class EvalManifest:
    """Collection of labeled pages plus manifest metadata."""

    version: int
    samples: tuple[LabeledPage, ...]
    manifest_path: Path

    def existing_samples(self) -> list[LabeledPage]:
        """Return only samples whose source file exists on disk."""
        return [s for s in self.samples if s.source_path.is_file()]


def _parse_expected(raw: dict) -> ExpectedCrop:
    bbox = raw.get("bbox_xyxy")
    parsed_bbox = tuple(bbox) if bbox is not None else None
    if parsed_bbox is not None and len(parsed_bbox) != 4:
        raise ValueError(f"bbox_xyxy must have 4 integers, got {bbox!r}")
    return ExpectedCrop(
        bbox_xyxy=parsed_bbox,  # type: ignore[arg-type]
        min_iou=float(raw.get("min_iou", 0.50)),
        max_text_overlap=float(raw.get("max_text_overlap", 0.20)),
        max_area_ratio=raw.get("max_area_ratio"),
        min_area_ratio=raw.get("min_area_ratio"),
        expected_crop_profile=raw.get("expected_crop_profile"),
        expect_figure=bool(raw.get("expect_figure", True)),
    )


def _parse_sample(raw: dict, manifest_dir: Path) -> LabeledPage:
    source = Path(raw["source_path"])
    if not source.is_absolute():
        source = (manifest_dir / source).resolve()

    return LabeledPage(
        id=str(raw["id"]),
        source_path=source,
        page_index=int(raw.get("page_index", 0)),
        document_tags=tuple(raw.get("document_tags", [])),
        notes=str(raw.get("notes", "")),
        expected=_parse_expected(raw.get("expected", {})),
    )


def load_manifest(path: Path | str) -> EvalManifest:
    """Load ``eval/manifest.json`` (or any compatible manifest file)."""
    manifest_path = Path(path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dir = manifest_path.parent
    samples = tuple(
        _parse_sample(item, manifest_dir) for item in data.get("samples", [])
    )
    return EvalManifest(
        version=int(data.get("version", 1)),
        samples=samples,
        manifest_path=manifest_path,
    )
