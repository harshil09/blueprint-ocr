#!/usr/bin/env python3
"""CLI entry point for labeled figure-crop evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.harness import run_figure_eval
from src.logger import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate figure cropping against the labeled test set."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "eval" / "manifest.json",
        help="Path to eval manifest JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for per-sample outputs and eval_report.json",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write eval_report.json",
    )
    args = parser.parse_args()

    setup_logging()
    report = run_figure_eval(
        args.manifest,
        output_dir=args.output,
        write_report=not args.no_write,
    )

    print(f"Evaluated: {report.evaluated}/{report.total_samples} samples")
    print(f"Passed:    {report.passed}")
    print(f"Failed:    {report.failed}")
    print(f"Pass rate: {report.pass_rate * 100:.1f}%")
    if report.skipped_missing:
        print(f"Skipped (missing file): {report.skipped_missing}")
    if report.failure_counts:
        print("Failure breakdown:")
        for cat, count in sorted(report.failure_counts.items()):
            print(f"  {cat}: {count}")

    return 0 if report.failed == 0 and report.skipped_missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
