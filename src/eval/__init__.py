"""Figure-crop evaluation: labeled test set, metrics, and failure categories."""

from src.eval.categories import FailureCategory
from src.eval.harness import EvalReport, run_figure_eval
from src.eval.labels import EvalManifest, LabeledPage, load_manifest
from src.eval.metrics import PageEvalResult, evaluate_page_crop

__all__ = [
    "EvalManifest",
    "EvalReport",
    "FailureCategory",
    "LabeledPage",
    "PageEvalResult",
    "evaluate_page_crop",
    "load_manifest",
    "run_figure_eval",
]
