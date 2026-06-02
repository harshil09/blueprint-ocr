"""Central logging setup for the extraction pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src import config

_CONFIGURED = False


def setup_logging(
    level: str | int | None = None,
    log_file: Path | str | None = None,
) -> None:
    """Configure root logging once for the application."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = level if level is not None else config.LOG_LEVEL
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    file_path = log_file if log_file is not None else config.LOG_FILE
    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured."""
    setup_logging()
    return logging.getLogger(name)
