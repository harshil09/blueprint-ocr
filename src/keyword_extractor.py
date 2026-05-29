"""Unsupervised keyword extraction from OCR text (YAKE)."""

from __future__ import annotations

import re

import yake

import config


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_keywords(
    text: str,
    max_keywords: int | None = None,
    language: str = "en",
) -> list[dict]:
    """
    Extract ranked keywords/phrases from unstructured document text.

    Returns list of {"keyword": str, "score": float} sorted by relevance.
    Lower YAKE score = more important.
    """
    text = _normalize_text(text)
    if not text:
        return []

    limit = max_keywords or config.MAX_KEYWORDS
    kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=config.KEYWORD_NGRAM_MAX,
        dedupLim=config.KEYWORD_DEDUP_THRESHOLD,
        top=limit,
        features=None,
    )
    pairs = kw_extractor.extract_keywords(text)
    results = [{"keyword": kw, "score": float(score)} for kw, score in pairs]
    return results


def keywords_as_strings(
    text: str,
    max_keywords: int | None = None,
) -> list[str]:
    """Return keyword strings only, in ranked order."""
    return [item["keyword"] for item in extract_keywords(text, max_keywords)]
