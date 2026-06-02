"""Extract keywords from OCR text using YAKE."""

from __future__ import annotations

import re

import yake

from src import config


def extract_keywords(text: str, max_keywords: int | None = None, language: str = "en") -> list[dict]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    limit = max_keywords if max_keywords is not None else config.MAX_KEYWORDS
    if limit is None:
        # YAKE requires an integer `top`; use a very high ceiling to emulate
        # uncapped extraction for practical document sizes.
        limit = 10_000
    extractor = yake.KeywordExtractor(
        lan=language,
        n=config.KEYWORD_NGRAM_MAX,
        dedupLim=config.KEYWORD_DEDUP_THRESHOLD,
        top=limit,
        features=None,
    )
    pairs = extractor.extract_keywords(text)
    return [{"keyword": kw, "score": float(score)} for kw, score in pairs]


def keywords_as_strings(text: str, max_keywords: int | None = None) -> list[str]:
    ranked = extract_keywords(text, max_keywords)
    return [item["keyword"] for item in ranked]
