"""Extract keywords from OCR text using YAKE."""

from __future__ import annotations

import re

import yake

from src import config


def extract_keywords(text: str, max_keywords: int | None = None, language: str = "en") -> list[dict]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    limit = max_keywords or config.MAX_KEYWORDS
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
