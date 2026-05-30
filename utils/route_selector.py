"""Generic scoring-based selection among candidate route strings (no domain rules).

Intended for optional ranking or experiments only. Do **not** use :func:`select_route`
as the primary business router when the app already fixes the route via API stage,
``generation_route``, or explicit user choice—prefer those sources in handlers.
"""

from __future__ import annotations

import math
import re
from typing import List


def _score_route(text: str) -> float:
    """Score in [0, 1] using length, lexical diversity, and simple structure signals."""
    s = text.strip()
    if not s:
        return 0.0

    words = re.findall(r"\b\w+\b", s, flags=re.UNICODE)
    n_words = len(words)
    if n_words == 0:
        # Fallback: non-word characters still contribute a small length signal
        return min(1.0, len(s) / 800.0) * 0.25

    diversity = len(set(w.lower() for w in words)) / n_words
    length_factor = min(1.0, math.log1p(n_words) / math.log1p(80))
    lines = s.count("\n") + 1
    structure = min(1.0, math.log1p(lines) / math.log1p(12))
    punct = sum(1 for c in s if c in ",.;:!?|/-—")
    punct_factor = min(1.0, punct / max(n_words, 1))

    # Weights favor lexical diversity and length so typical multi-sentence routes
    # can exceed the default ``score_threshold`` without domain-specific keywords.
    return (
        0.40 * diversity
        + 0.32 * length_factor
        + 0.20 * structure
        + 0.08 * punct_factor
    )


def select_route(routes: List[str], score_threshold: float = 0.7) -> str:
    """Pick the highest-scoring route string; requires best score >= ``score_threshold``."""
    if routes is None or not isinstance(routes, list):
        raise ValueError("routes must be a non-empty list of strings")
    if not routes:
        raise ValueError("routes must not be empty")
    if not (0.0 <= score_threshold <= 1.0):
        raise ValueError("score_threshold must be between 0.0 and 1.0")

    stripped: List[str] = []
    for i, item in enumerate(routes):
        if not isinstance(item, str):
            raise ValueError(f"routes[{i}] must be str, got {type(item).__name__}")
        stripped.append(item.strip())

    if all(not t for t in stripped):
        raise ValueError("All routes are empty or whitespace-only")

    scores = [_score_route(t) for t in stripped]
    best_idx = max(range(len(scores)), key=lambda j: scores[j])
    best_score = scores[best_idx]
    if best_score < score_threshold:
        raise ValueError(
            f"No route met score_threshold={score_threshold}; best was {best_score:.4f}"
        )
    return routes[best_idx].strip()
