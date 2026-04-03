from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GradeResult:
    score: float  # normalized in [0.0, 1.0]
    rationale: str


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    return text


def grade_diagnosis(predicted: str, actual: str) -> GradeResult:
    """Deterministic grader producing a score in [0.0, 1.0]."""

    p = _normalize(predicted)
    a = _normalize(actual)

    if not p:
        return GradeResult(score=0.0, rationale="empty prediction")

    if p == a:
        return GradeResult(score=1.0, rationale="exact match")

    # simple partial credit: any meaningful token overlap
    p_tokens = set(p.split())
    a_tokens = set(a.split())
    overlap = len(p_tokens & a_tokens)

    if overlap > 0:
        return GradeResult(score=0.5, rationale=f"partial token overlap={overlap}")

    # allow common synonyms for influenza
    if a == "influenza" and p in {"flu", "influenza virus", "seasonal flu"}:
        return GradeResult(score=1.0, rationale="synonym match")

    return GradeResult(score=0.0, rationale="no match")
