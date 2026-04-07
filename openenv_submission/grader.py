from __future__ import annotations

import re
from dataclasses import dataclass

from .tasks import CASE_BY_NAME, list_task_names


@dataclass(frozen=True)
class GradeResult:
    # NOTE: Hackathon Phase 2 deep validation requires scores to be strictly
    # within (0, 1) (i.e. not 0.0 and not 1.0).
    score: float
    rationale: str
    is_correct: bool = False


# Keep bounds visible and stable (also avoids rounding to 0.00 / 1.00 in logs).
MIN_SCORE = 0.01
MAX_SCORE = 0.99


def clamp_score(score: float) -> float:
    """Clamp any raw score into the open interval (0, 1)."""

    if score <= 0.0:
        return MIN_SCORE
    if score >= 1.0:
        return MAX_SCORE
    # Also keep scores inside our open-interval bounds.
    if score < MIN_SCORE:
        return MIN_SCORE
    if score > MAX_SCORE:
        return MAX_SCORE
    return float(score)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    return text


def grade_diagnosis(predicted: str, actual: str) -> GradeResult:
    """Deterministic diagnosis grader.

    Returns:
        GradeResult with score strictly in (0, 1).
    """

    p = _normalize(predicted)
    a = _normalize(actual)

    if not p:
        return GradeResult(score=MIN_SCORE, rationale="empty prediction", is_correct=False)

    if p == a:
        return GradeResult(score=MAX_SCORE, rationale="exact match", is_correct=True)

    # allow common synonyms for influenza
    if a == "influenza" and p in {"flu", "influenza virus", "seasonal flu"}:
        return GradeResult(score=MAX_SCORE, rationale="synonym match", is_correct=True)

    # simple partial credit: any meaningful token overlap
    p_tokens = set(p.split())
    a_tokens = set(a.split())
    overlap = len(p_tokens & a_tokens)

    if overlap > 0:
        return GradeResult(score=clamp_score(0.5), rationale=f"partial token overlap={overlap}", is_correct=False)

    return GradeResult(score=MIN_SCORE, rationale="no match", is_correct=False)


def grade_task(task_name: str, predicted: str) -> GradeResult:
    """Grade a prediction for a named task."""

    case = CASE_BY_NAME.get(task_name)
    if case is None:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list_task_names()}")
    return grade_diagnosis(predicted=predicted, actual=case.target_diagnosis)


def grade(task_name: str, predicted: str) -> float:
    """Convenience API returning the numeric score for a task."""

    return float(grade_task(task_name=task_name, predicted=predicted).score)


# Explicit per-task grader functions (some validators look for these by name).
def grade_easy_fever_cough(predicted: str) -> GradeResult:
    return grade_task("easy_fever_cough", predicted)


def grade_medium_flu_vs_dengue(predicted: str) -> GradeResult:
    return grade_task("medium_flu_vs_dengue", predicted)


def grade_hard_dengue_like(predicted: str) -> GradeResult:
    return grade_task("hard_dengue_like", predicted)


# Mapping of task_name -> grader callable.
TASK_GRADERS = {
    name: (lambda predicted, _name=name: grade_task(_name, predicted))
    for name in list_task_names()
}
