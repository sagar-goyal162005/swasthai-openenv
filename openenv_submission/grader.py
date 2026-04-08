from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .tasks import CASE_BY_NAME, Case, list_task_names


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

    # Synonym maps for accepted alternative labels
    _SYNONYMS = {
        "influenza": {"flu", "influenza virus", "seasonal flu", "influenza a", "influenza b"},
        "common cold": {"cold", "upper respiratory infection", "uri", "viral uri", "rhinitis"},
        "dengue": {"dengue fever", "dengue hemorrhagic fever", "dhf", "break-bone fever"},
        "malaria": {"plasmodium", "plasmodium falciparum", "plasmodium vivax", "malarial fever"},
        "typhoid": {"typhoid fever", "enteric fever", "salmonella typhi"},
        "pneumonia": {"bacterial pneumonia", "community acquired pneumonia", "cap", "lobar pneumonia", "lung infection"},
        "covid-19": {"covid", "coronavirus", "sars-cov-2", "covid 19", "corona"},
        "chikungunya": {"chikungunya fever", "chikungunya virus", "chik", "chikv"},
    }

    synonyms = _SYNONYMS.get(a, set())
    if p in synonyms:
        return GradeResult(score=MAX_SCORE, rationale="synonym match", is_correct=True)

    # Severity-aware partial credit
    p_tokens = set(p.split())
    a_tokens = set(a.split())
    overlap = len(p_tokens & a_tokens)

    # Also check if prediction is close to any synonym
    synonym_partial = any(
        len(p_tokens & set(syn.split())) > 0 for syn in synonyms
    )

    if overlap > 0:
        ratio = overlap / max(len(a_tokens), 1)
        return GradeResult(score=clamp_score(0.4 + 0.3 * ratio), rationale=f"partial token overlap={overlap} ratio={ratio:.2f}", is_correct=False)

    if synonym_partial:
        return GradeResult(score=clamp_score(0.35), rationale="partial synonym overlap", is_correct=False)

    return GradeResult(score=MIN_SCORE, rationale="no match", is_correct=False)


def grade_task(task_name: str, predicted: str) -> GradeResult:
    """Grade a prediction for a named task."""

    case = CASE_BY_NAME.get(task_name)
    if case is None:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list_task_names()}")
    return grade_diagnosis(predicted=predicted, actual=case.target_diagnosis)


def grade_result(task_name: str, predicted: str) -> GradeResult:
    """Public API returning a structured grade (score + rationale)."""

    return grade_task(task_name=task_name, predicted=predicted)


def grade(task_name: str, predicted: str) -> float:
    """Public API returning only the numeric score for a task.

    Some validators expect task graders to return a raw float.
    """

    return float(grade_result(task_name=task_name, predicted=predicted).score)


# Explicit per-task grader functions.
#
# IMPORTANT: Hackathon deep validation expects each task grader to return a
# numeric score strictly within (0, 1). Returning a dataclass here can cause
# those checks to fail.
def grade_easy_fever_cough(predicted: str) -> float:
    return float(grade("easy_fever_cough", predicted))


def grade_medium_flu_vs_dengue(predicted: str) -> float:
    return float(grade("medium_flu_vs_dengue", predicted))


def grade_hard_dengue_like(predicted: str) -> float:
    return float(grade("hard_dengue_like", predicted))


def grade_expert_malaria_mimic(predicted: str) -> float:
    return float(grade("expert_malaria_mimic", predicted))


def grade_expert_typhoid_enteric(predicted: str) -> float:
    return float(grade("expert_typhoid_enteric", predicted))


def grade_medium_pneumonia(predicted: str) -> float:
    return float(grade("medium_pneumonia", predicted))


def grade_hard_covid_respiratory(predicted: str) -> float:
    return float(grade("hard_covid_respiratory", predicted))


def grade_expert_chikungunya(predicted: str) -> float:
    return float(grade("expert_chikungunya", predicted))


def grade_easy_fever_cough_result(predicted: str) -> GradeResult:
    return grade_result("easy_fever_cough", predicted)


def grade_medium_flu_vs_dengue_result(predicted: str) -> GradeResult:
    return grade_result("medium_flu_vs_dengue", predicted)


def grade_hard_dengue_like_result(predicted: str) -> GradeResult:
    return grade_result("hard_dengue_like", predicted)


def grade_expert_malaria_mimic_result(predicted: str) -> GradeResult:
    return grade_result("expert_malaria_mimic", predicted)


def grade_expert_typhoid_enteric_result(predicted: str) -> GradeResult:
    return grade_result("expert_typhoid_enteric", predicted)


def grade_medium_pneumonia_result(predicted: str) -> GradeResult:
    return grade_result("medium_pneumonia", predicted)


def grade_hard_covid_respiratory_result(predicted: str) -> GradeResult:
    return grade_result("hard_covid_respiratory", predicted)


def grade_expert_chikungunya_result(predicted: str) -> GradeResult:
    return grade_result("expert_chikungunya", predicted)


# Mapping of task_name -> grader callable returning a float score.
TASK_GRADERS = {
    name: (lambda predicted, _name=name: grade(_name, predicted))
    for name in list_task_names()
}


# ---------------------------------------------------------------------------
# Trajectory grading — evaluate the *quality* of questions asked, not just
# the final diagnosis.
# ---------------------------------------------------------------------------

def grade_trajectory(
    case: Case,
    asked_fact_keys: Sequence[str],
    diagnosis_correct: bool,
    steps_taken: int,
    max_steps: int = 8,
) -> float:
    """Score an entire episode trajectory.

    Components:
    - diagnosis_weight (60%): 0.99 if correct, else MIN_SCORE
    - question_quality (25%): fraction of key_questions the agent asked
    - efficiency (15%): bonus for finishing in fewer steps

    Returns a score strictly in (0, 1).
    """
    # Diagnosis component
    dx_score = MAX_SCORE if diagnosis_correct else MIN_SCORE

    # Question quality — what fraction of key questions did agent ask?
    if case.key_questions:
        asked_set = set(asked_fact_keys)
        hits = sum(1 for kq in case.key_questions if kq in asked_set)
        q_score = hits / len(case.key_questions)
    else:
        q_score = 0.5  # no key questions defined → neutral

    # Efficiency — fewer steps is better
    eff_score = max(0.0, 1.0 - (steps_taken / max_steps))

    raw = 0.60 * dx_score + 0.25 * q_score + 0.15 * eff_score
    return clamp_score(raw)


def time_decay_factor(step: int, max_steps: int = 8) -> float:
    """Reward decay multiplier: earlier diagnoses get higher reward.

    Step 1 → 1.0, step max_steps → 0.5.
    """
    if max_steps <= 1:
        return 1.0
    decay = 1.0 - 0.5 * ((step - 1) / (max_steps - 1))
    return max(0.5, min(1.0, decay))
