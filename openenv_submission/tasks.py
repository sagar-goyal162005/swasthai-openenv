from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class Case:
    """A single patient case.

    `public_symptoms` are visible to the agent at reset.
    `hidden_facts` are only retrievable by asking relevant questions.
    """

    name: str
    public_symptoms: List[str]
    hidden_facts: Dict[str, str]
    target_diagnosis: str


# 3 tasks (easy -> medium -> hard) with deterministic graders
CASES: List[Case] = [
    Case(
        name="easy_fever_cough",
        public_symptoms=["fever", "cough", "sore throat"],
        hidden_facts={
            "duration": "2 days",
            "breathlessness": "no",
            "rash": "no",
            "travel": "no",
            "body_pain": "mild",
        },
        target_diagnosis="common cold",
    ),
    Case(
        name="medium_flu_vs_dengue",
        public_symptoms=["fever", "body pain", "fatigue", "headache"],
        hidden_facts={
            "duration": "4 days",
            "rash": "no",
            "platelets": "normal",
            "breathlessness": "no",
            "travel": "no",
        },
        target_diagnosis="influenza",
    ),
    Case(
        name="hard_dengue_like",
        public_symptoms=["high fever", "severe headache", "rash", "joint pain"],
        hidden_facts={
            "duration": "5 days",
            "platelets": "low",
            "bleeding": "minor gum bleeding",
            "travel": "recent mosquito exposure",
            "breathlessness": "no",
        },
        target_diagnosis="dengue",
    ),
]


CASE_BY_NAME = {c.name: c for c in CASES}


def list_task_names() -> List[str]:
    return [c.name for c in CASES]


# --- Hackathon deep-validation compatibility ---
#
# Some validators (separate from `openenv validate`) expect a module-level `TASKS`
# list where each task is a dict containing a callable `grader`.
#
# We keep the existing `Case`/`CASES` representation used by the environment, and
# provide a parallel `TASKS` structure that points to deterministic per-task
# grader callables.


def _call_grader_module(fn_name: str, predicted: str) -> float:
    # Local import to avoid circular import at module import time.
    from . import grader as _grader

    fn = getattr(_grader, fn_name)
    return float(fn(predicted))


def easy_fever_cough_task_grader(predicted: str) -> float:
    return _call_grader_module("grade_easy_fever_cough", predicted)


def medium_flu_vs_dengue_task_grader(predicted: str) -> float:
    return _call_grader_module("grade_medium_flu_vs_dengue", predicted)


def hard_dengue_like_task_grader(predicted: str) -> float:
    return _call_grader_module("grade_hard_dengue_like", predicted)


def _task_dict(case: Case, grader: Callable[[str], float]) -> Dict[str, Any]:
    return {
        "name": case.name,
        "public_symptoms": list(case.public_symptoms),
        "hidden_facts": dict(case.hidden_facts),
        "answer": case.target_diagnosis,
        "grader": grader,
    }


TASKS: List[Dict[str, Any]] = [
    _task_dict(CASE_BY_NAME["easy_fever_cough"], easy_fever_cough_task_grader),
    _task_dict(CASE_BY_NAME["medium_flu_vs_dengue"], medium_flu_vs_dengue_task_grader),
    _task_dict(CASE_BY_NAME["hard_dengue_like"], hard_dengue_like_task_grader),
]
