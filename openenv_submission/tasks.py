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


# 5 tasks (easy -> medium -> hard -> expert) with deterministic graders
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
            "appetite": "slightly reduced",
            "temperature": "99.5F",
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
            "chills": "yes, with sweating",
            "temperature": "102F",
            "appetite": "very poor",
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
            "temperature": "104F",
            "appetite": "none",
            "dehydration": "moderate",
        },
        target_diagnosis="dengue",
    ),
    Case(
        name="expert_malaria_mimic",
        public_symptoms=["high fever", "chills", "sweating", "headache"],
        hidden_facts={
            "duration": "7 days",
            "platelets": "low",
            "bleeding": "no",
            "travel": "visited endemic malaria zone 2 weeks ago",
            "rash": "no",
            "temperature": "103F with cyclic pattern",
            "appetite": "very poor",
            "fatigue": "severe, bedridden",
            "spleen": "enlarged on palpation",
        },
        target_diagnosis="malaria",
    ),
    Case(
        name="expert_typhoid_enteric",
        public_symptoms=["sustained fever", "abdominal pain", "weakness"],
        hidden_facts={
            "duration": "10 days",
            "platelets": "borderline low",
            "bleeding": "no",
            "travel": "consumed street food and untreated water",
            "rash": "faint rose spots on abdomen",
            "temperature": "stepladder pattern reaching 104F",
            "appetite": "absent",
            "fatigue": "extreme",
            "diarrhea": "yes, alternating with constipation",
            "spleen": "mildly enlarged",
        },
        target_diagnosis="typhoid",
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


def expert_malaria_mimic_task_grader(predicted: str) -> float:
    return _call_grader_module("grade_expert_malaria_mimic", predicted)


def expert_typhoid_enteric_task_grader(predicted: str) -> float:
    return _call_grader_module("grade_expert_typhoid_enteric", predicted)


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
    _task_dict(CASE_BY_NAME["expert_malaria_mimic"], expert_malaria_mimic_task_grader),
    _task_dict(CASE_BY_NAME["expert_typhoid_enteric"], expert_typhoid_enteric_task_grader),
]
