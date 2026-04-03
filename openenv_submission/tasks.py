from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


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
