from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class Case:
    """A single patient case.

    `public_symptoms` are visible to the agent at reset.
    `hidden_facts` are only retrievable by asking relevant questions.
    `key_questions` are the diagnostically important fact keys — used for
    trajectory grading (did the agent ask the *right* questions?).
    `variations` maps hidden-fact keys to alternative values that can be
    selected via a seed for randomized patient presentations.
    """

    name: str
    public_symptoms: List[str]
    hidden_facts: Dict[str, str]
    target_diagnosis: str
    key_questions: List[str] = field(default_factory=list)
    variations: Dict[str, List[str]] = field(default_factory=dict)


def apply_variations(case: Case, seed: Optional[int] = None) -> Case:
    """Return a copy of *case* with hidden facts randomized by *seed*.

    If seed is None the canonical case is returned unchanged.
    """
    if seed is None or not case.variations:
        return case
    rng = random.Random(seed)
    facts = dict(case.hidden_facts)
    for key, options in case.variations.items():
        if key in facts:
            facts[key] = rng.choice(options)
    return Case(
        name=case.name,
        public_symptoms=list(case.public_symptoms),
        hidden_facts=facts,
        target_diagnosis=case.target_diagnosis,
        key_questions=list(case.key_questions),
        variations=dict(case.variations),
    )


# 8 tasks (easy -> medium -> hard -> expert) with deterministic graders
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
        key_questions=["duration", "breathlessness", "rash"],
        variations={"duration": ["1 day", "2 days", "3 days"], "temperature": ["98.6F", "99.5F", "100F"]},
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
        key_questions=["platelets", "rash", "travel", "duration"],
        variations={"duration": ["3 days", "4 days", "5 days"], "temperature": ["101F", "102F", "103F"]},
    ),
    Case(
        name="medium_pneumonia",
        public_symptoms=["fever", "productive cough", "chest pain", "shortness of breath"],
        hidden_facts={
            "duration": "5 days",
            "breathlessness": "yes, worsening",
            "rash": "no",
            "travel": "no",
            "body_pain": "chest wall tenderness",
            "appetite": "reduced",
            "temperature": "102.5F",
            "sputum": "yellowish-green, thick",
            "oxygen": "SpO2 93%",
            "auscultation": "crackles in right lower lobe",
        },
        target_diagnosis="pneumonia",
        key_questions=["breathlessness", "sputum", "oxygen", "auscultation"],
        variations={"oxygen": ["SpO2 91%", "SpO2 93%", "SpO2 95%"], "sputum": ["yellowish-green, thick", "rusty colored", "blood-tinged"]},
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
        key_questions=["platelets", "bleeding", "travel"],
        variations={"platelets": ["low", "very low", "critically low"], "bleeding": ["minor gum bleeding", "petechiae on arms", "nosebleed"]},
    ),
    Case(
        name="hard_covid_respiratory",
        public_symptoms=["fever", "dry cough", "loss of taste", "fatigue"],
        hidden_facts={
            "duration": "6 days",
            "breathlessness": "yes, on exertion",
            "rash": "no",
            "travel": "attended large indoor gathering 5 days ago",
            "body_pain": "moderate myalgia",
            "temperature": "101F",
            "appetite": "poor",
            "oxygen": "SpO2 95%",
            "smell": "complete loss of smell (anosmia)",
            "sputum": "dry, non-productive",
        },
        target_diagnosis="covid-19",
        key_questions=["smell", "travel", "oxygen", "breathlessness"],
        variations={"oxygen": ["SpO2 94%", "SpO2 95%", "SpO2 97%"], "duration": ["4 days", "6 days", "8 days"]},
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
        key_questions=["travel", "spleen", "temperature", "platelets"],
        variations={"temperature": ["103F with cyclic pattern", "104F with tertian pattern", "103F with quotidian pattern"]},
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
        key_questions=["travel", "diarrhea", "rash", "spleen"],
        variations={"diarrhea": ["yes, alternating with constipation", "profuse watery", "pea-soup consistency"]},
    ),
    Case(
        name="expert_chikungunya",
        public_symptoms=["high fever", "severe joint pain", "rash", "swollen joints"],
        hidden_facts={
            "duration": "4 days",
            "platelets": "normal to slightly low",
            "bleeding": "no",
            "travel": "lives in tropical area with active chikungunya outbreak",
            "rash": "maculopapular rash on trunk and limbs",
            "temperature": "103F, sudden onset",
            "appetite": "poor",
            "fatigue": "moderate",
            "joint_swelling": "symmetric, affecting wrists, ankles, and small joints",
            "conjunctivitis": "mild redness in both eyes",
        },
        target_diagnosis="chikungunya",
        key_questions=["travel", "joint_swelling", "rash", "conjunctivitis"],
        variations={"joint_swelling": ["symmetric, affecting wrists, ankles, and small joints", "predominantly in hands and feet", "polyarthralgia with morning stiffness"]},
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


def _make_task_grader(task_name: str) -> Callable[[str], float]:
    grader_fn_name = f"grade_{task_name}"
    return lambda predicted, _fn=grader_fn_name: _call_grader_module(_fn, predicted)


# Generate per-task grader functions dynamically
easy_fever_cough_task_grader = _make_task_grader("easy_fever_cough")
medium_flu_vs_dengue_task_grader = _make_task_grader("medium_flu_vs_dengue")
medium_pneumonia_task_grader = _make_task_grader("medium_pneumonia")
hard_dengue_like_task_grader = _make_task_grader("hard_dengue_like")
hard_covid_respiratory_task_grader = _make_task_grader("hard_covid_respiratory")
expert_malaria_mimic_task_grader = _make_task_grader("expert_malaria_mimic")
expert_typhoid_enteric_task_grader = _make_task_grader("expert_typhoid_enteric")
expert_chikungunya_task_grader = _make_task_grader("expert_chikungunya")


def _task_dict(case: Case, grader: Callable[[str], float]) -> Dict[str, Any]:
    return {
        "name": case.name,
        "public_symptoms": list(case.public_symptoms),
        "hidden_facts": dict(case.hidden_facts),
        "answer": case.target_diagnosis,
        "grader": grader,
    }


TASKS: List[Dict[str, Any]] = [
    _task_dict(CASE_BY_NAME[name], _make_task_grader(name))
    for name in list_task_names()
]
