from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action as OEAction
from openenv.core.env_server.types import EnvironmentMetadata
from openenv.core.env_server.types import Observation as OEObservation
from openenv.core.env_server.types import State as OEState

from ..grader import clamp_score, grade_diagnosis, grade_trajectory, time_decay_factor
from ..tasks import CASE_BY_NAME, Case, apply_variations, list_task_names


class SwasthAIAction(OEAction):
    type: Literal["ask", "diagnose"]
    content: str
    confidence: Optional[float] = None


class SwasthAIObservation(OEObservation):
    task: str
    public_symptoms: List[str]
    history: List[str] = Field(default_factory=list)
    last_answer: Optional[str] = None
    vitals: Optional[Dict[str, str]] = None


class SwasthAIState(OEState):
    task: str
    steps: int
    max_steps: int
    asked: List[str]
    retrieved_facts: List[str] = Field(default_factory=list)
    wrong_diagnoses: int = 0
    last_action_error: Optional[str] = None


class SwasthAIEnvironment(Environment[SwasthAIAction, SwasthAIObservation, SwasthAIState]):
    """OpenEnv-core compatible server environment with progressive rewards.

    Features: time-decay, trajectory grading, wrong-diagnosis penalty,
    seed-based variations, structured vitals, confidence weighting.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, max_steps: int = 8):
        super().__init__()
        self._max_steps = max_steps
        self._case: Optional[Case] = None
        self._steps: int = 0
        self._history: List[str] = []
        self._asked: List[str] = []
        self._retrieved_fact_keys: List[str] = []
        self._last_answer: Optional[str] = None
        self._last_action_error: Optional[str] = None
        self._wrong_diagnoses: int = 0

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="swasthai",
            description="Healthcare triage & diagnosis workflow environment (SwasthAI).",
            version="1.0.0",
        )

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> SwasthAIObservation:
        task_name = kwargs.get("task_name") or kwargs.get("task")
        if task_name is None:
            task_name = list_task_names()[0]

        case = CASE_BY_NAME.get(str(task_name))
        if case is None:
            raise ValueError(f"Unknown task '{task_name}'. Available: {list_task_names()}")

        self._steps = 0
        self._case = apply_variations(case, seed)
        self._history = []
        self._asked = []
        self._retrieved_fact_keys = []
        self._last_answer = None
        self._last_action_error = None
        self._wrong_diagnoses = 0

        return SwasthAIObservation(
            task=self._case.name,
            public_symptoms=self._case.public_symptoms,
            history=[],
            last_answer=None,
            vitals=None,
            reward=None,
            done=False,
            metadata={},
        )

    def step(self, action: SwasthAIAction, timeout_s: Optional[float] = None, **kwargs: Any) -> SwasthAIObservation:
        if self._case is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        self._steps += 1
        self._last_action_error = None
        reward_value = 0.0
        done = False
        metadata: Dict[str, Any] = {}

        try:
            if action.type == "ask":
                q = (action.content or "").strip()
                if not q:
                    self._last_action_error = "empty question"
                    answer = "Please ask a non-empty question."
                    reward_value = clamp_score(0.0)
                else:
                    answer, reward_value = self._answer_question(q)

                self._history.append(f"Q: {q}")
                self._history.append(f"A: {answer}")
                self._asked.append(q)
                self._last_answer = answer

            elif action.type == "diagnose":
                pred = (action.content or "").strip()
                gr = grade_diagnosis(predicted=pred, actual=self._case.target_diagnosis)
                reward_value = float(gr.score)
                metadata["grade_rationale"] = gr.rationale
                metadata["is_correct"] = bool(gr.is_correct)

                decay = time_decay_factor(self._steps, self._max_steps)
                reward_value = reward_value * decay

                if action.confidence is not None:
                    conf = max(0.0, min(1.0, action.confidence))
                    metadata["agent_confidence"] = conf
                    if gr.is_correct:
                        reward_value = reward_value * (0.8 + 0.2 * conf)
                    else:
                        reward_value = reward_value * (1.0 - 0.3 * conf)

                self._history.append(f"DX: {pred}")
                self._last_answer = None

                if gr.is_correct:
                    done = True
                    traj_score = grade_trajectory(
                        case=self._case,
                        asked_fact_keys=self._retrieved_fact_keys,
                        diagnosis_correct=True,
                        steps_taken=self._steps,
                        max_steps=self._max_steps,
                    )
                    metadata["trajectory_score"] = traj_score
                else:
                    self._wrong_diagnoses += 1
                    penalty = 0.05 * self._wrong_diagnoses
                    reward_value = max(reward_value - penalty, 0.0)

            else:
                self._last_action_error = f"invalid action.type={action.type}"
                reward_value = clamp_score(0.0)

        except Exception as e:
            self._last_action_error = f"exception: {type(e).__name__}: {e}"
            reward_value = clamp_score(0.0)

        if self._steps >= self._max_steps:
            done = True

        reward_value = clamp_score(reward_value)

        if self._last_action_error:
            metadata["last_action_error"] = self._last_action_error

        metadata["wrong_diagnoses"] = self._wrong_diagnoses
        metadata["retrieved_facts"] = list(self._retrieved_fact_keys)

        vitals = self._extract_vitals()

        return SwasthAIObservation(
            task=self._case.name,
            public_symptoms=self._case.public_symptoms,
            history=list(self._history),
            last_answer=self._last_answer,
            vitals=vitals,
            reward=reward_value,
            done=done,
            metadata=metadata,
        )

    @property
    def state(self) -> SwasthAIState:
        if self._case is None:
            return SwasthAIState(task="<not-reset>", steps=self._steps, max_steps=self._max_steps, asked=[])
        return SwasthAIState(
            task=self._case.name,
            steps=self._steps,
            max_steps=self._max_steps,
            asked=list(self._asked),
            retrieved_facts=list(self._retrieved_fact_keys),
            wrong_diagnoses=self._wrong_diagnoses,
            last_action_error=self._last_action_error,
        )

    def close(self) -> None:
        return

    def _extract_vitals(self) -> Optional[Dict[str, str]]:
        if self._case is None:
            return None
        vitals: Dict[str, str] = {}
        for key in ("temperature", "oxygen", "platelets"):
            if key in self._case.hidden_facts and key in self._retrieved_fact_keys:
                vitals[key] = self._case.hidden_facts[key]
        return vitals if vitals else None

    _HIGH_VALUE_KEYS = {
        "platelets", "bleeding", "travel", "rash", "diarrhea", "spleen",
        "oxygen", "sputum", "auscultation", "smell", "conjunctivitis", "joint_swelling",
    }

    def _answer_question(self, question: str) -> tuple[str, float]:
        assert self._case is not None
        q = question.lower()

        key_map = {
            "how long": "duration",
            "duration": "duration",
            "days": "duration",
            "rash": "rash",
            "platelet": "platelets",
            "bleeding": "bleeding",
            "body pain": "body_pain",
            "fatigue": "fatigue",
            "breath": "breathlessness",
            "travel": "travel",
            "mosquito": "travel",
            "appetite": "appetite",
            "temperature": "temperature",
            "temp": "temperature",
            "chills": "chills",
            "sweat": "chills",
            "diarrhea": "diarrhea",
            "stool": "diarrhea",
            "constipation": "diarrhea",
            "spleen": "spleen",
            "abdomen": "spleen",
            "dehydration": "dehydration",
            "water": "dehydration",
            "food": "appetite",
            "eat": "appetite",
            "sputum": "sputum",
            "phlegm": "sputum",
            "mucus": "sputum",
            "oxygen": "oxygen",
            "spo2": "oxygen",
            "saturation": "oxygen",
            "lung": "auscultation",
            "auscultation": "auscultation",
            "crackle": "auscultation",
            "smell": "smell",
            "anosmia": "smell",
            "taste": "smell",
            "joint": "joint_swelling",
            "swelling": "joint_swelling",
            "eye": "conjunctivitis",
            "conjunctiv": "conjunctivitis",
            "red eye": "conjunctivitis",
            "gathering": "travel",
            "contact": "travel",
            "exposure": "travel",
        }

        matched_key: Optional[str] = None
        for needle, key in key_map.items():
            if needle in q and key in self._case.hidden_facts:
                matched_key = key
                break

        if matched_key is None:
            return "I don't have that information based on your question.", clamp_score(0.0)

        if matched_key in self._retrieved_fact_keys:
            answer = self._case.hidden_facts[matched_key]
            return f"{matched_key}: {answer} (already asked)", clamp_score(0.01)

        self._retrieved_fact_keys.append(matched_key)
        answer = self._case.hidden_facts[matched_key]
        if matched_key in self._HIGH_VALUE_KEYS:
            return f"{matched_key}: {answer}", clamp_score(0.10)
        return f"{matched_key}: {answer}", clamp_score(0.05)
