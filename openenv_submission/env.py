from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from .grader import clamp_score, grade_diagnosis, grade_trajectory, time_decay_factor
from .tasks import CASE_BY_NAME, Case, apply_variations, list_task_names


class Observation(BaseModel):
    task: str
    public_symptoms: List[str]
    history: List[str] = Field(default_factory=list)
    last_answer: Optional[str] = None
    vitals: Optional[Dict[str, str]] = None


class Action(BaseModel):
    type: Literal["ask", "diagnose"]
    content: str
    confidence: Optional[float] = None


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)


class SwasthAIEnvState(BaseModel):
    task: str
    steps: int
    max_steps: int
    asked: List[str]
    retrieved_facts: List[str] = Field(default_factory=list)
    wrong_diagnoses: int = 0
    last_action_error: Optional[str] = None


class SwasthAIEnv:
    """A real-world diagnosis workflow environment with progressive rewards.

    The agent can:
      - ask(question): get a factual answer derived from hidden facts
      - diagnose(label, confidence?): get graded score in (0, 1)

    Features:
      - Time-decay: earlier correct diagnosis → higher reward
      - Trajectory grading: bonus for asking diagnostically relevant questions
      - Wrong-diagnosis penalty: repeated wrong guesses penalized
      - Seed-based variations: same task, different presentations
      - Structured vitals in observations

    Episode ends when:
      - correct diagnosis
      - max_steps reached
    """

    benchmark_name = "swasthai"

    def __init__(self, max_steps: int = 8):
        self._max_steps = max_steps
        self._case: Optional[Case] = None
        self._steps: int = 0
        self._history: List[str] = []
        self._asked: List[str] = []
        self._retrieved_fact_keys: List[str] = []
        self._last_answer: Optional[str] = None
        self._last_action_error: Optional[str] = None
        self._wrong_diagnoses: int = 0
        self._seed: Optional[int] = None

    def reset(self, task_name: Optional[str] = None, seed: Optional[int] = None) -> Observation:
        self._steps = 0
        self._history = []
        self._asked = []
        self._retrieved_fact_keys = []
        self._last_answer = None
        self._last_action_error = None
        self._wrong_diagnoses = 0
        self._seed = seed

        if task_name is None:
            task_name = list_task_names()[0]

        case = CASE_BY_NAME.get(task_name)
        if case is None:
            raise ValueError(f"Unknown task '{task_name}'. Available: {list_task_names()}")

        self._case = apply_variations(case, seed)

        vitals = self._extract_vitals()
        return Observation(
            task=self._case.name,
            public_symptoms=self._case.public_symptoms,
            history=[],
            last_answer=None,
            vitals=vitals,
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._case is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        self._steps += 1
        self._last_action_error = None
        reward_value = 0.0
        done = False
        info: Dict[str, Any] = {}

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
                info["grade_rationale"] = gr.rationale
                info["is_correct"] = bool(gr.is_correct)

                # Time-decay: earlier correct diagnosis gets higher reward
                decay = time_decay_factor(self._steps, self._max_steps)
                reward_value = reward_value * decay

                # Confidence weighting (optional)
                if action.confidence is not None:
                    conf = max(0.0, min(1.0, action.confidence))
                    info["agent_confidence"] = conf
                    if gr.is_correct:
                        reward_value = reward_value * (0.8 + 0.2 * conf)
                    else:
                        reward_value = reward_value * (1.0 - 0.3 * conf)

                self._history.append(f"DX: {pred}")
                self._last_answer = None

                if gr.is_correct:
                    done = True
                    # Trajectory bonus
                    traj_score = grade_trajectory(
                        case=self._case,
                        asked_fact_keys=self._retrieved_fact_keys,
                        diagnosis_correct=True,
                        steps_taken=self._steps,
                        max_steps=self._max_steps,
                    )
                    info["trajectory_score"] = traj_score
                else:
                    self._wrong_diagnoses += 1
                    # Penalize repeated wrong diagnoses
                    penalty = 0.05 * self._wrong_diagnoses
                    reward_value = max(reward_value - penalty, 0.0)

            else:
                self._last_action_error = f"invalid action.type={action.type}"
                reward_value = clamp_score(0.0)

        except Exception as e:  # keep env robust
            self._last_action_error = f"exception: {type(e).__name__}: {e}"
            reward_value = clamp_score(0.0)

        if self._steps >= self._max_steps:
            done = True

        vitals = self._extract_vitals()
        obs = Observation(
            task=self._case.name,
            public_symptoms=self._case.public_symptoms,
            history=list(self._history),
            last_answer=self._last_answer,
            vitals=vitals,
        )

        # Ensure reward strictly within (0,1)
        reward_value = clamp_score(reward_value)

        if self._last_action_error:
            info["last_action_error"] = self._last_action_error

        info["wrong_diagnoses"] = self._wrong_diagnoses
        info["retrieved_facts"] = list(self._retrieved_fact_keys)

        return obs, reward_value, done, info

    def state(self) -> SwasthAIEnvState:
        if self._case is None:
            return SwasthAIEnvState(task="<not-reset>", steps=self._steps, max_steps=self._max_steps, asked=[])
        return SwasthAIEnvState(
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

    def _extract_vitals(self) -> Dict[str, str]:
        """Return structured vitals from retrieved hidden facts."""
        if self._case is None:
            return {}
        vitals: Dict[str, str] = {}
        vital_keys = {"temperature", "oxygen", "platelets"}
        for key in vital_keys:
            if key in self._case.hidden_facts and key in self._retrieved_fact_keys:
                vitals[key] = self._case.hidden_facts[key]
        return vitals if vitals else {}

    # High-value facts are more diagnostically useful — reward them more.
    _HIGH_VALUE_KEYS = {
        "platelets", "bleeding", "travel", "rash", "diarrhea", "spleen",
        "oxygen", "sputum", "auscultation", "smell", "conjunctivitis", "joint_swelling",
    }

    def _answer_question(self, question: str) -> Tuple[str, float]:
        """Return (answer, reward) where reward is strictly within (0,1).

        Progressive reward shaping:
        - High-value diagnostic facts: 0.10
        - Standard facts:              0.05
        - Repeated / irrelevant:       ~0.0
        """

        assert self._case is not None
        q = question.lower()

        # map common intents to hidden facts
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
            # pneumonia / covid / chikungunya
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

        # Penalize repeated questions — diminishing returns
        if matched_key in self._retrieved_fact_keys:
            answer = self._case.hidden_facts[matched_key]
            return f"{matched_key}: {answer} (already asked)", clamp_score(0.01)

        self._retrieved_fact_keys.append(matched_key)
        answer = self._case.hidden_facts[matched_key]
        # Progressive reward: high-value diagnostic facts get more reward
        if matched_key in self._HIGH_VALUE_KEYS:
            return f"{matched_key}: {answer}", clamp_score(0.10)
        return f"{matched_key}: {answer}", clamp_score(0.05)
