from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from .grader import clamp_score, grade_diagnosis
from .tasks import CASE_BY_NAME, Case, list_task_names


class Observation(BaseModel):
    task: str
    public_symptoms: List[str]
    history: List[str] = Field(default_factory=list)
    last_answer: Optional[str] = None


class Action(BaseModel):
    type: Literal["ask", "diagnose"]
    content: str


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)


class SwasthAIEnvState(BaseModel):
    task: str
    steps: int
    max_steps: int
    asked: List[str]
    last_action_error: Optional[str] = None


class SwasthAIEnv:
    """A minimal real-world diagnosis workflow environment.

    The agent can:
      - ask(question): get a factual answer derived from hidden facts
      - diagnose(label): get graded score in [0.0, 1.0]

    Episode ends when:
      - correct diagnosis (score==1.0)
      - max_steps reached
    """

    benchmark_name = "swasthai"

    def __init__(self, max_steps: int = 8):
        self._max_steps = max_steps
        self._case: Optional[Case] = None
        self._steps: int = 0
        self._history: List[str] = []
        self._asked: List[str] = []
        self._last_answer: Optional[str] = None
        self._last_action_error: Optional[str] = None

    def reset(self, task_name: Optional[str] = None) -> Observation:
        self._steps = 0
        self._history = []
        self._asked = []
        self._last_answer = None
        self._last_action_error = None

        if task_name is None:
            task_name = list_task_names()[0]

        case = CASE_BY_NAME.get(task_name)
        if case is None:
            raise ValueError(f"Unknown task '{task_name}'. Available: {list_task_names()}")

        self._case = case
        return Observation(task=case.name, public_symptoms=case.public_symptoms, history=[], last_answer=None)

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

                self._history.append(f"DX: {pred}")
                self._last_answer = None

                if gr.is_correct:
                    done = True

            else:
                self._last_action_error = f"invalid action.type={action.type}"
                reward_value = clamp_score(0.0)

        except Exception as e:  # keep env robust
            self._last_action_error = f"exception: {type(e).__name__}: {e}"
            reward_value = clamp_score(0.0)

        if self._steps >= self._max_steps:
            done = True

        obs = Observation(
            task=self._case.name,
            public_symptoms=self._case.public_symptoms,
            history=list(self._history),
            last_answer=self._last_answer,
        )

        # Ensure reward strictly within (0,1)
        reward_value = clamp_score(reward_value)

        if self._last_action_error:
            info["last_action_error"] = self._last_action_error

        return obs, reward_value, done, info

    def state(self) -> SwasthAIEnvState:
        if self._case is None:
            return SwasthAIEnvState(task="<not-reset>", steps=self._steps, max_steps=self._max_steps, asked=[])
        return SwasthAIEnvState(
            task=self._case.name,
            steps=self._steps,
            max_steps=self._max_steps,
            asked=list(self._asked),
            last_action_error=self._last_action_error,
        )

    def close(self) -> None:
        # no external resources
        return

    def _answer_question(self, question: str) -> Tuple[str, float]:
        """Return (answer, reward) where reward is strictly within (0,1)."""

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
        }

        matched_key: Optional[str] = None
        for needle, key in key_map.items():
            if needle in q and key in self._case.hidden_facts:
                matched_key = key
                break

        if matched_key is None:
            # neutral answer; no negative rewards (reward must be 0..1)
            return "I don't have that information based on your question.", clamp_score(0.0)

        answer = self._case.hidden_facts[matched_key]
        # small dense reward for retrieving relevant info
        return f"{matched_key}: {answer}", clamp_score(0.05)
