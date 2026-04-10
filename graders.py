"""Deterministic task graders for swasthai-openenv.

These graders are intentionally tolerant to different evaluator payload shapes.
Each grader returns a normalized score in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


TASK_GRADERS: Dict[str, str] = {
    "easy_fever_cough": "graders.grade_easy_fever_cough",
    "medium_flu_vs_dengue": "graders.grade_medium_flu_vs_dengue",
    "medium_pneumonia": "graders.grade_medium_pneumonia",
    "hard_dengue_like": "graders.grade_hard_dengue_like",
    "hard_covid_respiratory": "graders.grade_hard_covid_respiratory",
    "expert_malaria_mimic": "graders.grade_expert_malaria_mimic",
    "expert_typhoid_enteric": "graders.grade_expert_typhoid_enteric",
    "expert_chikungunya": "graders.grade_expert_chikungunya",
}

TASK_GRADERS_COLON: Dict[str, str] = {
    "easy_fever_cough": "graders:grade_easy_fever_cough",
    "medium_flu_vs_dengue": "graders:grade_medium_flu_vs_dengue",
    "medium_pneumonia": "graders:grade_medium_pneumonia",
    "hard_dengue_like": "graders:grade_hard_dengue_like",
    "hard_covid_respiratory": "graders:grade_hard_covid_respiratory",
    "expert_malaria_mimic": "graders:grade_expert_malaria_mimic",
    "expert_typhoid_enteric": "graders:grade_expert_typhoid_enteric",
    "expert_chikungunya": "graders:grade_expert_chikungunya",
}


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _extract_numeric(candidate: Any) -> Optional[float]:
    if isinstance(candidate, (int, float)):
        return _clamp01(float(candidate))
    return None


def _extract_from_mapping(data: Dict[str, Any]) -> Optional[float]:
    for key in ("normalized_score", "score", "final_score"):
        val = _extract_numeric(data.get(key))
        if val is not None:
            return val

    info = data.get("info")
    if isinstance(info, dict):
        for key in ("normalized_score", "score", "final_score"):
            val = _extract_numeric(info.get(key))
            if val is not None:
                return val

    nested = data.get("state")
    if isinstance(nested, dict):
        val = _extract_from_mapping(nested)
        if val is not None:
            return val

    nested = data.get("result")
    if isinstance(nested, dict):
        val = _extract_from_mapping(nested)
        if val is not None:
            return val

    trajectory = data.get("trajectory")
    if isinstance(trajectory, Iterable) and not isinstance(trajectory, (str, bytes)):
        step_rewards: List[float] = []
        for step in trajectory:
            if isinstance(step, dict):
                numeric_reward = _extract_numeric(step.get("reward"))
                if numeric_reward is not None:
                    step_rewards.append(numeric_reward)
        if step_rewards:
            return _clamp01(sum(step_rewards) / len(step_rewards))

    return None


def _extract_score(payload: Any) -> Optional[float]:
    val = _extract_numeric(payload)
    if val is not None:
        return val

    if isinstance(payload, dict):
        return _extract_from_mapping(payload)

    # Dataclass/Pydantic-like objects.
    for attr in ("normalized_score", "score", "final_score"):
        if hasattr(payload, attr):
            val = _extract_numeric(getattr(payload, attr))
            if val is not None:
                return val

    if hasattr(payload, "info"):
        info = getattr(payload, "info")
        if isinstance(info, dict):
            val = _extract_from_mapping({"info": info})
            if val is not None:
                return val

    if hasattr(payload, "model_dump"):
        try:
            dumped = payload.model_dump()
            if isinstance(dumped, dict):
                return _extract_from_mapping(dumped)
        except Exception:
            pass

    return None


def _grade_from_inputs(task_id: str, *args: Any, **kwargs: Any) -> float:
    candidates = []
    candidates.extend(args)
    candidates.extend(kwargs.values())

    for candidate in candidates:
        score = _extract_score(candidate)
        if score is not None:
            return _clamp01(score)

    # Deterministic neutral fallback if evaluator passes unsupported payload.
    return 0.0


def grade_easy_fever_cough(*args: Any, **kwargs: Any) -> float:
    return _grade_from_inputs("easy_fever_cough", *args, **kwargs)


def grade_medium_flu_vs_dengue(*args: Any, **kwargs: Any) -> float:
    return _grade_from_inputs("medium_flu_vs_dengue", *args, **kwargs)


def grade_medium_pneumonia(*args: Any, **kwargs: Any) -> float:
    return _grade_from_inputs("medium_pneumonia", *args, **kwargs)


def grade_hard_dengue_like(*args: Any, **kwargs: Any) -> float:
    return _grade_from_inputs("hard_dengue_like", *args, **kwargs)


def grade_hard_covid_respiratory(*args: Any, **kwargs: Any) -> float:
    return _grade_from_inputs("hard_covid_respiratory", *args, **kwargs)


def grade_expert_malaria_mimic(*args: Any, **kwargs: Any) -> float:
    return _grade_from_inputs("expert_malaria_mimic", *args, **kwargs)


def grade_expert_typhoid_enteric(*args: Any, **kwargs: Any) -> float:
    return _grade_from_inputs("expert_typhoid_enteric", *args, **kwargs)


def grade_expert_chikungunya(*args: Any, **kwargs: Any) -> float:
    return _grade_from_inputs("expert_chikungunya", *args, **kwargs)


def grade_task(task_id: str, *args: Any, **kwargs: Any) -> float:
    if task_id == "easy_fever_cough":
        return grade_easy_fever_cough(*args, **kwargs)
    if task_id == "medium_flu_vs_dengue":
        return grade_medium_flu_vs_dengue(*args, **kwargs)
    if task_id == "medium_pneumonia":
        return grade_medium_pneumonia(*args, **kwargs)
    if task_id == "hard_dengue_like":
        return grade_hard_dengue_like(*args, **kwargs)
    if task_id == "hard_covid_respiratory":
        return grade_hard_covid_respiratory(*args, **kwargs)
    if task_id == "expert_malaria_mimic":
        return grade_expert_malaria_mimic(*args, **kwargs)
    if task_id == "expert_typhoid_enteric":
        return grade_expert_typhoid_enteric(*args, **kwargs)
    if task_id == "expert_chikungunya":
        return grade_expert_chikungunya(*args, **kwargs)
    return 0.0
