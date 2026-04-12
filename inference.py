#!/usr/bin/env python3
"""OpenEnv inference script template.

This script is designed to satisfy the OpenEnv submission contract:
- Reads model/env settings from environment variables.
- Uses OpenAI Client for LLM calls.
- Emits strict [START]/[STEP]/[END] stdout lines.
"""

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

# MANDATORY vars for most leaderboard setups.
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

# Allowed defaults (per requirement).
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = os.getenv("SWASTHAI_TASK") or os.getenv("SWASTHAI_TASKS") or "all"
BENCHMARK = os.getenv("SWASTHAI_BENCHMARK", "swasthai_v1")

MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "250"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.70"))
USE_LLM = os.getenv("USE_LLM", "1").strip().lower() not in {"0", "false", "no"}

MAX_TOTAL_REWARD = max(float(MAX_STEPS), 1.0)

DIAGNOSIS_LABELS: Tuple[str, ...] = (
    "common cold", "influenza", "dengue", "malaria", "typhoid",
    "pneumonia", "covid-19", "chikungunya",
)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert clinical triage AI in a simulated diagnostic environment.
    Output exactly one command as valid JSON:
    - {"type": "ask", "content": "<targeted clinical question>"}
    - {"type": "diagnose", "content": "<diagnosis label>", "confidence": <0.0-1.0>}
    Rules:
    - No markdown, no backticks, no explanations outside JSON.
    - Ask the most discriminating question first.
    - Do NOT repeat questions already asked.
    - After 4 questions, you MUST diagnose.
    """
).strip()


def _to_bool_str(value: bool) -> str:
    return "true" if value else "false"


def _sanitize_single_line(value: str) -> str:
    return " ".join((value or "").split())


def _format_error(err: Optional[str]) -> str:
    if err is None or err == "":
        return "null"
    return _sanitize_single_line(err)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str], task: Optional[str] = None
) -> None:
    task_part = f"task={_sanitize_single_line(task)} " if task else ""
    print(
        "[STEP] "
        f"{task_part}"
        f"step={step} "
        f"action={_sanitize_single_line(action)} "
        f"reward={reward:.2f} "
        f"done={_to_bool_str(done)} "
        f"error={_format_error(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_to_bool_str(success)} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _infer_diagnosis(obs_text: str) -> str:
    text = obs_text.lower()
    if "joint_swelling" in text or ("swollen" in text and "joint" in text):
        return "chikungunya"
    if "anosmia" in text or "loss of smell" in text or "loss of taste" in text:
        return "covid-19"
    if "crackle" in text or "sputum" in text:
        return "pneumonia"
    if "stepladder" in text or "street food" in text or "untreated water" in text:
        return "typhoid"
    if "cyclic" in text or "endemic" in text or "malaria zone" in text:
        return "malaria"
    if "platelets: low" in text or "bleeding" in text or "mosquito" in text:
        return "dengue"
    if "cough" in text or "sore throat" in text:
        return "common cold"
    return "influenza"


def _pick_question(step: int, asked: List[str]) -> str:
    questions = [
        "How long have you had these symptoms?",
        "Do you have any rash?",
        "Have you had recent mosquito exposure or travel?",
        "Do you have difficulty breathing?",
        "Do you have chills or sweating episodes?",
        "What is your oxygen saturation?",
        "Have you lost your sense of smell or taste?",
        "Do you have joint swelling?",
    ]
    asked_lower = {a.lower() for a in asked}
    for q in questions:
        if q.lower() not in asked_lower:
            return q
    return questions[0]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_openai_client_class() -> Any:
    from openai import OpenAI
    return OpenAI


def _load_env_classes() -> Any:
    from openenv_submission.server.environment import SwasthAIAction, SwasthAIEnvironment
    return SwasthAIAction, SwasthAIEnvironment


def get_model_message(
    client: Optional[Any], step: int, last_obs: str, last_reward: float, history: List[str]
) -> str:
    if client is None:
        return _heuristic_action(step, last_obs, history)

    history_block = "\n".join(history[-6:]) if history else "None"
    user_prompt = textwrap.dedent(
        f"""
        Step: {step}
        Last observation: {last_obs!r}
        Last reward: {last_reward:.2f}
        Recent history:
        {history_block}

        Return only your next action as JSON.
        """
    ).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            return _heuristic_action(step, last_obs, history)
        return _sanitize_single_line(text)
    except Exception:
        return _heuristic_action(step, last_obs, history)


def _heuristic_action(step: int, last_obs: str, history: List[str]) -> str:
    asked = [h.split("action=")[1].split(" ")[0] if "action=" in h else "" for h in history]
    if step >= MAX_STEPS - 1 or len(history) >= 4:
        dx = _infer_diagnosis(last_obs + " " + " ".join(history))
        return json.dumps({"type": "diagnose", "content": dx, "confidence": 0.8})
    q = _pick_question(step, asked)
    return json.dumps({"type": "ask", "content": q})


def _extract_last_error(result: Any) -> Optional[str]:
    md = getattr(result, "metadata", None)
    if isinstance(md, dict):
        value = md.get("last_action_error")
        if value is not None:
            return str(value)
    return None


def _load_known_tasks() -> List[str]:
    try:
        from openenv_submission.tasks import list_task_names
        return list(list_task_names())
    except Exception:
        return []


def _resolve_task_sequence(task_spec: str) -> List[str]:
    available = _load_known_tasks()
    spec = (task_spec or "").strip()

    if not spec:
        return available[:1] if available else ["easy_fever_cough"]

    low = spec.lower()
    if low == "auto":
        return available[:1] if available else ["easy_fever_cough"]
    if low in {"all", "*"}:
        return available if available else ["easy_fever_cough"]

    if "," in spec:
        requested = [item.strip() for item in spec.split(",") if item.strip()]
        if available:
            filtered = [item for item in requested if item in available]
            return filtered or available[:1]
        return requested

    return [spec]


def _ensure_minimum_task_coverage(task_sequence: List[str], min_tasks: int = 3) -> List[str]:
    available = _load_known_tasks()
    unique: List[str] = []
    seen = set()

    for item in task_sequence:
        if item and item not in seen:
            unique.append(item)
            seen.add(item)

    for item in available:
        if len(unique) >= min_tasks:
            break
        if item not in seen:
            unique.append(item)
            seen.add(item)

    return unique


async def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env = None
    client: Optional[Any] = None

    try:
        action_cls, env_cls = _load_env_classes()
        env = env_cls(max_steps=MAX_STEPS)

        task_sequence = _resolve_task_sequence(TASK_NAME)
        task_sequence = _ensure_minimum_task_coverage(task_sequence, min_tasks=3)

        should_use_llm = USE_LLM and bool(API_KEY)
        if API_KEY and API_KEY.lower().startswith("dummy"):
            should_use_llm = False

        if should_use_llm:
            try:
                openai_cls = _load_openai_client_class()
                client = openai_cls(base_url=API_BASE_URL, api_key=API_KEY)
            except Exception:
                client = None

        episode_scores: List[float] = []
        for task_name in task_sequence:
            history: List[str] = []
            task_rewards: List[float] = []

            # Per-task [START]
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

            try:
                result = env.reset(task_name=task_name)
            except TypeError:
                result = env.reset()

            last_obs = str(getattr(result, "public_symptoms", ""))
            last_reward = 0.0
            task_steps = 0

            for step in range(1, MAX_STEPS + 1):
                if bool(getattr(result, "done", False)):
                    break

                action_text = get_model_message(client, step, last_obs, last_reward, history)

                # Parse action
                try:
                    m = _JSON_RE.search(action_text)
                    if m:
                        action_data = json.loads(m.group(0))
                        action = action_cls(**action_data)
                    else:
                        action = action_cls(type="diagnose", content=action_text[:120])
                except Exception:
                    action = action_cls(type="diagnose", content="unknown")

                try:
                    result = env.step(action)
                    reward = float(getattr(result, "reward", 0.0) or 0.0)
                    done = bool(getattr(result, "done", False))
                    error = _extract_last_error(result)
                except Exception as exc:
                    reward = 0.0
                    done = True
                    error = str(exc)

                rewards.append(reward)
                task_rewards.append(reward)
                steps_taken += 1
                task_steps += 1
                last_obs = str(getattr(result, "history", ""))
                last_reward = reward

                log_step(
                    step=step,
                    action=action_text,
                    reward=reward,
                    done=done,
                    error=error,
                    task=task_name,
                )
                history.append(f"step={step} action={action_text!r} reward={reward:.2f}")

                if done:
                    break

            raw_task_score = sum(task_rewards) / MAX_TOTAL_REWARD
            task_score = min(max(raw_task_score, 0.0), 1.0)
            episode_scores.append(task_score)

            # Per-task [END]
            task_success = task_score >= SUCCESS_SCORE_THRESHOLD
            log_end(success=task_success, steps=task_steps, score=task_score, rewards=task_rewards)

        if episode_scores:
            score = min(max(sum(episode_scores) / len(episode_scores), 0.0), 1.0)
        else:
            raw_score = sum(rewards) / MAX_TOTAL_REWARD
            score = min(max(raw_score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception:
        raw_score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(raw_score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
