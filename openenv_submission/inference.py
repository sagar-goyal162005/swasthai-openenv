from __future__ import annotations

import json
import os
import re
import textwrap
from typing import List, Optional, Tuple

try:
    # openai>=1.0.0
    from openai import OpenAI  # type: ignore

    _OPENAI_V1 = True
except Exception:  # pragma: no cover
    # openai<1.0.0
    import openai  # type: ignore

    OpenAI = None  # type: ignore
    _OPENAI_V1 = False

from openenv_submission.env import Action, SwasthAIEnv
from openenv_submission.tasks import list_task_names


def _env_vars() -> Tuple[str, str, str]:
    api_base_url = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    model_name = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
    hf_token = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
    return api_base_url, model_name, hf_token


def _build_prompt(task: str, symptoms: List[str], history: List[str]) -> str:
    return textwrap.dedent(
        f"""
        You are a clinical triage assistant in a simulated environment.

        Task: Identify the most likely diagnosis.
        Visible symptoms: {symptoms}

        Conversation history (Q/A + any prior diagnoses):
        {"\n".join(history) if history else "(none)"}

        Choose ONE action for the next step, in strict JSON:
        {{"type": "ask", "content": "<one short question>"}}
        OR
        {{"type": "diagnose", "content": "<diagnosis label>"}}

        Rules:
        - Ask at most 3 questions then diagnose.
        - Keep content concise.
        - Output ONLY JSON, no markdown.
        """
    ).strip()


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_action(raw: str) -> Action:
    m = _JSON_RE.search(raw.strip())
    if not m:
        # fallback: diagnose with raw text
        return Action(type="diagnose", content=raw.strip()[:120])

    try:
        data = json.loads(m.group(0))
        return Action.model_validate(data)
    except Exception:
        return Action(type="diagnose", content=raw.strip()[:120])


def run_episode(task_name: str, max_steps: int = 8) -> int:
    api_base_url, model_name, hf_token = _env_vars()

    if _OPENAI_V1:
        client = OpenAI(api_key=hf_token, base_url=api_base_url)  # type: ignore[misc]
    else:
        # openai<1.0.0 compatibility
        openai.api_key = hf_token  # type: ignore[name-defined]
        openai.api_base = api_base_url  # type: ignore[name-defined]
        client = None

    env = SwasthAIEnv(max_steps=max_steps)
    obs = env.reset(task_name)

    print(f"[START] task={task_name} env={env.benchmark_name} model={model_name}")

    rewards: List[str] = []
    success = False
    last_error: Optional[str] = None

    try:
        for step_idx in range(max_steps):
            prompt = _build_prompt(task=obs.task, symptoms=obs.public_symptoms, history=obs.history)

            try:
                messages = [
                    {"role": "system", "content": "You must output strict JSON only."},
                    {"role": "user", "content": prompt},
                ]

                if _OPENAI_V1:
                    resp = client.chat.completions.create(  # type: ignore[union-attr]
                        model=model_name,
                        temperature=0.0,
                        max_tokens=200,
                        messages=messages,
                    )
                    raw = (resp.choices[0].message.content or "").strip()
                else:
                    resp = openai.ChatCompletion.create(  # type: ignore[name-defined]
                        model=model_name,
                        temperature=0.0,
                        max_tokens=200,
                        messages=messages,
                    )
                    raw = (resp["choices"][0]["message"]["content"] or "").strip()

                action = _parse_action(raw)
                last_error = None
            except Exception as e:
                # keep evaluation format stable even if LLM call fails
                last_error = f"llm_error: {type(e).__name__}: {e}"
                action = Action(type="diagnose", content="unknown")

            obs, reward, done, info = env.step(action)
            rewards.append(f"{reward:.2f}")

            # action must be a single string
            action_str = (
                f"ask({json.dumps(action.content)})" if action.type == "ask" else f"diagnose({json.dumps(action.content)})"
            )

            step_error = info.get("last_action_error") or last_error
            if not step_error:
                err_field = "null"
            else:
                # keep it single-token-ish for strict parsers
                err_field = str(step_error).replace("\n", " ").replace("\r", " ")
                err_field = re.sub(r"\s+", "_", err_field)[:240]
            print(
                f"[STEP] step={step_idx + 1} action={action_str} reward={reward:.2f} done={str(done).lower()} error={err_field}"
            )

            if done:
                # success is "correct diagnosis at any point"
                success = reward >= 1.0
                break

    finally:
        env.close()
        print(f"[END] success={str(success).lower()} steps={len(rewards)} rewards={','.join(rewards)}")

    return 0


def main() -> int:
    task_name = os.getenv("OPENENV_TASK") or list_task_names()[0]
    return run_episode(task_name)


if __name__ == "__main__":
    raise SystemExit(main())
