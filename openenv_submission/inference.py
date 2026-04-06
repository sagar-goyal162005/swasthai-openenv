from __future__ import annotations

import json
import os
import re
import textwrap
from typing import List, Optional, Sequence, Tuple

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


_DIAGNOSIS_LABELS: Tuple[str, ...] = ("common cold", "influenza", "dengue")

_QUESTION_NEEDLES: Tuple[str, ...] = (
    "how long",
    "duration",
    "days",
    "rash",
    "platelet",
    "bleeding",
    "body pain",
    "fatigue",
    "breath",
    "travel",
    "mosquito",
)


def _extract_questions(history: Sequence[str]) -> List[str]:
    qs: List[str] = []
    for line in history:
        if line.startswith("Q: "):
            q = line[3:].strip()
            if q:
                qs.append(q)
    return qs


def _is_supported_question(question: str) -> bool:
    q = (question or "").lower()
    return any(needle in q for needle in _QUESTION_NEEDLES)


def _pick_next_question(symptoms: Sequence[str], asked: Sequence[str]) -> str:
    candidates: List[str] = []
    s = " ".join(symptoms).lower()

    candidates.append("How long have you had these symptoms?")
    candidates.append("Do you have body pain?")

    if "cough" in s or "sore throat" in s or "fever" in s:
        candidates.append("Do you have any difficulty breathing?")

    # Dengue-vs-flu disambiguators
    candidates.append("Do you have any rash?")
    candidates.append("Have you had recent mosquito exposure or travel?")
    candidates.append("Do you know your platelet count?")
    candidates.append("Any bleeding like gum bleeding?")

    asked_set = {a.strip().lower() for a in asked}
    for q in candidates:
        if q.strip().lower() not in asked_set:
            return q
    return "How long have you had these symptoms?"


def _heuristic_diagnosis(symptoms: Sequence[str], history: Sequence[str]) -> str:
    s = " ".join(symptoms).lower()
    h = "\n".join(history).lower()

    # Strong dengue indicators
    if "platelets: low" in h or "bleeding" in h or "mosquito" in h or "rash" in s:
        return "dengue"

    # URI/common cold style symptoms
    if "cough" in s or "sore throat" in s:
        return "common cold"

    # Default influenza-like illness
    return "influenza"


def _env_vars() -> Tuple[str, str, Optional[str]]:
    api_base_url = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    model_name = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
    hf_token = os.getenv("HF_TOKEN")
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

        Allowed diagnosis labels (must match exactly): {list(_DIAGNOSIS_LABELS)}

        Rules:
        - Do NOT repeat a question already asked.
        - Ask at most 3 questions total, then diagnose.
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

    if _OPENAI_V1 and hf_token:
        client = OpenAI(api_key=hf_token, base_url=api_base_url)  # type: ignore[misc]
    else:
        # openai<1.0.0 compatibility
        if not _OPENAI_V1:
            openai.api_key = hf_token  # type: ignore[name-defined]
            openai.api_base = api_base_url  # type: ignore[name-defined]
        client = None

    env = SwasthAIEnv(max_steps=max_steps)
    obs = env.reset(task_name)

    print(f"[START] task={task_name} env={env.benchmark_name} model={model_name}")

    rewards: List[str] = []
    success = False
    last_error: Optional[str] = None
    final_score: float = 0.0

    try:
        for step_idx in range(max_steps):
            asked_questions = _extract_questions(obs.history)
            max_asks = 3
            prompt = _build_prompt(task=obs.task, symptoms=obs.public_symptoms, history=obs.history)

            try:
                try:
                    # If token is missing, skip LLM call and use deterministic fallback.
                    if not hf_token:
                        raise RuntimeError("missing_hf_token")

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
                except BaseException as e:  # pragma: no cover
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise
                    last_error = f"llm_error: {type(e).__name__}: {e}"
                    action = Action(type="diagnose", content="unknown")
            except Exception as e:  # ultra-defensive
                last_error = f"fatal_error: {type(e).__name__}: {e}"
                action = Action(type="diagnose", content="unknown")

            # Enforce ask-limit + de-duplication + last-step diagnosis.
            if (len(asked_questions) >= max_asks) or (step_idx >= max_steps - 1):
                action = Action(type="diagnose", content=_heuristic_diagnosis(obs.public_symptoms, obs.history))
            else:
                if action.type == "ask":
                    q_norm = (action.content or "").strip().lower()
                    if (not q_norm) or (q_norm in {q.strip().lower() for q in asked_questions}) or (not _is_supported_question(q_norm)):
                        action = Action(type="ask", content=_pick_next_question(obs.public_symptoms, asked_questions))
                else:
                    pred = (action.content or "").strip().lower()
                    if pred not in _DIAGNOSIS_LABELS:
                        # If LLM isn't available or returns an invalid label, use a safe exact-label fallback.
                        action = Action(type="diagnose", content=_heuristic_diagnosis(obs.public_symptoms, obs.history))

            obs, reward, done, info = env.step(action)
            final_score = float(reward)
            rewards.append(f"{reward:.2f}")

            # action must be a single string
            action_str = (
                f"ask({json.dumps(action.content)})" if action.type == "ask" else f"diagnose({json.dumps(action.content)})"
            )

            step_error = info.get("last_action_error") or last_error
            if not step_error:
                err_field = "null"
            else:
                # raw error string but keep it on one line
                err_field = str(step_error).replace("\n", " ").replace("\r", " ")
            print(
                f"[STEP] step={step_idx + 1} action={action_str} reward={reward:.2f} done={str(done).lower()} error={err_field}"
            )

            if done:
                # success is "correct diagnosis at any point"
                success = reward >= 1.0
                break

    finally:
        env.close()
        print(
            f"[END] success={str(success).lower()} steps={len(rewards)} score={final_score:.2f} rewards={','.join(rewards)}"
        )

    return 0


def main() -> int:
    task_name = os.getenv("OPENENV_TASK") or list_task_names()[0]
    return run_episode(task_name)


if __name__ == "__main__":
    raise SystemExit(main())
