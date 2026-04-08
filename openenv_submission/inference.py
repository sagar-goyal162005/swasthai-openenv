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


_DIAGNOSIS_LABELS: Tuple[str, ...] = (
    "common cold", "influenza", "dengue", "malaria", "typhoid",
    "pneumonia", "covid-19", "chikungunya",
)

_QUESTION_NEEDLES: Tuple[str, ...] = (
    "how long", "duration", "days",
    "rash", "platelet", "bleeding",
    "body pain", "fatigue", "breath",
    "travel", "mosquito",
    "appetite", "temperature", "chills",
    "diarrhea", "spleen", "abdomen",
    "dehydration", "stool", "constipation",
    "food", "eat",
    # pneumonia / covid / chikungunya
    "sputum", "phlegm", "mucus",
    "oxygen", "spo2", "saturation",
    "lung", "auscultation", "crackle",
    "smell", "anosmia", "taste",
    "joint", "swelling",
    "eye", "conjunctiv",
    "gathering", "contact", "exposure",
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

    # Malaria / typhoid disambiguators
    candidates.append("Do you have chills or sweating episodes?")
    candidates.append("What is your temperature pattern?")
    candidates.append("Do you have diarrhea or constipation?")
    candidates.append("Is your spleen or abdomen tender?")
    candidates.append("How is your appetite and food intake?")

    # Pneumonia / COVID disambiguators
    candidates.append("Do you have sputum or phlegm? What color?")
    candidates.append("What is your oxygen saturation (SpO2)?")
    candidates.append("Have you lost your sense of smell or taste?")
    candidates.append("Did you attend any large gatherings or have close contact with sick people?")

    # Chikungunya disambiguators
    candidates.append("Do you have joint swelling, especially in wrists or ankles?")
    candidates.append("Do you have any eye redness or conjunctivitis?")

    asked_set = {a.strip().lower() for a in asked}
    for q in candidates:
        if q.strip().lower() not in asked_set:
            return q
    return "How long have you had these symptoms?"


def _heuristic_diagnosis(symptoms: Sequence[str], history: Sequence[str]) -> str:
    s = " ".join(symptoms).lower()
    h = "\n".join(history).lower()

    # Chikungunya — joint swelling + rash + tropical outbreak
    if ("joint_swelling" in h or "swollen joints" in s) and ("outbreak" in h or "conjunctivitis" in h):
        return "chikungunya"

    # COVID-19 — loss of smell/taste + gathering exposure
    if "anosmia" in h or "loss of smell" in h or ("loss of taste" in s and "gathering" in h):
        return "covid-19"

    # Pneumonia — productive cough + crackles/sputum + low SpO2
    if ("crackle" in h or "sputum" in h) and ("cough" in s or "chest" in s):
        return "pneumonia"

    # Typhoid indicators — stepladder fever, diarrhea, street food
    if ("stepladder" in h or "street food" in h or "untreated water" in h
            or ("diarrhea" in h and "abdominal" in s)):
        return "typhoid"

    # Malaria indicators — cyclic fever, endemic zone, spleen
    if ("cyclic" in h or "endemic" in h or "malaria zone" in h
            or ("spleen" in h and "chills" in s)):
        return "malaria"

    # Strong dengue indicators
    if "platelets: low" in h or "bleeding" in h or "mosquito" in h:
        return "dengue"

    # URI/common cold style symptoms
    if "cough" in s or "sore throat" in s:
        if "shortness of breath" in s or "chest pain" in s:
            return "pneumonia"
        return "common cold"

    # Default influenza-like illness
    return "influenza"


def _heuristic_confidence(symptoms: Sequence[str], history: Sequence[str]) -> float:
    """Estimate confidence based on how much evidence was gathered."""
    h = "\n".join(history).lower()
    evidence_count = h.count("a: ") - h.count("already asked")
    if evidence_count >= 4:
        return 0.9
    if evidence_count >= 2:
        return 0.7
    return 0.5


def _env_vars() -> Tuple[str, str, Optional[str]]:
    api_base_url = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    model_name = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
    hf_token = os.getenv("HF_TOKEN")
    return api_base_url, model_name, hf_token


_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert clinical triage AI in a simulated diagnostic environment.
Your goal is to efficiently identify the correct diagnosis by asking targeted
questions and then providing your diagnosis with a confidence score.

Think step by step:
1. Analyze the visible symptoms and conversation history
2. Identify which key clinical facts are still unknown
3. Choose the most discriminating question OR diagnose if you have enough evidence

Output ONLY valid JSON, no markdown, no explanation outside the JSON.
""").strip()

_FEW_SHOT_EXAMPLES = textwrap.dedent("""\
Example interactions:
- Symptoms: [fever, cough, sore throat] → ask about duration → 2 days → ask about breathing → no → diagnose "common cold" confidence 0.9
- Symptoms: [high fever, chills, sweating, headache] → ask about travel → endemic malaria zone → ask about temperature → cyclic → diagnose "malaria" confidence 0.85
- Symptoms: [fever, dry cough, loss of taste] → ask about smell → anosmia → diagnose "covid-19" confidence 0.9
""").strip()


def _build_prompt(task: str, symptoms: List[str], history: List[str]) -> str:
    history_block = "\n".join(history) if history else "(none)"
    return textwrap.dedent(
        f"""
        {_FEW_SHOT_EXAMPLES}

        ---

        Current case:
        Visible symptoms: {symptoms}

        Conversation history:
        {history_block}

        Choose ONE action in strict JSON:
        {{"type": "ask", "content": "<targeted clinical question>"}}
        OR
        {{"type": "diagnose", "content": "<diagnosis label>", "confidence": <0.0-1.0>}}

        Allowed diagnosis labels (must match exactly): {list(_DIAGNOSIS_LABELS)}

        Rules:
        - Ask the MOST discriminating question first (e.g., travel history, platelet count, smell/taste).
        - Do NOT repeat a question already asked.
        - Ask at most 4 questions total, then you MUST diagnose.
        - Include a confidence score (0.0-1.0) when diagnosing.
        - Consider differential diagnosis: rule out similar diseases before deciding.
        - Output ONLY JSON, no markdown or extra text.
        """
    ).strip()


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_action(raw: str) -> Action:
    m = _JSON_RE.search(raw.strip())
    if not m:
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
            max_asks = 4
            prompt = _build_prompt(task=obs.task, symptoms=obs.public_symptoms, history=obs.history)

            try:
                try:
                    if not hf_token:
                        raise RuntimeError("missing_hf_token")

                    messages = [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]

                    if _OPENAI_V1:
                        resp = client.chat.completions.create(  # type: ignore[union-attr]
                            model=model_name,
                            temperature=0.0,
                            max_tokens=250,
                            messages=messages,
                        )
                        raw = (resp.choices[0].message.content or "").strip()
                    else:
                        resp = openai.ChatCompletion.create(  # type: ignore[name-defined]
                            model=model_name,
                            temperature=0.0,
                            max_tokens=250,
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
                dx = _heuristic_diagnosis(obs.public_symptoms, obs.history)
                conf = _heuristic_confidence(obs.public_symptoms, obs.history)
                action = Action(type="diagnose", content=dx, confidence=conf)
            else:
                if action.type == "ask":
                    q_norm = (action.content or "").strip().lower()
                    if (not q_norm) or (q_norm in {q.strip().lower() for q in asked_questions}) or (not _is_supported_question(q_norm)):
                        action = Action(type="ask", content=_pick_next_question(obs.public_symptoms, asked_questions))
                else:
                    pred = (action.content or "").strip().lower()
                    if pred not in _DIAGNOSIS_LABELS:
                        dx = _heuristic_diagnosis(obs.public_symptoms, obs.history)
                        conf = _heuristic_confidence(obs.public_symptoms, obs.history)
                        action = Action(type="diagnose", content=dx, confidence=conf)

            obs, reward, done, info = env.step(action)
            final_score = float(reward)
            rewards.append(f"{reward:.2f}")

            action_str = (
                f"ask({json.dumps(action.content)})" if action.type == "ask" else f"diagnose({json.dumps(action.content)})"
            )

            step_error = info.get("last_action_error") or last_error
            if not step_error:
                err_field = "null"
            else:
                err_field = str(step_error).replace("\n", " ").replace("\r", " ")

            extra = ""
            if info.get("trajectory_score") is not None:
                extra += f" trajectory={info['trajectory_score']:.2f}"
            if info.get("agent_confidence") is not None:
                extra += f" confidence={info['agent_confidence']:.2f}"

            print(
                f"[STEP] step={step_idx + 1} action={action_str} reward={reward:.2f} done={str(done).lower()} error={err_field}{extra}"
            )

            if done:
                success = bool(info.get("is_correct", False))
                break

    finally:
        env.close()
        print(
            f"[END] success={str(success).lower()} steps={len(rewards)} score={final_score:.2f} rewards={','.join(rewards)}"
        )

    return 0


def main() -> int:
    single = os.getenv("OPENENV_TASK")
    if single:
        return run_episode(single)
    # Run ALL tasks so the validator sees 3+ tasks with graders.
    exit_code = 0
    for task_name in list_task_names():
        rc = run_episode(task_name)
        if rc != 0:
            exit_code = rc
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
