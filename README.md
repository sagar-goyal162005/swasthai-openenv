---
title: swasthai-openenv
sdk: docker
app_port: 7860
license: apache-2.0
---

# SwasthAI OpenEnv (Hackathon Layer)

This folder is a **standalone OpenEnv-style environment** built on top of the SwasthAI idea (healthcare triage), without changing the existing frontend/backend.

## Architecture

```
inference.py          → runs all 5 tasks, emits [START]/[STEP]/[END] logs
openenv_submission/
  ├── env.py          → standalone SwasthAIEnv (reset/step/state)
  ├── grader.py       → deterministic per-task graders with synonym & partial credit
  ├── tasks.py        → 5 patient cases (easy → expert) with hidden facts
  ├── inference.py    → LLM agent + heuristic fallback
  ├── openenv.yaml    → OpenEnv spec (tasks, graders, entrypoint)
  └── server/
       ├── app.py         → FastAPI server via openenv-core
       └── environment.py → OpenEnv Environment interface implementation
```

## What this environment simulates
A **clinician-style diagnostic workflow**:
- The agent sees **public symptoms** at episode start.
- It can **ask questions** to retrieve hidden clinical facts (duration, platelet count, rash, travel history, spleen status, etc.).
- It must then **produce a diagnosis** from the evidence gathered.
- The environment rewards **efficient information-gathering** and **accurate diagnosis**.

## Tasks (5 — easy → expert)

| Task | Difficulty | Target Diagnosis | Key Differentiators |
|------|-----------|-----------------|---------------------|
| `easy_fever_cough` | Easy | common cold | Short duration, cough, sore throat |
| `medium_flu_vs_dengue` | Medium | influenza | Body pain, fatigue, normal platelets |
| `hard_dengue_like` | Hard | dengue | Low platelets, bleeding, mosquito exposure |
| `expert_malaria_mimic` | Expert | malaria | Cyclic fever, endemic travel, enlarged spleen |
| `expert_typhoid_enteric` | Expert | typhoid | Stepladder fever, diarrhea, street food exposure |

Each task has **7–10 hidden facts** the agent must discover through targeted questions.

## Action space
- `{"type": "ask", "content": "<question>"}` — retrieve a hidden clinical fact
- `{"type": "diagnose", "content": "<diagnosis label>"}` — submit a diagnosis

## Observation space
- `task`: task name
- `public_symptoms`: visible symptoms list
- `history`: list of conversation turns (Q/A pairs and diagnoses)
- `last_answer`: most recent answer from the environment

## Rewards (progressive shaping)
All rewards are normalized to the open interval **(0.0, 1.0)**.

| Action | Condition | Reward |
|--------|-----------|--------|
| `ask` | High-value fact (platelets, bleeding, travel, rash, diarrhea, spleen) | **0.10** |
| `ask` | Standard fact (duration, temperature, appetite, etc.) | **0.05** |
| `ask` | Repeated question | **0.01** |
| `ask` | Irrelevant / unrecognized | **0.01** |
| `diagnose` | Exact match or synonym | **0.99** |
| `diagnose` | Partial token overlap | **0.40–0.70** (ratio-based) |
| `diagnose` | Partial synonym overlap | **0.35** |
| `diagnose` | No match | **0.01** |

## Grading logic
- **Deterministic** — no randomness, fully reproducible
- **Synonym-aware** — accepts common medical synonyms (e.g., "flu" → influenza, "enteric fever" → typhoid)
- **Partial credit** — token overlap ratio gives proportional score
- **Severity-scaled** — scores clamped to open interval (0, 1) for validator compatibility

## RL framing (why this is reinforcement learning)
This is an RL-style environment: an agent interacts over multiple steps by choosing actions (`ask` vs `diagnose`) based on observations (symptoms + conversation history) to maximize cumulative reward. Rewards are **progressively shaped**:
- **Dense rewards** for information-gathering (higher for diagnostically relevant facts)
- **Diminishing returns** for repeated questions
- **Terminal reward** for correct diagnosis
- Episodes terminate on correct diagnosis or at the step limit (default: 8)

## Run locally (Python)
From repo root:

```bash
python inference.py
```

## Run via Docker
From repo root:

```bash
docker build -t swasthai-openenv .
docker run --rm -p 7860:7860 swasthai-openenv
```

This starts the **OpenEnv server** (HF Spaces needs a long-running HTTP process).

To run the baseline evaluation script inside the container instead:

```bash
docker run --rm swasthai-openenv python /app/inference.py
```

## Required environment variables for the hackathon inference
- `API_BASE_URL` (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` (default: `Qwen/Qwen2.5-72B-Instruct`)
- `HF_TOKEN` (HF / API key)

Note: the folder is named `openenv_submission/` (not `openenv/`) to avoid clashing with the installed Python package `openenv` that the validator uses.

## Deploy to Hugging Face Spaces (Docker)

Create a new Space:
- **SDK**: Docker
- **Visibility**: Public

Build context:
- Hugging Face Spaces will build using the **root** [Dockerfile](../Dockerfile).

Space settings → Variables and secrets:
- Add **Secret**: `HF_TOKEN`
- Optional **Variables**:
	- `API_BASE_URL` (default: `https://router.huggingface.co/v1`)
	- `MODEL_NAME` (default: `Qwen/Qwen2.5-72B-Instruct`)
	- `MAX_STEPS` (default: `8`)

After the image builds:
- Visiting the Space root URL should return **HTTP 200**.
- The OpenEnv runtime endpoints will handle `reset/step/state`.

Optional (debugging):
- Set `ENABLE_WEB_INTERFACE=true` to enable the built-in OpenEnv web UI (typically available under `/web`).

## Validate (local)

```bash
cd openenv_submission
openenv validate
```
