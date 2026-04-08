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
inference.py          → runs all 8 tasks, emits [START]/[STEP]/[END] logs
openenv_submission/
  ├── env.py          → SwasthAIEnv with progressive rewards, time-decay, trajectory grading
  ├── grader.py       → deterministic per-task graders with synonym maps & partial credit
  ├── tasks.py        → 8 patient cases (easy → expert) with hidden facts & variations
  ├── inference.py    → LLM agent with CoT prompting + heuristic fallback
  ├── openenv.yaml    → OpenEnv spec (8 tasks, graders, entrypoint)
  └── server/
       ├── app.py         → FastAPI server via openenv-core
       └── environment.py → OpenEnv Environment interface implementation
tests/
  ├── test_grader.py  → 38 unit tests for grading logic
  └── test_env.py     → 28 unit tests for environment behavior
```

## What this environment simulates
A **clinician-style diagnostic workflow**:
- The agent sees **public symptoms** at episode start.
- It can **ask questions** to retrieve hidden clinical facts (duration, platelet count, rash, travel history, SpO2, sputum, smell, etc.).
- It must then **produce a diagnosis** with an optional **confidence score**.
- The environment rewards **efficient information-gathering**, **asking the right questions**, and **accurate early diagnosis**.

## Tasks (8 — easy → expert)

| Task | Difficulty | Target Diagnosis | Key Differentiators |
|------|-----------|-----------------|---------------------|
| `easy_fever_cough` | Easy | common cold | Short duration, cough, sore throat |
| `medium_flu_vs_dengue` | Medium | influenza | Body pain, fatigue, normal platelets |
| `medium_pneumonia` | Medium | pneumonia | Productive cough, low SpO2, crackles |
| `hard_dengue_like` | Hard | dengue | Low platelets, bleeding, mosquito exposure |
| `hard_covid_respiratory` | Hard | covid-19 | Loss of smell/taste, gathering exposure |
| `expert_malaria_mimic` | Expert | malaria | Cyclic fever, endemic travel, enlarged spleen |
| `expert_typhoid_enteric` | Expert | typhoid | Stepladder fever, diarrhea, street food exposure |
| `expert_chikungunya` | Expert | chikungunya | Joint swelling, outbreak area, conjunctivitis |

Each task has **7–10 hidden facts** and **seed-based variations** for different presentations.

## Action space
- `{"type": "ask", "content": "<question>"}` — retrieve a hidden clinical fact
- `{"type": "diagnose", "content": "<label>", "confidence": 0.85}` — submit diagnosis with optional confidence

## Observation space
- `task`: task name
- `public_symptoms`: visible symptoms list
- `history`: list of conversation turns (Q/A pairs and diagnoses)
- `last_answer`: most recent answer from the environment
- `vitals`: structured dict of retrieved vital signs (temperature, SpO2, platelets)

## Rewards (progressive shaping)
All rewards are normalized to the open interval **(0.0, 1.0)**.

| Action | Condition | Reward |
|--------|-----------|--------|
| `ask` | High-value fact (platelets, bleeding, travel, rash, SpO2, sputum, smell, etc.) | **0.10** |
| `ask` | Standard fact (duration, temperature, appetite, etc.) | **0.05** |
| `ask` | Repeated question | **0.01** |
| `ask` | Irrelevant / unrecognized | **0.01** |
| `diagnose` | Exact match or synonym × time-decay | **up to 0.99** |
| `diagnose` | Partial token overlap × time-decay | **0.40–0.70** |
| `diagnose` | Wrong diagnosis (penalized per attempt) | **decreasing** |

## Advanced features

- **Time-decay**: Earlier correct diagnosis → higher reward (step 1 = 1.0×, step 8 = 0.5×)
- **Trajectory grading**: 60% diagnosis + 25% question quality + 15% efficiency
- **Confidence weighting**: High-confidence correct answers get bonus; high-confidence wrong answers penalized
- **Wrong-diagnosis penalty**: Each wrong attempt reduces reward by 0.05 (cumulative)
- **Seed-based variations**: Same task with different hidden fact values (e.g., platelet count: low/very low/critically low)
- **Structured vitals**: Retrieved vital signs appear in `obs.vitals` dict

## Grading logic
- **Deterministic** — no randomness, fully reproducible
- **Synonym-aware** — 8 disease synonym maps (e.g., "flu" → influenza, "corona" → covid-19, "chikv" → chikungunya)
- **Partial credit** — token overlap ratio gives proportional score
- **Severity-scaled** — scores clamped to open interval (0, 1) for validator compatibility

## RL framing (why this is reinforcement learning)
This is an RL-style environment: an agent interacts over multiple steps by choosing actions (`ask` vs `diagnose`) based on observations (symptoms + conversation history + vitals) to maximize cumulative reward. Rewards are **progressively shaped**:
- **Dense rewards** for information-gathering (higher for diagnostically relevant facts)
- **Diminishing returns** for repeated questions
- **Time-decayed terminal reward** for correct diagnosis
- **Trajectory bonus** for asking the right diagnostic questions
- **Penalty** for overconfident wrong diagnoses
- Episodes terminate on correct diagnosis or at the step limit (default: 8)

## Testing

```bash
python -m pytest tests/ -v
```

66 unit tests covering grading logic, environment behavior, all 8 tasks, synonyms, confidence, trajectory scoring, time-decay, seed reproducibility, and edge cases.

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
