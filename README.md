---
title: swasthai-openenv
sdk: docker
app_port: 7860
license: apache-2.0
---

# SwasthAI OpenEnv (Hackathon Layer)

This folder is a **standalone OpenEnv-style environment** built on top of the SwasthAI idea (healthcare triage), without changing the existing frontend/backend.

## What this environment simulates
A clinician-style workflow:
- The agent sees **public symptoms**.
- It can **ask questions** to retrieve hidden facts (duration, platelet count, rash, etc.).
- It must then **produce a diagnosis**.

## Tasks (3+)
- `easy_fever_cough` → common cold
- `medium_flu_vs_dengue` → influenza
- `hard_dengue_like` → dengue

## Action space
- `{"type": "ask", "content": "<question>"}`
- `{"type": "diagnose", "content": "<diagnosis label>"}`

## Observation space
- `task`: task name
- `public_symptoms`: visible symptoms list
- `history`: list of conversation turns (Q/A)

## Rewards
All rewards are normalized to **[0.0, 1.0]**.
- `ask` → `0.05` for retrieving a relevant hidden fact, otherwise `0.0`
- `diagnose` → `1.0` exact/synonym match, `0.5` partial token overlap, else `0.0`

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
