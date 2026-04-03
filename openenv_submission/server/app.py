from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI

from openenv.core.env_server import create_app

from .environment import SwasthAIAction, SwasthAIEnvironment, SwasthAIObservation


def _env_factory() -> SwasthAIEnvironment:
    # One env instance per session (OpenEnv handles this via factory)
    max_steps = int(os.getenv("MAX_STEPS", "8"))
    return SwasthAIEnvironment(max_steps=max_steps)


# OpenEnv-compatible server app (includes WebSocket + HTTP API contract)
app: FastAPI = create_app(
    env=_env_factory,
    action_cls=SwasthAIAction,
    observation_cls=SwasthAIObservation,
    env_name="swasthai",
)


@app.get("/")
def root() -> dict:
    # Ping endpoint for hackathon HF Space availability checks
    return {"ok": True, "env": "swasthai"}


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("SPACE_PORT", "7860")))
    uvicorn.run(app, host=host, port=port, log_level=os.getenv("LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
