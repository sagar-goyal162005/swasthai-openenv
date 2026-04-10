from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional

from openenv_submission.server.environment import (
    SwasthAIAction,
    SwasthAIEnvironment,
    SwasthAIObservation,
)
from openenv_submission.tasks import list_task_names
from graders import TASK_GRADERS, TASK_GRADERS_COLON

app = FastAPI(title="SwasthAI OpenEnv", version="1.0.0")
_env = SwasthAIEnvironment()


def _task_rows() -> list[dict[str, object]]:
    return [
        {
            "id": task_id,
            "task_id": task_id,
            "name": task_id,
            "task_name": task_id,
            "grader": TASK_GRADERS.get(task_id),
            "grader_fn": TASK_GRADERS.get(task_id),
            "grader_path": TASK_GRADERS.get(task_id),
            "agent_grader": TASK_GRADERS_COLON.get(task_id),
            "grader_colon": TASK_GRADERS_COLON.get(task_id),
            "score_range": [0.0, 1.0],
        }
        for task_id in list_task_names()
    ]


class ResetRequest(BaseModel):
    task_name: Optional[str] = None


@app.get("/")
async def root() -> dict:
    return {
        "name": "swasthai",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata() -> dict:
    rows = _task_rows()
    graders_count = sum(1 for row in rows if row.get("grader") or row.get("grader_fn"))
    return {
        "name": "swasthai",
        "description": "Healthcare triage & diagnosis workflow environment where an agent asks targeted clinical questions and diagnoses patients under step constraints.",
        "tasks_count": str(len(rows)),
        "tasks_with_graders": str(graders_count),
    }


@app.get("/schema")
async def schema() -> dict:
    return {
        "action": SwasthAIAction.model_json_schema(),
        "observation": SwasthAIObservation.model_json_schema(),
        "state": {"type": "object"},
    }


@app.post("/reset")
async def reset(payload: Optional[ResetRequest] = None) -> dict:
    task_name = payload.task_name if payload else None
    obs = _env.reset(task_name=task_name)
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else vars(obs),
        "reward": None,
        "done": False,
    }


@app.get("/tasks")
async def tasks(format: Optional[str] = Query(default=None)) -> object:
    rows = _task_rows()
    if (format or "").lower() in {"object", "dict", "wrapped"}:
        return {
            "tasks": rows,
            "count": len(rows),
            "tasks_with_graders": sum(1 for row in rows if row.get("grader") or row.get("grader_fn")),
        }
    return rows


@app.post("/step")
async def step(action: SwasthAIAction) -> dict:
    obs = _env.step(action)
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else vars(obs),
        "reward": getattr(obs, "reward", 0.0),
        "done": getattr(obs, "done", False),
    }


@app.get("/state")
async def state() -> dict:
    s = _env.state
    return s.model_dump() if hasattr(s, "model_dump") else vars(s)


@app.post("/close")
async def close() -> dict:
    _env.close()
    return {"status": "closed"}
