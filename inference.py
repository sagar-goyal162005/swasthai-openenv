from __future__ import annotations

import os

from openai import OpenAI  # noqa: F401

from openenv_submission.inference import main


# Required env vars for the hackathon validator (do not set defaults for HF_TOKEN)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — only needed if you use from_docker_image() in your inference
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


if __name__ == "__main__":
    raise SystemExit(main())
