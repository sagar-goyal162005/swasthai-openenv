FROM python:3.11-slim

WORKDIR /app

# Copy the full submission bundle so `openenv validate` (repo-mode)
# can see repo-root artifacts like `pyproject.toml`, `uv.lock`, and `server/app.py`.
COPY . /app

RUN pip install --no-cache-dir -r /app/openenv_submission/requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "openenv_submission.server.app"]
