FROM python:3.11-slim

WORKDIR /app

COPY openenv_submission/ /app/openenv_submission/
COPY openenv.yaml /app/openenv.yaml
COPY inference.py /app/inference.py

RUN pip install --no-cache-dir -r /app/openenv_submission/requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "openenv_submission.server.app"]
