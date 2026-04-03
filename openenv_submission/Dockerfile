FROM python:3.11-slim

WORKDIR /app

# Copy only the submission layer and root inference entry
COPY openenv_submission/ /app/openenv_submission/
COPY inference.py /app/inference.py

RUN pip install --no-cache-dir -r /app/openenv_submission/requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "openenv_submission.server.app"]
