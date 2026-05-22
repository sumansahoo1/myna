# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────
COPY app/ ./app/

# ── Runtime ────────────────────────────────────────────────────────────
EXPOSE 8000

ENV WHISPER_MODEL=small
ENV WHISPER_DEVICE=auto
ENV TRANSCRIBER_PROVIDER=local
ENV DIARIZER_PROVIDER=local

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
