# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (layer cached) ─────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────
COPY app/ ./app/

# ── Runtime dirs ───────────────────────────────────────────────────────
RUN mkdir -p /app/storage/videos /app/storage/tmp

# ── Log output unbuffered ──────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1

# ── Defaults (override via docker-compose or -e) ───────────────────────
ENV WHISPER_MODEL=small
ENV WHISPER_DEVICE=cpu
ENV TRANSCRIBER_PROVIDER=local
ENV DIARIZER_PROVIDER=local

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
