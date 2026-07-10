# syntax=docker/dockerfile:1

# ── CPU target (default) ────────────────────────────────────────────────
FROM python:3.11-slim AS cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install PyTorch CPU-only to keep image small
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Patch pyannote.audio: use_auth_token → token for hf_hub_download compatibility
RUN python -c "import pyannote.audio.core.model as m; src=open(m.__file__).read(); open(m.__file__,'w').write(src.replace('use_auth_token=use_auth_token','token=use_auth_token'))"
RUN python -c "import pyannote.audio.core.pipeline as p; src=open(p.__file__).read(); open(p.__file__,'w').write(src.replace('use_auth_token=use_auth_token','token=use_auth_token'))"

COPY app/ ./app/
RUN mkdir -p /app/storage/videos /app/storage/tmp

ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=small
ENV WHISPER_DEVICE=cpu
ENV TRANSCRIBER_PROVIDER=local
ENV DIARIZER_PROVIDER=local

EXPOSE 8000
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ── GPU target (NVIDIA CUDA) ─────────────────────────────────────────────
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04 AS gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# Patch pyannote.audio: use_auth_token → token for hf_hub_download compatibility
RUN python -c "import pyannote.audio.core.model as m; src=open(m.__file__).read(); open(m.__file__,'w').write(src.replace('use_auth_token=use_auth_token','token=use_auth_token'))"
RUN python -c "import pyannote.audio.core.pipeline as p; src=open(p.__file__).read(); open(p.__file__,'w').write(src.replace('use_auth_token=use_auth_token','token=use_auth_token'))"

COPY app/ ./app/
RUN mkdir -p /app/storage/videos /app/storage/tmp

ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=small
ENV WHISPER_DEVICE=cuda
ENV TRANSCRIBER_PROVIDER=local
ENV DIARIZER_PROVIDER=local

EXPOSE 8000
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
