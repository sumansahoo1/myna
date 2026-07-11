# Myna

**Turn meeting videos into speaker-labeled transcripts — entirely on your machine. No cloud. No API keys. No per-minute pricing.**

Drop in a video. Get back who said what, when.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen)
![Status](https://img.shields.io/badge/status-active-success)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Environment Variables Reference](#environment-variables-reference)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Support & Contact](#support--contact)
- [Acknowledgments](#acknowledgments)

---

## Features

### 🎙️ **High-Accuracy Transcription**
- Powered by **faster-whisper** (CTranslate2-optimized Whisper) with GPU-batched inference
- Configurable model size — `tiny` (fast, 39 MB) through `large-v3` (accurate, 3 GB)
- Batch size configurable via `INFERENCE_BATCH_SIZE` (default 8) for GPU throughput
- Automatic language detection for every transcript

### 👥 **Speaker Diarization**
- **pyannote.audio** identifies who spoke when — not just a wall of text
- Every segment tagged with speaker labels (`SPEAKER_00`, `SPEAKER_01`, etc.)
- Works without a HuggingFace token — all speakers default to `UNKNOWN`
- With `HF_TOKEN` set: full speaker diarization via `pyannote/speaker-diarization-3.1`

### 🔇 **Smart VAD Pre-Filtering**
- **Silero VAD** detects speech regions before transcription — skips silence, noise, and dead air
- Reduces transcription workload and improves accuracy
- Gated by `ENABLE_VAD` (default on). Falls back gracefully on failure

### ⚡ **Two Processing Modes**
- **`/upload`** — Fire-and-forget. Returns immediately, processes in background. Poll for results.
- **`/upload-and-diarize`** — Synchronous. Holds the connection until processing completes. Returns everything in one response.

### 🐳 **Zero-Dependency Deploy**
- Ships as a **single Docker container** with `ffmpeg`, all Python dependencies, and model support
- **CPU** and **GPU** profiles — pick one flag
- Persistent volumes for uploaded videos, database, and cached models (no re-download on restart)
- Healthcheck endpoint for orchestration

### 🔒 **100% Local**
- Transcription and diarization run on your hardware — nothing leaves the machine
- No cloud API calls, no usage limits, no per-minute pricing
- SQLite database (zero config, no external DB required)

### 🛡️ **Resilient Pipeline**
- GPU memory management: models loaded sequentially (never both at once) to avoid OOM
- Automatic CUDA cache clearing between stages
- Retry with exponential backoff on transient GPU errors (OOM, CUDA errors)
- Partial persistence — transcript saved even if diarization fails

---

## Architecture

```
myna/
├── app/                    # FastAPI application
│   ├── main.py             # Entry point, lifespan, router registration
│   ├── config.py           # Settings from environment variables
│   ├── database.py         # SQLAlchemy engine, session, migrations
│   ├── models.py           # ORM models (Meeting, TranscriptSegment)
│   ├── schemas.py          # Pydantic request/response models
│   ├── transcription.py    # Audio extraction + faster-whisper transcription
│   ├── diarization.py      # Speaker diarization (pyannote.audio)
│   ├── merge.py            # Merge transcript segments with speaker turns
│   ├── vad.py              # Silero VAD speech detection
│   └── routers/
│       └── videos.py       # All API route handlers
├── storage/
│   ├── videos/             # Uploaded video files (UUID-named)
│   ├── meetings.db         # SQLite database (auto-created)
│   └── tmp/                # Temporary WAV files (cleaned after processing)
├── tests/                  # Pytest test suite
├── docker-compose.yml      # CPU + GPU service profiles
├── Dockerfile              # Multi-stage: CPU (python:3.11-slim) + GPU (nvidia/cuda)
├── requirements.txt
└── .env.example
```

### Pipeline Flow

```
 POST /api/v1/upload  (multipart video file)
          │
          ▼
   Validate extension ──✗──→ 400 error
          │✓
          ▼
   Generate UUIDs, save video, create DB row (status: pending)
   Schedule background task
          │
          ▼
   ┌─────────────────────────────────────────────────────────┐
   │  _process_meeting_video()                               │
   │                                                         │
   │  1. Extract   ──ffmpeg──→  mono 16kHz WAV              │
   │  2. VAD       ──Silero VAD──→  speech regions           │
   │  3. Transcribe ──faster-whisper (batched GPU)──→ segments │
   │     ↳ Retry on CUDA/OOM errors (exponential backoff)    │
   │  4. Diarize   ──pyannote.audio──→  speaker turns        │
   │     ↳ Retry on CUDA/OOM errors                          │
   │  5. Merge     ──overlap matching──→  labeled segments   │
   │  6. Persist   ──SQLite──→  segments table               │
   │                                                         │
   │  On failure → status = failed, error saved              │
   │  Finally   → temp WAV deleted, GPU cache cleared        │
   └─────────────────────────────────────────────────────────┘
          │
          ▼
   GET /api/v1/meetings/{id}/segments  → speaker-labeled transcript
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | FastAPI (Python) |
| **ASGI Server** | Uvicorn |
| **Transcription** | faster-whisper (CTranslate2) — `tiny` through `large-v3` |
| **Speaker Diarization** | pyannote.audio (`speaker-diarization-3.1`) |
| **Voice Activity Detection** | Silero VAD (`torch.hub`) |
| **Audio Processing** | ffmpeg + ffprobe (system) |
| **Database** | SQLite via SQLAlchemy (auto-migrated, no Alembic) |
| **ML Framework** | PyTorch + torchaudio |
| **File Handling** | aiofiles (async streaming uploads) |
| **Containerization** | Docker (multi-stage: CPU + GPU) |
| **Orchestration** | Docker Compose with CPU/GPU profiles |
| **Testing** | pytest + httpx (TestClient) |
| **CI** | GitHub Actions |

---

## Project Structure

```
myna/
│
├── .github/workflows/
│   └── ci.yml                          # CI: lint, test, Docker build
│
├── app/                                # FastAPI Application
│   ├── main.py                         # App entry point, /health endpoint, lifespan
│   ├── config.py                       # Settings from env vars with sensible defaults
│   ├── database.py                     # SQLAlchemy engine, session, PRAGMA migrations
│   ├── models.py                       # Meeting & TranscriptSegment ORM models
│   ├── schemas.py                      # Pydantic request/response schemas
│   ├── transcription.py                # Audio extraction + faster-whisper with batched inference
│   ├── diarization.py                  # pyannote.audio speaker diarization + NoopDiarizer fallback
│   ├── merge.py                        # Overlap-based segment-to-speaker assignment
│   ├── vad.py                          # Silero VAD speech region detection
│   └── routers/
│       └── videos.py                   # All /api/v1/ route handlers + pipeline orchestrator
│
├── storage/                            # Runtime data (gitignored)
│   ├── videos/                         # Uploaded video files (UUID-named)
│   ├── meetings.db                     # SQLite database (auto-created on startup)
│   └── tmp/                            # Temporary WAV files (cleaned after processing)
│
├── tests/                              # Pytest test suite
│   ├── conftest.py                     # Fixtures: TestClient, mock transcriber/diarizer, temp dirs
│   ├── test_videos_router.py           # Route-level integration tests
│   ├── test_pipeline.py                # Pipeline logic unit tests
│   ├── test_diarization.py             # Diarization unit tests
│   ├── test_transcription.py           # Transcription unit tests
│   ├── test_vad.py                     # VAD unit tests
│   └── test_merge.py                   # Segment merging unit tests
│
├── docker-compose.yml                  # Services: api (CPU) + api-gpu (GPU)
├── Dockerfile                          # Multi-stage build: cpu + gpu targets
├── requirements.txt                    # Python dependencies (pinned)
├── .env.example                        # Environment template
├── .gitignore
├── PROJECT_STRUCTURE.md                # Detailed architecture & design decisions
├── AGENTS.md                           # AI agent instructions
└── README.md                           # This file
```

### Key Modules Explained

- **`app/main.py`**: FastAPI app factory with lifespan (creates dirs, inits DB), IST-configured logging, and `/health` endpoint.
- **`app/routers/videos.py`**: All 6 API endpoints plus `_process_meeting_video()` — the full pipeline orchestrator with GPU retry logic.
- **`app/transcription.py`**: Provider pattern (`get_transcriber()`) selecting `local` (faster-whisper with `BatchedInferencePipeline`) or `hosted` (stub). Also handles `ffmpeg` audio extraction.
- **`app/diarization.py`**: Provider pattern (`get_diarizer()`) selecting `LocalPyannoteDializer` or `NoopDiarizer` based on `HF_TOKEN` presence.
- **`app/vad.py`**: Silero VAD loaded via `torch.hub.load(trust_repo=True)` on first use. Returns speech regions + audio array to avoid double decode.
- **`app/database.py`**: Manual PRAGMA-based migrations (`_MEETINGS_MIGRATIONS` list). New columns added via `ALTER TABLE` on startup — no Alembic.
- **`app/config.py`**: `Settings` class loaded from environment via `python-dotenv`. Auto-detects CUDA for device selection.

---

## Prerequisites

| Tool | Required for | Version |
|---|---|---|
| **Python** | Local dev | ≥ 3.10 |
| **pip** | Package management | Latest |
| **ffmpeg + ffprobe** | Audio extraction & duration detection | On PATH |
| **Docker** | Containerized deployment | Latest |
| **Docker Compose** | Multi-service orchestration | v2+ |
| **HuggingFace Token** | Speaker diarization (optional) | Free at [hf.co/settings/tokens](https://huggingface.co/settings/tokens) |
| **NVIDIA GPU + CUDA** | GPU acceleration (optional) | CUDA 12.1+ |

> **HuggingFace Token Setup**: To enable speaker diarization, get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept the model terms for:
> - [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
> - [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)

---

## Getting Started

### Quick Start (2 Minutes)

#### 1. Clone the Repository

```bash
git clone https://github.com/sumansahoo1/myna.git
cd myna
```

#### 2. Configure Environment

```bash
cp .env.example .env
```

For basic transcription without speaker labels, no edits needed. For diarization, uncomment and set `HF_TOKEN`.

#### 3. Start with Docker

```bash
# CPU (default)
docker compose --profile cpu up

# GPU (NVIDIA)
docker compose --profile gpu up
```

#### 4. Try It

Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API docs.

Or from the terminal:

```bash
curl -F "video=@meeting.mp4" http://localhost:8000/api/v1/upload
```

---

## Local Development

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/sumansahoo1/myna.git
cd myna
cp .env.example .env
```

Open `.env` and configure:

| Variable | Where to get it |
|---|---|
| `HF_TOKEN` | [HuggingFace Settings](https://huggingface.co/settings/tokens) → New token (read) |
| `WHISPER_MODEL` | Model size — `tiny`, `base`, `small`, `medium`, `large-v3` (default: `small`) |
| `WHISPER_DEVICE` | `cpu` or `cuda` (auto-detected if not set) |

### 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** PyTorch in `requirements.txt` is CPU-only by default. For GPU, install CUDA-compatible torch manually:
> ```bash
> pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Run the Server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Verify

```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger UI.

### 5. Run Tests

```bash
pytest tests/ -v
```

CI runs the full suite on every push and PR via GitHub Actions.

---

## Docker Deployment

### CPU Deployment

```bash
docker compose --profile cpu up -d
```

- 2 CPU cores, 4 GB RAM limit
- Models cached in `myna-models` volume (no re-download on restart)
- Videos and database persisted in `myna-storage` volume

### GPU Deployment

```bash
BUILD_TARGET=gpu docker compose --profile gpu up -d
```

- 4 CPU cores, 8 GB RAM, 1 NVIDIA GPU
- CUDA 12.1 runtime with PyTorch cu121 wheels
- Longer start period (30s) for GPU warmup

### Volume Management

| Volume | Purpose | Notes |
|---|---|---|
| `myna-storage` | Videos, database | Persists across restarts |
| `myna-models` | Model cache (`/root/.cache`) | faster-whisper + pyannote models cached here |
| `tmpfs` | Temporary WAV files | Ephemeral, 2 GB (CPU) / 4 GB (GPU) |

### Healthcheck

The container self-monitors via `/health` endpoint every 30 seconds. Configured with 3 retries and a 15s start period.

---

## API Reference

**Base URL:** `http://localhost:8000`

All endpoints are served under `/api/v1`. Interactive docs available at `/docs`.

### Endpoints

| Method | Path | Description | Mode |
|---|---|---|---|
| `GET` | `/health` | Liveness check | — |
| `POST` | `/api/v1/upload` | Upload video, process in background | Async |
| `POST` | `/api/v1/upload-and-diarize` | Upload video, block until done, return segments | Sync |
| `GET` | `/api/v1/meetings/{meeting_id}` | Meeting status, timestamps, file info | — |
| `GET` | `/api/v1/meetings/{meeting_id}/transcript` | Full transcript as plain text with detected language | — |
| `GET` | `/api/v1/meetings/{meeting_id}/segments` | Every utterance as a timestamped segment with speaker label | — |

**Supported video formats:** `mp4`, `avi`, `mov`, `webm`, `mkv`, `wmv`, `flv`, `m4v`, `mpeg`, `mpg`, `3gp`

---

### `POST /api/v1/upload`

Upload a video file for background processing. Returns immediately with meeting and video IDs. Poll the status and transcript endpoints to retrieve results.

**Request:**
```http
POST /api/v1/upload
Content-Type: multipart/form-data
```

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "video=@meeting.mp4"
```

**Response** `200`:
```json
{
  "meeting_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "video_id": "f9e8d7c6-b5a4-3210-fedc-ba0987654321",
  "message": "Video stored successfully"
}
```

**Response** `400` (invalid format):
```json
{
  "detail": "Invalid video format. Allowed: .3gp, .avi, .flv, .m4v, .mkv, .mov, .mp4, .mpeg, .mpg, .webm, .wmv"
}
```

---

### `POST /api/v1/upload-and-diarize`

Upload a video and wait for the full pipeline (transcription + diarization). Returns complete results in a single response. Blocks until processing finishes.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/upload-and-diarize" \
  -F "video=@meeting.mp4"
```

**Response** `200`:
```json
{
  "meeting_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "transcription_status": "completed",
  "diarization_status": "completed",
  "language": "en",
  "transcript_text": "Good morning everyone. Let's go over the Q3 results...",
  "segments": [
    {
      "id": 1,
      "start_sec": 0.5,
      "end_sec": 4.2,
      "speaker_label": "SPEAKER_00",
      "text": " Good morning everyone."
    },
    {
      "id": 2,
      "start_sec": 4.8,
      "end_sec": 10.1,
      "speaker_label": "SPEAKER_00",
      "text": " Let's go over the Q3 results."
    },
    {
      "id": 3,
      "start_sec": 11.0,
      "end_sec": 15.5,
      "speaker_label": "SPEAKER_01",
      "text": " Revenue is up 12% year over year."
    }
  ],
  "error": null
}
```

**Response** `200` (partial — diarization unavailable):
```json
{
  "meeting_id": "a1b2c3d4-...",
  "transcription_status": "completed",
  "diarization_status": "failed",
  "language": "en",
  "transcript_text": "Good morning everyone...",
  "segments": [
    {
      "id": 1,
      "start_sec": 0.5,
      "end_sec": 4.2,
      "speaker_label": "UNKNOWN",
      "text": " Good morning everyone."
    }
  ],
  "error": "HF_TOKEN not set"
}
```

---

### `GET /api/v1/meetings/{meeting_id}`

Check meeting status and metadata.

**Response** `200`:
```json
{
  "meeting_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "video_id": "f9e8d7c6-b5a4-3210-fedc-ba0987654321",
  "filename": "f9e8d7c6-b5a4-3210-fedc-ba0987654321.mp4",
  "transcription_status": "completed",
  "diarization_status": "completed",
  "created_at": "2026-07-11T10:30:00+05:30"
}
```

**Transcription status values:** `pending` → `processing` → `completed` / `failed`

---

### `GET /api/v1/meetings/{meeting_id}/transcript`

Get the full transcript as a single text block with detected language.

**Response** `200`:
```json
{
  "meeting_id": "a1b2c3d4-...",
  "transcription_status": "completed",
  "language": "en",
  "transcript_text": "Good morning everyone. Let's go over the Q3 results. Revenue is up 12% year over year.",
  "error": null
}
```

**Response** `200` (still processing):
```json
{
  "meeting_id": "a1b2c3d4-...",
  "transcription_status": "processing",
  "language": null,
  "transcript_text": null,
  "error": null
}
```

---

### `GET /api/v1/meetings/{meeting_id}/segments`

Get timestamped, speaker-labeled segments.

**Response** `200`:
```json
{
  "meeting_id": "a1b2c3d4-...",
  "transcription_status": "completed",
  "diarization_status": "completed",
  "segments": [
    {
      "id": 1,
      "start_sec": 0.5,
      "end_sec": 4.2,
      "speaker_label": "SPEAKER_00",
      "text": " Good morning everyone."
    },
    {
      "id": 2,
      "start_sec": 4.8,
      "end_sec": 10.1,
      "speaker_label": "SPEAKER_00",
      "text": " Let's go over the Q3 results."
    },
    {
      "id": 3,
      "start_sec": 11.0,
      "end_sec": 15.5,
      "speaker_label": "SPEAKER_01",
      "text": " Revenue is up 12% year over year."
    }
  ]
}
```

---

## Environment Variables Reference

Copy `.env.example` to `.env`. All variables have sensible defaults.

### `app/.env` (loaded via python-dotenv)

```env
# ── HuggingFace ────────────────────────────────────────
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx

# ── Whisper (transcription) ────────────────────────────
WHISPER_MODEL=small
WHISPER_DEVICE=auto
TRANSCRIBER_PROVIDER=local

# ── Diarization (speaker identification) ──────────────
DIARIZER_PROVIDER=local
DIARIZATION_DEVICE=auto

# ── VAD ────────────────────────────────────────────────
ENABLE_VAD=true

# ── GPU Inference ─────────────────────────────────────
INFERENCE_BATCH_SIZE=8

# ── Storage (optional overrides) ──────────────────────
# STORAGE_DIR=/custom/path/to/storage
# DATABASE_URL=sqlite:///custom/path/db.sqlite
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `HF_TOKEN` | For diarization | — | HuggingFace token for pyannote.audio models |
| `WHISPER_MODEL` | No | `small` | Model size — `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | No | auto-detect | `cpu` or `cuda`. Auto-detects CUDA if available. |
| `TRANSCRIBER_PROVIDER` | No | `local` | `local` (faster-whisper) or `hosted` (stub) |
| `DIARIZER_PROVIDER` | No | `local` | `local` (pyannote) or `hosted` (stub) |
| `DIARIZATION_DEVICE` | No | auto-detect | `cpu` or `cuda` for pyannote pipeline |
| `ENABLE_VAD` | No | `true` | Toggle Silero VAD pre-filtering (`true`/`false`) |
| `INFERENCE_BATCH_SIZE` | No | `8` | GPU batch size for batched inference |
| `STORAGE_DIR` | No | `./storage` | Override video storage directory |
| `DATABASE_URL` | No | `sqlite:///./storage/meetings.db` | Override database URL |

The `.env` file is gitignored — never committed.

---

## Roadmap

### Phase 2 (Q3 2026)
- [ ] **WebSocket progress updates** — Real-time status push during processing
- [ ] **ARQ task queue** — Replace in-process BackgroundTasks with Redis-backed queue
- [ ] **PostgreSQL support** — Optional database backend for multi-instance deployments
- [ ] **ONNX runtime** — ONNX-exported models for lower latency inference

### Phase 3 (Q4 2026)
- [ ] **Multiple file upload** — Batch process entire meeting folders
- [ ] **Language override** — Force transcription to a specific language
- [ ] **Segment search** — Full-text search across all transcripts
- [ ] **Export formats** — SRT, VTT, JSON subtitle formats

### Completed
- [x] ~~**Background processing pipeline**~~ — Done
- [x] ~~**GPU-batched inference**~~ — Done (BatchedInferencePipeline)
- [x] ~~**VAD pre-filtering**~~ — Done (Silero VAD)
- [x] ~~**Docker CPU + GPU profiles**~~ — Done
- [x] ~~**GPU retry with backoff**~~ — Done
- [x] ~~**CI/CD pipeline**~~ — Done (GitHub Actions)

---

## Contributing

Contributions welcome! Here's how:

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make your changes** and test thoroughly (`pytest tests/ -v`)
4. **Commit with conventional messages**: `git commit -m 'feat: description'`
5. **Push to your fork**: `git push origin feature/your-feature`
6. **Open a Pull Request** with a description of your changes

### Contribution Guidelines

- Follow existing code style and conventions
- Add or update tests for new functionality
- Ensure `pytest tests/ -v` passes locally
- Update documentation if you change API behaviour
- Keep PRs focused — one feature or fix per PR

### Report Issues

Found a bug? Open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behaviour
- Python version, OS, GPU info (if applicable)
- Relevant log output

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

## Support & Contact

- 🐛 **Issues**: [GitHub Issues](https://github.com/sumansahoo1/myna/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/sumansahoo1/myna/discussions)
- 📧 **Email**: [suman@sumansahoo.com](mailto:suman@sumansahoo.com)

---

## Acknowledgments

- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — CTranslate2-optimized Whisper for fast, accurate transcription
- **[pyannote.audio](https://github.com/pyannote/pyannote-audio)** — State-of-the-art speaker diarization
- **[Silero VAD](https://github.com/snakers4/silero-vad)** — Lightweight voice activity detection
- **[FastAPI](https://fastapi.tiangolo.com/)** — Modern, fast web framework
- **[SQLAlchemy](https://www.sqlalchemy.org/)** — Python SQL toolkit and ORM
- **[ffmpeg](https://ffmpeg.org/)** — Swiss army knife for audio/video processing

---

**Made with ❤️ by [Suman Sahoo](https://github.com/sumansahoo1)**

⭐ If you find this project helpful, please consider giving it a star on GitHub!
