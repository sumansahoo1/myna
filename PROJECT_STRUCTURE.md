# Myna — Project Structure & Details

## Overview

**Myna** is a FastAPI-based video backend that receives video uploads, stores them locally, transcribes speech to text (with speaker diarization), and serves the results via REST APIs. It uses SQLite for persistence and runs transcription/diarization entirely locally using open-source models.

---

## Directory Layout

```
myna/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point, lifespan, router registration
│   ├── config.py            # Paths, settings, environment-based configuration
│   ├── database.py          # SQLAlchemy engine, session, table migrations
│   ├── models.py            # ORM models (Meeting, TranscriptSegment)
│   ├── schemas.py           # Pydantic request/response models
│   ├── transcription.py     # Audio transcription (faster-whisper)
│   ├── diarization.py       # Speaker diarization (pyannote.audio)
│   ├── merge.py             # Merge transcription segments with speaker turns
│   └── routers/
│       ├── __init__.py
│       └── videos.py        # API route handlers
├── storage/
│   ├── meetings.db          # SQLite database (auto-created)
│   ├── videos/              # Uploaded video files (UUID-named)
│   └── tmp/                 # Temporary WAV files (cleaned after processing)
├── .gitignore
├── PROJECT_STRUCTURE.md
├── README.md
└── requirements.txt
```

---

## Setup & Running

**Prerequisites:** Python 3.10+, `ffmpeg` installed on the machine.

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Processing Pipeline (Data Flow)

```
POST /api/v1/upload  (multipart video file)
         │
         ▼
  Validate extension ──✗──→ 400 error
         │✓
         ▼
  Generate UUIDs, save file to storage/videos/{video_id}.ext
  Create Meeting DB row (status: pending)
  Schedule background task
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  _process_meeting_video()  (background)             │
  │                                                     │
  │  1. Extract audio  ──ffmpeg──→ mono 16kHz WAV      │
  │  2. Transcribe     ──faster-whisper──→ segments    │
  │  3. Diarize        ──pyannote.audio──→ speaker turns│
  │  4. Merge          ──overlap matching──→ labeled    │
  │  5. Persist segments & update status                │
  │                                                     │
  │  On failure → status = failed, error saved          │
  │  Finally    → temp WAV deleted                      │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  GET /api/v1/meetings/{id}/segments  → speaker-labeled transcript
```

### Stage Details

| Stage | File | Dependencies |
|---|---|---|
| Audio Extraction | `app/transcription.py:88-114` | `ffmpeg` (system) |
| Transcription | `app/transcription.py:51-85` | `faster-whisper` model (cached on first run) |
| Diarization | `app/diarization.py:54-101` | `pyannote.audio` + HuggingFace token (optional) |
| Merge | `app/merge.py:7-29` | Output of transcription + diarization |

The merge step matches each transcript segment to the speaker whose diarization turn has the greatest time overlap. Segments with no overlap fall back to `UNKNOWN`.

---

## Data Models

### Meetings Table (`app/models.py:8-23`)

| Column | Type | Description |
|---|---|---|
| meeting_id | String(36) PK | UUID |
| video_id | String(36) UNIQUE | UUID of stored video file |
| filename | String(255) | Stored filename (`{video_id}.{ext}`) |
| transcription_status | String(20) | `pending` / `processing` / `completed` / `failed` |
| transcript_text | Text | Full concatenated transcript text |
| transcript_language | String(16) | Detected language code (e.g. `en`) |
| transcript_error | Text | Error message if transcription failed |
| diarization_status | String(20) | `pending` / `processing` / `completed` / `failed` |
| diarization_error | Text | Error message if diarization failed |
| created_at | DateTime(timezone) | Auto-set on row creation |

### TranscriptSegments Table (`app/models.py:25-35`)

| Column | Type | Description |
|---|---|---|
| id | Integer PK | Auto-increment |
| meeting_id | String(36) FK | References `meetings.meeting_id` |
| start_sec | Float | Segment start time in seconds |
| end_sec | Float | Segment end time in seconds |
| speaker_label | String(32) | e.g. `SPEAKER_00`, `SPEAKER_01`, or `UNKNOWN` |
| text | Text | Transcribed text for this segment |

Relationship: `Meeting` has a one-to-many relationship with `TranscriptSegment`, ordered by `start_sec`.

---

## API Endpoints

| Method | Path | Returns | Statuses |
|---|---|---|---|
| POST | `/api/v1/upload` | `{meeting_id, video_id, message}` | Background processing starts |
| GET | `/api/v1/meetings/{meeting_id}` | `{meeting_id, video_id, filename, statuses, created_at}` | — |
| GET | `/api/v1/meetings/{meeting_id}/transcript` | `{meeting_id, status, language, transcript_text, error}` | pending → processing → completed/failed |
| GET | `/api/v1/meetings/{meeting_id}/segments` | `{meeting_id, statuses, segments[]}` | Full segment list with speaker labels |

### Upload Example (curl)

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@/path/to/video.mp4"
```

### Supported Video Formats

`mp4`, `avi`, `mov`, `webm`, `mkv`, `wmv`, `flv`, `m4v`, `mpeg`, `mpg`, `3gp`

---

## Configuration (`app/config.py`)

### Settings Class

All settings have sensible defaults and can be overridden via environment variables:

| Setting | Env Var | Default | Values |
|---|---|---|---|
| `transcriber_provider` | `TRANSCRIBER_PROVIDER` | `local` | `local` \| `hosted` |
| `whisper_model` | `WHISPER_MODEL` | `small` | `tiny` \| `base` \| `small` \| `medium` \| `large-v3` |
| `whisper_device` | `WHISPER_DEVICE` | `cpu` | `cpu` \| `cuda` |
| `diarizer_provider` | `DIARIZER_PROVIDER` | `local` | `local` \| `hosted` |
| `hf_token` | `HF_TOKEN` | `None` | HuggingFace token for pyannote |

### Provider System

`get_transcriber()` and `get_diarizer()` are factory functions (`app/transcription.py:31-40`, `app/diarization.py:22-32`) that select the implementation:

- **`local`** — uses open-source models running on the machine (faster-whisper / pyannote).
- **`hosted`** — returns a stub that raises `RuntimeError`. Placeholder for future cloud-based services.

### Diarization Behavior

- **With `HF_TOKEN` set:** `LocalPyannoteDializer` runs full speaker diarization using `pyannote/speaker-diarization-3.1`.
- **Without `HF_TOKEN`:** `NoopDiarizer` is used, returning empty speaker turns. All segments are labeled `UNKNOWN`.
- Token must have accepted the model conditions at [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0).

---

## Dependencies (`requirements.txt`)

| Package | Purpose |
|---|---|
| `fastapi[standard]` | Web framework |
| `uvicorn[standard]` | ASGI server |
| `sqlalchemy` | ORM / database abstraction |
| `python-multipart` | File upload parsing |
| `aiofiles` | Async file I/O (video upload streaming) |
| `faster-whisper` | Local transcription (CTranslate2-optimized Whisper) |
| `pyannote.audio` | Local speaker diarization |
| `requests` | HTTP client utility |

**System requirement:** `ffmpeg` must be installed and available on `PATH` for audio extraction.

---

## .gitignore Highlights

```
storage/         # Excludes meetings.db, uploaded videos, temp WAV files
*.db             # Catch-all for database files
.env             # Environment secrets (HF_TOKEN, etc.)
__pycache__/     # Python bytecode
*.py[cod]
.venv/, venv/    # Virtual environments
*.egg-info/, dist/, build/  # Build artifacts
```

---

## Design Decisions

- **SQLite + manual migrations** (`app/database.py:28-37`): New columns are added via `ALTER TABLE` using PRAGMA introspection at startup. No Alembic dependency. Suitable for MVP/single-server deployments.
- **BackgroundTasks** (`app/routers/videos.py:77`): Transcription/diarization runs in FastAPI's in-process `BackgroundTasks`. No external task queue. For production, this should be migrated to Celery/ARQ.
- **Temporary audio files** (`app/transcription.py:88-114`): Extracted WAV files are written to `storage/tmp/` and always cleaned up in the `finally` block.
- **No authentication**: The API has no auth layer. Assumes deployment behind a trusted reverse proxy or VPN.
- **Diarization is optional**: Functions fully without `HF_TOKEN` — all speakers default to `UNKNOWN`.
- **Error isolation** (`app/routers/videos.py:185-196`): If the pipeline fails mid-way, only the stages that were in `processing` state are marked `failed`. A preceding completed stage retains its success status.
- **Provider stubs**: The `hosted` provider implementations serve as API contracts for future cloud-based transcription/diarization services.
