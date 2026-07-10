# Myna — Agent Instructions

## Commands

Activate the venv before running Python commands (`.env` is auto-loaded by `python-dotenv` in `app/config.py`):

```bash
source .venv/bin/activate

# Dev server (requires python 3.10+, ffmpeg on PATH)
# Use `python -m uvicorn` not bare `uvicorn` — venv binary has wrong shebang
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Docker (CPU target by default; BUILD_TARGET=gpu for CUDA)
docker compose up           # CPU
docker compose --profile gpu up   # GPU

# Run all tests
pytest tests/ -v

# Run single test file
pytest tests/test_videos_router.py -v
```

## Architecture

Single FastAPI app (`app.main:app`). `app/routers/videos.py` holds all endpoints under `/api/v1`. No auth layer (assumes trusted reverse proxy/VPN).

Processing pipeline: **VAD → batched transcribe + diarize (parallel) → merge → persist**.
- `ENABLE_VAD` env var toggles VAD pre-filtering (default `true`).
- `ENABLE_CHUNKING` toggles BatchedInferencePipeline (GPU-batched) vs single-file transcription (default `true`).
- Transcription (BatchedInferencePipeline or single-file) + diarization run in `ThreadPoolExecutor(max_workers=2)` — wall-clock = max of the two.
- Pipeline runs via FastAPI `BackgroundTasks` (same process, no external queue).

Provider pattern: `get_transcriber()` / `get_diarizer()` select `local` or `hosted` based on env vars. `hosted` provider stubs raise `RuntimeError` — not implemented.

## Configuration

`.env` loaded via `python-dotenv` at `app/config.py` import time. `Settings` is a plain class (not pydantic-settings). See `.env.example`.

- `HF_TOKEN` — without it, diarization uses `NoopDiarizer` (all speakers = `UNKNOWN`).
- `WHISPER_DEVICE` auto-detects CUDA via `torch.cuda.is_available()` when not set.
- `CHUNK_DURATION_SEC`, `CHUNK_OVERLAP_SEC`, `CHUNK_BATCH_SIZE` tune chunking behavior.

## Database

SQLite at `storage/meetings.db` (auto-created on startup). **No Alembic** — manual PRAGMA-based migrations in `app/database.py:_MEETINGS_MIGRATIONS`. To add a column: append tuple `(col_name, col_def)` to that list.

`check_same_thread=False` required because FastAPI async handlers may use sessions across threads.

## Gotchas

- **ffmpeg AND ffprobe must be on PATH.** `chunking.py` uses ffprobe for duration detection.
- **Dockerfile patches pyannote.audio source files** (`use_auth_token` → `token`). If pyannote version changes, verify the patch still works.
- **`storage/` is gitignored.** DB, uploaded videos, temp WAVs never committed.
- **Logging timestamps configured to IST** in `app/main.py:_ist_converter`.
- **VAD loads via `torch.hub.load(trust_repo=True)`** on first use, not at startup. May download model files.
- **`ARCHITECTURE_REPORT.md`** describes planned optimizations (ARQ queue, PostgreSQL, ONNX, WebSocket progress) — **none of these are implemented.** Do not assume they exist.
- **`PROJECT_STRUCTURE.md`** is the canonical detailed reference for data models, API, and design decisions.

## Tests

Test database: in-memory SQLite with `StaticPool` (see `tests/conftest.py`). Each test gets fresh tables via `_setup_test_db` autouse fixture.

Key fixtures in `conftest.py`:
- `client` — FastAPI `TestClient` with DB override and `_process_meeting_video` **mocked to no-op**. Route-level tests do NOT run the real pipeline.
- `mock_transcriber` / `mock_diarizer` — for unit-testing pipeline logic.
- `tmp_video_dir` — redirects `settings.video_storage_dir` to a temp path.

CI runs `pytest tests/ -v --tb=short` on every push/PR. Must install `ffmpeg` before running tests.

## GitHub — Permission Required

**NEVER perform irreversible GitHub actions without explicit permission.** This includes:
- `git push` (local or via MCP)
- Creating/updating/merging/closing pull requests
- Creating/commenting/reacting on issues or PRs
- Pushing commits, branches, or tags to the remote
- Deleting branches, files, or releases on GitHub
- Any GitHub MCP write tool (`github_issue_write`, `github_create_pull_request`, `github_push_files`, `github_add_issue_comment`, etc.)

**NEVER commit changes unless the user explicitly asks you to.** This includes `git add`, `git commit`, `git stash`, or any other mutating git operation — even local ones. Wait for an explicit instruction like "commit this" before touching git at all.

Ask before anything that touches the remote.
