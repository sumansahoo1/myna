# Myna

Turn meeting videos into speaker-labeled transcripts — entirely on your machine. No cloud. No API keys. No per-minute pricing.

Drop in a video. Get back who said what, when.

**Why Myna:**

- **Fully local** — transcription and speaker identification run on your hardware. Nothing leaves the machine.
- **Speaker-aware** — not just a wall of text. Every segment tagged with who spoke.
- **Two modes** — fire-and-forget background jobs, or synchronous calls that return results inline.
- **Zero-dependency deploy** — ships as a single Docker container with `ffmpeg` and all models bundled.

## Quick start

```bash
cp .env.example .env
docker compose up
```

That's it. API running at `http://localhost:8000`.

For local development without Docker:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Requires Python 3.10+ and `ffmpeg` on your machine.

## Try it

Interactive API docs — explore every endpoint, make requests, see responses:

→ [http://localhost:8000/docs](http://localhost:8000/docs)

Or from the terminal:

```bash
curl -F "video=@meeting.mp4" http://localhost:8000/api/v1/upload
```

## API

Six endpoints. Two workflows.

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/api/v1/upload` | Upload video, process in background. Poll transcript and segments when ready. |
| `POST` | `/api/v1/upload-and-diarize` | Upload video, block until transcription and diarization complete. Returns everything in one response. |
| `GET` | `/api/v1/meetings/{id}` | Meeting status, timestamps, file info |
| `GET` | `/api/v1/meetings/{id}/transcript` | Full transcript as plain text, with detected language |
| `GET` | `/api/v1/meetings/{id}/segments` | Every utterance as a timestamped segment with speaker label |

**Supported formats:** mp4, avi, mov, webm, mkv, wmv, flv, m4v, mpeg, mpg, 3gp.

## How it works

```
Upload → ffmpeg extracts audio → VAD detects speech regions → faster-whisper transcribes → pyannote identifies speakers → segments merged
```

1. **Extract** — `ffmpeg` converts the video to mono 16kHz WAV.
2. **VAD** — Silero VAD detects speech regions, skipping silence/noise. Gated by `ENABLE_VAD` (default on).
3. **Transcribe** — `faster-whisper` produces timestamped text segments. Model size is configurable (`tiny` through `large-v3`).
4. **Diarize & merge** — `pyannote.audio` figures out who spoke when, then each transcript segment gets labeled with the speaker that had the most time overlap. Without a HuggingFace token, all speakers default to `UNKNOWN`.

**`/upload`** returns immediately and processes in the background — poll for results.

**`/upload-and-diarize`** holds the connection until processing finishes and returns segments inline. Good for low-latency use cases or when you want the result in a single request.

## Configuration

Copy `.env.example` to `.env`. Everything has sensible defaults:

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL` | `small` | Model size — `tiny`/`base` for speed, `large-v3` for accuracy |
| `WHISPER_DEVICE` | auto | Auto-detects CUDA. Set explicitly to `cpu` or `cuda` to override. |
| `HF_TOKEN` | — | HuggingFace token to unlock speaker diarization. Get one free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept the model terms for [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) |
| `ENABLE_VAD` | `true` | Toggle Silero VAD pre-filtering — skips silence/noise before transcription |
| `INFERENCE_BATCH_SIZE` | `8` | GPU batch size for faster-whisper batched inference |
| `DIARIZATION_DEVICE` | auto | `cpu` or `cuda` for pyannote pipeline. Auto-detects like `WHISPER_DEVICE`. |

## Storage

```
storage/videos/<video_id>.<ext>   # uploaded videos
storage/meetings.db               # SQLite (meetings, transcripts, segments)
storage/tmp/                      # temporary WAV files, auto-cleaned after processing
```

## Tests

```bash
pytest tests/ -v
```

CI runs the full suite on every push and PR via GitHub Actions.

---

Deeper dive into architecture, data models, and design decisions → [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).
