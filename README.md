# Myna Video Backend

FastAPI backend that receives video uploads, stores them locally, and returns meeting IDs. Uses a SQLite database to store meeting_id ↔ video_id mappings.

## Setup

```bash
cd myna
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Transcription (local, free)

This MVP transcribes videos **locally** using `faster-whisper` and requires **`ffmpeg`** installed on the machine.

Provider switching is controlled by env vars:

- `TRANSCRIBER_PROVIDER=local` (default)
- `WHISPER_MODEL=small` (default; try `base`/`small` for CPU MVP)
- `WHISPER_DEVICE=cpu` (default; set `cuda` if available)

## Endpoints

### `POST /api/v1/upload`

Upload a video file. Returns `meeting_id` and `video_id`. Transcription runs in background.

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@/path/to/video.mp4"
```

**Response:**
```json
{
  "meeting_id": "550e8400-e29b-41d4-a716-446655440000",
  "video_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "message": "Video stored successfully"
}
```

### `GET /api/v1/meetings/{meeting_id}`

Retrieve meeting info (meeting_id, video_id, filename) by meeting ID.

### `GET /api/v1/meetings/{meeting_id}/transcript`

Retrieve transcription status and transcript (when ready).

## Supported Video Formats

mp4, avi, mov, webm, mkv, wmv, flv, m4v, mpeg, mpg, 3gp

## Storage

- Videos: `storage/videos/<video_id>.<ext>`
- Database: `storage/meetings.db` (SQLite)
