# Myna Architecture Report: From Minutes to Seconds

**Goal**: Process long videos (1-4 hours) from transcription to speaker-labeled diarization in seconds, not minutes/hours.

**Current baseline** (Whisper `small` on CPU, single-threaded):
| Video length | Approx. processing time |
|---|---|
| 10 min | 20-30 min |
| 1 hour | 2-3 hours |
| 2 hours | 4-6 hours |

**Target** (optimized architecture on GPU):
| Video length | Target processing time |
|---|---|
| 10 min | 5-10 seconds |
| 1 hour | 30-60 seconds |
| 2 hours | 60-120 seconds |

---

## Root Cause Analysis

### Bottleneck 1: Sequential Pipeline (`app/routers/videos.py:269-292`)
Transcription and diarization run one after the other on the same WAV. Both are independent — they share no intermediate state. Current wall-clock time = `T_transcribe + T_diarize`. Can be `max(T_transcribe, T_diarize)`.

### Bottleneck 2: No Chunked/Parallel Processing
Entire audio file is fed as a single unit. For a 2-hour video (7200s of audio), Whisper processes all 7200s sequentially. Faster-whisper supports batched inference internally but not across file chunks. Chunking the audio into 30s segments and processing N chunks in parallel yields `T_total = T_chunk * (total_chunks / parallel_workers)`.

### Bottleneck 3: Single-Process BackgroundTasks (`app/routers/videos.py:91`)
FastAPI `BackgroundTasks` run in the same process as the web server. Python GIL means only one CPU-bound task executes at a time. Multiple concurrent uploads queue behind each other. No horizontal scaling possible.

### Bottleneck 4: CPU-Only (Default)
`WHISPER_DEVICE=cpu` is the hardcoded default in Dockerfile and config. No GPU passthrough in docker-compose. Whisper `large-v3` on CPU is ~0.05x realtime; on GPU (A100) it's ~50x realtime. A 1000x difference.

### Bottleneck 5: O(n*m) Merge (`app/merge.py:20-41`)
For n transcript segments and m speaker turns, nested loop = O(n*m). For a 2-hour meeting with 4000 segments and 500 speaker turns: 2M overlap calculations. Two-pointer sweep reduces this to O(n+m).

### Bottleneck 6: No VAD Pre-Filtering
Silence, music, noise all get processed by Whisper and pyannote. In a typical meeting, 30-50% of audio is silence. VAD pre-filtering skips these regions, cutting input size proportionally.

### Bottleneck 7: Blocking ffmpeg (`app/transcription.py:130`)
`subprocess.run()` blocks the thread. While not a GIL issue (releases GIL during subprocess), it prevents async pipeline integration.

### Bottleneck 8: SQLite Write Contention
SQLite allows only one writer at a time. Multiple concurrent processing tasks will serialize on DB writes. WAL mode + aiosqlite helps; PostgreSQL is better.

### Bottleneck 9: Cold-Start Model Loading (`app/transcription.py:67-78`, `app/diarization.py:84-103`)
Models load lazily on first request. Whisper `small` takes ~5s to load; pyannote diarization takes ~10s. For single requests this is amortized, but for serverless/cold-start scenarios it's a killer.

### Bottleneck 10: No Progress/Streaming API
Users poll `GET /meetings/{id}` to check status. No WebSocket or SSE for real-time progress. No partial results streaming.

---

## Architectural Changes Required

### Phase 1: Critical — Immediate 10-50x Speedup

#### 1.1 GPU Acceleration
**Files**: `docker-compose.yml`, `Dockerfile`, `app/config.py`

```yaml
# docker-compose.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

```dockerfile
# Dockerfile — use CUDA base image
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04
# Install Python 3.11 + system deps + torch with CUDA
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```python
# app/config.py — change default
whisper_device: str = "cuda"  # was "cpu"
```

**Impact**: 50-500x faster transcription (model-dependent). Diarization also GPU-accelerated.

#### 1.2 Parallel Transcription + Diarization
**File**: `app/routers/videos.py`

Replace sequential calls:
```python
# CURRENT (lines 269-292)
transcriber = get_transcriber()
transcript_result = transcriber.transcribe(audio_path)
# ...
diarizer = get_diarizer()
speaker_turns = diarizer.diarize(audio_path)
```

With concurrent execution:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=2) as executor:
    fut_transcribe = executor.submit(_run_transcription, audio_path)
    fut_diarize = executor.submit(_run_diarization, audio_path)
    
    for future in as_completed([fut_transcribe, fut_diarize]):
        result = future.result()
        # handle whichever finishes first
```

**Impact**: Wall-clock time = `max(T_transcribe, T_diarize)`, ~2x speedup when both are similar duration.

#### 1.3 Chunked Audio Processing
**New file**: `app/chunking.py`

```python
@dataclass
class AudioChunk:
    start_sec: float
    end_sec: float
    wav_path: Path

def split_audio(audio_path: Path, chunk_duration: float = 30.0, overlap: float = 5.0) -> list[AudioChunk]:
    """Split WAV into overlapping chunks using ffmpeg."""
    ...

def stitch_transcripts(chunks: list[AudioChunk], results: list[TranscriptResult]) -> TranscriptResult:
    """Merge overlapping transcript chunks, deduplicating overlap regions."""
    ...
```

Pipeline changes:
1. Split audio into 30s chunks with 5s overlap
2. Transcribe all chunks in parallel (batch inference via faster-whisper)
3. Stitch results, resolving overlap
4. Diarize on full audio (diarization needs global context)

**Impact**: For a 1-hour video split into 30s chunks (120 chunks), with batch_size=16 on GPU: `T = 120/16 * T_chunk ≈ 7.5 * 2s = 15s` for transcription. Without chunking: ~60s. **4x speedup**.

#### 1.4 VAD Pre-Filtering
**New file**: `app/vad.py`

```python
# Use Silero VAD (ONNX runtime, sub-millisecond per frame)
import torch

def filter_speech(audio_path: Path, sample_rate: int = 16000) -> list[tuple[float, float]]:
    """
    Returns list of (start, end) speech segments.
    Non-speech regions are skipped.
    """
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    # ... process audio, return speech regions
```

Integration: feed only speech regions to Whisper and pyannote.

**Impact**: 30-50% less audio to process → 1.4-2x speedup for typical meetings.

### Phase 2: High Impact — Another 3-5x Speedup

#### 2.1 Task Queue Architecture
**Files**: Replace `BackgroundTasks` with ARQ (Redis-based)

```
┌─────────┐     ┌──────────┐     ┌──────────────┐
│ FastAPI │────▶│  Redis   │────▶│ ARQ Worker 1 │──▶ GPU 1
│ (API)   │     │ (Queue)  │     │ ARQ Worker 2 │──▶ GPU 2
└─────────┘     └──────────┘     │ ARQ Worker N │──▶ GPU N
                                 └──────────────┘
```

```python
# app/worker.py — ARQ worker
from arq import create_pool
from arq.worker import func

@func
async def process_meeting(ctx, meeting_id: str):
    """Called by ARQ worker, runs full pipeline with GPU."""
    ...
```

```python
# app/routers/videos.py — enqueue instead of BackgroundTasks
from arq import ArqRedis

@router.post("/upload")
async def upload_video(..., arq: ArqRedis = Depends(get_arq)):
    meeting = Meeting(...)
    db.commit()
    await arq.enqueue_job('process_meeting', meeting_id)
    return VideoUploadResponse(...)
```

**Benefits**:
- Horizontal scaling: N workers = N concurrent pipeline executions
- Retry logic built-in
- Progress tracking via Redis
- Priority queues (short videos first)
- Graceful shutdown (workers finish current job)

**New dependencies**: `arq`, `redis[hiredis]`

#### 2.2 O(n+m) Merge Algorithm
**File**: `app/merge.py`

```python
def assign_speakers(
    transcript_segments: list[TranscriptSegment],
    speaker_turns: list[SpeakerTurn],
) -> list[dict]:
    """Two-pointer sweep. Both inputs must be sorted by start_sec."""
    result = []
    turn_idx = 0
    n_turns = len(speaker_turns)
    
    for seg in transcript_segments:
        # Advance turn_idx to the first turn that could overlap
        while turn_idx < n_turns and speaker_turns[turn_idx].end_sec < seg.start_sec:
            turn_idx += 1
        
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        j = turn_idx
        
        # Check all turns that overlap with this segment
        while j < n_turns and speaker_turns[j].start_sec <= seg.end_sec:
            overlap = _overlap(seg.start_sec, seg.end_sec,
                             speaker_turns[j].start_sec, speaker_turns[j].end_sec)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_turns[j].speaker_label
            j += 1
        
        result.append({
            "start_sec": seg.start_sec,
            "end_sec": seg.end_sec,
            "text": seg.text,
            "speaker_label": best_speaker,
        })
    
    return result
```

**Impact**: For 4000 segments × 500 turns: from 2M operations to ~4000 operations. **500x faster merge**, significant for long videos.

#### 2.3 Model Preloading at Startup
**File**: `app/main.py`

```python
from contextlib import asynccontextmanager

_transcriber: Transcriber | None = None
_diarizer: Diarizer | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _transcriber, _diarizer
    loop = asyncio.get_running_loop()
    _transcriber = await loop.run_in_executor(None, get_transcriber)
    _diarizer = await loop.run_in_executor(None, get_diarizer)
    logger.info("models preloaded")
    yield
    # cleanup if needed

app = FastAPI(lifespan=lifespan)
```

```python
# app/routers/videos.py — use preloaded models
def _run_transcription(audio_path: Path) -> TranscriptResult:
    from app.main import _transcriber
    return _transcriber.transcribe(audio_path)  # type: ignore[union-attr]
```

**Impact**: Eliminates 5-15s cold-start penalty on first request.

#### 2.4 Async ffmpeg
**File**: `app/transcription.py`

```python
import asyncio

async def _extract_audio_to_wav_async(video_path: Path) -> Path:
    """Non-blocking ffmpeg audio extraction."""
    cmd = [...]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode().strip()}")
    return out_path
```

**Impact**: Event loop stays responsive during extraction. Enables WebSocket progress updates.

### Phase 3: Advanced — Production-Grade

#### 3.1 WhisperX-Style Forced Alignment
**New file**: `app/alignment.py`

Use wav2vec2 forced alignment for word-level timestamps:
```python
from whisperx.alignment import align

def align_words(audio_path: Path, transcript_segments: list[dict]) -> list[dict]:
    """Post-process segments with phoneme-level alignment for precise word timestamps."""
    model_a, metadata = load_align_model(language_code="en", device="cuda")
    result = align(transcript_segments, model_a, metadata, audio_path, device="cuda")
    return result
```

**New dependency**: `whisperx` (or just its alignment module)

**Benefit**: Word-level timestamps for precise speaker attribution. Essential for subtitle generation.

#### 3.2 Streaming Results via WebSocket
**New file**: `app/routers/ws.py`

```python
@router.websocket("/ws/meetings/{meeting_id}")
async def meeting_progress(websocket: WebSocket, meeting_id: str):
    await websocket.accept()
    # Subscribe to Redis pub/sub for this meeting
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"meeting:{meeting_id}:progress")
    
    async for message in pubsub.listen():
        if message["type"] == "message":
            await websocket.send_text(message["data"])
```

Pipeline publishes progress events:
```python
await redis.publish(f"meeting:{meeting_id}:progress",
    json.dumps({"stage": "transcribing", "progress": 0.45, "eta_seconds": 12}))
```

**Benefit**: Real-time UI updates. Users see progress, not polls.

#### 3.3 Speaker Embedding Cache
**New file**: `app/speakers.py`

```python
import numpy as np
import pickle
from pathlib import Path

class SpeakerCache:
    """Cache speaker embeddings to avoid re-computing for known speakers."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def get(self, audio_fingerprint: str) -> str | None:
        """Return cached speaker label if embedding matches."""
        ...
    
    def put(self, audio_fingerprint: str, speaker_label: str):
        """Cache speaker embedding for future sessions."""
        ...
```

**Benefit**: For recurring meetings (daily standups), speaker identification becomes near-instant.

#### 3.4 ONNX Export for Pyannote
Export pyannote segmentation model to ONNX for optimized inference:

```python
# Build-time: export to ONNX
import torch
from pyannote.audio import Model

model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=HF_TOKEN)
torch.onnx.export(model, dummy_input, "segmentation.onnx", opset_version=17)

# Runtime: load ONNX with onnxruntime-gpu
import onnxruntime as ort
session = ort.InferenceSession("segmentation.onnx", providers=['CUDAExecutionProvider'])
```

**Benefit**: 2-3x faster diarization inference via ONNX Runtime with TensorRT.

#### 3.5 Distributed Multi-GPU Processing
**File**: `docker-compose.yml` (worker scaling)

```yaml
services:
  api:
    # ... FastAPI unchanged
  worker-gpu0:
    build: .
    command: arq app.worker.WorkerSettings
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
  worker-gpu1:
    build: .
    command: arq app.worker.WorkerSettings
    environment:
      - CUDA_VISIBLE_DEVICES=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
```

**Benefit**: N GPUs = N concurrent video processing jobs.

#### 3.6 PostgreSQL Migration
**File**: `app/database.py`

```python
# Replace SQLite with PostgreSQL for concurrent access
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/myna")

engine = create_async_engine(DATABASE_URL, pool_size=20, max_overflow=10)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
```

**Benefit**: Concurrent writes from multiple workers. Better query performance on large datasets.

#### 3.7 Faster Whisper Model Selection
**File**: `app/config.py`

```python
# Use optimized model variants
whisper_model: str = "large-v3-turbo"  # Faster than large-v3, similar accuracy
# Alternative: "distil-large-v3" for even faster with slight accuracy trade-off
```

**Benefit**: `large-v3-turbo` is 2x faster than `large-v3` with <1% WER difference.

---

## Target Architecture Diagram

```
                        ┌──────────────────────┐
  Video Upload ────────▶│   FastAPI (async)     │
                        │   - Upload handler     │
                        │   - WebSocket push     │
                        │   - REST endpoints     │
                        └──────┬───────────────┘
                               │ enqueue job
                        ┌──────▼───────────────┐
                        │       Redis            │
                        │  - Job queue (ARQ)     │
                        │  - Progress pub/sub    │
                        │  - Speaker cache       │
                        └──────┬───────────────┘
                               │ dequeue
                 ┌─────────────┼─────────────┐
          ┌──────▼──────┐ ┌───▼────────┐ ┌──▼──────────┐
          │ ARQ Worker 1 │ │ ARQ Wkr 2  │ │ ARQ Wkr N   │
          │   GPU 0      │ │  GPU 1     │ │  GPU N      │
          └──────────────┘ └────────────┘ └─────────────┘
                 │               │               │
          ┌──────▼───────────────▼───────────────▼──────┐
          │              PostgreSQL                      │
          │  - meetings, segments, speakers              │
          └─────────────────────────────────────────────┘

Worker Pipeline (per job):
  1. Extract audio (async ffmpeg)
  2. VAD → speech regions
  3. ┌─ Chunk audio (30s windows) ─┐
     │  Parallel batch transcribe  │  (GPU)
     │  Stitch chunks              │
     └─────────────────────────────┘
  4. ┌─ Diarize full audio ────────┐  (GPU, parallel with 3)
     │  Extract speaker turns      │
     └─────────────────────────────┘
  5. Two-pointer merge (O(n+m))
  6. Persist to PostgreSQL
  7. Publish completion via Redis pub/sub
```

---

## File Change Summary

| File | Change | Phase |
|---|---|---|
| `app/config.py` | GPU default, new model name, Redis/ARQ config | 1, 2 |
| `docker-compose.yml` | GPU device reservations, worker services, Redis, PostgreSQL | 1, 2, 3 |
| `Dockerfile` | CUDA base image, ONNX runtime, Redis client | 1, 3 |
| `app/routers/videos.py` | Parallel pipeline, ARQ enqueue, async ffmpeg | 1, 2 |
| `app/routers/ws.py` | **New** — WebSocket progress endpoint | 3 |
| `app/chunking.py` | **New** — Audio chunking + stitch | 1 |
| `app/vad.py` | **New** — Silero VAD integration | 1 |
| `app/merge.py` | Two-pointer algorithm | 2 |
| `app/transcription.py` | Async ffmpeg, batched chunk transcribe | 1, 2 |
| `app/diarization.py` | ONNX model support | 3 |
| `app/database.py` | PostgreSQL + async session | 3 |
| `app/alignment.py` | **New** — wav2vec2 forced alignment | 3 |
| `app/speakers.py` | **New** — Speaker embedding cache | 3 |
| `app/worker.py` | **New** — ARQ worker entry point | 2 |
| `app/main.py` | Lifespan model preloading, ARQ/Redis init | 2 |
| `requirements.txt` | Add: `arq`, `redis[hiredis]`, `onnxruntime-gpu`, `whisperx`, `aiosqlite`/`asyncpg`, `pydub` | 1, 2, 3 |

---

## Effort Estimate

| Phase | Effort | Speedup (cumulative) |
|---|---|---|
| Phase 1 (GPU + Parallel + Chunking + VAD) | 2-3 days | 50-100x |
| Phase 2 (ARQ + Merge + Preload + Async) | 2-3 days | 100-200x |
| Phase 3 (Alignment + WS + ONNX + Postgres + Multi-GPU) | 5-7 days | 200-500x |

**Total**: ~2 weeks for full implementation.

---

## Risk / Trade-offs

1. **GPU cost**: Requires NVIDIA GPU (A10/A100/T4). Cloud GPU instances are $0.50-2.00/hr.
2. **Chunking accuracy**: Splitting audio at word boundaries can degrade accuracy. Overlap + stitch resolves most cases.
3. **VAD false negatives**: May skip quiet speech. Tune VAD threshold per use case.
4. **ARQ/Redis dependency**: Adds operational complexity. Worth it for production scale.
5. **ONNX export fragility**: pyannote ONNX export is not officially supported. May break on version updates.
6. **Multi-GPU complexity**: Requires `CUDA_VISIBLE_DEVICES` routing. Works best on dedicated GPU servers.
