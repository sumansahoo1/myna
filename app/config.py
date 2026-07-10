import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "storage" / "videos"
VIDEO_STORAGE_DIR = VIDEOS_DIR  # alias
DATABASE_URL = f"sqlite:///{BASE_DIR / 'storage' / 'meetings.db'}"

# Allowed video formats (extensions)
ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".webm",
    ".mkv",
    ".wmv",
    ".flv",
    ".m4v",
}


class Settings:
    database_url: str = os.getenv("DATABASE_URL", DATABASE_URL)
    video_storage_dir: Path = VIDEOS_DIR
    allowed_extensions: set = ALLOWED_VIDEO_EXTENSIONS
    transcriber_provider: str = "local"  # "local" | "hosted"
    whisper_model: str = "small"  # tiny|base|small|medium|large-v3|large-v3-turbo
    whisper_device: str = _detect_device()  # auto-detect GPU
    diarizer_provider: str = "local"  # "local" | "hosted"
    hf_token: str | None = None  # HuggingFace token for pyannote models
    # Phase 1: chunking & VAD
    chunk_duration_sec: float = 30.0  # seconds per audio chunk
    chunk_overlap_sec: float = 5.0  # overlap between chunks
    chunk_batch_size: int = 16  # max chunks per transcribe batch
    enable_vad: bool = True  # skip silence via Silero VAD
    enable_chunking: bool = True  # split long audio and batch-transcribe


settings = Settings()
