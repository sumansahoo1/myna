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
    ".mpeg",
    ".mpg",
    ".3gp",
}


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    return int(val)


class Settings:
    database_url: str = os.getenv("DATABASE_URL", DATABASE_URL)
    video_storage_dir: Path = Path(os.getenv("STORAGE_DIR", VIDEOS_DIR))
    allowed_extensions: set = ALLOWED_VIDEO_EXTENSIONS
    transcriber_provider: str = os.getenv("TRANSCRIBER_PROVIDER", "local")
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    whisper_device: str = os.getenv("WHISPER_DEVICE") or _detect_device()
    diarization_device: str = os.getenv("DIARIZATION_DEVICE") or _detect_device()
    diarizer_provider: str = os.getenv("DIARIZER_PROVIDER", "local")
    hf_token: str | None = os.getenv("HF_TOKEN") or None
    # VAD
    enable_vad: bool = _env_bool("ENABLE_VAD", True)
    # Batched inference
    batch_size: int = _env_int("INFERENCE_BATCH_SIZE", 8)


settings = Settings()
