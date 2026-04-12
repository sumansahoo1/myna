from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "storage" / "videos"
VIDEO_STORAGE_DIR = VIDEOS_DIR  # alias
DATABASE_URL = f"sqlite:///{BASE_DIR / 'storage' / 'meetings.db'}"

# Allowed video formats (extensions)
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv", ".flv", ".m4v"}


class Settings:
    database_url: str = DATABASE_URL
    video_storage_dir: Path = VIDEOS_DIR
    allowed_extensions: set = ALLOWED_VIDEO_EXTENSIONS
    transcriber_provider: str = "local"  # "local" | "hosted"
    whisper_model: str = "small"  # tiny|base|small|medium|large-v3
    whisper_device: str = "cpu"  # cpu|cuda
    diarizer_provider: str = "local"  # "local" | "hosted"
    hf_token: str | None = None  # HuggingFace token for pyannote models


settings = Settings()
