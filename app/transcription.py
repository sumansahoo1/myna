from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.config import settings


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    language: str | None = None


class Transcriber:
    def transcribe(self, video_path: Path) -> TranscriptResult:
        raise NotImplementedError


def get_transcriber() -> Transcriber:
    provider = os.getenv("TRANSCRIBER_PROVIDER", settings.transcriber_provider).strip().lower()
    if provider == "local":
        return LocalFasterWhisperTranscriber(
            model_name=os.getenv("WHISPER_MODEL", settings.whisper_model),
            device=os.getenv("WHISPER_DEVICE", settings.whisper_device),
        )
    if provider == "hosted":
        return HostedTranscriberStub()
    raise ValueError(f"Unknown transcriber provider: {provider}")


class HostedTranscriberStub(Transcriber):
    def transcribe(self, video_path: Path) -> TranscriptResult:
        raise RuntimeError(
            "Hosted transcription provider is not configured yet. "
            "Set TRANSCRIBER_PROVIDER=local to use local transcription."
        )


class LocalFasterWhisperTranscriber(Transcriber):
    def __init__(self, model_name: str, device: str):
        self._model_name = model_name
        self._device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self._model_name, device=self._device)
        return self._model

    def transcribe(self, video_path: Path) -> TranscriptResult:
        audio_path = _extract_audio_to_wav(video_path)
        try:
            model = self._get_model()
            segments, info = model.transcribe(str(audio_path))
            text = " ".join(seg.text.strip() for seg in segments).strip()
            language = getattr(info, "language", None)
            return TranscriptResult(text=text, language=language)
        finally:
            audio_path.unlink(missing_ok=True)


def _extract_audio_to_wav(video_path: Path) -> Path:
    """
    Extract audio from video into a temporary wav file.
    Requires `ffmpeg` to be installed on the machine.
    """
    tmp_dir = settings.video_storage_dir.parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"{uuid.uuid4()}.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return out_path

