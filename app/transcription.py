from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings


@dataclass(frozen=True)
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str


@dataclass
class TranscriptResult:
    text: str
    language: str | None = None
    segments: list[TranscriptSegment] = field(default_factory=list)


class Transcriber:
    def transcribe(self, audio_path: Path) -> TranscriptResult:
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
    def transcribe(self, audio_path: Path) -> TranscriptResult:
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

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        model = self._get_model()
        raw_segments, info = model.transcribe(str(audio_path), word_timestamps=False)

        segments: list[TranscriptSegment] = []
        parts: list[str] = []
        for seg in raw_segments:
            txt = seg.text.strip()
            if not txt:
                continue
            segments.append(TranscriptSegment(
                start_sec=round(seg.start, 3),
                end_sec=round(seg.end, 3),
                text=txt,
            ))
            parts.append(txt)

        return TranscriptResult(
            text=" ".join(parts),
            language=getattr(info, "language", None),
            segments=segments,
        )


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

