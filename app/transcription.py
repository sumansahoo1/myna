from __future__ import annotations

import logging
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


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

    def transcribe_chunks(self, chunks: list) -> list[TranscriptResult]:
        """Transcribe multiple audio chunks. Override for batched optimization."""
        results: list[TranscriptResult] = []
        for chunk in chunks:
            result = self.transcribe(chunk.wav_path)
            results.append(_offset_segments(result, chunk.start_sec))
        return results


def get_transcriber() -> Transcriber:
    provider = (
        os.getenv("TRANSCRIBER_PROVIDER", settings.transcriber_provider).strip().lower()
    )
    if provider == "local":
        model = os.getenv("WHISPER_MODEL", settings.whisper_model)
        device = os.getenv("WHISPER_DEVICE", settings.whisper_device)
        logger.info(
            "transcriber: local faster-whisper model=%s device=%s", model, device
        )
        return LocalFasterWhisperTranscriber(
            model_name=model,
            device=device,
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

    def transcribe_chunks(self, chunks: list) -> list[TranscriptResult]:
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
            logger.info(
                "loading faster-whisper model=%s device=%s",
                self._model_name,
                self._device,
            )
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self._model_name, device=self._device)
            logger.info("faster-whisper model loaded")
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
            segments.append(
                TranscriptSegment(
                    start_sec=round(seg.start, 3),
                    end_sec=round(seg.end, 3),
                    text=txt,
                )
            )
            parts.append(txt)

        return TranscriptResult(
            text=" ".join(parts),
            language=getattr(info, "language", None),
            segments=segments,
        )

    def transcribe_chunks(self, chunks: list) -> list[TranscriptResult]:
        """Transcribe each chunk and offset timestamps to absolute audio time."""
        model = self._get_model()
        results: list[TranscriptResult] = []
        for chunk in chunks:
            raw_segments, info = model.transcribe(
                str(chunk.wav_path), word_timestamps=False
            )
            segments: list[TranscriptSegment] = []
            parts: list[str] = []
            offset = chunk.start_sec
            for seg in raw_segments:
                txt = seg.text.strip()
                if not txt:
                    continue
                segments.append(
                    TranscriptSegment(
                        start_sec=round(seg.start + offset, 3),
                        end_sec=round(seg.end + offset, 3),
                        text=txt,
                    )
                )
                parts.append(txt)
            results.append(
                TranscriptResult(
                    text=" ".join(parts),
                    language=getattr(info, "language", None),
                    segments=segments,
                )
            )
        return results


def _offset_segments(result: TranscriptResult, offset: float) -> TranscriptResult:
    return TranscriptResult(
        text=result.text,
        language=result.language,
        segments=[
            TranscriptSegment(
                start_sec=round(s.start_sec + offset, 3),
                end_sec=round(s.end_sec + offset, 3),
                text=s.text,
            )
            for s in result.segments
        ],
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
    logger.info("ffmpeg extracting audio: %s", video_path.name)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("ffmpeg failed for %s: %s", video_path.name, proc.stderr.strip())
        raise RuntimeError(
            f"ffmpeg failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    logger.info("ffmpeg done → %s", out_path)
    return out_path
