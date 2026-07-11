from __future__ import annotations

import logging
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

    def transcribe_batched(
        self,
        audio_path: Path,
        speech_regions: list | None = None,
        language: str | None = None,
    ) -> TranscriptResult:
        """Transcribe full audio with batched GPU inference. Override for optimization."""
        return self.transcribe(audio_path)


def get_transcriber() -> Transcriber:
    provider = settings.transcriber_provider.strip().lower()
    if provider == "local":
        model = settings.whisper_model
        device = settings.whisper_device
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

    def transcribe_batched(
        self,
        audio_path: Path,
        speech_regions: list | None = None,
        language: str | None = None,
    ) -> TranscriptResult:
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

    def transcribe_batched(
        self,
        audio_path: Path,
        speech_regions: list | None = None,
        language: str | None = None,
    ) -> TranscriptResult:
        """Transcribe full audio using BatchedInferencePipeline for true GPU batching."""
        from faster_whisper import BatchedInferencePipeline

        model = self._get_model()
        pipeline = BatchedInferencePipeline(model)

        kwargs: dict = {"batch_size": 8}

        if speech_regions:
            sampling_rate = 16000
            clip_timestamps = [
                {
                    "start": int(r.start_sec * sampling_rate),
                    "end": int(r.end_sec * sampling_rate),
                }
                for r in speech_regions
            ]
            kwargs["clip_timestamps"] = clip_timestamps
            kwargs["vad_filter"] = False
        # else: let BatchedInferencePipeline run its own VAD (vad_filter=True by default)

        if language:
            kwargs["language"] = language

        logger.info(
            "batched transcribe: audio=%s speech_regions=%d batch_size=%d",
            audio_path.name,
            len(speech_regions) if speech_regions else 0,
            kwargs["batch_size"],
        )

        raw_segments, info = pipeline.transcribe(str(audio_path), **kwargs)

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
