from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerTurn:
    start_sec: float
    end_sec: float
    speaker_label: str


class Diarizer:
    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        raise NotImplementedError


def get_diarizer() -> Diarizer:
    provider = settings.diarizer_provider.strip().lower()
    hf_token = (settings.hf_token or "").strip()

    device = settings.whisper_device

    if provider == "local":
        if not hf_token:
            logger.info(
                "diarizer: no HF_TOKEN, using NoopDiarizer (all speakers=UNKNOWN)"
            )
            return NoopDiarizer()
        logger.info("diarizer: local pyannote with HF_TOKEN device=%s", device)
        return LocalPyannoteDializer(
            hf_token=hf_token, num_speakers=None, device=device
        )
    if provider == "hosted":
        logger.info("diarizer: hosted stub selected")
        return HostedDiarizerStub()
    raise ValueError(f"Unknown diarizer provider: {provider}")


class NoopDiarizer(Diarizer):
    """
    Used when no HF_TOKEN is set. Skips diarization entirely.
    All transcript segments will be labeled UNKNOWN.
    Set HF_TOKEN env var to enable real speaker diarization.
    """

    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        return []


class HostedDiarizerStub(Diarizer):
    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        raise RuntimeError(
            "Hosted diarization provider is not configured yet. "
            "Set DIARIZER_PROVIDER=local to use local diarization."
        )


class LocalPyannoteDializer(Diarizer):
    """
    Uses pyannote.audio for speaker diarization.

    Requirements:
      - pip install pyannote.audio
      - A Hugging Face token with access to:
          pyannote/speaker-diarization-3.1
          pyannote/segmentation-3.0
        Accept model conditions at:
          https://hf.co/pyannote/speaker-diarization-3.1
          https://hf.co/pyannote/segmentation-3.0
      - Set HF_TOKEN env var or HUGGINGFACE_TOKEN in config.
    """

    def __init__(
        self, hf_token: str, num_speakers: int | None = None, device: str = "cpu"
    ):
        self._hf_token = hf_token
        self._num_speakers = num_speakers
        self._device = device
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            logger.info(
                "loading pyannote diarization pipeline: pyannote/speaker-diarization-3.1"
            )
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self._hf_token,
            )
            if pipeline is None:
                raise RuntimeError(
                    "Failed to load pyannote/speaker-diarization-3.1. "
                    "Verify HF_TOKEN is set and you accepted the user conditions at "
                    "https://hf.co/pyannote/speaker-diarization-3.1"
                )
            if self._device == "cuda":
                import torch

                pipeline.to(torch.device("cuda"))
                logger.info("pyannote pipeline moved to cuda")
            self._pipeline = pipeline
            logger.info("pyannote pipeline loaded")
        return self._pipeline

    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        logger.info("diarizing: %s", audio_path.name)
        pipeline = self._get_pipeline()
        kwargs = {}
        if self._num_speakers:
            kwargs["num_speakers"] = self._num_speakers

        diarization = pipeline(str(audio_path), **kwargs)

        turns: list[SpeakerTurn] = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    start_sec=round(segment.start, 3),
                    end_sec=round(segment.end, 3),
                    speaker_label=speaker,
                )
            )
        logger.info("diarization done: %d speaker turns found", len(turns))
        return turns
