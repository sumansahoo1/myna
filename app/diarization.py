from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.config import settings


@dataclass(frozen=True)
class SpeakerTurn:
    start_sec: float
    end_sec: float
    speaker_label: str


class Diarizer:
    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        raise NotImplementedError


def get_diarizer() -> Diarizer:
    provider = os.getenv("DIARIZER_PROVIDER", settings.diarizer_provider).strip().lower()
    hf_token = os.getenv("HF_TOKEN", settings.hf_token or "").strip()

    if provider == "local":
        if not hf_token:
            return NoopDiarizer()
        return LocalPyannoteDializer(hf_token=hf_token, num_speakers=None)
    if provider == "hosted":
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

    def __init__(self, hf_token: str, num_speakers: int | None = None):
        self._hf_token = hf_token
        self._num_speakers = num_speakers
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self._hf_token,
            )
        return self._pipeline

    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
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
        return turns
