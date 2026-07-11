from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechRegion:
    start_sec: float
    end_sec: float


_VAD_MODEL: Optional[object] = None
_VAD_UTILS: Optional[object] = None


def _load_vad():
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is not None:
        return _VAD_MODEL, _VAD_UTILS

    logger.info("loading Silero VAD model")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    _VAD_MODEL = model
    _VAD_UTILS = utils
    logger.info("Silero VAD model loaded")
    return model, utils


def detect_speech_regions(
    audio_path: Path,
    sample_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
) -> tuple[list[SpeechRegion], np.ndarray]:
    """
    Returns (list[SpeechRegion], numpy audio array at sample_rate Hz).
    Caller can reuse the returned audio to avoid re-decoding for transcription.
    Non-speech (silence, music, noise) is excluded from regions.
    """
    model, utils = _load_vad()
    get_speech_timestamps = utils[0]

    wav = _read_audio_manual(audio_path)
    timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )

    regions = [
        SpeechRegion(
            start_sec=ts["start"] / sample_rate,
            end_sec=ts["end"] / sample_rate,
        )
        for ts in timestamps
    ]

    logger.info(
        "VAD: found %d speech regions in %s (%.1fs total speech)",
        len(regions),
        audio_path.name,
        sum(r.end_sec - r.start_sec for r in regions),
    )
    return regions, wav.numpy()


def _read_audio_manual(path: Path) -> "torch.Tensor":
    import torchaudio

    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    return waveform.squeeze(0)
