"""Unit tests for app.vad — speech detection module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.vad import SpeechRegion, detect_speech_regions, _read_audio_manual


# ── SpeechRegion dataclass ───────────────────────────────────────────────


class TestSpeechRegion:
    def test_fields(self):
        r = SpeechRegion(start_sec=1.5, end_sec=3.0)
        assert r.start_sec == 1.5
        assert r.end_sec == 3.0

    def test_frozen(self):
        r = SpeechRegion(0, 5)
        with pytest.raises(Exception):
            r.start_sec = 10  # type: ignore[misc]

    def test_equality(self):
        assert SpeechRegion(1, 2) == SpeechRegion(1, 2)
        assert SpeechRegion(1, 2) != SpeechRegion(2, 3)


# ── _read_audio_manual ──────────────────────────────────────────────────


class TestReadAudioManual:
    def test_loads_mono_wav(self, tmp_path: Path):
        """Valid mono 16kHz WAV loads correctly."""
        wav = _make_sine_wav(
            tmp_path / "test.wav", duration=1.0, freq=440, stereo=False
        )
        waveform = _read_audio_manual(wav)
        assert waveform.dim() == 1
        assert waveform.shape[0] > 0

    def test_converts_stereo_to_mono(self, tmp_path: Path):
        """Stereo WAV is averaged to mono."""
        wav = _make_sine_wav(
            tmp_path / "stereo.wav", duration=0.5, freq=440, stereo=True
        )
        waveform = _read_audio_manual(wav)
        assert waveform.dim() == 1

    def test_resamples_non_16khz(self, tmp_path: Path):
        """Audio at 44.1kHz is resampled to 16kHz."""
        wav = _make_sine_wav(
            tmp_path / "44k.wav", duration=0.2, freq=440, sample_rate=44100
        )
        waveform = _read_audio_manual(wav)
        expected_samples = int(0.2 * 16000)
        assert abs(waveform.shape[0] - expected_samples) < 1000


# ── detect_speech_regions (mocked VAD model) ─────────────────────────────


class TestDetectSpeechRegions:
    def test_returns_speech_regions(self, tmp_path: Path):
        """Mocks VAD model, verifies correct SpeechRegion conversion."""
        wav = _make_sine_wav(tmp_path / "speech.wav", duration=2.0)

        # Mock timestamps: samples at 16000 Hz
        mock_ts = [
            {"start": 1600, "end": 8000},  # 0.1s → 0.5s
            {"start": 16000, "end": 28000},  # 1.0s → 1.75s
        ]

        with patch("app.vad._load_vad") as mock_load:
            mock_model = MagicMock()
            mock_utils = (MagicMock(return_value=mock_ts), MagicMock(), MagicMock())
            mock_load.return_value = (mock_model, mock_utils)

            with patch("app.vad._read_audio_manual", return_value=MagicMock()):
                regions = detect_speech_regions(wav)

        assert len(regions) == 2
        assert regions[0].start_sec == pytest.approx(0.1)
        assert regions[0].end_sec == pytest.approx(0.5)
        assert regions[1].start_sec == pytest.approx(1.0)
        assert regions[1].end_sec == pytest.approx(1.75)

    def test_no_speech_returns_empty(self, tmp_path: Path):
        """Empty timestamp list → empty regions."""
        wav = _make_sine_wav(tmp_path / "silent.wav", duration=1.0)

        with patch("app.vad._load_vad") as mock_load:
            mock_model = MagicMock()
            mock_utils = (MagicMock(return_value=[]), MagicMock(), MagicMock())
            mock_load.return_value = (mock_model, mock_utils)

            with patch("app.vad._read_audio_manual", return_value=MagicMock()):
                regions = detect_speech_regions(wav)

        assert regions == []

    def test_model_loaded_once(self, tmp_path: Path):
        """VAD model is cached after first load."""
        wav = _make_sine_wav(tmp_path / "once.wav", duration=1.0)
        from app import vad

        # Reset globals
        vad._VAD_MODEL = None
        vad._VAD_UTILS = None

        call_count = 0
        mock_model = MagicMock()
        mock_utils = (MagicMock(return_value=[]), MagicMock(), MagicMock())
        orig = vad._load_vad

        def fake_load():
            nonlocal call_count
            if vad._VAD_MODEL is not None:
                return vad._VAD_MODEL, vad._VAD_UTILS
            call_count += 1
            vad._VAD_MODEL = mock_model
            vad._VAD_UTILS = mock_utils
            return vad._VAD_MODEL, vad._VAD_UTILS

        vad._load_vad = fake_load
        try:
            with patch("app.vad._read_audio_manual", return_value=MagicMock()):
                detect_speech_regions(wav)
                detect_speech_regions(wav)
        finally:
            vad._load_vad = orig

        assert call_count == 1

    def test_passes_vad_parameters(self, tmp_path: Path):
        """min_speech_duration_ms etc. forwarded to VAD."""
        wav = _make_sine_wav(tmp_path / "params.wav", duration=1.0)
        mock_get_ts = MagicMock(return_value=[])

        with patch("app.vad._load_vad") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = (
                mock_model,
                (mock_get_ts, MagicMock(), MagicMock()),
            )

            with patch("app.vad._read_audio_manual", return_value=MagicMock()):
                detect_speech_regions(
                    wav,
                    min_speech_duration_ms=500,
                    min_silence_duration_ms=200,
                    speech_pad_ms=50,
                )

        mock_get_ts.assert_called_once()
        kwargs = mock_get_ts.call_args.kwargs
        assert kwargs["min_speech_duration_ms"] == 500
        assert kwargs["min_silence_duration_ms"] == 200
        assert kwargs["speech_pad_ms"] == 50


# ── helpers ──────────────────────────────────────────────────────────────


def _make_sine_wav(
    path: Path,
    duration: float = 1.0,
    freq: float = 440.0,
    sample_rate: int = 16000,
    stereo: bool = False,
) -> Path:
    """Generate a synthetic WAV file with a sine tone."""
    import struct
    import math
    import wave

    n_samples = int(duration * sample_rate)
    n_channels = 2 if stereo else 1

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)

        for i in range(n_samples):
            value = int(16000 * math.sin(2 * math.pi * freq * i / sample_rate))
            packed = struct.pack("<h", value)
            if stereo:
                packed += packed  # same value in both channels
            wf.writeframes(packed)

    return path
