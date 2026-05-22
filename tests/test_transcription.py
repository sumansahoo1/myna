"""Unit tests for app.transcription — factory, stubs, and data classes."""

from pathlib import Path

import pytest

from app.transcription import (
    HostedTranscriberStub,
    LocalFasterWhisperTranscriber,
    TranscriptResult,
    TranscriptSegment,
    get_transcriber,
)


class TestTranscriptSegment:
    def test_fields(self):
        seg = TranscriptSegment(start_sec=0.0, end_sec=1.5, text="hello")
        assert seg.start_sec == 0.0
        assert seg.end_sec == 1.5
        assert seg.text == "hello"


class TestTranscriptResult:
    def test_default_language_is_none(self):
        result = TranscriptResult(text="hello", segments=[])
        assert result.language is None

    def test_text_and_segments(self):
        segs = [TranscriptSegment(0, 1, "a"), TranscriptSegment(1, 2, "b")]
        result = TranscriptResult(text="ab", language="en", segments=segs)
        assert result.text == "ab"
        assert result.language == "en"
        assert len(result.segments) == 2


class TestHostedTranscriberStub:
    def test_raises_runtime_error(self):
        stub = HostedTranscriberStub()
        with pytest.raises(RuntimeError, match="not configured"):
            stub.transcribe(Path("/fake/path.wav"))


class TestGetTranscriber:
    def test_local_returns_local_transcriber(self, monkeypatch):
        monkeypatch.setenv("TRANSCRIBER_PROVIDER", "local")
        result = get_transcriber()
        assert isinstance(result, LocalFasterWhisperTranscriber)

    def test_hosted_returns_stub(self, monkeypatch):
        monkeypatch.setenv("TRANSCRIBER_PROVIDER", "hosted")
        result = get_transcriber()
        assert isinstance(result, HostedTranscriberStub)

    def test_default_is_local(self, monkeypatch):
        monkeypatch.delenv("TRANSCRIBER_PROVIDER", raising=False)
        result = get_transcriber()
        assert isinstance(result, LocalFasterWhisperTranscriber)

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("TRANSCRIBER_PROVIDER", "nonexistent")
        with pytest.raises(ValueError, match="Unknown transcriber provider"):
            get_transcriber()
