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

    def test_transcribe_chunks_raises_runtime_error(self):
        stub = HostedTranscriberStub()
        with pytest.raises(RuntimeError, match="not configured"):
            stub.transcribe_chunks([])


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


# ── _offset_segments ─────────────────────────────────────────────────────


class TestOffsetSegments:
    def test_shifts_timestamps(self):
        from app.transcription import _offset_segments

        result = TranscriptResult(
            text="hello",
            language="en",
            segments=[TranscriptSegment(0, 5, "hello")],
        )
        offset = _offset_segments(result, 30.0)
        assert offset.segments[0].start_sec == 30.0
        assert offset.segments[0].end_sec == 35.0

    def test_preserves_text_and_language(self):
        from app.transcription import _offset_segments

        result = TranscriptResult("bonjour", "fr", [TranscriptSegment(1, 2, "bonjour")])
        offset = _offset_segments(result, 10.0)
        assert offset.text == "bonjour"
        assert offset.language == "fr"

    def test_multiple_segments(self):
        from app.transcription import _offset_segments

        result = TranscriptResult(
            "a b",
            "en",
            [
                TranscriptSegment(0, 1, "a"),
                TranscriptSegment(2, 3, "b"),
            ],
        )
        offset = _offset_segments(result, 100.0)
        assert offset.segments[0].start_sec == 100.0
        assert offset.segments[1].start_sec == 102.0
        assert offset.segments[1].end_sec == 103.0


# ── transcribe_chunks (via base Transcriber) ─────────────────────────────


class TestTranscribeChunks:
    def test_base_uses_per_chunk_transcribe(self):
        """Base Transcriber.transcribe_chunks calls transcribe per chunk."""
        from app.transcription import Transcriber

        call_paths: list[Path] = []

        class FakeTranscriber(Transcriber):
            def transcribe(self, audio_path):
                call_paths.append(audio_path)
                return TranscriptResult("x", segments=[TranscriptSegment(0, 1, "x")])

        transcriber = FakeTranscriber()
        chunks = [
            type("Chunk", (), {"wav_path": Path("/a.wav"), "start_sec": 0.0})(),
            type("Chunk", (), {"wav_path": Path("/b.wav"), "start_sec": 10.0})(),
        ]

        results = transcriber.transcribe_chunks(chunks)
        assert len(results) == 2
        assert call_paths == [Path("/a.wav"), Path("/b.wav")]

    def test_offsets_applied(self):
        """Each chunk result has timestamps offset by chunk.start_sec."""
        from app.transcription import Transcriber

        class FakeTranscriber(Transcriber):
            def transcribe(self, audio_path):
                return TranscriptResult("x", segments=[TranscriptSegment(0, 5, "x")])

        transcriber = FakeTranscriber()
        chunks = [
            type("Chunk", (), {"wav_path": Path("/a.wav"), "start_sec": 0.0})(),
            type("Chunk", (), {"wav_path": Path("/b.wav"), "start_sec": 25.0})(),
        ]

        results = transcriber.transcribe_chunks(chunks)
        assert results[0].segments[0].start_sec == 0.0
        assert results[0].segments[0].end_sec == 5.0
        assert results[1].segments[0].start_sec == 25.0
        assert results[1].segments[0].end_sec == 30.0
