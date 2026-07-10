"""Unit tests for app.chunking — audio splitting and transcript stitching."""

import math
import struct
import uuid
import wave
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.chunking import (
    AudioChunk,
    split_audio,
    stitch_transcripts,
    _get_audio_duration,
    _extract_chunk_ffmpeg,
)
from app.transcription import TranscriptResult, TranscriptSegment


# ── AudioChunk dataclass ─────────────────────────────────────────────────


class TestAudioChunk:
    def test_fields(self):
        c = AudioChunk(
            chunk_id=1,
            start_sec=10.0,
            end_sec=40.0,
            duration_sec=30.0,
            wav_path=Path("/tmp/a.wav"),
        )
        assert c.chunk_id == 1
        assert c.start_sec == 10.0
        assert c.end_sec == 40.0
        assert c.duration_sec == 30.0
        assert c.wav_path == Path("/tmp/a.wav")

    def test_frozen(self):
        c = AudioChunk(0, 0, 30, 30, Path("x.wav"))
        with pytest.raises(Exception):
            c.start_sec = 5  # type: ignore[misc]

    def test_equality(self):
        a = AudioChunk(0, 0, 10, 10, Path("a.wav"))
        b = AudioChunk(0, 0, 10, 10, Path("a.wav"))
        assert a == b


# ── _get_audio_duration ──────────────────────────────────────────────────


class TestGetAudioDuration:
    def test_valid_wav(self, tmp_path: Path):
        wav = _make_wav(tmp_path / "d.wav", duration=3.75)
        dur = _get_audio_duration(wav)
        assert dur == pytest.approx(3.75, abs=0.1)

    def test_missing_file_raises(self):
        with pytest.raises(RuntimeError, match="ffprobe failed"):
            _get_audio_duration(Path("/nonexistent/abc.wav"))


# ── split_audio ──────────────────────────────────────────────────────────


class TestSplitAudio:
    def test_short_audio_returns_single_chunk(self, tmp_path: Path):
        """Audio shorter than chunk_duration → one chunk with original path."""
        wav = _make_wav(tmp_path / "short.wav", duration=10.0)
        chunks = split_audio(wav, chunk_duration=30.0, overlap=5.0)
        assert len(chunks) == 1
        assert chunks[0].wav_path == wav
        assert chunks[0].start_sec == 0.0
        assert chunks[0].duration_sec == pytest.approx(10.0, abs=0.5)

    def test_splits_uniform(self, tmp_path: Path):
        """Long audio split into overlapping chunks."""
        wav = _make_wav(tmp_path / "long.wav", duration=95.0)
        chunks = split_audio(wav, chunk_duration=30.0, overlap=5.0)
        # 95s / (30 - 5) = 3.8 → 4 chunks
        assert len(chunks) >= 3
        for i, c in enumerate(chunks):
            assert c.wav_path.suffix == ".wav"
            assert c.chunk_id == i
            assert c.duration_sec <= 30.0 + 0.5
        # chunks should be sequential with overlap
        for i in range(len(chunks) - 1):
            gap = chunks[i + 1].start_sec - chunks[i].end_sec
            assert gap < 0  # negative = overlap

    def test_last_chunk_clamped(self, tmp_path: Path):
        """Last chunk clamped to total duration."""
        wav = _make_wav(tmp_path / "clamp.wav", duration=50.0)
        chunks = split_audio(wav, chunk_duration=30.0, overlap=5.0)
        last = chunks[-1]
        assert last.end_sec == pytest.approx(50.0, abs=0.5)
        assert last.duration_sec <= 30.0

    def test_speech_aligned_chunks(self, tmp_path: Path):
        """When speech_regions provided, only speech gets chunks."""
        wav = _make_wav(tmp_path / "speech.wav", duration=120.0)
        regions = [
            MagicMock(start_sec=10.0, end_sec=25.0),
            MagicMock(start_sec=60.0, end_sec=90.0),
        ]
        chunks = split_audio(
            wav, chunk_duration=30.0, overlap=5.0, speech_regions=regions
        )
        assert len(chunks) >= 2
        # All chunks should be within speech regions
        for c in chunks:
            in_region = any(
                r.start_sec <= c.start_sec and c.end_sec <= r.end_sec + 0.5
                for r in regions
            )
            assert in_region, f"Chunk {c.start_sec}-{c.end_sec} outside speech regions"

    def test_fallback_on_broken_file(self, tmp_path: Path):
        """split_audio wrapped in try/except at pipeline level, not tested here."""
        pass  # tested in test_videos_router integration tests


# ── stitch_transcripts ───────────────────────────────────────────────────


class TestStitchTranscripts:
    def test_single_chunk_passthrough(self):
        """One chunk → result returned as-is."""
        chunk = AudioChunk(0, 0, 30, 30, Path("x.wav"))
        result = TranscriptResult(
            text="hello world",
            language="en",
            segments=[
                TranscriptSegment(0, 2, "hello"),
                TranscriptSegment(3, 5, "world"),
            ],
        )
        stitched = stitch_transcripts([chunk], [result])
        assert stitched.text == "hello world"
        assert stitched.language == "en"
        assert len(stitched.segments) == 2

    def test_deduplicates_overlap(self):
        """Segments entirely in overlap of previous chunk are dropped."""
        c1 = AudioChunk(0, 0, 30, 30, Path("a.wav"))
        c2 = AudioChunk(1, 25, 55, 30, Path("b.wav"))  # 5s overlap with c1

        r1 = TranscriptResult(
            "first",
            "en",
            [
                TranscriptSegment(0, 10, "hello"),
                TranscriptSegment(11, 29, "world"),
            ],
        )
        r2 = TranscriptResult(
            "second",
            "en",
            [
                TranscriptSegment(25, 28, "dup"),  # fully in overlap → dropped
                TranscriptSegment(
                    29, 35, "valid"
                ),  # starts in overlap, ends after → kept
                TranscriptSegment(40, 50, "later"),
            ],
        )

        stitched = stitch_transcripts([c1, c2], [r1, r2])
        assert len(stitched.segments) == 4  # 2 from r1 + 2 from r2 (dup dropped)
        texts = [s.text for s in stitched.segments]
        assert "dup" not in texts
        assert "valid" in texts
        assert "later" in texts

    def test_multiple_chunks_sort_order(self):
        """Stitched segments sorted by start_sec."""
        c1 = AudioChunk(0, 0, 10, 10, Path("a.wav"))
        c2 = AudioChunk(1, 6, 16, 10, Path("b.wav"))
        c3 = AudioChunk(2, 12, 22, 10, Path("c.wav"))

        r1 = TranscriptResult("a", "en", [TranscriptSegment(0, 3, "a1")])
        r2 = TranscriptResult("b", "en", [TranscriptSegment(6, 9, "b1")])
        r3 = TranscriptResult("c", "en", [TranscriptSegment(12, 15, "c1")])

        stitched = stitch_transcripts([c1, c2, c3], [r1, r2, r3])
        starts = [s.start_sec for s in stitched.segments]
        assert starts == sorted(starts)

    def test_preserves_language(self):
        """Language from first chunk propagated."""
        c1 = AudioChunk(0, 0, 10, 10, Path("a.wav"))
        c2 = AudioChunk(1, 10, 20, 10, Path("b.wav"))
        r1 = TranscriptResult("text", "fr", [TranscriptSegment(0, 5, "bonjour")])
        r2 = TranscriptResult("text", "fr", [TranscriptSegment(10, 15, "monde")])
        stitched = stitch_transcripts([c1, c2], [r1, r2])
        assert stitched.language == "fr"

    def test_empty_chunks_returns_empty(self):
        result = stitch_transcripts([], [])
        assert result.text == ""
        assert result.segments == []
        assert result.language is None

    def test_boundary_exact_overlap(self):
        """Segment ending at boundary is dropped (covered by previous chunk)."""
        c1 = AudioChunk(0, 0, 30, 30, Path("a.wav"))
        c2 = AudioChunk(1, 25, 55, 30, Path("b.wav"))
        r1 = TranscriptResult("x", "en", [TranscriptSegment(0, 25, "kept")])
        r2 = TranscriptResult("y", "en", [TranscriptSegment(25, 30, "dropped")])
        stitched = stitch_transcripts([c1, c2], [r1, r2])
        # "dropped" segment ends at 30 which is boundary (c1.end_sec) → dropped
        assert len(stitched.segments) == 1
        assert stitched.segments[0].text == "kept"


# ── helpers ──────────────────────────────────────────────────────────────


def _make_wav(path: Path, duration: float = 3.0, sample_rate: int = 16000) -> Path:
    """Generate a valid mono 16kHz WAV with sine tone."""
    n_samples = int(duration * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(8000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            wf.writeframes(struct.pack("<h", value))
    return path
