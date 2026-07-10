from __future__ import annotations

import logging
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.transcription import TranscriptSegment, TranscriptResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioChunk:
    chunk_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    wav_path: Path


def split_audio(
    audio_path: Path,
    chunk_duration: float | None = None,
    overlap: float | None = None,
    speech_regions: list | None = None,
) -> list[AudioChunk]:
    """
    Split a WAV file into overlapping chunks.

    If speech_regions provided (from VAD), chunks are aligned to speech regions
    and silence-only regions are skipped entirely.

    Returns list of AudioChunk sorted by start_sec.
    """
    chunk_dur = chunk_duration or settings.chunk_duration_sec
    ov = overlap or settings.chunk_overlap_sec
    tmp_dir = settings.video_storage_dir.parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Get total duration via ffprobe
    total_duration = _get_audio_duration(audio_path)
    logger.info(
        "splitting audio: duration=%.1fs chunk=%.1fs overlap=%.1fs",
        total_duration,
        chunk_dur,
        ov,
    )

    if total_duration <= chunk_dur:
        return [
            AudioChunk(
                chunk_id=0,
                start_sec=0.0,
                end_sec=total_duration,
                duration_sec=total_duration,
                wav_path=audio_path,
            )
        ]

    if speech_regions:
        return _split_by_speech(audio_path, speech_regions, chunk_dur, ov, tmp_dir)

    return _split_uniform(audio_path, total_duration, chunk_dur, ov, tmp_dir)


def _split_uniform(
    audio_path: Path,
    total_dur: float,
    chunk_dur: float,
    overlap: float,
    tmp_dir: Path,
) -> list[AudioChunk]:
    if overlap >= chunk_dur:
        raise ValueError(
            f"overlap ({overlap}s) must be less than chunk_duration ({chunk_dur}s)"
        )

    chunks: list[AudioChunk] = []
    chunk_id = 0
    cursor = 0.0

    while cursor < total_dur:
        end = min(cursor + chunk_dur, total_dur)
        dur = end - cursor
        if dur < 1.0:
            break

        out_path = tmp_dir / f"chunk_{uuid.uuid4().hex[:8]}.wav"
        _extract_chunk_ffmpeg(audio_path, cursor, dur, out_path)

        chunks.append(
            AudioChunk(
                chunk_id=chunk_id,
                start_sec=cursor,
                end_sec=end,
                duration_sec=dur,
                wav_path=out_path,
            )
        )
        chunk_id += 1
        cursor += chunk_dur - overlap

    logger.info("split audio into %d uniform chunks", len(chunks))
    return chunks


def _split_by_speech(
    audio_path: Path,
    speech_regions: list,
    chunk_dur: float,
    overlap: float,
    tmp_dir: Path,
) -> list[AudioChunk]:
    """Split audio into chunks aligned to VAD-detected speech regions."""
    if overlap >= chunk_dur:
        raise ValueError(
            f"overlap ({overlap}s) must be less than chunk_duration ({chunk_dur}s)"
        )

    chunks: list[AudioChunk] = []
    chunk_id = 0

    for region in speech_regions:
        region_start = region.start_sec
        region_end = region.end_sec
        cursor = region_start

        while cursor < region_end:
            end = min(cursor + chunk_dur, region_end)
            dur = end - cursor
            if dur < 0.5:
                break

            out_path = tmp_dir / f"chunk_{uuid.uuid4().hex[:8]}.wav"
            _extract_chunk_ffmpeg(audio_path, cursor, dur, out_path)

            chunks.append(
                AudioChunk(
                    chunk_id=chunk_id,
                    start_sec=cursor,
                    end_sec=end,
                    duration_sec=dur,
                    wav_path=out_path,
                )
            )
            chunk_id += 1
            cursor += chunk_dur - overlap

    logger.info("split audio into %d speech-aligned chunks", len(chunks))
    return chunks


def _extract_chunk_ffmpeg(src: Path, start_sec: float, duration_sec: float, out: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg chunk extract failed: {proc.stderr.strip()}")


def _get_audio_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
    return float(proc.stdout.strip())


def stitch_transcripts(
    chunks: list[AudioChunk],
    chunk_results: list[TranscriptResult],
) -> TranscriptResult:
    """
    Merge transcript results from overlapping chunks into a single result.

    For overlap regions between consecutive chunks, segments from the later
    chunk that lie entirely within the previous chunk's territory (end_sec
    <= previous_chunk.end_sec) are dropped — they were already transcribed.
    Timestamps are already absolute (adjusted during chunk transcription).
    """
    all_segments: list[TranscriptSegment] = []
    boundary = -1.0

    for chunk, result in zip(chunks, chunk_results):
        for seg in result.segments:
            if seg.end_sec <= boundary:
                continue
            all_segments.append(seg)
        boundary = chunk.end_sec

    # Sort by start time
    all_segments.sort(key=lambda s: s.start_sec)

    full_text = " ".join(s.text for s in all_segments)
    language = chunk_results[0].language if chunk_results else None

    return TranscriptResult(
        text=full_text,
        language=language,
        segments=all_segments,
    )
