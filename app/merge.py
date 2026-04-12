from __future__ import annotations

from app.diarization import SpeakerTurn
from app.transcription import TranscriptSegment


def assign_speakers(
    transcript_segments: list[TranscriptSegment],
    speaker_turns: list[SpeakerTurn],
) -> list[dict]:
    """
    Match each transcript segment to the speaker whose turn has the
    greatest time overlap with that segment. Falls back to "UNKNOWN"
    when no speaker turn covers the segment at all.

    Returns a list of dicts ready for bulk-inserting into TranscriptSegment rows.
    """
    result = []

    for seg in transcript_segments:
        best_speaker = _find_best_speaker(seg.start_sec, seg.end_sec, speaker_turns)
        result.append({
            "start_sec": seg.start_sec,
            "end_sec": seg.end_sec,
            "text": seg.text,
            "speaker_label": best_speaker,
        })

    return result


def _find_best_speaker(
    start: float,
    end: float,
    turns: list[SpeakerTurn],
) -> str:
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for turn in turns:
        overlap = _overlap(start, end, turn.start_sec, turn.end_sec)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker_label

    return best_speaker


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))
