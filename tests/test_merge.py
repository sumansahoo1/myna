"""Unit tests for app.merge — pure functions, no dependencies."""

from app.diarization import SpeakerTurn
from app.merge import _overlap, _find_best_speaker, assign_speakers
from app.transcription import TranscriptSegment


# ── _overlap ────────────────────────────────────────────────────────────


class TestOverlap:
    def test_no_overlap(self):
        """Non-overlapping intervals return zero."""
        assert _overlap(0, 5, 10, 15) == 0.0

    def test_touching_edges_no_overlap(self):
        """Intervals that touch at endpoints have zero overlap."""
        assert _overlap(0, 5, 5, 10) == 0.0

    def test_partial_overlap(self):
        """Partial overlap returns the correct duration."""
        assert _overlap(0, 5, 3, 8) == 2.0

    def test_complete_containment(self):
        """One interval fully inside another."""
        assert _overlap(2, 8, 3, 5) == 2.0

    def test_identical_intervals(self):
        """Identical intervals return their full duration."""
        assert _overlap(3, 7, 3, 7) == 4.0

    def test_reverse_args(self):
        """Overlap is symmetric."""
        assert _overlap(3, 8, 0, 5) == _overlap(0, 5, 3, 8)

    def test_zero_length_segment(self):
        """A zero-length interval has zero overlap."""
        assert _overlap(5, 5, 4, 6) == 0.0


# ── _find_best_speaker ─────────────────────────────────────────────────


class TestFindBestSpeaker:
    def test_single_speaker(self):
        """One speaker turn that fully covers the segment."""
        turns = [SpeakerTurn(0, 10, "SPEAKER_00")]
        assert _find_best_speaker(2, 5, turns) == "SPEAKER_00"

    def test_multiple_speakers_picks_greatest_overlap(self):
        """Chooses the speaker with the largest overlap."""
        turns = [
            SpeakerTurn(0, 4, "SPEAKER_00"),
            SpeakerTurn(4, 8, "SPEAKER_01"),
            SpeakerTurn(8, 12, "SPEAKER_02"),
        ]
        # Segment 3-7 overlaps SPEAKER_00 by 1s, SPEAKER_01 by 3s
        assert _find_best_speaker(3, 7, turns) == "SPEAKER_01"

    def test_no_speaker_turns(self):
        """Empty speaker list falls back to UNKNOWN."""
        assert _find_best_speaker(0, 5, []) == "UNKNOWN"

    def test_no_overlap_with_any_speaker(self):
        """Segment with no overlap falls back to UNKNOWN."""
        turns = [SpeakerTurn(10, 20, "SPEAKER_00")]
        assert _find_best_speaker(0, 5, turns) == "UNKNOWN"

    def test_first_speaker_wins_tie(self):
        """When two speakers tie, the first one encountered is chosen."""
        turns = [
            SpeakerTurn(0, 5, "SPEAKER_00"),
            SpeakerTurn(0, 5, "SPEAKER_01"),
        ]
        assert _find_best_speaker(0, 5, turns) == "SPEAKER_00"


# ── assign_speakers ────────────────────────────────────────────────────


class TestAssignSpeakers:
    def test_empty_segments(self):
        """No segment input returns empty list."""
        assert assign_speakers([], [SpeakerTurn(0, 10, "A")]) == []

    def test_empty_speaker_turns(self):
        """No speaker turns → all segments get UNKNOWN labels."""
        segments = [
            TranscriptSegment(0, 5, "hello"),
            TranscriptSegment(5, 10, "world"),
        ]
        result = assign_speakers(segments, [])
        assert all(r["speaker_label"] == "UNKNOWN" for r in result)

    def test_happy_path(self):
        """Segments are correctly matched to speakers."""
        segments = [
            TranscriptSegment(0, 2, "First"),
            TranscriptSegment(3, 5, "Second"),
        ]
        turns = [
            SpeakerTurn(0, 2.5, "SPEAKER_A"),
            SpeakerTurn(2.5, 6, "SPEAKER_B"),
        ]
        result = assign_speakers(segments, turns)
        assert len(result) == 2
        assert result[0] == {"start_sec": 0, "end_sec": 2, "text": "First", "speaker_label": "SPEAKER_A"}
        assert result[1] == {"start_sec": 3, "end_sec": 5, "text": "Second", "speaker_label": "SPEAKER_B"}

    def test_result_keys(self):
        """Each result dict has the expected keys."""
        result = assign_speakers(
            [TranscriptSegment(0, 1, "test")],
            [SpeakerTurn(0, 1, "SPEAKER_00")],
        )
        assert list(result[0].keys()) == ["start_sec", "end_sec", "text", "speaker_label"]
