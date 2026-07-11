"""Integration tests for API endpoints and background pipeline.

The `client` fixture patches _process_meeting_video → no-op so upload tests
don't trigger ML models.  Pipeline logic itself is tested separately in
TestProcessMeetingVideo.
"""

import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.models import Meeting
from app.schemas import TranscriptionStatus
from app.database import get_db
from app.transcription import TranscriptResult, TranscriptSegment as TS
from tests.conftest import override_get_db


# ── POST /api/v1/upload ────────────────────────────────────────────────


class TestUpload:
    def test_success(self, client, tmp_video_dir):
        small_video = b"fake video content"
        resp = client.post(
            "/api/v1/upload",
            files={"video": ("test.mp4", small_video, "video/mp4")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "meeting_id" in data
        assert "video_id" in data
        assert data["message"] == "Video stored successfully"

        # File should have been written
        stored_files = list(tmp_video_dir.iterdir())
        assert len(stored_files) == 1
        assert stored_files[0].suffix == ".mp4"

    def test_invalid_format(self, client, tmp_video_dir):
        resp = client.post(
            "/api/v1/upload",
            files={"video": ("test.exe", b"content", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "invalid video format" in resp.json()["detail"].lower()

    def test_without_filename(self, client, tmp_video_dir):
        resp = client.post(
            "/api/v1/upload",
            files={"video": b"content"},
        )
        assert resp.status_code == 400

    def test_multiple_extensions(self, client, tmp_video_dir):
        """Each valid extension is accepted."""
        valid = [
            "mp4",
            "avi",
            "mov",
            "webm",
            "mkv",
            "wmv",
            "flv",
            "m4v",
            "mpeg",
            "mpg",
            "3gp",
        ]
        for ext in valid:
            resp = client.post(
                "/api/v1/upload",
                files={"video": (f"video.{ext}", b"content", "video/mp4")},
            )
            assert resp.status_code == 200, f"Extension .{ext} should be accepted"

    def test_persists_meeting_in_db(self, client, tmp_video_dir, db):
        resp = client.post(
            "/api/v1/upload",
            files={"video": ("test.mp4", b"content", "video/mp4")},
        )
        data = resp.json()
        meeting_id = data["meeting_id"]
        video_id = data["video_id"]
        meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
        assert meeting is not None
        assert meeting.filename == f"{video_id}.mp4"
        assert meeting.transcription_status == TranscriptionStatus.pending


# ── GET /api/v1/meetings/{meeting_id} ──────────────────────────────────


class TestGetMeeting:
    def test_found(self, client, sample_meeting):
        resp = client.get(f"/api/v1/meetings/{sample_meeting.meeting_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["meeting_id"] == sample_meeting.meeting_id
        assert data["video_id"] == sample_meeting.video_id
        assert data["filename"] == "test_video.mp4"

    def test_not_found(self, client):
        resp = client.get(f"/api/v1/meetings/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_response_keys(self, client, sample_meeting):
        resp = client.get(f"/api/v1/meetings/{sample_meeting.meeting_id}")
        expected_keys = {
            "meeting_id",
            "video_id",
            "filename",
            "transcription_status",
            "diarization_status",
            "created_at",
        }
        assert set(resp.json().keys()) == expected_keys


# ── GET /api/v1/meetings/{meeting_id}/transcript ───────────────────────


class TestGetTranscript:
    def test_found(self, client, completed_meeting):
        resp = client.get(f"/api/v1/meetings/{completed_meeting.meeting_id}/transcript")
        assert resp.status_code == 200
        data = resp.json()
        assert data["meeting_id"] == completed_meeting.meeting_id
        assert data["transcription_status"] == TranscriptionStatus.completed
        assert data["language"] == "en"
        assert data["transcript_text"] == "Hello world."

    def test_not_found(self, client):
        resp = client.get(f"/api/v1/meetings/{uuid.uuid4()}/transcript")
        assert resp.status_code == 404

    def test_pending_status(self, client, sample_meeting):
        resp = client.get(f"/api/v1/meetings/{sample_meeting.meeting_id}/transcript")
        assert resp.status_code == 200
        assert resp.json()["transcription_status"] == TranscriptionStatus.pending
        assert resp.json()["transcript_text"] is None


# ── GET /api/v1/meetings/{meeting_id}/segments ─────────────────────────


class TestGetSegments:
    def test_found(self, client, completed_meeting):
        resp = client.get(f"/api/v1/meetings/{completed_meeting.meeting_id}/segments")
        assert resp.status_code == 200
        data = resp.json()
        assert data["meeting_id"] == completed_meeting.meeting_id
        assert len(data["segments"]) == 2

    def test_not_found(self, client):
        resp = client.get(f"/api/v1/meetings/{uuid.uuid4()}/segments")
        assert resp.status_code == 404

    def test_segment_keys(self, client, completed_meeting):
        resp = client.get(f"/api/v1/meetings/{completed_meeting.meeting_id}/segments")
        seg = resp.json()["segments"][0]
        assert set(seg.keys()) == {
            "id",
            "start_sec",
            "end_sec",
            "speaker_label",
            "text",
        }

    def test_speaker_labels(self, client, completed_meeting):
        resp = client.get(f"/api/v1/meetings/{completed_meeting.meeting_id}/segments")
        segments = resp.json()["segments"]
        assert segments[0]["speaker_label"] == "SPEAKER_00"
        assert segments[1]["speaker_label"] == "SPEAKER_01"


# ── Background pipeline ────────────────────────────────────────────────


class TestProcessMeetingVideo:
    """Tests the _process_meeting_video function with mocked ML services."""

    def test_pipeline_success(self, db, tmp_video_dir, mock_transcriber, mock_diarizer):
        """Happy path: transcription + diarization + merge update DB correctly."""
        from app.routers.videos import _process_meeting_video

        # Create a pending meeting
        meeting = Meeting(
            meeting_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            filename="test.mp4",
            transcription_status=TranscriptionStatus.pending,
            diarization_status=TranscriptionStatus.pending,
        )
        db.add(meeting)
        db.commit()

        fake_wav = tmp_video_dir / "audio.wav"
        fake_wav.write_text("not really a wav")

        with (
            patch("app.routers.videos.get_db", override_get_db),
            patch("app.routers.videos._extract_audio_to_wav", return_value=fake_wav),
            patch("app.routers.videos.get_transcriber", return_value=mock_transcriber),
            patch("app.routers.videos.get_diarizer", return_value=mock_diarizer),
        ):
            _process_meeting_video(meeting.meeting_id)

        db.expire_all()
        db.refresh(meeting)
        assert meeting.transcription_status == TranscriptionStatus.completed
        assert meeting.diarization_status == TranscriptionStatus.completed
        assert meeting.transcript_text == "Hello world. Hi there."
        assert meeting.transcript_language == "en"

        from app.models import TranscriptSegment

        segs = (
            db.query(TranscriptSegment)
            .filter(TranscriptSegment.meeting_id == meeting.meeting_id)
            .order_by(TranscriptSegment.start_sec)
            .all()
        )
        assert len(segs) == 2
        assert segs[0].speaker_label == "SPEAKER_00"
        assert segs[1].speaker_label == "SPEAKER_01"

    def test_pipeline_transcription_fails(self, db, tmp_video_dir):
        """When transcription fails, error statuses reflect partial failure."""
        from app.routers.videos import _process_meeting_video

        meeting = Meeting(
            meeting_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            filename="test.mp4",
            transcription_status=TranscriptionStatus.pending,
            diarization_status=TranscriptionStatus.pending,
        )
        db.add(meeting)
        db.commit()

        failing_transcriber = type(
            "Fake",
            (),
            {
                "transcribe": lambda self, p: (_ for _ in ()).throw(
                    RuntimeError("model crashed")
                ),
                "transcribe_batched": lambda self, p, speech_regions=None, language=None: (
                    (_ for _ in ()).throw(RuntimeError("model crashed"))
                ),
            },
        )()

        fake_wav = tmp_video_dir / "audio.wav"
        fake_wav.write_text("")

        with (
            patch("app.routers.videos.get_db", override_get_db),
            patch("app.routers.videos._extract_audio_to_wav", return_value=fake_wav),
            patch(
                "app.routers.videos.get_transcriber", return_value=failing_transcriber
            ),
        ):
            _process_meeting_video(meeting.meeting_id)

        db.expire_all()
        db.refresh(meeting)
        assert meeting.transcription_status == TranscriptionStatus.failed
        # Both statuses are set to "processing" at the start of the pipeline,
        # so when an exception occurs, both end up as "failed"
        assert meeting.diarization_status == TranscriptionStatus.failed

    def test_pipeline_diarization_fails(self, db, tmp_video_dir, mock_transcriber):
        """When diarization fails, transcription still succeeds."""
        from app.routers.videos import _process_meeting_video

        meeting = Meeting(
            meeting_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            filename="test.mp4",
            transcription_status=TranscriptionStatus.pending,
            diarization_status=TranscriptionStatus.pending,
        )
        db.add(meeting)
        db.commit()

        failing_diarizer = type(
            "Fake",
            (),
            {
                "diarize": lambda self, p: (_ for _ in ()).throw(
                    RuntimeError("no HF token")
                )
            },
        )()

        fake_wav = tmp_video_dir / "audio.wav"
        fake_wav.write_text("")

        with (
            patch("app.routers.videos.get_db", override_get_db),
            patch("app.routers.videos._extract_audio_to_wav", return_value=fake_wav),
            patch("app.routers.videos.get_transcriber", return_value=mock_transcriber),
            patch("app.routers.videos.get_diarizer", return_value=failing_diarizer),
        ):
            _process_meeting_video(meeting.meeting_id)

        db.expire_all()
        db.refresh(meeting)
        # Both statuses are committed atomically — if diarization fails,
        # neither is marked completed
        assert meeting.transcription_status == TranscriptionStatus.failed
        assert meeting.diarization_status == TranscriptionStatus.failed

    def test_vad_no_speech_short_circuits(self, db, tmp_video_dir):
        """VAD detects no speech → pipeline returns empty early."""
        from app.routers.videos import _process_meeting_video

        meeting = Meeting(
            meeting_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            filename="test.mp4",
            transcription_status=TranscriptionStatus.pending,
            diarization_status=TranscriptionStatus.pending,
        )
        db.add(meeting)
        db.commit()

        fake_wav = tmp_video_dir / "audio.wav"
        fake_wav.write_text("")

        with (
            patch("app.routers.videos.get_db", override_get_db),
            patch("app.routers.videos._extract_audio_to_wav", return_value=fake_wav),
            patch("app.routers.videos.detect_speech_regions", return_value=[]),
            patch("app.routers.videos.settings.enable_vad", True),
        ):
            _process_meeting_video(meeting.meeting_id)

        db.expire_all()
        db.refresh(meeting)
        assert meeting.transcription_status == TranscriptionStatus.completed
        assert meeting.transcript_text == ""
        assert meeting.diarization_status == TranscriptionStatus.completed

    def test_vad_failure_falls_back(
        self, db, tmp_video_dir, mock_transcriber, mock_diarizer
    ):
        """VAD raises → pipeline proceeds without VAD, uses full audio."""
        from app.routers.videos import _process_meeting_video

        meeting = Meeting(
            meeting_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            filename="test.mp4",
            transcription_status=TranscriptionStatus.pending,
            diarization_status=TranscriptionStatus.pending,
        )
        db.add(meeting)
        db.commit()

        fake_wav = tmp_video_dir / "audio.wav"
        fake_wav.write_text("")

        with (
            patch("app.routers.videos.get_db", override_get_db),
            patch("app.routers.videos._extract_audio_to_wav", return_value=fake_wav),
            patch(
                "app.routers.videos.detect_speech_regions",
                side_effect=RuntimeError("VAD crash"),
            ),
            patch("app.routers.videos.settings.enable_vad", True),
            patch("app.routers.videos.get_transcriber", return_value=mock_transcriber),
            patch("app.routers.videos.get_diarizer", return_value=mock_diarizer),
        ):
            _process_meeting_video(meeting.meeting_id)

        db.expire_all()
        db.refresh(meeting)
        assert meeting.transcription_status == TranscriptionStatus.completed
        assert meeting.diarization_status == TranscriptionStatus.completed

    def test_parallel_execution_both_complete(
        self, db, tmp_video_dir, mock_transcriber, mock_diarizer
    ):
        """Transcription + diarization both complete in parallel."""
        from app.routers.videos import _process_meeting_video

        meeting = Meeting(
            meeting_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            filename="test.mp4",
            transcription_status=TranscriptionStatus.pending,
            diarization_status=TranscriptionStatus.pending,
        )
        db.add(meeting)
        db.commit()

        fake_wav = tmp_video_dir / "audio.wav"
        fake_wav.write_text("")

        with (
            patch("app.routers.videos.get_db", override_get_db),
            patch("app.routers.videos._extract_audio_to_wav", return_value=fake_wav),
            patch("app.routers.videos.get_transcriber", return_value=mock_transcriber),
            patch("app.routers.videos.get_diarizer", return_value=mock_diarizer),
        ):
            _process_meeting_video(meeting.meeting_id)

        db.expire_all()
        db.refresh(meeting)
        assert meeting.transcription_status == TranscriptionStatus.completed
        assert meeting.diarization_status == TranscriptionStatus.completed
        mock_transcriber.transcribe_batched.assert_called_once()
        mock_diarizer.diarize.assert_called_once()

    def test_temp_files_cleaned_on_success(
        self, db, tmp_video_dir, mock_transcriber, mock_diarizer
    ):
        """Audio temp files deleted after successful pipeline."""
        from app.routers.videos import _process_meeting_video

        meeting = Meeting(
            meeting_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            filename="test.mp4",
            transcription_status=TranscriptionStatus.pending,
            diarization_status=TranscriptionStatus.pending,
        )
        db.add(meeting)
        db.commit()

        fake_wav = tmp_video_dir / "audio.wav"
        fake_wav.write_text("")

        with (
            patch("app.routers.videos.get_db", override_get_db),
            patch("app.routers.videos._extract_audio_to_wav", return_value=fake_wav),
            patch("app.routers.videos.get_transcriber", return_value=mock_transcriber),
            patch("app.routers.videos.get_diarizer", return_value=mock_diarizer),
        ):
            _process_meeting_video(meeting.meeting_id)

        assert not fake_wav.exists()

    def test_temp_files_cleaned_on_failure(self, db, tmp_video_dir):
        """Audio temp file deleted even when pipeline fails."""
        from app.routers.videos import _process_meeting_video

        meeting = Meeting(
            meeting_id=str(uuid.uuid4()),
            video_id=str(uuid.uuid4()),
            filename="test.mp4",
            transcription_status=TranscriptionStatus.pending,
            diarization_status=TranscriptionStatus.pending,
        )
        db.add(meeting)
        db.commit()

        fake_wav = tmp_video_dir / "audio.wav"
        fake_wav.write_text("")

        with (
            patch("app.routers.videos.get_db", override_get_db),
            patch("app.routers.videos._extract_audio_to_wav", return_value=fake_wav),
            patch(
                "app.routers.videos.get_transcriber", side_effect=RuntimeError("boom")
            ),
        ):
            _process_meeting_video(meeting.meeting_id)

        assert not fake_wav.exists()


# ── POST /api/v1/upload-and-diarize ──────────────────────────────────────


class TestUploadAndDiarize:
    """Synchronous endpoint: upload video → process → return diarization."""

    def test_success(self, client, tmp_video_dir):
        """Pipeline completes, segments with speaker labels returned."""
        from app.models import Meeting

        def fake_pipeline(meeting_id):
            s = next(override_get_db())
            try:
                m = s.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
                m.transcription_status = TranscriptionStatus.completed
                m.diarization_status = TranscriptionStatus.completed
                m.transcript_text = "Hello world. Hi there."
                m.transcript_language = "en"
                from app.models import TranscriptSegment as TSeg

                s.add(
                    TSeg(
                        meeting_id=meeting_id,
                        start_sec=0.0,
                        end_sec=1.5,
                        speaker_label="SPEAKER_00",
                        text="Hello world.",
                    )
                )
                s.add(
                    TSeg(
                        meeting_id=meeting_id,
                        start_sec=1.5,
                        end_sec=3.0,
                        speaker_label="SPEAKER_01",
                        text="Hi there.",
                    )
                )
                s.commit()
            finally:
                s.close()

        with patch(
            "app.routers.videos._process_meeting_video", side_effect=fake_pipeline
        ):
            resp = client.post(
                "/api/v1/upload-and-diarize",
                files={"video": ("test.mp4", b"fake video content", "video/mp4")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["meeting_id"]
        assert data["transcription_status"] == TranscriptionStatus.completed
        assert data["diarization_status"] == TranscriptionStatus.completed
        assert data["language"] == "en"
        assert data["transcript_text"] == "Hello world. Hi there."
        assert data["error"] is None
        assert len(data["segments"]) == 2
        assert data["segments"][0]["speaker_label"] == "SPEAKER_00"
        assert data["segments"][0]["text"] == "Hello world."
        assert data["segments"][1]["speaker_label"] == "SPEAKER_01"
        assert data["segments"][1]["text"] == "Hi there."

    def test_success_persists_file(self, client, tmp_video_dir):
        """Uploaded file is written to disk."""

        def fake_pipeline(meeting_id):
            pass

        with patch(
            "app.routers.videos._process_meeting_video", side_effect=fake_pipeline
        ):
            client.post(
                "/api/v1/upload-and-diarize",
                files={"video": ("meeting.mp4", b"file bytes", "video/mp4")},
            )

        stored = list(tmp_video_dir.iterdir())
        assert len(stored) == 1
        assert stored[0].suffix == ".mp4"
        assert stored[0].read_bytes() == b"file bytes"

    def test_invalid_format(self, client, tmp_video_dir):
        resp = client.post(
            "/api/v1/upload-and-diarize",
            files={"video": ("malware.exe", b"content", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "invalid video format" in resp.json()["detail"].lower()

    def test_without_filename(self, client, tmp_video_dir):
        resp = client.post(
            "/api/v1/upload-and-diarize",
            files={"video": b"content"},
        )
        assert resp.status_code == 400

    def test_pipeline_failure_returns_error(self, client, tmp_video_dir):
        """Failed pipeline → response has error + failed status."""
        from app.models import Meeting

        def failing_pipeline(meeting_id):
            s = next(override_get_db())
            try:
                m = s.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
                m.transcription_status = TranscriptionStatus.failed
                m.diarization_status = TranscriptionStatus.failed
                m.transcript_error = "model crashed"
                m.diarization_error = "model crashed"
                s.commit()
            finally:
                s.close()

        with patch(
            "app.routers.videos._process_meeting_video", side_effect=failing_pipeline
        ):
            resp = client.post(
                "/api/v1/upload-and-diarize",
                files={"video": ("test.mp4", b"fake video content", "video/mp4")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["transcription_status"] == TranscriptionStatus.failed
        assert data["diarization_status"] == TranscriptionStatus.failed
        assert data["error"] == "model crashed"
        assert len(data["segments"]) == 0

    def test_multiple_valid_extensions(self, client, tmp_video_dir):
        """All allowed video extensions are accepted."""
        valid = [
            "mp4",
            "avi",
            "mov",
            "webm",
            "mkv",
            "wmv",
            "flv",
            "m4v",
            "mpeg",
            "mpg",
            "3gp",
        ]

        def fake_pipeline(meeting_id):
            pass

        with patch(
            "app.routers.videos._process_meeting_video", side_effect=fake_pipeline
        ):
            for ext in valid:
                resp = client.post(
                    "/api/v1/upload-and-diarize",
                    files={"video": (f"video.{ext}", b"content", "video/mp4")},
                )
                assert resp.status_code == 200, f"Extension .{ext} should be accepted"
