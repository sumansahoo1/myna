import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.config import settings
from app.main import app
from app.models import Meeting, TranscriptSegment
from app.schemas import TranscriptionStatus
from app.transcription import TranscriptResult, TranscriptSegment as TS

# ── Test database ──────────────────────────────────────────────────────

TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _setup_test_db():
    """Create all tables before each test, drop after."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db():
    """Provide a bare test DB session (not wired into the app)."""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    """FastAPI TestClient with in-memory DB and a no-op background task."""
    app.dependency_overrides[get_db] = override_get_db

    with (
        patch("app.routers.videos._process_meeting_video") as mock_pipeline,
        TestClient(app) as c,
    ):
        mock_pipeline.return_value = None
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def sample_meeting(db) -> Meeting:
    """Pre-created Meeting row, status pending."""
    meeting = Meeting(
        meeting_id=str(uuid.uuid4()),
        video_id=str(uuid.uuid4()),
        filename="test_video.mp4",
        transcription_status=TranscriptionStatus.pending,
        diarization_status=TranscriptionStatus.pending,
    )
    db.add(meeting)
    db.commit()
    return meeting


@pytest.fixture
def completed_meeting(db) -> Meeting:
    """Pre-created Meeting row with completed transcription."""
    meeting = Meeting(
        meeting_id=str(uuid.uuid4()),
        video_id=str(uuid.uuid4()),
        filename="completed_video.mp4",
        transcription_status=TranscriptionStatus.completed,
        diarization_status=TranscriptionStatus.completed,
        transcript_text="Hello world.",
        transcript_language="en",
    )
    db.add(meeting)
    db.commit()

    segments = [
        TranscriptSegment(
            meeting_id=meeting.meeting_id,
            start_sec=0.0,
            end_sec=1.5,
            speaker_label="SPEAKER_00",
            text="Hello world.",
        ),
        TranscriptSegment(
            meeting_id=meeting.meeting_id,
            start_sec=1.5,
            end_sec=3.0,
            speaker_label="SPEAKER_01",
            text="Hi there.",
        ),
    ]
    for seg in segments:
        db.add(seg)
    db.commit()
    return meeting


@pytest.fixture
def tmp_video_dir(monkeypatch, tmp_path) -> Path:
    """Redirect video storage to a temporary directory."""
    d = tmp_path / "videos"
    d.mkdir(parents=True)
    monkeypatch.setattr(settings, "video_storage_dir", d)
    return d


@pytest.fixture
def mock_transcriber():
    """Returns a Transcriber mock that produces known segments."""
    mock = MagicMock()
    result = TranscriptResult(
        text="Hello world. Hi there.",
        language="en",
        segments=[
            TS(start_sec=0.0, end_sec=1.5, text="Hello world."),
            TS(start_sec=1.5, end_sec=3.0, text="Hi there."),
        ],
    )
    mock.transcribe.return_value = result
    mock.transcribe_batched.return_value = result
    return mock


@pytest.fixture
def mock_diarizer():
    """Returns a Diarizer mock that produces known speaker turns."""
    from app.diarization import SpeakerTurn

    mock = MagicMock()
    mock.diarize.return_value = [
        SpeakerTurn(start_sec=0.0, end_sec=1.8, speaker_label="SPEAKER_00"),
        SpeakerTurn(start_sec=1.8, end_sec=3.5, speaker_label="SPEAKER_01"),
    ]
    return mock
