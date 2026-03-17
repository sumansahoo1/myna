from datetime import datetime

from pydantic import BaseModel


class TranscriptionStatus:
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class MeetingCreate(BaseModel):
    meeting_id: str
    video_id: str
    filename: str


class MeetingResponse(BaseModel):
    meeting_id: str
    video_id: str
    filename: str
    transcription_status: str | None = None
    created_at: datetime | None = None

    class Config:
        from_attributes = True


class VideoUploadResponse(BaseModel):
    meeting_id: str
    video_id: str
    message: str = "Video stored successfully"


class TranscriptResponse(BaseModel):
    meeting_id: str
    transcription_status: str
    language: str | None = None
    transcript_text: str | None = None
    error: str | None = None
