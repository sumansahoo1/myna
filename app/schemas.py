from pydantic import BaseModel
from datetime import datetime


class MeetingCreate(BaseModel):
    meeting_id: str
    video_id: str
    filename: str


class MeetingResponse(BaseModel):
    meeting_id: str
    video_id: str
    filename: str
    created_at: datetime | None = None

    class Config:
        from_attributes = True


class VideoUploadResponse(BaseModel):
    meeting_id: str
    video_id: str
    message: str = "Video stored successfully"
