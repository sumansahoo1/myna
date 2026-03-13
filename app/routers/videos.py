import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import Meeting
from app.schemas import MeetingResponse, VideoUploadResponse

router = APIRouter()

ALLOWED_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv", ".flv",
    ".m4v", ".mpeg", ".mpg", ".3gp",
}


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_valid_video_format(filename: str) -> bool:
    return get_file_extension(filename) in ALLOWED_EXTENSIONS


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not is_valid_video_format(video.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    meeting_id = str(uuid.uuid4())
    video_id = str(uuid.uuid4())
    ext = get_file_extension(video.filename)
    stored_filename = f"{video_id}{ext}"

    video_path = settings.video_storage_dir / stored_filename
    video_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(video_path, "wb") as buffer:
        while content := await video.read(1024 * 1024):
            await buffer.write(content)

    db = next(get_db())
    try:
        meeting = Meeting(meeting_id=meeting_id, video_id=video_id, filename=stored_filename)
        db.add(meeting)
        db.commit()
        db.refresh(meeting)
    finally:
        db.close()

    return VideoUploadResponse(meeting_id=meeting_id, video_id=video_id)


@router.get("/meetings/{meeting_id}", response_model=MeetingResponse)
def get_meeting(meeting_id: str):
    db = next(get_db())
    try:
        meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        return MeetingResponse.model_validate(meeting)
    finally:
        db.close()
