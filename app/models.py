from sqlalchemy import Column, Float, ForeignKey, Integer, String, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class Meeting(Base):
    __tablename__ = "meetings"

    meeting_id = Column(String(36), primary_key=True, index=True)
    video_id = Column(String(36), unique=True, nullable=False)
    filename = Column(String(255), nullable=False)
    transcription_status = Column(String(20), nullable=False, default="pending")
    transcript_text = Column(Text, nullable=True)
    transcript_language = Column(String(16), nullable=True)
    transcript_error = Column(Text, nullable=True)
    diarization_status = Column(String(20), nullable=False, default="pending")
    diarization_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    segments = relationship("TranscriptSegment", back_populates="meeting", order_by="TranscriptSegment.start_sec")


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id = Column(String(36), ForeignKey("meetings.meeting_id"), nullable=False, index=True)
    start_sec = Column(Float, nullable=False)
    end_sec = Column(Float, nullable=False)
    speaker_label = Column(String(32), nullable=True)
    text = Column(Text, nullable=False)

    meeting = relationship("Meeting", back_populates="segments")
