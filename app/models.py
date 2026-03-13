from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func

from .database import Base


class Meeting(Base):
    __tablename__ = "meetings"

    meeting_id = Column(String(36), primary_key=True, index=True)
    video_id = Column(String(36), unique=True, nullable=False)
    filename = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
