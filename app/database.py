from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

_MEETINGS_MIGRATIONS = [
    ("transcription_status", "TEXT NOT NULL DEFAULT 'pending'"),
    ("transcript_text", "TEXT"),
    ("transcript_language", "TEXT"),
    ("transcript_error", "TEXT"),
]


def init_db():
    from app.models import Meeting  # noqa: F401 - registers model with Base
    from app.config import BASE_DIR
    (BASE_DIR / "storage" / "videos").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _migrate_meetings_table()


def _migrate_meetings_table() -> None:
    # MVP-friendly SQLite migration to add new columns without Alembic.
    with engine.begin() as conn:
        cols = conn.execute(text("PRAGMA table_info(meetings)")).fetchall()
        existing = {row[1] for row in cols}  # row[1] = column name

        for col_name, col_def in _MEETINGS_MIGRATIONS:
            if col_name in existing:
                continue
            conn.execute(text(f"ALTER TABLE meetings ADD COLUMN {col_name} {col_def}"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
