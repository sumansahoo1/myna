import logging
import sys
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import init_db
from app.routers import videos

IST = timezone(timedelta(hours=5, minutes=30))


def _ist_converter(*args):
    return datetime.now(IST).timetuple()


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
handler.formatter.converter = _ist_converter  # type: ignore[attr-defined]

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
    force=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.config import settings

    settings.video_storage_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(
    title="Myna Video Backend",
    description="Backend for receiving and storing video uploads with meeting IDs",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(videos.router, prefix="/api/v1", tags=["videos"])
