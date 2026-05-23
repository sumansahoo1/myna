from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import init_db
from app.routers import videos


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
