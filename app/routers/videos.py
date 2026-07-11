import asyncio
import gc
import logging
import random
import time
import uuid
from pathlib import Path

import aiofiles
import torch
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.diarization import get_diarizer
from app.merge import assign_speakers
from app.models import Meeting, TranscriptSegment as SegmentModel
from app.schemas import (
    DiarizeUploadResponse,
    MeetingResponse,
    SegmentResponse,
    SegmentsListResponse,
    TranscriptResponse,
    TranscriptionStatus,
    VideoUploadResponse,
)
from app.transcription import TranscriptResult, get_transcriber, _extract_audio_to_wav
from app.vad import detect_speech_regions

logger = logging.getLogger(__name__)

router = APIRouter()


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_valid_video_format(filename: str) -> bool:
    return get_file_extension(filename) in settings.allowed_extensions


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not is_valid_video_format(video.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video format. Allowed: {', '.join(sorted(settings.allowed_extensions))}",
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

    meeting = Meeting(
        meeting_id=meeting_id,
        video_id=video_id,
        filename=stored_filename,
        transcription_status=TranscriptionStatus.pending,
        diarization_status=TranscriptionStatus.pending,
    )
    db.add(meeting)
    db.commit()
    db.refresh(meeting)

    background_tasks.add_task(_process_meeting_video, meeting_id)

    return VideoUploadResponse(meeting_id=meeting_id, video_id=video_id)


@router.get("/meetings/{meeting_id}", response_model=MeetingResponse)
def get_meeting(meeting_id: str, db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return MeetingResponse.model_validate(meeting)


@router.get("/meetings/{meeting_id}/transcript", response_model=TranscriptResponse)
def get_transcript(meeting_id: str, db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    return TranscriptResponse(
        meeting_id=meeting.meeting_id,
        transcription_status=meeting.transcription_status,
        language=meeting.transcript_language,
        transcript_text=meeting.transcript_text,
        error=meeting.transcript_error,
    )


@router.get("/meetings/{meeting_id}/segments", response_model=SegmentsListResponse)
def get_segments(meeting_id: str, db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    segments = (
        db.query(SegmentModel)
        .filter(SegmentModel.meeting_id == meeting_id)
        .order_by(SegmentModel.start_sec)
        .all()
    )

    return SegmentsListResponse(
        meeting_id=meeting_id,
        transcription_status=meeting.transcription_status,
        diarization_status=meeting.diarization_status,
        segments=[
            SegmentResponse(
                id=seg.id,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                speaker_label=seg.speaker_label,
                text=seg.text,
            )
            for seg in segments
        ],
    )


@router.post("/upload-and-diarize", response_model=DiarizeUploadResponse)
async def upload_and_diarize(
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not is_valid_video_format(video.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video format. Allowed: {', '.join(sorted(settings.allowed_extensions))}",
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

    meeting = Meeting(
        meeting_id=meeting_id,
        video_id=video_id,
        filename=stored_filename,
        transcription_status=TranscriptionStatus.pending,
        diarization_status=TranscriptionStatus.pending,
    )
    db.add(meeting)
    db.commit()
    db.refresh(meeting)

    logger.info(
        "upload-and-diarize: meeting_id=%s file=%s starting pipeline",
        meeting_id,
        stored_filename,
    )

    # Run pipeline synchronously in thread pool to avoid blocking event loop.
    await asyncio.to_thread(_process_meeting_video, meeting_id)

    logger.info(
        "upload-and-diarize: meeting_id=%s pipeline done, reading results", meeting_id
    )

    # Refresh and read results.
    db.refresh(meeting)
    segments = (
        db.query(SegmentModel)
        .filter(SegmentModel.meeting_id == meeting_id)
        .order_by(SegmentModel.start_sec)
        .all()
    )

    error = meeting.transcript_error or meeting.diarization_error

    return DiarizeUploadResponse(
        meeting_id=meeting_id,
        transcription_status=meeting.transcription_status,
        diarization_status=meeting.diarization_status,
        language=meeting.transcript_language,
        transcript_text=meeting.transcript_text,
        segments=[
            SegmentResponse(
                id=seg.id,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                speaker_label=seg.speaker_label,
                text=seg.text,
            )
            for seg in segments
        ],
        error=error,
    )


def _free_gpu():
    """Release GPU memory: trigger GC and clear PyTorch CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _retry_gpu(fn, *args, max_retries: int = 3, **kwargs):
    """Call *fn* with backoff and CUDA cache clearing on GPU failures.
    Retries transient errors (OOM, CUDA errors). Non-GPU errors propagate immediately."""
    base_delay = 2.0
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            is_gpu = any(kw in msg for kw in ("cuda", "out of memory", "gpu"))
            if not is_gpu or attempt == max_retries:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            logger.warning(
                "_retry_gpu: attempt %d/%d — %s: %s — retrying in %.1fs",
                attempt + 1,
                max_retries,
                type(e).__name__,
                e,
                delay,
            )
            torch.cuda.empty_cache()
            time.sleep(delay)
    raise last_error  # type: ignore[misc]


def _process_meeting_video(meeting_id: str) -> None:
    """
    Background pipeline (sequential GPU):
      1. Extract audio once
      2. VAD → detect speech regions (skip silence)
      3. Transcribe (batched GPU) → retry on OOM/CUDA errors
      4. Diarize (GPU) only if transcription succeeded → retry on OOM/CUDA errors
      5. Merge → assign speakers to segments
      6. Persist results (partial persistence: transcript saved even if diarization fails)
    """
    db = next(get_db())
    audio_path = None

    try:
        meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
        if not meeting:
            logger.warning(
                "_process_meeting_video: meeting_id=%s not found", meeting_id
            )
            return

        meeting.transcription_status = TranscriptionStatus.processing
        meeting.diarization_status = TranscriptionStatus.processing
        meeting.transcript_error = None
        meeting.diarization_error = None
        db.commit()

        video_path = settings.video_storage_dir / meeting.filename
        logger.info(
            "_process_meeting_video: meeting_id=%s extracting audio from %s",
            meeting_id,
            video_path.name,
        )
        audio_path = _extract_audio_to_wav(video_path)
        logger.info(
            "_process_meeting_video: meeting_id=%s audio extracted → %s",
            meeting_id,
            audio_path,
        )

        # --- VAD: detect speech regions ---
        speech_regions = None
        vad_audio_array = None
        if settings.enable_vad:
            try:
                speech_regions, vad_audio_array = detect_speech_regions(audio_path)
                if not speech_regions:
                    logger.info(
                        "_process_meeting_video: meeting_id=%s VAD found no speech, "
                        "returning empty transcript",
                        meeting_id,
                    )
                    meeting.transcription_status = TranscriptionStatus.completed
                    meeting.transcript_text = ""
                    meeting.transcript_language = None
                    meeting.diarization_status = TranscriptionStatus.completed
                    db.commit()
                    db.query(SegmentModel).filter(
                        SegmentModel.meeting_id == meeting_id
                    ).delete()
                    db.commit()
                    return
            except Exception as e:
                logger.warning(
                    "_process_meeting_video: meeting_id=%s VAD failed (%s), "
                    "proceeding without VAD",
                    meeting_id,
                    e,
                )
                speech_regions = None

        # --- sequential GPU: transcribe → diarize ---
        # Never load both GPU models at once — OOM risk.
        logger.info(
            "_process_meeting_video: meeting_id=%s batched transcribe on full audio",
            meeting_id,
        )

        transcriber = get_transcriber()
        diarizer = get_diarizer()

        transcript_ok = False
        diarize_ok = False
        transcript_result = None
        speaker_turns = None

        try:
            transcript_result = _retry_gpu(
                transcriber.transcribe_batched,
                audio_path,
                speech_regions,
                None,  # language
                vad_audio_array,
            )
            meeting.transcription_status = TranscriptionStatus.completed
            meeting.transcript_text = transcript_result.text
            meeting.transcript_language = transcript_result.language
            db.commit()
            transcript_ok = True
            logger.info(
                "_process_meeting_video: meeting_id=%s transcription done language=%s segments=%d",
                meeting_id,
                transcript_result.language,
                len(transcript_result.segments),
            )
        except Exception as e:
            logger.error(
                "_process_meeting_video: meeting_id=%s transcription failed: %s",
                meeting_id,
                e,
            )
            meeting.transcription_status = TranscriptionStatus.failed
            meeting.transcript_error = str(e)
            db.commit()

        # Free whisper GPU model before diarization (or on failure).
        del transcriber
        _free_gpu()

        if transcript_ok:
            try:
                speaker_turns = _retry_gpu(diarizer.diarize, audio_path)
                meeting.diarization_status = TranscriptionStatus.completed
                diarize_ok = True
                db.commit()
                logger.info(
                    "_process_meeting_video: meeting_id=%s diarization done speakers=%d",
                    meeting_id,
                    len(set(t.speaker_label for t in speaker_turns)),
                )
            except Exception as e:
                logger.error(
                    "_process_meeting_video: meeting_id=%s diarization failed: %s",
                    meeting_id,
                    e,
                )
                meeting.diarization_status = TranscriptionStatus.failed
                meeting.diarization_error = str(e)
                db.commit()

            del diarizer
            _free_gpu()

        # --- merge & persist segments ---
        if transcript_ok and diarize_ok:
            logger.info(
                "_process_meeting_video: meeting_id=%s merging segments with speakers",
                meeting_id,
            )
            merged = assign_speakers(transcript_result.segments, speaker_turns)
        elif transcript_ok:
            logger.info(
                "_process_meeting_video: meeting_id=%s persisting transcript-only segments (no diarization)",
                meeting_id,
            )
            merged = [
                {
                    "start_sec": seg.start_sec,
                    "end_sec": seg.end_sec,
                    "text": seg.text,
                    "speaker_label": "UNKNOWN",
                }
                for seg in transcript_result.segments
            ]
        else:
            merged = []

        if merged:
            db.query(SegmentModel).filter(
                SegmentModel.meeting_id == meeting_id
            ).delete()
            for row in merged:
                db.add(SegmentModel(meeting_id=meeting_id, **row))
            db.commit()
            logger.info(
                "_process_meeting_video: meeting_id=%s persisted %d segments",
                meeting_id,
                len(merged),
            )

    except Exception as e:
        logger.error("_process_meeting_video: meeting_id=%s FAILED: %s", meeting_id, e)
        db.rollback()
        meeting2 = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
        if meeting2:
            error_str = str(e)
            if meeting2.transcription_status == TranscriptionStatus.processing:
                meeting2.transcription_status = TranscriptionStatus.failed
                meeting2.transcript_error = error_str
            if meeting2.diarization_status == TranscriptionStatus.processing:
                meeting2.diarization_status = TranscriptionStatus.failed
                meeting2.diarization_error = error_str
            db.commit()
    finally:
        if audio_path:
            audio_path.unlink(missing_ok=True)
        _free_gpu()
        db.close()
