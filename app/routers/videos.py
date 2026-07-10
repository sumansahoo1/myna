import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.chunking import AudioChunk, split_audio, stitch_transcripts
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

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".webm",
    ".mkv",
    ".wmv",
    ".flv",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".3gp",
}


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_valid_video_format(filename: str) -> bool:
    return get_file_extension(filename) in ALLOWED_EXTENSIONS


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


def _process_meeting_video(meeting_id: str) -> None:
    """
    Background pipeline:
      1. Extract audio once
      2. VAD → detect speech regions (skip silence)
      3. Split audio into overlapping chunks (if enabled)
      4. Transcribe (chunked or full) | Diarize  —  in parallel
      5. Merge → assign speakers to segments
      6. Persist results
    """
    db = next(get_db())
    audio_path = None
    chunk_paths: set[Path] = set()

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
        if settings.enable_vad:
            try:
                speech_regions = detect_speech_regions(audio_path)
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

        # --- chunking ---
        chunks: list[AudioChunk] = []
        use_chunks = False
        if settings.enable_chunking:
            try:
                chunks = split_audio(audio_path, speech_regions=speech_regions)
                chunk_paths.update(
                    c.wav_path for c in chunks if c.wav_path != audio_path
                )
                use_chunks = len(chunks) > 1
            except Exception as e:
                logger.warning(
                    "_process_meeting_video: meeting_id=%s chunking failed (%s), "
                    "falling back to single-file transcription",
                    meeting_id,
                    e,
                )

        # --- parallel: transcribe + diarize ---
        transcriber = get_transcriber()
        diarizer = get_diarizer()

        with ThreadPoolExecutor(max_workers=2) as executor:
            if use_chunks:
                logger.info(
                    "_process_meeting_video: meeting_id=%s chunked transcribe %d chunks",
                    meeting_id,
                    len(chunks),
                )
                fut_transcribe = executor.submit(
                    _transcribe_chunks, transcriber, chunks
                )
            else:
                fut_transcribe = executor.submit(transcriber.transcribe, audio_path)

            fut_diarize = executor.submit(diarizer.diarize, audio_path)

            # --- transcription result ---
            raw = fut_transcribe.result()
            if use_chunks:
                transcript_result: TranscriptResult = stitch_transcripts(chunks, raw)
            else:
                transcript_result: TranscriptResult = raw

            meeting.transcription_status = TranscriptionStatus.completed
            meeting.transcript_text = transcript_result.text
            meeting.transcript_language = transcript_result.language
            db.commit()
            logger.info(
                "_process_meeting_video: meeting_id=%s transcription done language=%s segments=%d",
                meeting_id,
                transcript_result.language,
                len(transcript_result.segments),
            )

            # --- diarization result ---
            speaker_turns: list = fut_diarize.result()

            meeting.diarization_status = TranscriptionStatus.completed
            db.commit()
            logger.info(
                "_process_meeting_video: meeting_id=%s diarization done speakers=%d",
                meeting_id,
                len(set(t.speaker_label for t in speaker_turns)),
            )

        # --- merge & persist segments ---
        logger.info(
            "_process_meeting_video: meeting_id=%s merging segments with speakers",
            meeting_id,
        )
        merged = assign_speakers(transcript_result.segments, speaker_turns)

        db.query(SegmentModel).filter(SegmentModel.meeting_id == meeting_id).delete()
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
        for cp in chunk_paths:
            if cp != audio_path:
                cp.unlink(missing_ok=True)
        db.close()


def _transcribe_chunks(transcriber, chunks: list[AudioChunk]) -> list:
    """Transcribe chunks and return list of TranscriptResult (one per chunk)."""
    return transcriber.transcribe_chunks(chunks)
