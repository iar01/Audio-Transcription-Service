"""Asynchronous transcription REST API (``/api/v1/transcriptions``) and SSE progress stream.

| Method | Path                                   | Success | Errors               |
|--------|----------------------------------------|---------|----------------------|
| POST   | /api/v1/transcriptions                 | 202     | 400, 413, 415, 422, 503 |
| GET    | /api/v1/transcriptions/{id}            | 200     | 404                  |
| GET    | /api/v1/transcriptions/{id}/progress   | 200     | 404                  |
| GET    | /api/v1/transcriptions/{id}/result     | 200     | 404, 409             |
| GET    | /api/v1/transcriptions/{id}/result/{fmt} | 200   | 400, 404, 409        |
| POST   | /api/v1/transcriptions/{id}/retry      | 202     | 404, 409, 503        |
| DELETE | /api/v1/transcriptions/{id}            | 200/202 | 404, 409             |
| GET    | /api/v1/transcriptions/{id}/events     | 200 (text/event-stream) | 404 |
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse

from api.errors import ApiError, error_body, not_found
from config import get_settings
from database.models import JobStatus
from filestore import audio_key, get_storage
from filestore.base import StorageError
from services import job_service
from services.transcription_service import ffmpeg_available
from transcribe_pipeline.errors import (
    CorruptAudioError,
    EmptyFileError,
    FFmpegNotFoundError,
    FileSizeLimitExceededError,
    UnsupportedFormatError,
    ValidationError,
)
from transcribe_pipeline.validate import SUPPORTED_EXTENSIONS, validate_audio_file

logger = logging.getLogger("api.transcriptions")

router = APIRouter(prefix="/api/v1/transcriptions", tags=["transcriptions"])

COPY_BUFFER = 1024 * 1024
DOWNLOAD_TYPES = {"txt": "text/plain; charset=utf-8", "srt": "application/x-subrip; charset=utf-8",
                  "json": "application/json"}


def safe_display_name(name: str) -> str:
    """Sanitize a user-supplied filename for display/metadata only (never used as a path)."""
    base = re.split(r"[\\/]", name)[-1]
    base = unicodedata.normalize("NFKC", base)
    base = "".join(ch for ch in base if ch.isprintable() and ch not in '<>:"|?*')
    base = base.strip().strip(".") or "audio"
    return base[:255]


def _enqueue_or_fail(job_id: str) -> None:
    from workers.transcription_tasks import QueueUnavailableError, enqueue_job

    try:
        enqueue_job(job_id)
    except QueueUnavailableError as err:
        logger.error("Job %s: could not publish to the task queue: %s", job_id, err)
        job_service.mark_job_failed(job_id, "The task queue is unavailable. Retry once the service is back.")
        raise ApiError(503, "queue_unavailable",
                       "The transcription queue is temporarily unavailable. The job was saved; retry it later.",
                       {"job_id": job_id, "status": JobStatus.FAILED}) from err


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

@router.post("", status_code=202)
def create_transcription(audio_file: UploadFile = File(...)) -> JSONResponse:
    settings = get_settings()
    if not audio_file.filename:
        raise ApiError(400, "missing_file", "No audio file selected.")

    display_name = safe_display_name(audio_file.filename)
    suffix = Path(display_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ApiError(415, "unsupported_format", f"Unsupported format '{suffix or 'none'}'. Allowed formats: {allowed}")

    job_id = uuid.uuid4().hex
    tmp_dir = settings.storage_path / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"upload_{job_id}{suffix}"

    try:
        size = 0
        with open(tmp_path, "wb") as out:
            while True:
                block = audio_file.file.read(COPY_BUFFER)
                if not block:
                    break
                size += len(block)
                if size > settings.max_upload_size_bytes:
                    raise ApiError(413, "file_too_large",
                                   f"File exceeds the maximum upload size of {settings.max_upload_size_mb} MB.")
                out.write(block)

        try:
            metadata = validate_audio_file(tmp_path, max_size_bytes=settings.max_upload_size_bytes,
                                           verify_stream=ffmpeg_available())
        except EmptyFileError as err:
            raise ApiError(400, "empty_file", "Uploaded file is empty.") from err
        except FileSizeLimitExceededError as err:
            raise ApiError(413, "file_too_large", "File exceeds the maximum upload size.") from err
        except UnsupportedFormatError as err:
            raise ApiError(415, "unsupported_format", "Unsupported audio format.") from err
        except CorruptAudioError as err:
            raise ApiError(422, "invalid_audio", "The file is corrupt or contains no valid audio stream.") from err
        except FFmpegNotFoundError as err:  # pragma: no cover - guarded by ffmpeg_available()
            raise ApiError(503, "ffmpeg_missing", "Audio validation is unavailable on the server.") from err
        except ValidationError as err:
            raise ApiError(400, "invalid_file", "The uploaded file is not valid.") from err

        key = audio_key("anonymous", job_id, suffix)
        try:
            get_storage().put_file(tmp_path, key, move=True)
        except StorageError as err:
            logger.exception("Job %s: failed to store upload", job_id)
            raise ApiError(500, "storage_error", "Could not store the uploaded file.") from err

        try:
            job_service.create_job(job_id=job_id, original_filename=display_name, audio_storage_key=key,
                                   file_size=size, duration=metadata.duration or None)
        except Exception as err:
            logger.exception("Job %s: failed to create database record", job_id)
            get_storage().delete(key)  # don't leave orphaned audio behind
            raise ApiError(500, "database_error", "Could not create the transcription job.") from err
        logger.info("Job %s: created for '%s' (%d bytes, duration=%.1fs)", job_id, display_name, size,
                    metadata.duration)
    finally:
        audio_file.file.close()
        tmp_path.unlink(missing_ok=True)

    _enqueue_or_fail(job_id)
    return JSONResponse(status_code=202, content={
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "message": "Transcription queued successfully",
        "links": {
            "self": f"/api/v1/transcriptions/{job_id}",
            "progress": f"/api/v1/transcriptions/{job_id}/progress",
            "events": f"/api/v1/transcriptions/{job_id}/events",
            "result": f"/api/v1/transcriptions/{job_id}/result",
        },
    })


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _snapshot_or_404(job_id: str) -> Dict[str, Any]:
    snap = job_service.job_snapshot(job_id)
    if snap is None:
        raise not_found(job_id)
    return snap


@router.get("/{job_id}")
def get_transcription(job_id: str) -> Dict[str, Any]:
    return _snapshot_or_404(job_id)


@router.get("/{job_id}/progress")
def get_progress(job_id: str) -> Dict[str, Any]:
    snap = _snapshot_or_404(job_id)
    return {k: snap[k] for k in ("job_id", "status", "progress", "chunks", "error", "updated_at")}


def _completed_job(job_id: str):  # type: ignore[no-untyped-def]
    job = job_service.get_job(job_id)
    if job is None:
        raise not_found(job_id)
    if job.status != JobStatus.COMPLETED or not job.transcript_storage_prefix:
        raise ApiError(409, "not_completed", f"Transcript is not available; job status is '{job.status}'.",
                       {"status": job.status})
    return job


@router.get("/{job_id}/result")
def get_result(job_id: str) -> Dict[str, Any]:
    job = _completed_job(job_id)
    try:
        data = json.loads(get_storage().get_bytes(f"{job.transcript_storage_prefix}/transcript.json"))
    except (StorageError, ValueError):
        logger.warning("Job %s: stored transcript missing; rebuilding from database", job_id)
        segments = job_service.load_merged_segments(job_id)
        data = {"source_file": job.original_filename, "engine": job.engine, "model": job.model,
                "language": job.language, "duration": job.duration,
                "text": " ".join(s.text.strip() for s in segments if s.text.strip()),
                "segments": [s.to_dict() for s in segments]}
    data.update({"job_id": job.id, "status": job.status, "filename": job.original_filename,
                 "completed_at": job_service.iso(job.completed_at)})
    return data


@router.get("/{job_id}/result/{fmt}")
def download_result(job_id: str, fmt: str) -> Response:
    fmt = fmt.lower()
    if fmt not in DOWNLOAD_TYPES:
        raise ApiError(400, "invalid_format", "Format must be one of: txt, srt, json.")
    job = _completed_job(job_id)
    try:
        content = get_storage().get_bytes(f"{job.transcript_storage_prefix}/transcript.{fmt}")
    except StorageError as err:
        raise ApiError(404, "transcript_missing", "The exported transcript file is not available.") from err
    stem = Path(job.original_filename).stem or "transcript"
    ascii_stem = stem.encode("ascii", "ignore").decode() or "transcript"
    return Response(content=content, media_type=DOWNLOAD_TYPES[fmt],
                    headers={"Content-Disposition": f'attachment; filename="{ascii_stem}.{fmt}"'})


# ---------------------------------------------------------------------------
# Retry / cancel
# ---------------------------------------------------------------------------

@router.post("/{job_id}/retry", status_code=202)
def retry_transcription(job_id: str) -> JSONResponse:
    job = job_service.get_job(job_id)
    if job is None:
        raise not_found(job_id)
    if not job_service.prepare_retry(job_id):
        raise ApiError(409, "not_retryable", f"Only failed or cancelled jobs can be retried (status '{job.status}').",
                       {"status": job.status})
    _enqueue_or_fail(job_id)
    counts = job_service.chunk_counts(job_id)
    return JSONResponse(status_code=202, content={
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "message": "Retry queued; completed chunks will not be reprocessed.",
        "chunks_completed": counts["completed"],
        "chunks_to_process": counts["total"] - counts["completed"],
    })


@router.delete("/{job_id}")
def cancel_transcription(job_id: str) -> JSONResponse:
    job = job_service.get_job(job_id)
    if job is None:
        raise not_found(job_id)
    previous = job_service.cancel_job(job_id)
    if previous is None:
        raise ApiError(409, "not_cancellable", f"Job cannot be cancelled in status '{job.status}'.",
                       {"status": job.status})
    if job.celery_task_id:
        try:
            from workers.celery_app import celery_app

            # Revoking only drops not-yet-started messages; running chunks are never killed
            # mid-write. Workers check the job status before claiming each chunk.
            celery_app.control.revoke(job.celery_task_id)
        except Exception:  # broker down: DB status already prevents further work
            logger.warning("Job %s: revoke failed; relying on database cancellation flag", job_id)
    running = previous == JobStatus.PROCESSING
    return JSONResponse(status_code=202 if running else 200, content={
        "job_id": job_id,
        "status": JobStatus.CANCELLED,
        "message": ("Cancellation requested; a chunk already being transcribed will finish but no new "
                    "chunks will start.") if running else "Transcription cancelled.",
    })


# ---------------------------------------------------------------------------
# Server-Sent Events
# ---------------------------------------------------------------------------

def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


async def _event_stream(request: Request, job_id: str, max_seconds: float) -> AsyncIterator[str]:
    settings = get_settings()
    last_status: Optional[str] = None
    last_progress: Optional[tuple] = None
    last_beat = time.monotonic()
    started = time.monotonic()
    yield "retry: 3000\n\n"
    while True:
        if await request.is_disconnected():
            return
        snap = await run_in_threadpool(job_service.job_snapshot, job_id)
        if snap is None:
            yield _sse("error", error_body("job_not_found", "Job no longer exists."))
            return
        status = snap["status"]
        progress_key = (snap["progress"], snap["chunks"]["completed"], snap["chunks"]["failed"])
        if status != last_status:
            yield _sse("status", snap)
            last_status = status
            last_progress = progress_key
            last_beat = time.monotonic()
        elif progress_key != last_progress:
            yield _sse("progress", snap)
            last_progress = progress_key
            last_beat = time.monotonic()

        if status in JobStatus.TERMINAL:
            yield _sse(status, snap)  # 'completed' | 'failed' | 'cancelled'
            return
        if time.monotonic() - started > max_seconds:
            return  # client EventSource reconnects automatically
        if time.monotonic() - last_beat >= settings.sse_heartbeat_interval:
            yield ": keep-alive\n\n"
            last_beat = time.monotonic()
        await asyncio.sleep(settings.sse_poll_interval)


@router.get("/{job_id}/events")
async def stream_events(request: Request, job_id: str) -> StreamingResponse:
    if await run_in_threadpool(job_service.get_job, job_id) is None:
        raise not_found(job_id)
    return StreamingResponse(
        _event_stream(request, job_id, max_seconds=3600),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
