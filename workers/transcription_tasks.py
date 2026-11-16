"""Celery tasks: job orchestration, per-chunk transcription, assembly, recovery, cleanup.

Flow::

    transcribe_job(job_id)        claim job, plan chunks once, fan out
      └─ transcribe_chunk(id) ×N  claim chunk, transcribe, persist segments, settle
           └─ finalize_job(job_id) (enqueued by whichever chunk finishes last)
                                   merge segments, export TXT/SRT/JSON, mark completed

Every task is idempotent: state lives in the database and transitions are
compare-and-set, so duplicate deliveries and recovery re-enqueues are harmless.
"""

from __future__ import annotations

import logging
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import Task

from chunking.merger import clamp_to_duration, select_owned_segments
from config import get_settings
from database.models import JobStatus
from filestore import get_storage, transcript_prefix
from services import job_service, transcription_service
from services.errors import is_permanent, public_message
from transcribe_pipeline.exporters import export_all
from transcribe_pipeline.models import TranscriptionResult
from transcribe_pipeline.validate import probe_audio_stream
from workers.celery_app import celery_app

logger = logging.getLogger("workers.tasks")

TRANSCRIPT_BASENAME = "transcript"


class QueueUnavailableError(Exception):
    """Raised when a task cannot be published to the broker (e.g. Redis is down)."""


def backoff_seconds(attempt: int) -> int:
    """Exponential backoff with jitter: base * 2^(attempt-1), capped."""
    settings = get_settings()
    delay = min(settings.celery_retry_backoff_max, settings.celery_retry_backoff_base * (2 ** max(0, attempt - 1)))
    return int(delay + random.uniform(0, settings.celery_retry_backoff_base))


def _publish(task: Task, args: List[Any]) -> str:
    try:
        result = task.apply_async(
            args=args,
            retry=True,
            retry_policy={"max_retries": 2, "interval_start": 0.2, "interval_step": 0.5, "interval_max": 1.0},
        )
    except Exception as err:  # kombu.OperationalError, redis ConnectionError, ...
        raise QueueUnavailableError(str(err)) from err
    return result.id


def enqueue_job(job_id: str) -> str:
    """Publish the orchestration task for a job; returns the Celery task id."""
    task_id = _publish(transcribe_job, [job_id])
    job_service.set_task_id(job_id, task_id)
    return task_id


def _enqueue_finalize(job_id: str) -> None:
    _publish(finalize_job, [job_id])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="volga.transcribe_job", max_retries=None)
def transcribe_job(self: Task, job_id: str) -> Dict[str, Any]:
    job = job_service.claim_job(job_id)
    if job is None:
        logger.info("Job %s: not claimable (missing, cancelled or finished); skipping", job_id)
        return {"job_id": job_id, "skipped": True}

    try:
        duration = job.duration
        storage = get_storage()
        with storage.local_path(job.audio_storage_key) as audio_path:
            if not duration and transcription_service.ffmpeg_available():
                duration = probe_audio_stream(audio_path).duration or None

        chunks = job_service.ensure_chunks(job_id, duration)
        runnable = job_service.runnable_chunk_ids(job_id)
        logger.info("Job %s: %d chunk(s), %d to run", job_id, len(chunks), len(runnable))

        for chunk_id in runnable:
            _publish(transcribe_chunk, [chunk_id])

        if not runnable and job_service.settle(job_id) == "finalize":
            _enqueue_finalize(job_id)
        return {"job_id": job_id, "chunks": len(chunks), "enqueued": len(runnable)}

    except Exception as err:
        if is_permanent(err):
            logger.error("Job %s: permanent failure: %s", job_id, err)
            job_service.mark_job_failed(job_id, public_message(err))
            return {"job_id": job_id, "failed": True}
        attempt = self.request.retries + 1
        if attempt > get_settings().celery_max_retries:
            logger.exception("Job %s: giving up after %d attempts", job_id, attempt - 1)
            job_service.mark_job_failed(job_id, public_message(err))
            return {"job_id": job_id, "failed": True}
        countdown = backoff_seconds(attempt)
        logger.warning("Job %s: transient failure (%s); retry %d in %ss", job_id, err, attempt, countdown)
        raise self.retry(exc=err, countdown=countdown)


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="volga.transcribe_chunk", max_retries=None)
def transcribe_chunk(self: Task, chunk_id: int) -> Dict[str, Any]:
    claim = job_service.claim_chunk(chunk_id)
    if claim is None:
        logger.info("Chunk %s: already done, owned by another worker, or job inactive; skipping", chunk_id)
        return {"chunk_id": chunk_id, "skipped": True}

    log_prefix = f"Job {claim.job_id} chunk {claim.chunk_index + 1}/{claim.total_chunks}"
    try:
        storage = get_storage()
        with storage.local_path(claim.audio_storage_key) as audio_path:
            if claim.is_full_file:
                segments = transcription_service.transcribe_full_file(audio_path)
            else:
                assert claim.audio_end is not None and claim.end_time is not None
                logger.info("%s: transcribing %.1fs-%.1fs", log_prefix, claim.audio_start, claim.audio_end)
                raw = transcription_service.transcribe_window(audio_path, claim.audio_start, claim.audio_end)
                segments = select_owned_segments(
                    raw, claim.start_time, claim.end_time,
                    is_first=claim.chunk_index == 0,
                    is_last=claim.chunk_index == claim.total_chunks - 1,
                )
    except Exception as err:
        if is_permanent(err):
            logger.error("%s: permanent failure: %s", log_prefix, err)
            job_service.fail_chunk(claim, public_message(err))
            _after_settle(claim.job_id)
            return {"chunk_id": chunk_id, "failed": True}

        attempt = job_service.release_chunk_for_retry(claim, public_message(err))
        if attempt is None:
            logger.error("%s: retries exhausted: %s", log_prefix, err)
            _after_settle(claim.job_id)
            return {"chunk_id": chunk_id, "failed": True}
        countdown = backoff_seconds(attempt)
        logger.warning("%s: transient failure (%s); retry %d in %ss", log_prefix, err, attempt, countdown)
        raise self.retry(exc=err, countdown=countdown)

    saved = job_service.complete_chunk(claim, segments)
    if saved:
        logger.info("%s: saved %d segment(s)", log_prefix, len(segments))
        _after_settle(claim.job_id)
    return {"chunk_id": chunk_id, "segments": len(segments), "saved": saved}


def _after_settle(job_id: str) -> None:
    if job_service.settle(job_id) == "finalize":
        _enqueue_finalize(job_id)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="volga.finalize_job", max_retries=None)
def finalize_job(self: Task, job_id: str) -> Dict[str, Any]:
    job = job_service.get_job(job_id)
    if job is None or job.status != JobStatus.PROCESSING:
        return {"job_id": job_id, "skipped": True}
    if job_service.settle(job_id) != "finalize":
        return {"job_id": job_id, "skipped": True, "reason": "chunks not complete"}

    temp_dir = tempfile.mkdtemp(prefix="volga_export_")
    try:
        segments = job_service.load_merged_segments(job_id)
        engine = transcription_service.get_engine()
        if job.duration:
            segments = clamp_to_duration(segments, job.duration)
        duration = job.duration or (segments[-1].end if segments else 0.0)
        result = TranscriptionResult(
            source_file=job.original_filename,
            engine=engine.name,
            model=engine.model_name,
            language="en",
            duration=duration,
            segments=segments,
        )
        written = export_all(result, temp_dir, base_name=TRANSCRIPT_BASENAME)
        prefix = transcript_prefix(job.user_id, job.id)
        storage = get_storage()
        for fmt, path in written.items():
            storage.put_file(Path(path), f"{prefix}/{TRANSCRIPT_BASENAME}.{fmt}")

        if job_service.mark_job_completed(job_id, engine.name, engine.model_name, "en", duration, prefix):
            logger.info("Job %s: completed (%d segments)", job_id, len(segments))
        return {"job_id": job_id, "segments": len(segments)}
    except Exception as err:
        attempt = self.request.retries + 1
        if is_permanent(err) or attempt > get_settings().celery_max_retries:
            logger.exception("Job %s: assembly failed", job_id)
            job_service.mark_job_failed(job_id, public_message(err))
            return {"job_id": job_id, "failed": True}
        raise self.retry(exc=err, countdown=backoff_seconds(attempt))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Recovery and retention
# ---------------------------------------------------------------------------

def run_recovery() -> Dict[str, int]:
    """Re-enqueue interrupted work. Safe to run from several workers at once."""
    plan = job_service.find_recoverable()
    for job_id in plan.jobs_to_enqueue:
        enqueue_job(job_id)
    for chunk_id in plan.chunks_to_enqueue:
        _publish(transcribe_chunk, [chunk_id])
    for job_id in plan.jobs_to_finalize:
        _enqueue_finalize(job_id)
    return {
        "jobs_requeued": len(plan.jobs_to_enqueue),
        "chunks_requeued": len(plan.chunks_to_enqueue),
        "jobs_finalized": len(plan.jobs_to_finalize),
    }


@celery_app.task(name="volga.recover_jobs")
def recover_jobs() -> Dict[str, int]:
    return run_recovery()


def run_cleanup(retention_days: Optional[int] = None) -> int:
    """Delete audio, transcripts and records of terminal jobs older than RETENTION_DAYS."""
    days = get_settings().retention_days if retention_days is None else retention_days
    storage = get_storage()
    removed = 0
    for job_id in job_service.expired_job_ids(days):
        job = job_service.get_job(job_id)
        if job is None:
            continue
        storage.delete(job.audio_storage_key)
        storage.delete_prefix(transcript_prefix(job.user_id, job.id))
        job_service.delete_job_record(job_id)
        removed += 1
    if removed:
        logger.info("Retention cleanup removed %d job(s) older than %d day(s)", removed, days)
    return removed


@celery_app.task(name="volga.cleanup_expired")
def cleanup_expired() -> int:
    return run_cleanup()
