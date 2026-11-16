"""Job and chunk state transitions.

All state changes are compare-and-set ``UPDATE ... WHERE status = ...``
statements, so concurrent workers, duplicate Celery deliveries and API calls
can't corrupt state: whoever loses the race sees ``rowcount == 0`` and backs off.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import case, delete, func, select, update
from sqlalchemy.exc import IntegrityError

from chunking.audio_splitter import plan_chunks, should_chunk
from chunking.merger import SourcedSegment, merge_segments
from config import get_settings
from database.models import (
    ChunkStatus,
    JobStatus,
    TranscriptionChunk,
    TranscriptionJob,
    TranscriptSegment,
    utcnow,
)
from database.session import session_scope
from transcribe_pipeline.models import Segment

logger = logging.getLogger(__name__)

PROGRESS_STARTED = 2.0
PROGRESS_CHUNKS_SPAN = 93.0  # chunks move progress from 5% to 98%; assembly finishes at 100%


def iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Creation and reads
# ---------------------------------------------------------------------------

def create_job(
    job_id: str,
    original_filename: str,
    audio_storage_key: str,
    file_size: int,
    duration: Optional[float],
    user_id: str = "anonymous",
) -> TranscriptionJob:
    with session_scope() as s:
        job = TranscriptionJob(
            id=job_id,
            user_id=user_id,
            original_filename=original_filename,
            audio_storage_key=audio_storage_key,
            file_size=file_size,
            duration=duration if duration and duration > 0 else None,
            status=JobStatus.QUEUED,
            progress=0.0,
        )
        s.add(job)
        s.flush()
        return job


def get_job(job_id: str) -> Optional[TranscriptionJob]:
    with session_scope() as s:
        return s.get(TranscriptionJob, job_id)


def chunk_counts(job_id: str) -> Dict[str, int]:
    with session_scope() as s:
        rows = s.execute(
            select(TranscriptionChunk.status, func.count())
            .where(TranscriptionChunk.job_id == job_id)
            .group_by(TranscriptionChunk.status)
        ).all()
    counts = {ChunkStatus.PENDING: 0, ChunkStatus.PROCESSING: 0, ChunkStatus.COMPLETED: 0, ChunkStatus.FAILED: 0}
    for status, n in rows:
        counts[status] = int(n)
    counts["total"] = sum(counts.values())
    return counts


def job_snapshot(job_id: str) -> Optional[Dict[str, Any]]:
    """Public, path-free view of a job (used by the REST API and SSE)."""
    job = get_job(job_id)
    if job is None:
        return None
    counts = chunk_counts(job_id)
    return {
        "job_id": job.id,
        "filename": job.original_filename,
        "status": job.status,
        "progress": round(float(job.progress), 1),
        "file_size": job.file_size,
        "duration": job.duration,
        "error": job.error_message,
        "retry_count": job.retry_count,
        "chunked": job.chunked,
        "chunks": {
            "total": counts["total"],
            "completed": counts[ChunkStatus.COMPLETED],
            "failed": counts[ChunkStatus.FAILED],
            "processing": counts[ChunkStatus.PROCESSING],
            "pending": counts[ChunkStatus.PENDING],
        },
        "created_at": iso(job.created_at),
        "started_at": iso(job.started_at),
        "completed_at": iso(job.completed_at),
        "updated_at": iso(job.updated_at),
        "can_retry": job.status in {JobStatus.FAILED, JobStatus.CANCELLED},
    }


def set_task_id(job_id: str, task_id: str) -> None:
    with session_scope() as s:
        s.execute(update(TranscriptionJob).where(TranscriptionJob.id == job_id).values(celery_task_id=task_id))


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

def claim_job(job_id: str) -> Optional[TranscriptionJob]:
    """Move a job queued -> processing. Returns the job if it should be (re)worked, else None.

    A job already in ``processing`` is returned as well: that is a duplicate delivery
    or a recovery re-enqueue, and the downstream steps are idempotent.
    """
    with session_scope() as s:
        now = utcnow()
        s.execute(
            update(TranscriptionJob)
            .where(TranscriptionJob.id == job_id, TranscriptionJob.status == JobStatus.QUEUED)
            .values(
                status=JobStatus.PROCESSING,
                started_at=func.coalesce(TranscriptionJob.started_at, now),
                progress=case((TranscriptionJob.progress < PROGRESS_STARTED, PROGRESS_STARTED),
                              else_=TranscriptionJob.progress),
                error_message=None,
                updated_at=now,
            )
        )
        job = s.get(TranscriptionJob, job_id)
        if job is None or job.status != JobStatus.PROCESSING:
            return None
        return job


def ensure_chunks(job_id: str, duration: Optional[float]) -> List[TranscriptionChunk]:
    """Create the job's chunk plan exactly once (idempotent under concurrent calls)."""
    settings = get_settings()
    with session_scope() as s:
        existing = s.scalars(
            select(TranscriptionChunk).where(TranscriptionChunk.job_id == job_id)
            .order_by(TranscriptionChunk.chunk_index)
        ).all()
        if existing:
            return list(existing)

    chunked = should_chunk(duration, settings.chunking_threshold, settings.chunking_enabled)
    rows: List[TranscriptionChunk] = []
    if chunked:
        assert duration is not None
        for plan in plan_chunks(duration, settings.audio_chunk_duration, settings.audio_chunk_overlap):
            rows.append(TranscriptionChunk(
                job_id=job_id, chunk_index=plan.index, start_time=plan.start, end_time=plan.end,
                audio_start=plan.audio_start, audio_end=plan.audio_end, is_full_file=False,
            ))
    else:
        rows.append(TranscriptionChunk(
            job_id=job_id, chunk_index=0, start_time=0.0, end_time=duration,
            audio_start=0.0, audio_end=duration, is_full_file=True,
        ))

    try:
        with session_scope() as s:
            s.add_all(rows)
            s.execute(
                update(TranscriptionJob).where(TranscriptionJob.id == job_id)
                .values(chunked=chunked, total_chunks=len(rows), duration=duration)
            )
        logger.info("Job %s: planned %d chunk(s) (chunked=%s, duration=%s)", job_id, len(rows), chunked, duration)
    except IntegrityError:
        logger.info("Job %s: chunk plan already created by a concurrent task", job_id)

    with session_scope() as s:
        return list(s.scalars(
            select(TranscriptionChunk).where(TranscriptionChunk.job_id == job_id)
            .order_by(TranscriptionChunk.chunk_index)
        ).all())


def runnable_chunk_ids(job_id: str) -> List[int]:
    """Chunks that still need work (pending, or processing but possibly orphaned)."""
    with session_scope() as s:
        return list(s.scalars(
            select(TranscriptionChunk.id).where(
                TranscriptionChunk.job_id == job_id,
                TranscriptionChunk.status.in_([ChunkStatus.PENDING, ChunkStatus.PROCESSING]),
            ).order_by(TranscriptionChunk.chunk_index)
        ).all())


def mark_job_failed(job_id: str, message: str) -> bool:
    with session_scope() as s:
        res = s.execute(
            update(TranscriptionJob)
            .where(TranscriptionJob.id == job_id,
                   TranscriptionJob.status.in_([JobStatus.QUEUED, JobStatus.PROCESSING]))
            .values(status=JobStatus.FAILED, error_message=message, updated_at=utcnow())
        )
        return res.rowcount > 0


def is_job_active(job_id: str) -> bool:
    job = get_job(job_id)
    return job is not None and job.status == JobStatus.PROCESSING


# ---------------------------------------------------------------------------
# Chunk lifecycle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimedChunk:
    chunk_id: int
    job_id: str
    chunk_index: int
    total_chunks: int
    start_time: float
    end_time: Optional[float]
    audio_start: float
    audio_end: Optional[float]
    is_full_file: bool
    retry_count: int
    audio_storage_key: str
    token: str


def claim_chunk(chunk_id: int) -> Optional[ClaimedChunk]:
    """Atomically claim a chunk for processing.

    Succeeds if the chunk is pending, or stuck in processing longer than
    ``CHUNK_STALE_TIMEOUT`` (its worker most likely crashed). Returns None if the
    chunk is done, owned by a live worker, or its job is not processing.
    """
    settings = get_settings()
    token = uuid.uuid4().hex
    now = utcnow()
    stale_cutoff = now - timedelta(seconds=settings.chunk_stale_timeout)
    with session_scope() as s:
        job_ok = (
            select(TranscriptionJob.id)
            .where(TranscriptionJob.id == TranscriptionChunk.job_id,
                   TranscriptionJob.status == JobStatus.PROCESSING)
            .exists()
        )
        res = s.execute(
            update(TranscriptionChunk)
            .where(
                TranscriptionChunk.id == chunk_id,
                job_ok,
                (TranscriptionChunk.status == ChunkStatus.PENDING)
                | ((TranscriptionChunk.status == ChunkStatus.PROCESSING)
                   & (TranscriptionChunk.started_at < stale_cutoff)),
            )
            .values(status=ChunkStatus.PROCESSING, claim_token=token, started_at=now, updated_at=now)
            .execution_options(synchronize_session=False)
        )
        if res.rowcount == 0:
            return None
        chunk = s.get(TranscriptionChunk, chunk_id)
        assert chunk is not None
        job = s.get(TranscriptionJob, chunk.job_id)
        assert job is not None
        return ClaimedChunk(
            chunk_id=chunk.id, job_id=chunk.job_id, chunk_index=chunk.chunk_index,
            total_chunks=job.total_chunks, start_time=chunk.start_time, end_time=chunk.end_time,
            audio_start=chunk.audio_start, audio_end=chunk.audio_end, is_full_file=chunk.is_full_file,
            retry_count=chunk.retry_count, audio_storage_key=job.audio_storage_key, token=token,
        )


def complete_chunk(claim: ClaimedChunk, segments: Sequence[Segment]) -> bool:
    """Persist a chunk's segments and mark it completed, in one transaction.

    Guarded by the claim token: if another worker re-claimed the chunk (e.g. after a
    stale timeout) only one of them can commit, so segments are never duplicated.
    """
    now = utcnow()
    with session_scope() as s:
        res = s.execute(
            update(TranscriptionChunk)
            .where(TranscriptionChunk.id == claim.chunk_id,
                   TranscriptionChunk.status == ChunkStatus.PROCESSING,
                   TranscriptionChunk.claim_token == claim.token)
            .values(status=ChunkStatus.COMPLETED, segment_count=len(segments), error_message=None,
                    completed_at=now, updated_at=now)
        )
        if res.rowcount == 0:
            s.rollback()
            logger.warning("Chunk %s: lost claim before commit; discarding duplicate result", claim.chunk_id)
            return False
        s.execute(delete(TranscriptSegment).where(TranscriptSegment.chunk_id == claim.chunk_id))
        s.add_all([
            TranscriptSegment(job_id=claim.job_id, chunk_id=claim.chunk_id, seq=i, start=seg.start,
                              end=seg.end, text=seg.text, confidence=seg.confidence)
            for i, seg in enumerate(segments)
        ])
    update_progress(claim.job_id)
    return True


def release_chunk_for_retry(claim: ClaimedChunk, message: str) -> Optional[int]:
    """Record a transient failure. Returns the new retry count, or None if retries are exhausted
    (the chunk is then marked failed)."""
    settings = get_settings()
    now = utcnow()
    with session_scope() as s:
        chunk = s.get(TranscriptionChunk, claim.chunk_id)
        if chunk is None or chunk.claim_token != claim.token or chunk.status != ChunkStatus.PROCESSING:
            return None
        new_count = chunk.retry_count + 1
        exhausted = new_count > settings.celery_max_retries
        chunk.retry_count = new_count
        chunk.error_message = message
        chunk.status = ChunkStatus.FAILED if exhausted else ChunkStatus.PENDING
        chunk.claim_token = None
        chunk.updated_at = now
        return None if exhausted else new_count


def fail_chunk(claim: ClaimedChunk, message: str) -> None:
    now = utcnow()
    with session_scope() as s:
        s.execute(
            update(TranscriptionChunk)
            .where(TranscriptionChunk.id == claim.chunk_id, TranscriptionChunk.claim_token == claim.token)
            .values(status=ChunkStatus.FAILED, error_message=message, claim_token=None, updated_at=now)
        )


def update_progress(job_id: str) -> None:
    counts = chunk_counts(job_id)
    total = counts["total"] or 1
    pct = 5.0 + PROGRESS_CHUNKS_SPAN * counts[ChunkStatus.COMPLETED] / total
    with session_scope() as s:
        # Never let progress move backwards (concurrent chunk completions).
        s.execute(
            update(TranscriptionJob)
            .where(TranscriptionJob.id == job_id, TranscriptionJob.status == JobStatus.PROCESSING)
            .values(progress=case((TranscriptionJob.progress < pct, pct), else_=TranscriptionJob.progress),
                    updated_at=utcnow())
        )


def settle(job_id: str) -> str:
    """Decide what happens next once a chunk has finished (successfully or not).

    Returns ``"finalize"`` when every chunk completed, ``"failed"`` when no chunk is
    still running but at least one failed permanently, otherwise ``"running"``.
    Must be called *after* the chunk's own transaction committed so the last
    finisher always observes the full picture.
    """
    counts = chunk_counts(job_id)
    if counts["total"] == 0:
        return "running"
    if counts[ChunkStatus.PENDING] or counts[ChunkStatus.PROCESSING]:
        return "running"
    if counts[ChunkStatus.FAILED]:
        n = counts[ChunkStatus.FAILED]
        msg = f"{n} of {counts['total']} chunk(s) failed. Completed chunks were kept; retry to process the rest."
        if counts["total"] == 1:
            with session_scope() as s:
                chunk_msg = s.scalar(select(TranscriptionChunk.error_message).where(
                    TranscriptionChunk.job_id == job_id))
            msg = chunk_msg or "Transcription failed."
        mark_job_failed(job_id, msg)
        return "failed"
    return "finalize"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def load_merged_segments(job_id: str) -> List[Segment]:
    with session_scope() as s:
        rows = s.execute(
            select(TranscriptSegment, TranscriptionChunk.chunk_index)
            .join(TranscriptionChunk, TranscriptSegment.chunk_id == TranscriptionChunk.id, isouter=True)
            .where(TranscriptSegment.job_id == job_id)
            .order_by(TranscriptSegment.start, TranscriptSegment.seq)
        ).all()
        items = [
            SourcedSegment(
                segment=Segment(start=r.TranscriptSegment.start, end=r.TranscriptSegment.end,
                                text=r.TranscriptSegment.text, confidence=r.TranscriptSegment.confidence),
                chunk_index=r.chunk_index,
            )
            for r in rows
        ]
    return merge_segments(items)


def mark_job_completed(job_id: str, engine: str, model: str, language: str,
                       duration: Optional[float], transcript_prefix: str) -> bool:
    now = utcnow()
    with session_scope() as s:
        res = s.execute(
            update(TranscriptionJob)
            .where(TranscriptionJob.id == job_id, TranscriptionJob.status == JobStatus.PROCESSING)
            .values(status=JobStatus.COMPLETED, progress=100.0, completed_at=now, updated_at=now,
                    engine=engine, model=model, language=language, duration=duration,
                    transcript_storage_prefix=transcript_prefix, error_message=None)
        )
        return res.rowcount > 0


# ---------------------------------------------------------------------------
# User actions: cancel / retry
# ---------------------------------------------------------------------------

def cancel_job(job_id: str) -> Optional[str]:
    """Cancel a queued/processing job. Returns the previous status, or None if not cancellable."""
    with session_scope() as s:
        job = s.get(TranscriptionJob, job_id)
        if job is None or job.status not in {JobStatus.QUEUED, JobStatus.PROCESSING}:
            return None
        previous = job.status
        res = s.execute(
            update(TranscriptionJob)
            .where(TranscriptionJob.id == job_id, TranscriptionJob.status == previous)
            .values(status=JobStatus.CANCELLED, error_message="Cancelled by user.", updated_at=utcnow())
        )
        return previous if res.rowcount else None


def prepare_retry(job_id: str) -> bool:
    """Reset failed (and orphaned) chunks to pending and requeue the job.

    Completed chunks and their segments are untouched, so no work is repeated.
    """
    now = utcnow()
    with session_scope() as s:
        res = s.execute(
            update(TranscriptionJob)
            .where(TranscriptionJob.id == job_id,
                   TranscriptionJob.status.in_([JobStatus.FAILED, JobStatus.CANCELLED]))
            .values(status=JobStatus.QUEUED, error_message=None, completed_at=None,
                    retry_count=TranscriptionJob.retry_count + 1, updated_at=now)
        )
        if res.rowcount == 0:
            return False
        s.execute(
            update(TranscriptionChunk)
            .where(TranscriptionChunk.job_id == job_id,
                   TranscriptionChunk.status.in_([ChunkStatus.FAILED, ChunkStatus.PROCESSING]))
            .values(status=ChunkStatus.PENDING, retry_count=0, claim_token=None, updated_at=now)
        )
    return True


# ---------------------------------------------------------------------------
# Recovery and retention
# ---------------------------------------------------------------------------

@dataclass
class RecoveryPlan:
    jobs_to_enqueue: List[str]
    chunks_to_enqueue: List[int]
    jobs_to_finalize: List[str]


def find_recoverable() -> RecoveryPlan:
    """Identify work interrupted by crashes/restarts.

    * queued jobs older than ``QUEUED_STALE_TIMEOUT`` (enqueue was lost, e.g. Redis was down)
    * processing jobs with no chunk plan yet (crashed during planning)
    * chunks stuck in processing longer than ``CHUNK_STALE_TIMEOUT`` (worker crashed),
      reset to pending
    * processing jobs whose chunks are all complete but were never finalized
    """
    settings = get_settings()
    now = utcnow()
    queued_cutoff = now - timedelta(seconds=settings.queued_stale_timeout)
    chunk_cutoff = now - timedelta(seconds=settings.chunk_stale_timeout)
    plan = RecoveryPlan([], [], [])
    with session_scope() as s:
        plan.jobs_to_enqueue.extend(s.scalars(
            select(TranscriptionJob.id).where(TranscriptionJob.status == JobStatus.QUEUED,
                                              TranscriptionJob.updated_at < queued_cutoff)
        ).all())

        processing_jobs = s.scalars(
            select(TranscriptionJob.id).where(TranscriptionJob.status == JobStatus.PROCESSING)
        ).all()
        for jid in processing_jobs:
            counts = s.execute(
                select(TranscriptionChunk.status, func.count()).where(TranscriptionChunk.job_id == jid)
                .group_by(TranscriptionChunk.status)
            ).all()
            by_status = {k: int(v) for k, v in counts}
            total = sum(by_status.values())
            if total == 0:
                job = s.get(TranscriptionJob, jid)
                if job is not None and job.updated_at is not None and \
                        job.updated_at.replace(tzinfo=None) < queued_cutoff.replace(tzinfo=None):
                    plan.jobs_to_enqueue.append(jid)
            elif by_status.get(ChunkStatus.COMPLETED, 0) == total:
                plan.jobs_to_finalize.append(jid)

        stale = s.scalars(
            select(TranscriptionChunk.id)
            .join(TranscriptionJob, TranscriptionJob.id == TranscriptionChunk.job_id)
            .where(TranscriptionJob.status == JobStatus.PROCESSING,
                   TranscriptionChunk.status == ChunkStatus.PROCESSING,
                   TranscriptionChunk.started_at < chunk_cutoff)
        ).all()
        if stale:
            s.execute(
                update(TranscriptionChunk)
                .where(TranscriptionChunk.id.in_(stale), TranscriptionChunk.status == ChunkStatus.PROCESSING,
                       TranscriptionChunk.started_at < chunk_cutoff)
                .values(status=ChunkStatus.PENDING, claim_token=None, updated_at=now)
            )
            plan.chunks_to_enqueue.extend(stale)

        pending = s.scalars(
            select(TranscriptionChunk.id)
            .join(TranscriptionJob, TranscriptionJob.id == TranscriptionChunk.job_id)
            .where(TranscriptionJob.status == JobStatus.PROCESSING,
                   TranscriptionChunk.status == ChunkStatus.PENDING,
                   TranscriptionChunk.updated_at < queued_cutoff)
        ).all()
        plan.chunks_to_enqueue.extend(c for c in pending if c not in plan.chunks_to_enqueue)
    return plan


def expired_job_ids(retention_days: int) -> List[str]:
    """Terminal jobs older than the retention period (0 = keep forever)."""
    if retention_days <= 0:
        return []
    cutoff = utcnow() - timedelta(days=retention_days)
    with session_scope() as s:
        return list(s.scalars(
            select(TranscriptionJob.id).where(TranscriptionJob.status.in_(list(JobStatus.TERMINAL)),
                                              TranscriptionJob.updated_at < cutoff)
        ).all())


def delete_job_record(job_id: str) -> None:
    with session_scope() as s:
        s.execute(delete(TranscriptSegment).where(TranscriptSegment.job_id == job_id))
        s.execute(delete(TranscriptionChunk).where(TranscriptionChunk.job_id == job_id))
        s.execute(delete(TranscriptionJob).where(TranscriptionJob.id == job_id))
