"""Celery application (Redis broker).

Reliability settings:
* ``task_acks_late`` + ``task_reject_on_worker_lost``: a message is acknowledged only
  after the task finishes, so a crashed worker's task is redelivered.
* ``worker_prefetch_multiplier=1``: a worker reserves one task at a time, so long
  transcriptions don't pile up behind a busy process.
* No result backend: the database is the source of truth for job state.

Start a worker with (pool/concurrency come from CELERY_POOL / CELERY_CONCURRENCY):
    celery -A workers.celery_app worker --loglevel=INFO
"""

from __future__ import annotations

import logging

from celery import Celery
from celery.signals import worker_process_init, worker_ready

from config import Settings, get_settings

def apply_settings(app: Celery, settings: Settings) -> None:
    """Apply Volga settings to a Celery app (also used by tests after changing env vars)."""
    app.conf.update(
        broker_url=settings.redis_url,
        task_default_queue=settings.celery_queue,
        task_serializer="json",
        accept_content=["json"],
        task_ignore_result=True,
        result_backend=None,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_acks_on_failure_or_timeout=True,
        worker_prefetch_multiplier=1,
        worker_concurrency=settings.celery_concurrency,
        worker_pool=settings.celery_pool,
        # Recycle prefork children periodically to bound memory growth (ignored by solo/threads).
        worker_max_tasks_per_child=50,
        worker_hijack_root_logger=False,
        broker_connection_retry_on_startup=True,
        broker_connection_timeout=5,
        broker_transport_options={
            # Must exceed the longest single task; otherwise Redis redelivers it while still running.
            "visibility_timeout": settings.celery_visibility_timeout,
        },
        task_always_eager=settings.celery_task_always_eager,
        task_eager_propagates=False,
        enable_utc=True,
    )
    # Forget pools bound to a previous broker URL (kombu keeps them in a global registry keyed
    # by connection, so a new URL gets fresh pools; don't close them, they may be shared).
    app._pool = None  # noqa: SLF001
    amqp = app.__dict__.get("amqp")  # cached property; only reset if already created
    if amqp is not None:
        amqp._producer_pool = None  # noqa: SLF001


celery_app = Celery("volga", include=["workers.transcription_tasks"])
apply_settings(celery_app, get_settings())

logger = logging.getLogger("workers.celery_app")


@worker_process_init.connect
def _reset_after_fork(**_kwargs: object) -> None:
    """Prefork children must not reuse DB connections inherited from the parent."""
    from database.session import reset_engine

    reset_engine()


@worker_ready.connect
def _on_worker_ready(**_kwargs: object) -> None:
    """Initialize the schema and recover jobs interrupted by a previous crash/restart."""
    from database.session import init_db
    from workers.transcription_tasks import run_recovery

    init_db()
    try:
        summary = run_recovery()
        logger.info("Startup recovery: %s", summary)
    except Exception:  # never prevent the worker from starting
        logger.exception("Startup recovery failed")
