"""Celery worker entry point used by the integration tests.

Wraps the transcription service so every unit of work is logged (pid, window,
start/end time) and takes ``TEST_ENGINE_DELAY`` seconds, which lets tests observe
concurrency and kill the worker mid-task. Usage::

    python -m tests.celery_test_worker <node-name>
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List

from services import transcription_service
from transcribe_pipeline.models import Segment

_orig_window = transcription_service.transcribe_window
_orig_full = transcription_service.transcribe_full_file


def _log(event: str, label: str) -> None:
    with open(os.environ["TEST_ENGINE_LOG"], "a", encoding="utf-8") as fh:
        fh.write(f"{os.getpid()} {event} {label} {time.time():.3f}\n")


def _slow_window(path: Path, audio_start: float, audio_end: float) -> List[Segment]:
    _log("start", f"{audio_start:.1f}")
    time.sleep(float(os.environ.get("TEST_ENGINE_DELAY", "0")))
    result = _orig_window(path, audio_start, audio_end)
    _log("end", f"{audio_start:.1f}")
    return result


def _slow_full(path: Path) -> List[Segment]:
    _log("start", "full")
    time.sleep(float(os.environ.get("TEST_ENGINE_DELAY", "0")))
    result = _orig_full(path)
    _log("end", "full")
    return result


transcription_service.transcribe_window = _slow_window  # type: ignore[assignment]
transcription_service.transcribe_full_file = _slow_full  # type: ignore[assignment]

if __name__ == "__main__":
    from workers.celery_app import celery_app

    node = sys.argv[1] if len(sys.argv) > 1 else "test"
    celery_app.worker_main(["worker", "--loglevel=INFO", "-n", f"{node}@%h", "--without-gossip",
                            "--without-mingle", "--without-heartbeat"])
