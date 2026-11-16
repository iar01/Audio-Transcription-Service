"""Operational commands for the asynchronous transcription system.

    python manage.py init-db     # create database tables (idempotent)
    python manage.py recover     # re-enqueue jobs/chunks interrupted by crashes or restarts
    python manage.py cleanup     # delete terminal jobs older than RETENTION_DAYS
    python manage.py status ID   # print a job's status as JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python manage.py", description="Volga operational commands.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db", help="Create database tables.")
    sub.add_parser("recover", help="Re-enqueue interrupted jobs and chunks (requires Redis).")
    cleanup = sub.add_parser("cleanup", help="Delete expired jobs, audio and transcripts.")
    cleanup.add_argument("--days", type=int, default=None, help="Override RETENTION_DAYS.")
    status = sub.add_parser("status", help="Show a job's status.")
    status.add_argument("job_id")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from database.session import init_db

    init_db()
    if args.command == "init-db":
        print("Database initialized.")
    elif args.command == "recover":
        from workers.transcription_tasks import QueueUnavailableError, run_recovery

        try:
            print(json.dumps(run_recovery()))
        except QueueUnavailableError as err:
            sys.stderr.write(f"Error: Redis is unavailable ({err}).\n")
            return 1
    elif args.command == "cleanup":
        from workers.transcription_tasks import run_cleanup

        print(f"Removed {run_cleanup(args.days)} expired job(s).")
    elif args.command == "status":
        from services.job_service import job_snapshot

        snap = job_snapshot(args.job_id)
        if snap is None:
            sys.stderr.write("Job not found.\n")
            return 1
        print(json.dumps(snap, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
