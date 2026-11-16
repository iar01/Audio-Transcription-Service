"""Command-line interface for the transcription pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from transcribe_pipeline.errors import PipelineError
from transcribe_pipeline.pipeline import TranscriptionPipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure terminal logging formatting and log level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the command line arguments parser."""
    parser = argparse.ArgumentParser(
        prog="python -m transcribe_pipeline.cli",
        description="Audio speech-to-text transcription pipeline CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'transcribe' subcommand
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe an audio file into JSON, SRT, and TXT outputs.",
    )
    transcribe_parser.add_argument(
        "audio",
        type=str,
        help="Path to the input audio file (WAV, MP3, M4A, FLAC, OGG).",
    )
    transcribe_parser.add_argument(
        "--out-dir",
        type=str,
        default="out",
        help="Target output directory for exported transcripts (default: out/).",
    )
    transcribe_parser.add_argument(
        "--engine",
        type=str,
        choices=["faster-whisper", "mock"],
        default="faster-whisper",
        help="Speech-to-text engine to execute (default: faster-whisper).",
    )
    transcribe_parser.add_argument(
        "--model",
        type=str,
        default="base",
        help="Whisper model size/variant (default: base).",
    )
    transcribe_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logs and stack trace on failure.",
    )
    transcribe_parser.add_argument(
        "--no-verify-stream",
        action="store_true",
        help="Bypass ffprobe stream inspection (useful if ffprobe is not installed).",
    )
    transcribe_parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Bypass ffmpeg audio conversion (useful if audio is already WAV or ffmpeg is not installed).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI.

    Returns:
        int: Process exit code (0 for success, non-zero on error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("transcribe_pipeline.cli")

    if args.command == "transcribe":
        try:
            pipeline = TranscriptionPipeline(
                engine_name=args.engine,
                model=args.model,
            )
            result = pipeline.run(
                audio_path=args.audio,
                out_dir=args.out_dir,
                verify_stream=not args.no_verify_stream,
                preprocess=not args.no_preprocess,
            )

            print(
                f"\nSUCCESS: Transcribed '{result.source_file}' in {len(result.segments)} segments.\n"
                f"Exported to directory: '{args.out_dir}/'"
            )
            return 0

        except PipelineError as err:
            if args.verbose:
                logger.exception("Pipeline execution failed:")
            else:
                sys.stderr.write(f"Error: {err}\n")
            return 1

        except Exception as err:
            if args.verbose:
                logger.exception("Unexpected error encountered:")
            else:
                sys.stderr.write(f"Unexpected error: {err}\n")
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
