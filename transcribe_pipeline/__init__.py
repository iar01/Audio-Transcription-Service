"""Audio transcription pipeline package."""

from transcribe_pipeline.engines import (
    FasterWhisperEngine,
    MockTranscriberEngine,
    TranscriberEngine,
)
from transcribe_pipeline.errors import (
    CorruptAudioError,
    EmptyFileError,
    EngineError,
    ExportError,
    FFmpegNotFoundError,
    FileMissingError,
    FileSizeLimitExceededError,
    PipelineError,
    PreprocessingError,
    UnsupportedFormatError,
    ValidationError,
)
from transcribe_pipeline.exporters import (
    export_all,
    export_json,
    export_srt,
    export_txt,
    format_srt_timestamp,
)
from transcribe_pipeline.models import AudioMetadata, Segment, TranscriptionResult
from transcribe_pipeline.pipeline import TranscriptionPipeline, run_pipeline
from transcribe_pipeline.preprocess import convert_audio_to_wav, preprocessed_audio
from transcribe_pipeline.validate import probe_audio_stream, validate_audio_file

__all__ = [
    # Models
    "Segment",
    "AudioMetadata",
    "TranscriptionResult",
    # Errors
    "PipelineError",
    "ValidationError",
    "FileMissingError",
    "EmptyFileError",
    "FileSizeLimitExceededError",
    "UnsupportedFormatError",
    "CorruptAudioError",
    "PreprocessingError",
    "FFmpegNotFoundError",
    "EngineError",
    "ExportError",
    # Validation & Preprocessing
    "validate_audio_file",
    "probe_audio_stream",
    "preprocessed_audio",
    "convert_audio_to_wav",
    # Engines
    "TranscriberEngine",
    "MockTranscriberEngine",
    "FasterWhisperEngine",
    # Exporters
    "format_srt_timestamp",
    "export_srt",
    "export_json",
    "export_txt",
    "export_all",
    # Pipeline Orchestrator
    "TranscriptionPipeline",
    "run_pipeline",
]
