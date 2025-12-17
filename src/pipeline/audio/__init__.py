"""
Audio-focused processing pipeline steps.

This package contains the "podcast/audio" workflow (download → transcribe → identify speakers
→ segment topics → generate combined summaries).

The legacy modules in `src.pipeline.*` remain as backward-compatible import shims so existing
commands keep working. New code should prefer importing from `src.pipeline.audio`.
"""

from src.pipeline.audio.download import download_podcasts
from src.pipeline.audio.generate_summaries import process_combined_summaries
from src.pipeline.audio.identify_speakers import identify_speakers, update_transcript_with_speakers
from src.pipeline.audio.segment_topics import segment_transcript
from src.pipeline.audio.transcribe import transcribe_audio

__all__ = [
    "download_podcasts",
    "transcribe_audio",
    "identify_speakers",
    "update_transcript_with_speakers",
    "segment_transcript",
    "process_combined_summaries",
]


