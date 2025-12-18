"""Audio pipeline LLM configuration helpers.

These helpers define *task-specific* default model lists and provide a single place
to tune timeouts / retries / fallback order for audio ingestion steps.

Environment variables (comma-separated model lists):
- LF_AUDIO_GROUPING_MODELS
- LF_AUDIO_COMBINED_SUMMARY_MODELS
- LF_AUDIO_SEGMENTATION_MODELS
- LF_AUDIO_SPEAKER_ID_MODELS

Shared knobs:
- LF_AUDIO_LLM_TIMEOUT_SECONDS (default: 60)
- LF_AUDIO_LLM_MAX_TOKENS (optional; if unset, provider defaults apply)
"""

from __future__ import annotations

import os
from typing import Any

from src.llm.factory import RetryConfig, build_chat_runnable, parse_model_list


def _timeout_seconds() -> int:
    return int(os.getenv("LF_AUDIO_LLM_TIMEOUT_SECONDS", "60"))


def _max_tokens() -> int | None:
    raw = os.getenv("LF_AUDIO_LLM_MAX_TOKENS")
    return int(raw) if raw else None


def _retry_cfg() -> RetryConfig:
    # Slightly conservative defaults for batch pipelines.
    return RetryConfig(
        max_attempts=int(os.getenv("LF_AUDIO_LLM_MAX_ATTEMPTS", "4")),
        initial_backoff_seconds=float(os.getenv("LF_AUDIO_LLM_INITIAL_BACKOFF_SECONDS", "1.0")),
        max_backoff_seconds=float(os.getenv("LF_AUDIO_LLM_MAX_BACKOFF_SECONDS", "20.0")),
        jitter_ratio=float(os.getenv("LF_AUDIO_LLM_JITTER_RATIO", "0.2")),
    )


def get_grouping_llm(*, temperature: float = 0.0) -> Any:
    """LLM for episode grouping (`generate_summaries.group_episodes`)."""
    models = parse_model_list(
        os.getenv("LF_AUDIO_GROUPING_MODELS"),
        default=[
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "claude-sonnet-4-5",
            "gpt-5.2",
            "accounts/fireworks/models/deepseek-v3p2",
        ],
    )
    return build_chat_runnable(
        model_names=models,
        temperature=temperature,
        max_tokens=_max_tokens(),
        timeout=_timeout_seconds(),
        retry=_retry_cfg(),
    )


def get_combined_summary_llm(*, temperature: float = 0.1) -> Any:
    """LLM for combined episode summaries (`generate_summaries.generate_combined_summary`)."""
    models = parse_model_list(
        os.getenv("LF_AUDIO_COMBINED_SUMMARY_MODELS"),
        default=[
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "claude-sonnet-4-5",
            "gpt-5.2",
            "accounts/fireworks/models/deepseek-v3p2",
        ],
    )
    return build_chat_runnable(
        model_names=models,
        temperature=temperature,
        max_tokens=_max_tokens(),
        timeout=_timeout_seconds(),
        retry=_retry_cfg(),
    )


def get_segmentation_llm(*, temperature: float = 0.0) -> Any:
    """LLM for topic segmentation (`segment_topics.segment_transcript`)."""
    models = parse_model_list(
        os.getenv("LF_AUDIO_SEGMENTATION_MODELS"),
        default=[
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "claude-haiku-4-5",
            "gpt-5.1-mini",
            "accounts/fireworks/models/gpt-oss-120b",
        ],
    )
    return build_chat_runnable(
        model_names=models,
        temperature=temperature,
        max_tokens=_max_tokens(),
        timeout=_timeout_seconds(),
        retry=_retry_cfg(),
    )


def get_speaker_id_llm(*, temperature: float = 0.0) -> Any:
    """LLM for speaker identification (`identify_speakers.identify_speakers`)."""
    models = parse_model_list(
        os.getenv("LF_AUDIO_SPEAKER_ID_MODELS"),
        default=[
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "claude-haiku-4-5",
            "gpt-5.1-mini",
            "accounts/fireworks/models/gpt-oss-120b",
        ],
    )
    return build_chat_runnable(
        model_names=models,
        temperature=temperature,
        max_tokens=_max_tokens(),
        timeout=_timeout_seconds(),
        retry=_retry_cfg(),
    )


