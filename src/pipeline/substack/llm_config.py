"""Substack pipeline LLM configuration helpers.

Environment variables:
- LF_SUBSTACK_SUMMARY_MODELS (comma-separated; primary + fallbacks)

Shared knobs:
- LF_SUBSTACK_LLM_TIMEOUT_SECONDS (default: 60)
- LF_SUBSTACK_LLM_MAX_TOKENS (optional)
"""

from __future__ import annotations

import os
from typing import Any

from src.llm.factory import RetryConfig, build_chat_runnable, parse_model_list


def _timeout_seconds() -> int:
    return int(os.getenv("LF_SUBSTACK_LLM_TIMEOUT_SECONDS", "60"))


def _max_tokens() -> int | None:
    raw = os.getenv("LF_SUBSTACK_LLM_MAX_TOKENS")
    return int(raw) if raw else None


def _retry_cfg() -> RetryConfig:
    return RetryConfig(
        max_attempts=int(os.getenv("LF_SUBSTACK_LLM_MAX_ATTEMPTS", "4")),
        initial_backoff_seconds=float(os.getenv("LF_SUBSTACK_LLM_INITIAL_BACKOFF_SECONDS", "1.0")),
        max_backoff_seconds=float(os.getenv("LF_SUBSTACK_LLM_MAX_BACKOFF_SECONDS", "20.0")),
        jitter_ratio=float(os.getenv("LF_SUBSTACK_LLM_JITTER_RATIO", "0.2")),
    )


def get_article_summary_llm(*, model: str | None = None, temperature: float = 0.1) -> Any:
    """LLM for `summarize_articles.summarize_article` (structured summary + tags)."""
    defaults = [
        "gemini-flash-latest",
        "claude-haiku-4-5",
        "gpt-5.1-mini",
        "claude-sonnet-4-5",
        "gpt-5.2",
        "accounts/fireworks/models/deepseek-v3p2",
    ]
    env_models = parse_model_list(os.getenv("LF_SUBSTACK_SUMMARY_MODELS"), default=defaults)
    models = [model] + [m for m in env_models if m != model] if model else env_models
    return build_chat_runnable(
        model_names=models,
        temperature=temperature,
        max_tokens=_max_tokens(),
        timeout=_timeout_seconds(),
        retry=_retry_cfg(),
    )


