"""Shared LLM utilities (providers, model registry, factories).

This package centralizes:
- Provider resolution (OpenAI / Anthropic / Google Gemini / Fireworks)
- Model registry + required env var validation
- Model list parsing for primary + fallback models
- Lightweight retry helpers for transient failures (rate limits, 5xx, timeouts)

Pipelines and agents should import from here to avoid duplicating provider-specific
initialization logic across the repo.
"""

from .factory import (  # noqa: F401
    ModelRegistry,
    RetryConfig,
    create_chat_model,
    create_chat_models,
    parse_model_list,
    retryable_invoke,
)


