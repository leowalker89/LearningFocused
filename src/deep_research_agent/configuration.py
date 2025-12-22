import os
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from langchain_core.runnables import RunnableConfig

# Default registry of known models and their providers/api key env vars.
DEFAULT_MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    # Google Gemini
    "gemini-3-pro-preview": {"provider": "google_genai", "env_var": "GOOGLE_API_KEY"},
    "gemini-3-flash-preview": {"provider": "google_genai", "env_var": "GOOGLE_API_KEY"},
    "gemini-flash-latest": {"provider": "google_genai", "env_var": "GOOGLE_API_KEY"},
    # Anthropic
    # Prefer stable aliases (always latest)
    "claude-sonnet-4-5": {"provider": "anthropic", "env_var": "ANTHROPIC_API_KEY"},
    "claude-haiku-4-5": {"provider": "anthropic", "env_var": "ANTHROPIC_API_KEY"},
    "claude-opus-4-5": {"provider": "anthropic", "env_var": "ANTHROPIC_API_KEY"},
    # OpenAI
    "gpt-5.2": {"provider": "openai", "env_var": "OPENAI_API_KEY"},
    "gpt-5.1": {"provider": "openai", "env_var": "OPENAI_API_KEY"},
    "gpt-5.1-mini": {"provider": "openai", "env_var": "OPENAI_API_KEY"},
    # Fireworks
    "accounts/fireworks/models/deepseek-v3p2": {"provider": "fireworks", "env_var": "FIREWORKS_API_KEY"},
    "accounts/fireworks/models/kimi-k2-thinking": {"provider": "fireworks", "env_var": "FIREWORKS_API_KEY"},
    "accounts/fireworks/models/gpt-oss-120b": {"provider": "fireworks", "env_var": "FIREWORKS_API_KEY"},
}

# Configurable default model family - set here to override all defaults at once
DEFAULT_MODEL_FAMILY = os.environ.get("DRA_DEFAULT_MODEL_FAMILY", "gemini-flash-latest")

MODEL_FAMILY_TO_DEFAULTS = {
    "gemini-flash-latest": {
        "planner_model": "gemini-flash-latest",
        "writer_model": "gemini-flash-latest",
        "researcher_model": "gemini-flash-latest",
        "research_model": "gemini-flash-latest",
        "compression_model": "gemini-flash-latest",
        "final_report_model": "gemini-flash-latest"
    },
    "gemini-3-pro-preview": {
        "planner_model": "gemini-3-pro-preview",
        "writer_model": "gemini-3-pro-preview",
        "researcher_model": "gemini-3-pro-preview",
        "research_model": "gemini-3-pro-preview",
        "compression_model": "gemini-3-pro-preview",
        "final_report_model": "gemini-3-pro-preview"
    },
    "gpt-5.2": {
        "planner_model": "gpt-5.2",
        "writer_model": "gpt-5.2",
        "researcher_model": "gpt-5.2",
        "research_model": "gpt-5.2",
        "compression_model": "gpt-5.2",
        "final_report_model": "gpt-5.2",
    },
    "claude-sonnet-4-5-20250929": {
        "planner_model": "claude-sonnet-4-5-20250929",
        "writer_model": "claude-sonnet-4-5-20250929",
        "researcher_model": "claude-sonnet-4-5-20250929",
        "research_model": "claude-sonnet-4-5-20250929",
        "compression_model": "claude-sonnet-4-5-20250929",
        "final_report_model": "claude-sonnet-4-5-20250929",
    },
}

@dataclass
class Configuration:
    """The configuration for the agent."""
    max_research_loops: int = 3

    planner_model: str = MODEL_FAMILY_TO_DEFAULTS[DEFAULT_MODEL_FAMILY]["planner_model"]
    writer_model: str = MODEL_FAMILY_TO_DEFAULTS[DEFAULT_MODEL_FAMILY]["writer_model"]
    researcher_model: str = MODEL_FAMILY_TO_DEFAULTS[DEFAULT_MODEL_FAMILY]["researcher_model"]

    # New fields for the complex graph
    allow_clarification: bool = True
    research_model: str = MODEL_FAMILY_TO_DEFAULTS[DEFAULT_MODEL_FAMILY]["research_model"]
    research_model_max_tokens: int = 100000
    max_structured_output_retries: int = 3
    max_concurrent_research_units: int = 3
    max_researcher_iterations: int = 5
    mcp_prompt: str = ""
    max_react_tool_calls: int = 10
    compression_model: str = MODEL_FAMILY_TO_DEFAULTS[DEFAULT_MODEL_FAMILY]["compression_model"]
    compression_model_max_tokens: int = 100000
    final_report_model: str = MODEL_FAMILY_TO_DEFAULTS[DEFAULT_MODEL_FAMILY]["final_report_model"]
    final_report_model_max_tokens: int = 100000
    # Model registry to map model names to providers and env var keys
    model_registry: Dict[str, Dict[str, str]] = field(default_factory=lambda: DEFAULT_MODEL_REGISTRY.copy())

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        config = config or {}
        configurable = config.get("configurable") or {}
        return cls(**{k: v for k, v in configurable.items() if k in cls.__annotations__})
