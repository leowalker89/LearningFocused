import os
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from langchain_core.runnables import RunnableConfig

# Default registry of known models and their providers/api key env vars.
DEFAULT_MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    # OpenAI
    "gpt-5": {"provider": "openai", "env_var": "OPENAI_API_KEY"},
    "gpt-5-mini": {"provider": "openai", "env_var": "OPENAI_API_KEY"},
    # Google Gemini
    "gemini-3-pro-preview": {"provider": "google_genai", "env_var": "GOOGLE_API_KEY"},
    "gemini-1.5-pro": {"provider": "google_genai", "env_var": "GOOGLE_API_KEY"},
    # Anthropic
    "claude-sonnet-4-20250514": {"provider": "anthropic", "env_var": "ANTHROPIC_API_KEY"},
    "claude-3-5-sonnet": {"provider": "anthropic", "env_var": "ANTHROPIC_API_KEY"},
}

# Configurable default model family - set here to override all defaults at once
DEFAULT_MODEL_FAMILY = os.environ.get("DRA_DEFAULT_MODEL_FAMILY", "gpt-5")  # Change "gpt-5" to "gemini-3-pro-preview" for Gemini by default

MODEL_FAMILY_TO_DEFAULTS = {
    "gpt-5": {
        "planner_model": "gpt-5",
        "writer_model": "gpt-5",
        "researcher_model": "gpt-5",
        "research_model": "gpt-5",
        "compression_model": "gpt-5",
        "final_report_model": "gpt-5"
    },
    "gemini-3-pro-preview": {
        "planner_model": "gemini-3-pro-preview",
        "writer_model": "gemini-3-pro-preview",
        "researcher_model": "gemini-3-pro-preview",
        "research_model": "gemini-3-pro-preview",
        "compression_model": "gemini-3-pro-preview",
        "final_report_model": "gemini-3-pro-preview"
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
