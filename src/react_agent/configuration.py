"""React Agent Configuration - Settings and defaults.

Usage:
    config = Configuration.from_model("gpt-5")  # Quick model switch
    config = Configuration.from_dict({"model": "gpt-5", "max_tokens": 8000})
    config = Configuration.from_runnable_config(runnable_config)  # LangChain compatible
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, cast
from langchain_core.runnables import RunnableConfig

# Default registry mapping model names to their provider and required API key environment variable.
DEFAULT_MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    # Google (primary)
    "gemini-3-pro-preview": {"provider": "google_genai", "env_var": "GOOGLE_API_KEY"},
    "gemini-3-flash-preview": {"provider": "google_genai", "env_var": "GOOGLE_API_KEY"},

    # Anthropic (secondary)
    # Prefer stable aliases (always latest)
    "claude-sonnet-4-5": {"provider": "anthropic", "env_var": "ANTHROPIC_API_KEY"},
    "claude-haiku-4-5": {"provider": "anthropic", "env_var": "ANTHROPIC_API_KEY"},
    "claude-opus-4-5": {"provider": "anthropic", "env_var": "ANTHROPIC_API_KEY"},

    # OpenAI (third)
    "gpt-5.2": {"provider": "openai", "env_var": "OPENAI_API_KEY"},
    "gpt-5.1": {"provider": "openai", "env_var": "OPENAI_API_KEY"},
    "gpt-5.1-mini": {"provider": "openai", "env_var": "OPENAI_API_KEY"},

    # Fireworks (fourth)
    "accounts/fireworks/models/deepseek-v3p2": {"provider": "fireworks", "env_var": "FIREWORKS_API_KEY"},
    "accounts/fireworks/models/kimi-k2-thinking": {"provider": "fireworks", "env_var": "FIREWORKS_API_KEY"},
    "accounts/fireworks/models/gpt-oss-120b": {"provider": "fireworks", "env_var": "FIREWORKS_API_KEY"},
}

# Configurable default model - set via env var or use default
DEFAULT_MODEL = os.environ.get("REACT_AGENT_DEFAULT_MODEL", "gemini-3-flash-preview")
DEFAULT_MAX_ITERATIONS = int(os.environ.get("REACT_AGENT_MAX_ITERATIONS", "25"))


@dataclass
class Configuration:
    """Configuration for the React agent."""
    model: str = DEFAULT_MODEL
    # LangGraph uses this as recursion_limit; 10 can be too low for tool + answer loops.
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_tokens: int = 4000
    temperature: float = 0.0
    timeout: int = 30
    model_registry: Dict[str, Dict[str, str]] = field(default_factory=lambda: DEFAULT_MODEL_REGISTRY.copy())
    
    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create from LangChain RunnableConfig."""
        config = config or {}
        configurable = config.get("configurable") or {}
        return cls(**{k: v for k, v in configurable.items() if k in cls.__annotations__})
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Union[str, int, float]]) -> "Configuration":
        """Create from dictionary. Useful for quick testing."""
        valid_fields = {k: v for k, v in config_dict.items() if k in cls.__annotations__}
        return cls(**valid_fields)  # type: ignore[arg-type]
    
    @classmethod
    def from_model(cls, model_name: str, **kwargs: Union[str, int, float]) -> "Configuration":
        """Create with specific model name. Convenient for quick model switching."""
        return cls(model=model_name, **kwargs)  # type: ignore[arg-type]
    
    @classmethod
    def from_any(
        cls, 
        config: Optional[Union[RunnableConfig, Dict[str, Union[str, int, float]], str, "Configuration"]] = None
    ) -> "Configuration":
        """Flexible factory: accepts RunnableConfig, dict, str (model name), Configuration, or None."""
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if isinstance(config, str):
            return cls.from_model(config)
        if isinstance(config, dict):
            dict_config: Dict[str, Union[str, int, float]] = cast(Dict[str, Union[str, int, float]], config)
            return cls.from_dict(dict_config)
        # Assume it's a RunnableConfig
        return cls.from_runnable_config(cast(RunnableConfig, config))
