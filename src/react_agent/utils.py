"""React Agent Utils - Helper functions for model creation and tool management.

Purpose:
    Utility functions for creating chat models, resolving API keys, and building tool legends.
"""

import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from src.react_agent.configuration import Configuration
from src.llm.factory import create_chat_model as _create_chat_model_shared

# Load environment variables from .env file
load_dotenv()


def _resolve_registry_entry(model_name: str, config: Any) -> Dict[str, str]:
    """Resolve model registry entry for a given model name.
    
    Args:
        model_name: Name of the model
        config: Configuration object or RunnableConfig
        
    Returns:
        Dictionary with provider and env_var keys, or empty dict if not found
    """
    cfg = Configuration.from_runnable_config(config)
    registry = getattr(cfg, "model_registry", {}) or {}
    return registry.get(model_name, {})


def get_api_key_for_model(model_name: str, config: Any) -> Optional[str]:
    """Get API key for a model from environment variables."""
    entry = _resolve_registry_entry(model_name, config)
    env_var = entry.get("env_var")
    return os.getenv(env_var) if env_var else None


def get_model_provider(model_name: str) -> Optional[str]:
    """Infer provider from model name."""
    name = (model_name or "").lower()
    if "gemini" in name:
        return "google_genai"
    if "claude" in name:
        return "anthropic"
    if "fireworks" in name or name.startswith("fireworks-ai/"):
        return "fireworks"
    if name.startswith("gpt") or name.startswith("o1"):
        return "openai"
    return None


def create_chat_model(
    model_name: str,
    config: Any = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
) -> Any:
    """Create chat model instance based on model name.

    This delegates to the shared `src.llm.factory` so pipelines + agents share one
    provider selection and env var validation implementation.
    """
    cfg = Configuration.from_runnable_config(config)
    registry_entry = _resolve_registry_entry(model_name, config)
    
    # Enable LangSmith tracing if available
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "react_agent")
    
    max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
    temperature = temperature if temperature is not None else cfg.temperature
    timeout = timeout if timeout is not None else cfg.timeout

    # Pass the agent's registry through so custom additions continue to work.
    return _create_chat_model_shared(
        model_name=model_name,
        registry=cfg.model_registry,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def build_tool_legend(tools: List[Any]) -> str:
    """Return a concise legend of available tools."""
    if not tools:
        return "Tools available: none."
    entries = []
    for tool in tools:
        name = getattr(tool, "name", None) or tool.get("name", "unknown")  # type: ignore[attr-defined]
        desc = getattr(tool, "description", "") or getattr(tool, "__doc__", "") or ""
        desc = desc.strip().replace("\n", " ")
        entries.append(f"- {name}: {desc}" if desc else f"- {name}")
    return "Tools available:\n" + "\n".join(entries)

