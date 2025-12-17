"""
Pipeline package.

- `src.pipeline.audio`: canonical audio/podcast pipeline implementation
- `src.pipeline.substack`: scaffold for Substack/article pipeline implementation

All audio implementations live under `src.pipeline.audio`.
"""

from src.pipeline import audio, substack

__all__ = ["audio", "substack"]


