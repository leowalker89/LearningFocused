"""React Agent Configuration - Settings and defaults.

Purpose:
    Dataclass holding configurable parameters for the react agent.
    Much simpler than deep_research_agent since we have fewer moving parts.
    
Key Fields:
    - model: LLM to use (default: gpt-5-mini for speed)
    - max_tokens: Token limit for responses
    - max_iterations: Safety limit on ReAct loops (prevent runaway)
    - temperature: Creativity vs determinism
    
Usage:
    config = Configuration.from_runnable_config(runnable_config)
"""

# TODO: Define Configuration dataclass

