## Deep Research Agent Overview

This folder hosts the multi-node LangGraph “deep research” agent that coordinates clarification, planning, tool-using research workers, and report writing. Use this as a reference when designing a simpler React-style agent.

- `graph.py` — Main LangGraph; nodes for clarification, brief generation, supervisor delegation, researcher loop, compression, and final report. Binds tools and injects a dynamic tool legend so prompts know about Chroma/Neo4j access.
- `prompts.py` — System/human prompt templates for clarification, research planning, researcher, compression, and final report (with tool legend placeholders).
- `state.py` — TypedDicts/Pydantic models for agent state, tool schemas (`ConductResearch`, `ResearchComplete`), and structured outputs.
- `configuration.py` — Config dataclass (model names, token limits, iteration caps, tool registry mapping to env vars).
- `tools.py` — Actual tools exposed to the agent: `search_knowledge_base` (Chroma vector DB over transcripts and summaries), `query_knowledge_graph` (Neo4j Cypher), `inspect_graph_schema`.
- `utils.py` — Helpers for model config, API key lookup, token-limit handling, and `build_tool_legend` for prompt awareness of available tools.
- `chat_cli.py` — Interactive CLI entrypoint (loads `.env`, streams graph events, colorized output).
- `run.py` — One-shot CLI entrypoint for a single query (non-interactive).

Related
- For a lighter-weight Q&A loop over the same tools/stores, see `src/react_agent/`.

