# React Agent

A lightweight LangChain ReAct agent for querying the Future of Education podcast knowledge base.

## Overview

This agent implements the **ReAct pattern** (Reason → Act/tool call → Observe → repeat) using `langchain.agents.create_agent`. It’s intended for fast, tool-grounded Q&A over the local stores (Chroma + Neo4j).

**When to use this vs deep_research_agent:**
- **React Agent**: Quick questions, single-topic lookups, testing tools
- **Deep Research Agent**: Complex multi-part research, comprehensive reports, cross-episode synthesis

## File Structure

| File | Purpose |
|------|---------|
| `graph.py` | Main agent using `langchain.agents.create_agent` |
| `prompts.py` | System prompt guiding agent behavior |
| `configuration.py` | Settings (model, max iterations, etc.) with model registry |
| `utils.py` | Helper functions for model creation and tool management |
| `tools.py` | Re-exports shared tools (Chroma, Neo4j) |
| `chat_cli.py` | Interactive terminal interface |
| `__init__.py` | Package exports |

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  ReAct Agent (create_agent)         │
│  ┌─────────────────────────────┐    │
│  │ 1. Reason: What do I need?  │    │
│  │ 2. Act: Call a tool         │◄───┼─── Tools:
│  │ 3. Observe: Read result     │    │    • search_knowledge_base (Chroma)
│  │ 4. Repeat or Answer         │    │    • query_knowledge_graph (Neo4j)
│  └─────────────────────────────┘    │    • inspect_graph_schema
└─────────────────────────────────────┘
    │
    ▼
  Answer
```

## Implementation Details

### Dynamic Schema Loading
The agent dynamically loads the Neo4j graph schema at startup (in `prompts.py`) to minimize tool calls. This allows the agent to write Cypher queries immediately without needing an initial `inspect_graph_schema` call, reducing latency and cost.

### Multi-Provider Support

Model creation is delegated to the shared factory in `src/llm/factory.py`, which supports multiple providers and validates the required API key env var for the chosen model.

Model names accept **stable aliases** (recommended) that are mapped to provider-specific IDs under the hood:
- **OpenAI**: `gpt-5`, `gpt-5-mini`
- **Anthropic**: `claude-sonnet-4-20250514` (alias), plus internal IDs like `claude-sonnet-4-5`
- **Google Gemini**: `gemini-flash-latest` (alias)

### Model Configuration

Models are configured via the `Configuration` class which supports:
- Model selection (default: `gemini-3-flash-preview`)
- Max tokens (default: 4000)
- Temperature (default: 0.0)
- Timeout (default: 30 seconds)
- Max iterations (default: 25)

Configuration can be overridden at runtime via `RunnableConfig`.

## Usage

### Interactive Chat

```bash
# Interactive chat
uv run python -m src.react_agent.chat_cli
```

### Programmatic Usage

```python
from src.react_agent.graph import react_agent, get_react_agent
from langchain_core.messages import HumanMessage

# Use default agent
result = await react_agent.ainvoke({
    "messages": [HumanMessage(content="What is Two Hour Learning?")]
})

# Create agent with custom config
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    configurable={
        "model": "claude-sonnet-4-5",
        "max_tokens": 8000,
        "temperature": 0.1
    }
)
custom_agent = get_react_agent(config)
result = await custom_agent.ainvoke({
    "messages": [HumanMessage(content="What is Two Hour Learning?")]
})
```

## Comparison with deep_research_agent

| Aspect | React Agent | Deep Research Agent |
|--------|-------------|---------------------|
| Framework | LangChain `create_agent` | LangGraph custom graph |
| Graph complexity | Single ReAct loop | Multi-node (clarify → brief → supervisor → researchers → report) |
| Use case | Quick Q&A | In-depth research |
| Tool access | Direct | Delegated via researchers |
| Output | Single response | Structured report |
| State | Minimal (messages) | Rich (brief, notes, iterations) |
| Model creation | Provider-specific classes | `init_chat_model` with configurable fields |

## Tools Available

The agent has access to these tools (re-exported from `deep_research_agent.tools`):

1. **search_knowledge_base(query)** - Semantic search over podcast transcripts and episode summaries in ChromaDB
2. **query_knowledge_graph(query)** - Run Cypher queries against the Neo4j knowledge graph
3. **inspect_graph_schema()** - Get the Neo4j schema (nodes, relationships) to help write Cypher

## Configuration

Default settings in `configuration.py`:
- Model: `gemini-3-flash-preview`
- Max iterations: 25 (safety limit)
- Max tokens: 4000
- Temperature: 0 (deterministic)
- Timeout: 30 seconds

Override via `RunnableConfig`:
```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    configurable={
        "model": "gpt-5",
        "max_iterations": 5,
        "max_tokens": 8000,
        "temperature": 0.1
    }
)
result = await react_agent.ainvoke(input, config=config)
```

Or set environment variable:
```bash
export REACT_AGENT_DEFAULT_MODEL="claude-sonnet-4-20250514"
```

## Dependencies

Dependencies are managed in the repo’s `pyproject.toml`. The key runtime pieces are LangChain (+ provider integrations) and `python-dotenv`.
