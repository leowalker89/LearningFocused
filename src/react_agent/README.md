# React Agent

A lightweight LangChain ReAct agent for querying the Future of Education podcast knowledge base.

## Overview

This agent implements the **ReAct pattern** (Reason + Act) using LangChain's `create_agent` from `langchain.agents`. It's designed to be a simpler, faster alternative to `deep_research_agent` for quick Q&A over the podcast data.

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

### Multi-Provider Support

The agent supports multiple LLM providers through provider-specific chat model classes:
- **OpenAI**: `ChatOpenAI` (`gpt-5`, `gpt-5-mini`)
- **Anthropic**: `ChatAnthropic` (`claude-sonnet-4-5`, `claude-haiku-4-5`)
- **Google Gemini**: `ChatGoogleGenerativeAI` (`gemini-3-pro-preview`, `gemini-flash-latest`)
- **Fireworks AI**: `ChatFireworks` (`accounts/fireworks/models/llama4-maverick-instruct-basic`, `accounts/fireworks/models/qwen-3-235b-instruct`)

The `create_chat_model()` utility in `utils.py` automatically creates the correct model instance based on the model name and validates required API keys from the registry.

### Model Configuration

Models are configured via the `Configuration` class which supports:
- Model selection (default: `gpt-5-mini`)
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
- Model: `gpt-5-mini` (fast, cost-effective)
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
export REACT_AGENT_DEFAULT_MODEL="claude-sonnet-4-5"
```

## Dependencies

Required packages (most already in `pyproject.toml`):
- `langchain` - Core LangChain (includes `langchain.agents.create_agent`)
- `langchain-openai` - OpenAI support (`ChatOpenAI`)
- `langchain-google-genai` - Gemini support (`ChatGoogleGenerativeAI`)
- `langchain-anthropic` - Anthropic support (`ChatAnthropic`)
- `langchain-fireworks` - Fireworks AI support (`ChatFireworks`)
- `python-dotenv` - Environment variable management

