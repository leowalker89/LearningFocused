# React Agent

A lightweight LangGraph ReAct agent for querying the Future of Education podcast knowledge base.

## Overview

This agent implements the **ReAct pattern** (Reason + Act) using LangGraph's `create_react_agent` helper. It's designed to be a simpler, faster alternative to `deep_research_agent` for quick Q&A over the podcast data.

**When to use this vs deep_research_agent:**
- **React Agent**: Quick questions, single-topic lookups, testing tools
- **Deep Research Agent**: Complex multi-part research, comprehensive reports, cross-episode synthesis

## File Structure

| File | Purpose |
|------|---------|
| `graph.py` | Main agent graph using `create_react_agent` |
| `prompts.py` | System prompt guiding agent behavior |
| `configuration.py` | Settings (model, max iterations, etc.) |
| `tools.py` | Re-exports shared tools (Chroma, Neo4j) |
| `chat_cli.py` | Interactive terminal interface |
| `__init__.py` | Package marker |

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  ReAct Agent (create_react_agent)   │
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

## Usage

```bash
# Interactive chat
uv run python -m src.react_agent.chat_cli

# Programmatic
from src.react_agent.graph import react_agent
from langchain_core.messages import HumanMessage

result = await react_agent.ainvoke({
    "messages": [HumanMessage(content="What is Two Hour Learning?")]
})
```

## Comparison with deep_research_agent

| Aspect | React Agent | Deep Research Agent |
|--------|-------------|---------------------|
| Graph complexity | Single node (ReAct loop) | Multi-node (clarify → brief → supervisor → researchers → report) |
| Use case | Quick Q&A | In-depth research |
| Tool access | Direct | Delegated via researchers |
| Output | Single response | Structured report |
| State | Minimal (messages) | Rich (brief, notes, iterations) |

## Tools Available

The agent has access to these tools (re-exported from `deep_research_agent.tools`):

1. **search_knowledge_base(query)** - Semantic search over podcast transcripts and episode summaries in ChromaDB
2. **query_knowledge_graph(query)** - Run Cypher queries against the Neo4j knowledge graph
3. **inspect_graph_schema()** - Get the Neo4j schema (nodes, relationships) to help write Cypher

## Configuration

Default settings in `configuration.py`:
- Model: `gpt-5-mini` (fast, cost-effective)
- Max iterations: 10 (safety limit)
- Temperature: 0 (deterministic)

Override via `RunnableConfig`:
```python
config = {"configurable": {"model": "gpt-5", "max_iterations": 5}}
result = await react_agent.ainvoke(input, config=config)
```

