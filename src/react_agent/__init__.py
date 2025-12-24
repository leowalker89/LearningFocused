"""React Agent - A simple LangChain ReAct agent for querying the podcast knowledge base.

This agent uses the tools (Chroma vector search, Neo4j knowledge graph) 
to answer questions about the Future of Education podcast episodes.

The agent includes conversation memory via LangGraph's MemorySaver checkpointer.
Pass a thread_id in config to enable memory persistence within a session:

    result = await react_agent.ainvoke(
        {"messages": [HumanMessage(content="...")]},
        config={"configurable": {"thread_id": "my-thread"}}
    )
"""

from src.react_agent.graph import react_agent, get_react_agent, memory
from src.react_agent.configuration import Configuration
from src.react_agent.tools import (
    tools,
    search_knowledge_base,
    search_knowledge_base_structured,
    query_knowledge_graph,
    inspect_graph_schema,
)

__all__ = [
    "react_agent",
    "get_react_agent",
    "memory",
    "Configuration",
    "tools",
    "search_knowledge_base",
    "search_knowledge_base_structured",
    "query_knowledge_graph",
    "inspect_graph_schema",
]
