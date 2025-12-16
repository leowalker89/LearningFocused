"""React Agent - A simple LangChain ReAct agent for querying the podcast knowledge base.

This agent uses the tools (Chroma vector search, Neo4j knowledge graph) 
to answer questions about the Future of Education podcast episodes.
"""

from src.react_agent.graph import react_agent, get_react_agent
from src.react_agent.configuration import Configuration
from src.react_agent.tools import tools, search_knowledge_base, query_knowledge_graph, inspect_graph_schema

__all__ = [
    "react_agent",
    "get_react_agent",
    "Configuration",
    "tools",
    "search_knowledge_base",
    "query_knowledge_graph",
    "inspect_graph_schema",
]
