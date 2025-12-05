from langchain_core.tools import tool
from typing import List, Dict, Optional
from src.database.chroma_manager import query_segments, query_summaries
from src.database.neo4j_manager import run_cypher_query, get_graph_schema

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the educational podcast transcripts and summaries for specific information.
    Use this to find quotes, definitions, or discussions on specific topics.
    Returns a formatted string of relevant segments and summaries.
    """
    # Query both collections (serial execution is fast enough for local Chroma)
    segments = query_segments(query, k=5)
    summaries = query_summaries(query, k=3)
    
    results = []
    
    if summaries:
        results.append("=== Episode Summaries ===")
        for doc in summaries:
            results.append(f"Source: {doc.metadata.get('group_title', 'Unknown')}\nContent: {doc.page_content}\n")
    
    if segments:
        results.append("\n=== Transcript Segments ===")
        for doc in segments:
            results.append(f"Episode: {doc.metadata.get('title', 'Unknown')}\nTopic: {doc.metadata.get('topic', 'General')}\nContent: {doc.page_content}\n")
            
    if not results:
        return "No relevant information found in the knowledge base."
        
    return "\n".join(results)

@tool
def query_knowledge_graph(query: str) -> str:
    """
    Run a Cypher query against the Neo4j knowledge graph.
    Use this to find structural relationships, e.g., "Who criticized Alpha parenting?" or "What concepts are related to Agency?".
    Always check the schema with inspect_graph_schema first if you are unsure of the structure.
    """
    try:
        results = run_cypher_query(query)
        if not results:
            return "No results found."
        return str(results)
    except Exception as e:
        return f"Query failed: {e}"

@tool
def inspect_graph_schema() -> str:
    """
    Get the schema of the Neo4j knowledge graph.
    Use this before writing Cypher queries to understand the available Node labels and Relationship types.
    """
    return get_graph_schema()

# Export list of tools
tools = [search_knowledge_base, query_knowledge_graph, inspect_graph_schema]
