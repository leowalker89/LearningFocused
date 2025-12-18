from langchain_core.tools import tool
from typing import List, Dict, Optional
from src.database.chroma_manager import query_segments, query_summaries
from src.database.neo4j_manager import run_cypher_query, get_graph_schema

@tool
def search_knowledge_base(query: str, max_segments: int = 5, max_summaries: int = 3) -> str:
    """
    Search the educational podcast transcripts and summaries for specific information.
    Use this to find quotes, definitions, or discussions on specific topics.
    
    Args:
        query: The search query string.
        max_segments: Maximum number of transcript segments to return (default: 5).
        max_summaries: Maximum number of episode summaries to return (default: 3).
        
    Returns:
        A formatted string of relevant segments and summaries.
    """
    # Query both collections (serial execution is fast enough for local Chroma)
    segments = query_segments(query, k=max_segments)
    summaries = query_summaries(query, k=max_summaries)
    
    results = []
    
    if summaries:
        results.append("=== High-level Summaries (episodes + articles) ===")
        for doc in summaries:
            doc_type = doc.metadata.get("type", "unknown")
            if str(doc_type).startswith("article_"):
                title = doc.metadata.get("title", "Unknown Article")
                results.append(
                    f"Article: {title}\nType: {doc_type}\nContent: {doc.page_content}\n"
                )
            else:
                results.append(
                    f"Source: {doc.metadata.get('group_title', 'Unknown')}\nType: {doc_type}\nContent: {doc.page_content}\n"
                )
    
    if segments:
        results.append("\n=== Detailed Segments (transcripts + articles) ===")
        for doc in segments:
            doc_type = doc.metadata.get("type", "unknown")
            if doc_type == "article_text":
                results.append(
                    f"Article: {doc.metadata.get('title', 'Unknown')}\nContent: {doc.page_content}\n"
                )
            else:
                results.append(
                    f"Episode: {doc.metadata.get('title', 'Unknown')}\nTopic: {doc.metadata.get('topic', 'General')}\nContent: {doc.page_content}\n"
                )
            
    if not results:
        return "No relevant information found in the knowledge base."
        
    return "\n".join(results)

@tool
def query_knowledge_graph(query: str) -> str:
    """
    Run a Cypher query against the Neo4j knowledge graph.
    Use this to find structural relationships, e.g., "Who criticized Alpha parenting?" or "What concepts are related to Agency?".
    
    IMPORTANT:
    1. Check the schema with inspect_graph_schema ONLY if you do not already have it or are unsure of the structure.
    2. ALWAYS use a LIMIT clause (e.g., LIMIT 20) to prevent overwhelming response sizes.
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
