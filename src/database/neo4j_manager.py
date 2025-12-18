"""
Neo4j knowledge graph builder.

LangSmith tracing
To enable traces for the LLM transform + Neo4j write steps, set (e.g., in `.env`):
- LANGCHAIN_TRACING_V2=true
- LANGCHAIN_API_KEY=...            # LangSmith API key
- LANGCHAIN_PROJECT=learningfocused-neo4j

Optional:
- LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
- NEO4J_GRAPH_LLM_TIMEOUT_SECONDS=120  # fail fast instead of hanging for many minutes
"""

import os
import json
import logging
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, cast
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument as N4jGraphDocument, Node as N4jNode, Relationship as N4jRelationship
from langchain_community.graphs.graph_document import GraphDocument as CommunityGraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

# LangSmith tracing (optional at runtime; enabled via env vars).
# If not configured, the decorator is a no-op and adds near-zero overhead.
try:
    from langsmith import traceable  # type: ignore
except Exception:  # pragma: no cover
    def traceable(*_args: Any, **_kwargs: Any):  # type: ignore
        def _decorator(fn):  # type: ignore
            return fn

        return _decorator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError("Missing Neo4j configuration. Please check your .env file.")

# Allowed schema
ALLOWED_NODES = ["Person", "Concept", "Organization", "Tool", "Event"]
ALLOWED_RELATIONSHIPS = [
    "ADVOCATES_FOR", 
    "CRITICIZES", 
    "IMPLEMENTS", 
    "ENABLES", 
    "FOUNDED", 
    "DISCUSSED"
]

def get_graph() -> Neo4jGraph:
    """Initialize and return the Neo4jGraph connection."""
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )

def run_cypher_query(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Execute a Cypher query against the knowledge graph."""
    graph = get_graph()
    try:
        # Neo4jGraph typing expects dict[Any, Any]; normalize Optional/typed dict to satisfy type checkers.
        return graph.query(query, params=cast(Dict[Any, Any], (params or {})))
    except Exception as e:
        logger.error(f"Cypher query failed: {e}")
        return []

def get_graph_schema() -> str:
    """Return the schema of the knowledge graph."""
    graph = get_graph()
    return graph.schema

def get_graph_transformer() -> LLMGraphTransformer:
    """Initialize the LLMGraphTransformer with Gemini."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        # Avoid "hang forever" behavior; override with env var if you want longer.
        timeout=int(os.getenv("NEO4J_GRAPH_LLM_TIMEOUT_SECONDS", "120")),
    )
    
    return LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS
    )

def load_segmented_transcripts(directory: Path) -> List[Document]:
    """Load all segmented transcripts and convert to Documents."""
    documents = []
    files = list(directory.glob("*_segmented.json"))
    
    if not files:
        logger.warning(f"No segmented transcripts found in {directory}")
        return []

    logger.info(f"Found {len(files)} segmented transcripts")
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            episode_id = data.get("episode_id", "Unknown")
            title = data.get("title", "Unknown")
            
            for segment in data.get("segments", []):
                # Construct metadata
                metadata = {
                    "source": str(file_path),
                    "episode_id": episode_id,
                    "title": title,
                    "topic": segment.get("topic"),
                    "start_time": segment.get("start_time"),
                    "end_time": segment.get("end_time"),
                    "speakers": segment.get("speakers", [])
                }
                
                # Use content or summary + content
                text_content = segment.get("content", "")
                if not text_content:
                    continue
                    
                doc = Document(
                    page_content=text_content,
                    metadata=metadata
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            
    return documents

def convert_to_neo4j_graph_documents(docs: List[Any]) -> List[N4jGraphDocument]:
    """
    Convert community/experimental GraphDocuments to langchain_neo4j GraphDocuments.
    Needed because LLMGraphTransformer produces the former, but Neo4jGraph expects the latter.
    """
    neo4j_docs = []
    for doc in docs:
        if isinstance(doc, N4jGraphDocument):
            neo4j_docs.append(doc)
            continue
            
        # Manually convert nodes
        nodes = [
            N4jNode(
                id=node.id, 
                type=node.type, 
                properties=node.properties
            ) for node in doc.nodes
        ]
        
        # Manually convert relationships
        # Note: We need to map the source/target nodes to the new Node objects we just created to be safe,
        # but the Relationship constructor usually takes Node objects. 
        # We can recreate them or assume equality check passes. 
        # Safer to just recreate the relationship structure.
        relationships = [
            N4jRelationship(
                source=N4jNode(id=rel.source.id, type=rel.source.type),
                target=N4jNode(id=rel.target.id, type=rel.target.type),
                type=rel.type,
                properties=rel.properties
            ) for rel in doc.relationships
        ]
        
        neo4j_docs.append(N4jGraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=doc.source
        ))
    return neo4j_docs


@traceable(name="neo4j_llm_graph_transform")
def _trace_llm_transform(llm_transformer: LLMGraphTransformer, batch: List[Document]) -> List[Any]:
    """LLM graph transformation for a batch (LangSmith-traced)."""
    return llm_transformer.convert_to_graph_documents(batch)

def process_and_store(documents: List[Document]):
    """Process documents through LLMGraphTransformer and store in Neo4j."""
    
    # Initialize Neo4j connection
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        logger.info("Connected to Neo4j")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        return

    llm_transformer = get_graph_transformer()
    
    total_docs = len(documents)
    logger.info(f"Processing {total_docs} documents...")
    
    # Process in batches to avoid memory issues and provide progress
    batch_size = 5
    total_batches = (total_docs + batch_size - 1) // batch_size
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Processing batch {batch_num}/{total_batches}")
        
        try:
            t0 = time.perf_counter()
            graph_documents_raw = _trace_llm_transform(llm_transformer, batch)
            t1 = time.perf_counter()
            logger.info("Batch %s LLM transform time: %.2fs", batch_num, t1 - t0)
            
            # Convert to correct type if necessary
            graph_documents = convert_to_neo4j_graph_documents(graph_documents_raw)
            
            # Add to Neo4j
            if graph_documents:
                t2 = time.perf_counter()
                graph.add_graph_documents(
                    graph_documents,
                    baseEntityLabel=True,
                    include_source=True,
                )
                t3 = time.perf_counter()
                logger.info("Batch %s Neo4j write time: %.2fs", batch_num, t3 - t2)
                logger.info(f"Added {len(graph_documents)} graph documents to Neo4j")
            
        except Exception as e:
            logger.error(f"Error processing batch starting at index {i}: {e}")

def update_knowledge_graph():
    """Main function to update the Neo4j knowledge graph."""
    from src.config import SEGMENTED_DIR
    
    if not SEGMENTED_DIR.exists():
        logger.error(f"Directory not found: {SEGMENTED_DIR}")
        return
        
    docs = load_segmented_transcripts(SEGMENTED_DIR)

    # Add a run identifier so Neo4j writes can be correlated back to this execution and
    # to the LangSmith project used for tracing.
    ingest_run_id = str(uuid.uuid4())
    langsmith_project = os.getenv("LANGCHAIN_PROJECT")
    for d in docs:
        d.metadata["ingest_run_id"] = ingest_run_id
        if langsmith_project:
            d.metadata["langsmith_project"] = langsmith_project
    logger.info("Neo4j ingest_run_id=%s langsmith_project=%s", ingest_run_id, langsmith_project)
    
    if docs:
        process_and_store(docs)
    else:
        logger.info("No documents to process.")

if __name__ == "__main__":
    update_knowledge_graph()
