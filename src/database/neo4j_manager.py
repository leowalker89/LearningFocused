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

def _parse_csv_env(name: str) -> list[str] | None:
    """Parse a comma-separated env var into a list of non-empty strings."""
    raw = os.getenv(name)
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]
    return parts or None


# Allowed schema (defaults; can be overridden by env vars below)
_ALLOWED_NODES_CORE = ["Person", "Concept", "Organization", "Tool", "Event"]
_ALLOWED_RELATIONSHIPS_CORE = [
    "ADVOCATES_FOR",
    "CRITICIZES",
    "IMPLEMENTS",
    "ENABLES",
    "FOUNDED",
    "DISCUSSED",
]

# A deliberately modest expansion: adds evidence/causality/comparison semantics that appear often in
# education research and practitioner writing.
_ALLOWED_RELATIONSHIPS_EXTENDED = _ALLOWED_RELATIONSHIPS_CORE + [
    "SUPPORTS",
    "CONTRADICTS",
    "EVIDENCE_FOR",
    "CAUSES",
    "INFLUENCES",
    "CORRELATES_WITH",
    "MEASURES",
    "RECOMMENDS",
    "COMPARED_TO",
    "APPLIES_TO",
]

def resolve_allowed_schema(schema_profile: str | None = None) -> tuple[list[str], list[str], str]:
    """Resolve allowed nodes/relationships for the graph transformer.

    Precedence:
    - explicit function argument (schema_profile) for programmatic/CLI control
    - env var `NEO4J_SCHEMA_PROFILE` (deployment control)
    - default: "core"

    Optional env overrides:
    - `NEO4J_ALLOWED_NODES` (comma-separated) replaces node defaults
    - `NEO4J_ALLOWED_RELATIONSHIPS` (comma-separated) replaces relationship defaults
    """
    # Default to "extended" so the graph is useful for learning science + research expansion
    # without requiring .env configuration.
    env_profile = os.getenv("NEO4J_SCHEMA_PROFILE", "extended")
    profile = (schema_profile if schema_profile is not None else env_profile).strip().lower()
    if profile not in {"core", "extended"}:
        profile = "extended"

    allowed_nodes = _parse_csv_env("NEO4J_ALLOWED_NODES") or _ALLOWED_NODES_CORE
    rel_defaults = _ALLOWED_RELATIONSHIPS_EXTENDED if profile == "extended" else _ALLOWED_RELATIONSHIPS_CORE
    allowed_relationships = _parse_csv_env("NEO4J_ALLOWED_RELATIONSHIPS") or rel_defaults
    return allowed_nodes, allowed_relationships, profile


def build_graph_extraction_spec(*, allowed_nodes: list[str], allowed_relationships: list[str]) -> str:
    """Build explicit extraction instructions to prepend to each Document."""
    return (
        "You are extracting a knowledge graph from educational content.\n\n"
        "Extract:\n"
        f"- Nodes limited to these types: {', '.join(allowed_nodes)}\n"
        "- Relationships limited to these types:\n"
        + "\n".join(f"  - {r}" for r in allowed_relationships)
        + "\n\n"
        "Rules:\n"
        "- Only create nodes/edges that are explicitly supported by the text (no guessing).\n"
        '- Prefer canonical, human-readable names for entities (e.g., "Alpha School", not abbreviations unless used in text).\n'
        "- If a relationship is ambiguous, omit it rather than inventing it.\n"
        '- If the text is mostly descriptive with no clear relationship, use "DISCUSSED" where appropriate.\n'
    )


def _wrap_for_graph_extraction(doc: Document, *, graph_extraction_spec: str) -> Document:
    """Prepend graph extraction instructions and basic provenance to the document text."""
    title = doc.metadata.get("title") or doc.metadata.get("doc_id") or doc.metadata.get("episode_id")
    canonical_url = doc.metadata.get("canonical_url")
    source_type = doc.metadata.get("source_type")
    topic = doc.metadata.get("topic")

    provenance = []
    if source_type:
        provenance.append(f"SourceType: {source_type}")
    if title:
        provenance.append(f"Title: {title}")
    if canonical_url:
        provenance.append(f"URL: {canonical_url}")
    if topic:
        provenance.append(f"Topic: {topic}")

    preamble = graph_extraction_spec + "\n"
    if provenance:
        preamble += "Provenance:\n" + "\n".join(f"- {p}" for p in provenance) + "\n\n"

    return Document(page_content=preamble + doc.page_content, metadata=doc.metadata)

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

def get_graph_transformer(*, allowed_nodes: list[str], allowed_relationships: list[str]) -> LLMGraphTransformer:
    """Initialize the LLMGraphTransformer with Gemini."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        # Avoid "hang forever" behavior; override with env var if you want longer.
        timeout=int(os.getenv("NEO4J_GRAPH_LLM_TIMEOUT_SECONDS", "120")),
    )
    
    return LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
    )

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

def process_and_store(documents: List[Document], *, llm_transformer: LLMGraphTransformer):
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

def update_knowledge_graph(
    *,
    include_audio: bool = True,
    include_articles: bool = True,
    audio_limit: int | None = None,
    article_limit: int | None = None,
    schema_profile: str | None = None,
):
    """Main function to update the Neo4j knowledge graph."""
    from src.config import SEGMENTED_DIR, SUBSTACK_METADATA_DIR, SUBSTACK_TEXT_DIR, ensure_data_dirs

    ensure_data_dirs()
    
    if not SEGMENTED_DIR.exists():
        logger.error(f"Directory not found: {SEGMENTED_DIR}")
        return
        
    # Pipeline-owned document loaders
    from src.pipeline.audio.index_neo4j import collect_audio_graph_documents
    from src.pipeline.substack.index_neo4j import collect_substack_graph_documents

    docs: list[Document] = []
    if include_audio:
        docs.extend(collect_audio_graph_documents(SEGMENTED_DIR, limit=audio_limit))
    if include_articles and SUBSTACK_TEXT_DIR.exists() and SUBSTACK_METADATA_DIR.exists():
        docs.extend(
            collect_substack_graph_documents(
                SUBSTACK_TEXT_DIR,
                SUBSTACK_METADATA_DIR,
                limit=article_limit,
            )
        )

    allowed_nodes, allowed_relationships, profile = resolve_allowed_schema(schema_profile)
    graph_spec = build_graph_extraction_spec(
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
    )
    docs = [_wrap_for_graph_extraction(d, graph_extraction_spec=graph_spec) for d in docs]
    llm_transformer = get_graph_transformer(
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
    )
    logger.info("Neo4j schema_profile=%s allowed_nodes=%d allowed_relationships=%d", profile, len(allowed_nodes), len(allowed_relationships))

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
        process_and_store(docs, llm_transformer=llm_transformer)
    else:
        logger.info("No documents to process.")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build/update Neo4j knowledge graph from pipeline artifacts.")
    p.add_argument("--schema-profile", choices=["core", "extended"], default=None)
    p.add_argument("--include-audio", action="store_true", help="Include audio transcript segments (default: on).")
    p.add_argument("--no-include-audio", action="store_true", help="Exclude audio transcript segments.")
    p.add_argument("--include-articles", action="store_true", help="Include Substack article text docs (default: on).")
    p.add_argument("--no-include-articles", action="store_true", help="Exclude Substack article text docs.")
    p.add_argument("--audio-limit", type=int, default=None, help="Max number of audio Documents to transform/write.")
    p.add_argument("--article-limit", type=int, default=None, help="Max number of article Documents to transform/write.")
    args = p.parse_args()

    include_audio = not args.no_include_audio
    include_articles = not args.no_include_articles
    update_knowledge_graph(
        include_audio=include_audio,
        include_articles=include_articles,
        audio_limit=args.audio_limit,
        article_limit=args.article_limit,
        schema_profile=args.schema_profile,
    )
