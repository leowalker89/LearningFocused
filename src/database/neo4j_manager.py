import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument as N4jGraphDocument, Node as N4jNode, Relationship as N4jRelationship
from langchain_community.graphs.graph_document import GraphDocument as CommunityGraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

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
        return graph.query(query, params=params)
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
        model="gemini-flash-latest", 
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
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
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
        
        try:
            # Convert to graph documents
            graph_documents_raw = llm_transformer.convert_to_graph_documents(batch)
            
            # Convert to correct type if necessary
            graph_documents = convert_to_neo4j_graph_documents(graph_documents_raw)
            
            # Add to Neo4j
            if graph_documents:
                graph.add_graph_documents(
                    graph_documents, 
                    baseEntityLabel=True, 
                    include_source=True
                )
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
    
    if docs:
        process_and_store(docs)
    else:
        logger.info("No documents to process.")

if __name__ == "__main__":
    update_knowledge_graph()
