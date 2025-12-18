import argparse
from typing import List, Dict, Any, Optional, cast
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Constants
from src.config import (
    CHROMA_DIR,
    ensure_data_dirs,
)

COLLECTION_NAME = "education_knowledge_engine"

def get_vector_store() -> Chroma:
    """Initialize and return the ChromaDB vector store."""
    ensure_data_dirs()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

def query_segments(
    query: str,
    k: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
    *,
    allowed_types: Optional[List[str]] = None,
) -> List[Document]:
    """
    Search for detailed content segments in the vector store.
    Defaults to transcript segments + short-form article text docs.
    
    Args:
        query: The search query string.
        k: Number of results to return.
        filter_metadata: Optional dictionary to filter by metadata (e.g., {'episode_id': '...'}).
    """
    vector_store = get_vector_store()
    
    types = allowed_types or ["transcript_segment", "article_text"]
    # Prefer server-side filtering, but fall back to client-side if the backend doesn't support $in.
    filter_dict: Dict[str, Any] = {"type": {"$in": types}}
    if filter_metadata:
        filter_dict.update(filter_metadata)

    try:
        return vector_store.similarity_search(query, k=k, filter=cast(Any, filter_dict))
    except Exception:
        docs = vector_store.similarity_search(query, k=k * 3)
        filtered = [d for d in docs if d.metadata.get("type") in types]
        if filter_metadata:
            def _match_meta(d: Document) -> bool:
                for mk, mv in filter_metadata.items():
                    if d.metadata.get(mk) != mv:
                        return False
                return True
            filtered = [d for d in filtered if _match_meta(d)]
        return filtered[:k]

def query_summaries(query: str, k: int = 5) -> List[Document]:
    """
    Search for episode and series summaries.
    Excludes transcript segments to focus on high-level content.
    """
    vector_store = get_vector_store()
    
    # Filter for anything that IS NOT a transcript_segment
    # Chroma/LangChain filter syntax for negation or multiple values can be backend-specific.
    # For robustness, we'll fetch results and filter, or assume the query steers the embedding.
    # But using the $ne operator if supported is better.
    try:
        return vector_store.similarity_search(
            query,
            k=k,
            filter=cast(Any, {"type": {"$ne": "transcript_segment"}}),
        )
    except Exception:
        # Fallback if $ne isn't supported by the version/backend: search all and filter client-side
        docs = vector_store.similarity_search(query, k=k * 2) # Fetch more to allow for filtering
        return [d for d in docs if d.metadata.get("type") != "transcript_segment"][:k]

def update_chroma_db(reset: bool = False):
    """Main function to update the ChromaDB vector store."""
    ensure_data_dirs()
    # Pipeline-owned indexers (kept out of this DB adapter file)
    from src.pipeline.audio.index_chroma import collect_audio_documents
    from src.pipeline.substack.index_chroma import collect_substack_documents

    # 1. Initialize Vector Store
    print(f"Initializing ChromaDB in {CHROMA_DIR}...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if reset and CHROMA_DIR.exists():
        print("Reset flag detected. Cleaning up existing DB.")
        # In a real app, you might use shutil.rmtree(CHROMA_DIR)
        
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    
    all_documents = []

    print("Collecting audio documents...")
    all_documents.extend(collect_audio_documents())

    print("Collecting Substack documents...")
    all_documents.extend(collect_substack_documents())
        
    # Batch Add to Chroma
    if all_documents:
        print(f"Adding {len(all_documents)} documents to ChromaDB...")
        # Add in batches to avoid hitting API limits if any
        batch_size = 100
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i + batch_size]
            vector_store.add_documents(documents=batch)
            print(f"  Indexed batch {i // batch_size + 1}/{(len(all_documents) + batch_size - 1) // batch_size}")
            
        print("Success! Embeddings generated.")
    else:
        print("No documents found to index.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for Knowledge Engine.")
    parser.add_argument("--reset", action="store_true", help="Delete existing ChromaDB collection before starting")
    args = parser.parse_args()
    
    update_chroma_db(args.reset)
