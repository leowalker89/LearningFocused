import os
import argparse
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Constants (matching create_embeddings.py)
CHROMA_PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "education_knowledge_engine"

def inspect_chroma(query: Optional[str] = None):
    print(f"\n=== ChromaDB Inspection ({CHROMA_PERSIST_DIRECTORY}) ===\n")
    
    if not os.path.exists(CHROMA_PERSIST_DIRECTORY):
        print(f"Error: Directory {CHROMA_PERSIST_DIRECTORY} not found. Have you run create_embeddings.py?")
        return

    try:
        # Initialize vector store connection
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        
        # 1. Collection Stats
        # There isn't a direct "count" method in the LangChain wrapper that's always consistent, 
        # but accessing the underlying collection usually works.
        # Note: LangChain's Chroma wrapper might vary by version, but ._collection is standard Chroma client.
        try:
            count = vector_store._collection.count()
            print(f"Total Documents in Collection '{COLLECTION_NAME}': {count}")
        except Exception as e:
            print(f"Could not retrieve exact count: {e}")

        if query:
            print(f"\n--- Similarity Search for: '{query}' ---")
            results = vector_store.similarity_search_with_score(query, k=3)
            
            if not results:
                print("No results found.")
            
            for i, (doc, score) in enumerate(results):
                print(f"\nResult {i+1} (Score: {score:.4f}):")
                print(f"Metadata: {doc.metadata}")
                # Print first 200 chars of content
                content_preview = doc.page_content.replace('\n', ' ')[:200]
                print(f"Content: {content_preview}...")
        else:
            # 2. Peek at Data (if no query provided)
            print("\n--- Random Sample (First 1 items) ---")
            # We can't easily "peek" random items via LangChain interface without a query,
            # but we can do a dummy search or use the underlying collection get().
            try:
                peek_data = vector_store._collection.get(limit=1)
                if peek_data and peek_data['ids']:
                    print(f"ID: {peek_data['ids'][0]}")
                    print(f"Metadata: {peek_data['metadatas'][0]}")
                    content_preview = peek_data['documents'][0].replace('\n', ' ')[:200]
                    print(f"Content: {content_preview}...")
                else:
                    print("Collection appears empty.")
            except Exception as e:
                print(f"Could not peek at data: {e}")

    except Exception as e:
        print(f"Failed to inspect ChromaDB: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ChromaDB vector store.")
    parser.add_argument("--query", type=str, help="Optional text query to test similarity search")
    args = parser.parse_args()
    
    inspect_chroma(args.query)

