import os
import json
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Constants
CHROMA_PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "education_knowledge_engine"

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON content from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_summary_documents(summary_data: Dict[str, Any]) -> List[Document]:
    """
    Convert a 'Combined Summary' object into multiple granular Documents for embedding.
    
    Strategy: Multi-Vector Representation
    1. Overview Vector: Title + High Level Summary + Why Listen
    2. Takeaway Vectors: Individual Key Takeaways
    3. Motivation Vector: Explicit "Why Listen" vector
    """
    documents = []
    
    # Extract core metadata
    group_title = summary_data.get("group_title", "Unknown Series")
    episode_count = summary_data.get("episode_count", 1)
    topics = summary_data.get("generated_content", {}).get("topics", [])
    
    content = summary_data.get("generated_content", {})
    high_level_summary = content.get("high_level_summary", "")
    why_listen = content.get("why_listen", "")
    key_takeaways = content.get("key_takeaways", [])
    
    # 1. Overview Vector
    # Content: [Group Title] + [High Level Summary]
    overview_text = f"Series Title: {group_title}\n\nOverview: {high_level_summary}"
    
    documents.append(Document(
        page_content=overview_text,
        metadata={
            "type": "series_overview",
            "group_title": group_title,
            "episode_count": episode_count,
            "topics": ", ".join(topics) # Store as string for simple filtering, or keep list if DB supports
        }
    ))
    
    # 2. Motivation Vector
    # Content: [Group Title] + Why Listen: [Text]
    if why_listen:
        motivation_text = f"Series Title: {group_title}\n\nWhy Listen: {why_listen}"
        documents.append(Document(
            page_content=motivation_text,
            metadata={
                "type": "series_motivation",
                "group_title": group_title,
                "topics": ", ".join(topics)
            }
        ))
        
    # 3. Takeaway Vectors (Granular)
    # Content: [Group Title] + Key Takeaway: [Text]
    for i, takeaway in enumerate(key_takeaways):
        takeaway_text = f"Series Title: {group_title}\n\nKey Takeaway: {takeaway}"
        documents.append(Document(
            page_content=takeaway_text,
            metadata={
                "type": "key_takeaway",
                "group_title": group_title,
                "takeaway_index": i + 1,
                "topics": ", ".join(topics)
            }
        ))
        
    return documents

def create_transcript_documents(transcript_data: Dict[str, Any]) -> List[Document]:
    """
    Convert a 'Segmented Transcript' object into Documents.
    Each topic segment becomes one Document.
    """
    documents = []
    
    episode_id = transcript_data.get("episode_id", "Unknown ID")
    title = transcript_data.get("title", "Unknown Title")
    
    segments = transcript_data.get("segments", [])
    
    for segment in segments:
        topic = segment.get("topic", "General")
        content = segment.get("content", "")
        summary = segment.get("summary", "")
        speakers = ", ".join(segment.get("speakers", []))
        start_time = segment.get("start_time")
        end_time = segment.get("end_time")
        
        # Combined text for embedding: Topic + Summary + Full Content
        # We include the summary to boost semantic matching for high-level queries
        combined_text = f"Episode: {title}\nTopic: {topic}\nSummary: {summary}\n\nTranscript:\n{content}"
        
        documents.append(Document(
            page_content=combined_text,
            metadata={
                "type": "transcript_segment",
                "episode_id": episode_id,
                "title": title,
                "topic": topic,
                "speakers": speakers,
                "start_time": start_time,
                "end_time": end_time
            }
        ))
        
    return documents

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for Knowledge Engine.")
    parser.add_argument("--summaries_dir", default="combined_summaries", help="Directory containing combined summary JSONs")
    parser.add_argument("--transcripts_dir", default="segmented_transcripts", help="Directory containing segmented transcript JSONs")
    parser.add_argument("--reset", action="store_true", help="Delete existing ChromaDB collection before starting")
    args = parser.parse_args()
    
    # 1. Initialize Vector Store
    print(f"Initializing ChromaDB in {CHROMA_PERSIST_DIRECTORY}...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if args.reset and os.path.exists(CHROMA_PERSIST_DIRECTORY):
        print("Reset flag detected. Cleaning up existing DB (manual deletion required for safe persistent storage usually, but here we just overwrite).")
        # In a real app, you might use shutil.rmtree(CHROMA_PERSIST_DIRECTORY)
        
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    
    all_documents = []
    
    # 2. Process Summaries
    print("Processing Combined Summaries...")
    if os.path.exists(args.summaries_dir):
        for filename in os.listdir(args.summaries_dir):
            if filename.endswith(".json") and not filename.startswith("episode_groupings"):
                file_path = os.path.join(args.summaries_dir, filename)
                try:
                    data = load_json_file(file_path)
                    docs = create_summary_documents(data)
                    all_documents.extend(docs)
                    print(f"  Loaded {len(docs)} vectors from {filename}")
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")
    else:
        print(f"  Warning: Directory {args.summaries_dir} not found.")

    # 3. Process Transcripts
    print("Processing Segmented Transcripts...")
    if os.path.exists(args.transcripts_dir):
        for filename in os.listdir(args.transcripts_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(args.transcripts_dir, filename)
                try:
                    data = load_json_file(file_path)
                    docs = create_transcript_documents(data)
                    all_documents.extend(docs)
                    print(f"  Loaded {len(docs)} vectors from {filename}")
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")
    else:
        print(f"  Warning: Directory {args.transcripts_dir} not found.")
        
    # 4. Batch Add to Chroma
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
    main()
