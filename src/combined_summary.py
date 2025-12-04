import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# --- Pydantic Models for LLM Output ---

class EpisodeGroup(BaseModel):
    group_id: str = Field(description="A unique identifier for this group (e.g., 'S2E278-279 Alpha Haters Debate')")
    filenames: List[str] = Field(description="List of filenames belonging to this group")
    reasoning: str = Field(description="Explanation of why these episodes are grouped, citing specific metadata clues (e.g. matching titles, sequential numbering, part labels)")

class GroupingResponse(BaseModel):
    groups: List[EpisodeGroup] = Field(description="List of episode groups identified")

class CombinedSummary(BaseModel):
    overview: str = Field(description="A cohesive narrative summary synthesizing the content from all episodes in the group.")
    themes: List[str] = Field(description="List of core themes and topics discussed.")
    key_takeaways: List[str] = Field(description="List of actionable or insightful takeaways for the listener.")
    value_proposition: str = Field(description="Explanation of why someone should listen to this content (the 'hook').")


# --- Helper Functions ---

def load_transcript(file_path: Path) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_transcript_for_llm(transcript_data: List[Dict[str, Any]]) -> str:
    """
    Formats the transcript into a readable string for the LLM.
    Includes timestamps to help the LLM identify boundaries.
    """
    formatted_text = ""
    for segment in transcript_data:
        # Use speaker_name if available, otherwise speaker ID
        speaker = segment.get("speaker_name", segment.get("speaker", "Unknown"))
        start = segment.get("start_time", 0)
        text = segment.get("text", "")
        formatted_text += f"[{start:.2f}s] {speaker}: {text}\n"
    return formatted_text

def get_all_filenames(transcripts_dir: Path) -> List[str]:
    return [f.name for f in transcripts_dir.glob("*.json")]

def load_metadata(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def group_episodes(filenames: List[str], metadata_dir: Path) -> List[EpisodeGroup]:
    print("Grouping episodes based on filenames and metadata...")
    
    # Prepare a rich list of available episodes with their metadata
    episodes_context = []
    for fname in filenames:
        meta_path = metadata_dir / fname
        meta = load_metadata(meta_path)
        episodes_context.append({
            "filename": fname,
            "title": meta.get("title", fname),
            "summary": meta.get("summary", "")[:200] + "..." if meta.get("summary") else "No summary", # Truncate for token efficiency
            "published": meta.get("published", "Unknown date"),
            "episode_number": meta.get("itunes_episode", "Unknown")
        })
    
    # Use Gemini Flash for grouping
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    parser = PydanticOutputParser(pydantic_object=GroupingResponse)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert content librarian.
Your task is to group podcast episodes that belong to the same multi-part series or specific topic conversation.

Guidelines:
1. Analyze the provided list of episodes, including their filenames, titles, summaries, and publication dates.
2. Group episodes that are explicitly labeled as "Part 1", "Part 2", etc.
3. Group episodes that share a very specific, unique topic or guest, even if "Part X" is missing from the title, especially if they were published sequentially.
4. Episodes that are standalone interviews or topics should be in their own single-episode group.
5. Every provided filename MUST be assigned to exactly one group.
6. Create a descriptive group_id for each group.

{format_instructions}
"""),
        ("user", "Episode List:\n{episodes_json}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        # Convert the context list to a formatted JSON string for the prompt
        episodes_json_str = json.dumps(episodes_context, indent=2)
        
        result = chain.invoke({
            "episodes_json": episodes_json_str,
            "format_instructions": parser.get_format_instructions()
        })
        return result.groups
    except Exception as e:
        print(f"Error grouping episodes: {e}")
        # Fallback: treat every file as a single group
        return [EpisodeGroup(group_id=f, filenames=[f], reasoning="Fallback error") for f in filenames]

# ... (load_metadata removed as it was moved up)


def generate_combined_summary(group: EpisodeGroup, transcripts_dir: Path, metadata_dir: Path) -> Optional[Dict[str, Any]]:
    print(f"Generating summary for group: {group.group_id} ({len(group.filenames)} episodes)...")
    
    full_text = ""
    episodes_data = []
    
    # Load and combine text
    # Sort filenames to ensure Part 1 comes before Part 2 if detectable
    sorted_filenames = sorted(group.filenames)
    
    for filename in sorted_filenames:
        transcript_path = transcripts_dir / filename
        metadata_path = metadata_dir / filename
        
        if not transcript_path.exists():
            print(f"Warning: File {filename} not found.")
            continue
            
        data = load_transcript(transcript_path)
        meta = load_metadata(metadata_path)
        
        episodes_data.append({
            "filename": filename,
            "title": meta.get("title", data.get("meta_data", {}).get("og_file_name", filename)),
            "publisher_summary": meta.get("summary", ""),
            "published_date": meta.get("published", ""),
            "full_metadata": meta
        })
        
        transcript_text = format_transcript_for_llm(data.get("transcript", []))
        full_text += f"\n\n--- Start of Episode: {filename} ---\n\n"
        full_text += transcript_text
        
    if not full_text:
        return None

    # Use Gemini Flash for summarization
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    parser = PydanticOutputParser(pydantic_object=CombinedSummary)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert podcast content curator.
Your task is to create a high-level summary for a group of related podcast episodes.
These episodes might be a multi-part series or a single episode.

Guidelines:
1. Synthesize the content from all provided transcripts into a cohesive narrative.
2. Identify the core themes and topics.
3. Extract key takeaways that provide value to the listener.
4. Explain WHY someone should listen to this (the "hook" or value proposition).
5. The output will be used for knowledge embedding and search, so be comprehensive yet concise.

{format_instructions}
"""),
        ("user", "Podcast Transcripts:\n{transcript_text}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "transcript_text": full_text,
            "format_instructions": parser.get_format_instructions()
        })
        
        # Construct the final "comprehensive" object
        return {
            "id": group.group_id, # Could use a UUID or hash
            "group_title": episodes_data[0]["title"] if episodes_data else group.group_id,
            "episode_count": len(episodes_data),
            "generated_content": result.dict(),
            "episodes": episodes_data,
            "type": "series" if len(episodes_data) > 1 else "episode"
        }
        
    except Exception as e:
        print(f"Error generating summary for {group.group_id}: {e}")
        return None

def save_groupings(groups: List[EpisodeGroup], output_dir: Path):
    """
    Saves the identified groupings to a JSON file for reference/verification.
    """
    grouping_data = [group.dict() for group in groups]
    output_path = output_dir / "episode_groupings.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(grouping_data, f, indent=2)
    print(f"Saved grouping configuration to {output_path}")

def process_combined_summaries(transcripts_dir: Path, metadata_dir: Path, output_dir: Path):
    """
    Main function to process and generate combined summaries for all transcripts.
    """
    # 1. Get all filenames
    filenames = get_all_filenames(transcripts_dir)
    if not filenames:
        print("No transcript files found.")
        return

    # 2. Group episodes
    groups = group_episodes(filenames, metadata_dir)
    print(f"Identified {len(groups)} groups.")
    
    # Save the groupings for verification
    save_groupings(groups, output_dir)
    
    # 3. Process each group
    for group in groups:
        # Check if summary already exists to avoid re-processing? 
        # For now, let's overwrite or maybe check based on group ID.
        # But group ID is dynamic based on LLM. 
        # Let's just process.
        
        summary_data = generate_combined_summary(group, transcripts_dir, metadata_dir)
        
        if summary_data:
            # Create a safe filename for output
            safe_name = "".join([c if c.isalnum() or c in (' ', '-', '_') else '' for c in group.group_id]).strip()
            safe_name = safe_name.replace(" ", "_")[:50] # Limit length
            output_path = output_dir / f"summary_{safe_name}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)
            print(f"Saved summary to {output_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    transcripts_dir = base_dir / "transcripts"
    metadata_dir = base_dir / "metadata_output"
    output_dir = base_dir / "combined_summaries"
    output_dir.mkdir(exist_ok=True)
    
    process_combined_summaries(transcripts_dir, metadata_dir, output_dir)
