import json
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Define the output structure for the LLM
class TopicSegment(BaseModel):
    topic_label: str = Field(description="A concise label for the topic discussed in this segment")
    start_time: float = Field(description="The start timestamp of the segment in seconds")
    end_time: float = Field(description="The end timestamp of the segment in seconds")
    summary: str = Field(description="A brief summary of what is discussed in this segment")

class SegmentationResponse(BaseModel):
    segments: List[TopicSegment] = Field(description="List of topic segments covering the entire transcript")

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

def segment_transcript(file_path: Path, output_dir: Path):
    print(f"Processing {file_path.name}...")
    
    data = load_transcript(file_path)
    transcript_text = format_transcript_for_llm(data.get("transcript", []))
    
    # Initialize Gemini
    # using gemini-1.5-pro for large context window and better reasoning
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    parser = PydanticOutputParser(pydantic_object=SegmentationResponse)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert content analyzer for educational podcasts. 
Your task is to segment the provided transcript into coherent topic chunks.

Guidelines:
1. Identify major topic transitions in the conversation.
2. Create segments that cover the ENTIRE transcript from start to finish without gaps.
3. Each segment should represent a distinct subject or discussion point.
4. For each segment, provide a concise topic label, exact start/end times based on the text provided, and a brief summary.
5. Timestamps should align with the provided [timestamp] markers in the text.
6. Ignore minor banter unless it constitutes a meaningful transition or intro/outro.
7. The output must be valid JSON matching the specified structure.

{format_instructions}
"""),
        ("user", "{transcript}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "transcript": transcript_text,
            "format_instructions": parser.get_format_instructions()
        })
        
        # Process results to combine with original data
        processed_segments = []
        original_transcript = data.get("transcript", [])
        
        for segment in result.segments:
            # Find transcript entries that fall within this time range
            segment_text = ""
            segment_speakers = set()
            
            for entry in original_transcript:
                entry_start = entry.get("start_time", 0)
                entry_end = entry.get("end_time", 0)
                
                # Check for overlap
                if entry_start >= segment.start_time and entry_end <= segment.end_time: # Strictly inside
                     segment_text += f"{entry.get('speaker_name', entry.get('speaker'))}: {entry.get('text')} "
                     segment_speakers.add(entry.get('speaker_name', entry.get('speaker')))
                elif (entry_start < segment.end_time and entry_end > segment.start_time): # Partial overlap
                     # Include it if it mostly belongs here? 
                     # For simplicity, we might just rely on the time ranges provided by LLM to slice the text.
                     # But reconstructing text from original entries is safer for accuracy.
                     # Let's simple check if the midpoint of the entry is in the segment
                     midpoint = (entry_start + entry_end) / 2
                     if segment.start_time <= midpoint <= segment.end_time:
                         segment_text += f"{entry.get('speaker_name', entry.get('speaker'))}: {entry.get('text')} "
                         segment_speakers.add(entry.get('speaker_name', entry.get('speaker')))

            processed_segments.append({
                "topic": segment.topic_label,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "summary": segment.summary,
                "speakers": list(segment_speakers),
                "content": segment_text.strip()
            })
            
        output_data = {
            "episode_id": data.get("meta_data", {}).get("og_file_name", "").split(" ")[0], # rigorous id extraction might be needed
            "title": data.get("meta_data", {}).get("og_file_name", ""),
            "original_meta": data.get("meta_data", {}),
            "segments": processed_segments
        }
        
        output_path = output_dir / f"{file_path.stem}_segmented.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Saved segmented transcript to {output_path}")
        
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

if __name__ == "__main__":
    # Setup paths
    base_dir = Path(__file__).parent.parent
    transcripts_dir = base_dir / "transcripts"
    output_dir = base_dir / "segmented_transcripts"
    output_dir.mkdir(exist_ok=True)
    
    # Process all json files in transcripts directory
    # For testing, maybe just pick one
    files = list(transcripts_dir.glob("*.json"))
    if not files:
        print("No transcript files found.")
    else:
        for file_path in files:
            segment_transcript(file_path, output_dir)

