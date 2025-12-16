import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    RSS_FEED_URL, 
    DOWNLOADS_DIR, 
    TRANSCRIPTS_DIR, 
    METADATA_DIR,
    SEGMENTED_DIR
)
from src.pipeline.download import download_podcasts
from src.pipeline.transcribe import transcribe_audio
from src.pipeline.identify_speakers import identify_speakers, update_transcript_with_speakers
from src.pipeline.segment_topics import segment_transcript

def main():
    print("Starting Podcast Processing Pipeline...")
    
    # 1. Download Podcasts
    print("\n=== STEP 1: DOWNLOADING PODCASTS ===")
    try:
        # Using limit=None to download all episodes as requested
        download_podcasts(RSS_FEED_URL, str(DOWNLOADS_DIR), metadata_dir=str(METADATA_DIR), limit=None)
    except Exception as e:
        print(f"Error in download step: {e}")

    # 2. Transcribe Audio
    print("\n=== STEP 2: TRANSCRIBING AUDIO ===")
    audio_files = list(DOWNLOADS_DIR.glob("*.mp3"))
    print(f"Found {len(audio_files)} audio files.")
    
    for audio_file in audio_files:
        try:
            # Construct expected transcript path
            base_name = audio_file.stem
            transcript_path = TRANSCRIPTS_DIR / f"{base_name}.json"
            
            if transcript_path.exists():
                print(f"Skipping transcription (already exists): {base_name}")
                continue
                
            print(f"Transcribing: {base_name}")
            transcribe_audio(str(audio_file), output_dir=str(TRANSCRIPTS_DIR))
            
        except Exception as e:
            print(f"Error transcribing {audio_file.name}: {e}")

    # 3. Identify Speakers
    print("\n=== STEP 3: IDENTIFYING SPEAKERS ===")
    transcript_files = list(TRANSCRIPTS_DIR.glob("*.json"))
    print(f"Found {len(transcript_files)} transcript files.")
    
    for transcript_file in transcript_files:
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if speakers already identified
            meta_data = data.get("meta_data", {})
            if "speaker_map" in meta_data:
                print(f"Skipping speaker ID (already identified): {transcript_file.stem}")
                continue
            
            print(f"Identifying speakers for: {transcript_file.stem}")
            speakers = identify_speakers(str(transcript_file))
            
            if speakers:
                update_transcript_with_speakers(str(transcript_file), speakers)
            else:
                print(f"No speakers identified for {transcript_file.stem}")
                
        except Exception as e:
            print(f"Error identifying speakers for {transcript_file.name}: {e}")

    # 4. Segment Topics
    print("\n=== STEP 4: SEGMENTING TOPICS ===")
    # Re-list transcript files in case new ones were created
    transcript_files = list(TRANSCRIPTS_DIR.glob("*.json"))
    
    for transcript_file in transcript_files:
        try:
            # Construct expected segmented file path
            # The segment_topics script appends "_segmented.json" to the stem
            segmented_path = SEGMENTED_DIR / f"{transcript_file.stem}_segmented.json"
            
            if segmented_path.exists():
                print(f"Skipping segmentation (already exists): {transcript_file.stem}")
                continue
                
            print(f"Segmenting topics for: {transcript_file.stem}")
            segment_transcript(transcript_file, SEGMENTED_DIR)
            
        except Exception as e:
            print(f"Error segmenting {transcript_file.name}: {e}")

    print("\nPipeline execution completed.")

if __name__ == "__main__":
    main()

