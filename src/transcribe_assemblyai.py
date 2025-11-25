import assemblyai as aai
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("ASSEMBLYAI_API_KEY"):
    print("Error: ASSEMBLYAI_API_KEY not found in environment variables.")
    print("Please create a .env file with your API key.")
    # You can also set it directly: aai.settings.api_key = "YOUR_API_KEY"

def transcribe_audio(file_path: str, output_dir: str = "transcripts") -> str:
    """
    Transcribes an audio file using AssemblyAI with speaker diarization and saves as JSON.
    
    Args:
        file_path: Path to the audio file.
        output_dir: Directory to save the transcription JSON.
        
    Returns:
        Path to the saved JSON file.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Transcribing: {file_path}")

    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path, config=config)
    
    if transcript.status == aai.TranscriptStatus.error:
        raise Exception(f"Transcription failed: {transcript.error}")
        
    # Build the JSON structure
    json_output: dict = {
        "meta_data": {
            "og_file_name": os.path.basename(file_path),
            "file_path": file_path,
            "date_transcribed": datetime.now().isoformat(),
            "duration_seconds": transcript.audio_duration
        },
        "transcript": []
    }

    # Populate transcript list
    if transcript.utterances:
        for u in transcript.utterances:
            json_output["transcript"].append({
                "speaker": u.speaker,
                "text": u.text,
                "start_time": u.start / 1000.0, # Convert ms to seconds
                "end_time": u.end / 1000.0      # Convert ms to seconds
            })
    else:
        json_output["transcript"].append({
            "speaker": None,
            "text": transcript.text,
            "start_time": 0.0,
            "end_time": transcript.audio_duration or 0.0
        })

    # Define output filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.json")

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    return output_path

def main():
    """
    Main entry point for testing.
    """
    # Simple CLI for testing
    import sys
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        try:
            output_file = transcribe_audio(audio_file)
            print(f"\n✅ Transcription complete. Saved to: {output_file}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python src/transcribe_assemblyai.py <path_to_audio_file>")

if __name__ == "__main__":
    main()

