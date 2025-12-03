import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Import our processing modules
from download_podcasts import download_podcasts
from transcribe_assemblyai import transcribe_audio
from identify_speakers import identify_speakers, update_transcript_with_speakers
from topic_segmentation import segment_transcript
from combined_summary import process_combined_summaries

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_pipeline():
    # Load environment variables
    load_dotenv()
    
    # Define directories
    base_dir = Path(__file__).parent.parent
    downloads_dir = base_dir / "podcast_downloads"
    transcripts_dir = base_dir / "transcripts"
    segmented_dir = base_dir / "segmented_transcripts"
    metadata_dir = base_dir / "metadata_output"
    combined_dir = base_dir / "combined_summaries"
    
    # RSS Feed Configuration
    RSS_FEED_URL = "https://rss.art19.com/future-of-education"

    # Create output directories if they don't exist
    transcripts_dir.mkdir(exist_ok=True)
    segmented_dir.mkdir(exist_ok=True)
    combined_dir.mkdir(exist_ok=True)
    
    # 0. Download New Episodes
    logger.info("Checking for new episodes...")
    try:
        # Download new episodes (limit=None to check all, or set a reasonable limit for daily checks)
        # Using limit=5 for now to avoid checking full history every time, but adjustable.
        download_podcasts(RSS_FEED_URL, str(downloads_dir), metadata_dir=str(metadata_dir), limit=5)
    except Exception as e:
        logger.error(f"Error checking/downloading podcasts: {e}")

    # Get list of mp3 files
    audio_files = list(downloads_dir.glob("*.mp3"))
    logger.info(f"Found {len(audio_files)} audio files to process.")
    
    for audio_file in audio_files:
        try:
            logger.info(f"Processing: {audio_file.name}")
            
            # 1. Transcription
            transcript_path = transcripts_dir / f"{audio_file.stem}.json"
            
            if not transcript_path.exists():
                logger.info(f"Transcript not found. Transcribing {audio_file.name}...")
                try:
                    transcribe_audio(str(audio_file), str(transcripts_dir))
                    
                    # 2. Speaker Identification (only need to run if we just transcribed)
                    # OR if we want to ensure speakers are identified even for existing transcripts that might lack them
                    # Let's check if we should run it.
                    logger.info("Identifying speakers...")
                    speakers = identify_speakers(str(transcript_path))
                    if speakers:
                        update_transcript_with_speakers(str(transcript_path), speakers)
                        
                except Exception as e:
                    logger.error(f"Failed to transcribe {audio_file.name}: {e}")
                    continue
            else:
                logger.info(f"Transcript already exists for {audio_file.name}. Skipping transcription.")
                # We could check if speaker map exists and run identification if missing, 
                # but for now let's assume existing transcripts are good or will be handled manually if needed.
                # Actually, let's just check quickly if "speaker_map" is in the file?
                # No, let's keep it simple. If transcript exists, we proceed to segmentation.
            
            # 3. Topic Segmentation
            segmented_path = segmented_dir / f"{audio_file.stem}_segmented.json"
            
            if not segmented_path.exists():
                logger.info(f"Segmented transcript not found. Segmenting {audio_file.name}...")
                try:
                    segment_transcript(transcript_path, segmented_dir)
                except Exception as e:
                    logger.error(f"Failed to segment {audio_file.name}: {e}")
            else:
                 logger.info(f"Segmented transcript already exists for {audio_file.name}. Skipping segmentation.")
                 
        except Exception as e:
            logger.error(f"Error processing {audio_file.name}: {e}")
    
    # 4. Combined Summary Generation (Runs after individual processing)
    try:
        logger.info("Starting combined summary generation...")
        process_combined_summaries(transcripts_dir, metadata_dir, combined_dir)
    except Exception as e:
        logger.error(f"Error during combined summary generation: {e}")

if __name__ == "__main__":
    process_pipeline()

