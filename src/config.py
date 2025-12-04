import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Root
# src/config.py -> parent is src/ -> parent is project_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DOWNLOADS_DIR = PROJECT_ROOT / "podcast_downloads"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
SEGMENTED_DIR = PROJECT_ROOT / "segmented_transcripts"
METADATA_DIR = PROJECT_ROOT / "metadata_output"
COMBINED_DIR = PROJECT_ROOT / "combined_summaries"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Ensure directories exist
for directory in [DOWNLOADS_DIR, TRANSCRIPTS_DIR, SEGMENTED_DIR, METADATA_DIR, COMBINED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# External Config
RSS_FEED_URL = "https://rss.art19.com/future-of-education"

# Database Config
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

