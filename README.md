# Future of Education Knowledge Engine

A queryable "Second Brain" for the Future of Education podcast archive, enabling deep exploration of educational concepts through semantic search and knowledge graphs.

## Quick Start

1.  **Install Dependencies** (using [uv](https://github.com/astral-sh/uv)):
    ```bash
    uv sync
    ```

2.  **Environment Setup**:
    Create a `.env` file with your API keys:
    ```bash
    ASSEMBLYAI_API_KEY=your_key_here
    ```

3.  **Run Pipeline**:
    ```bash
    # Download podcasts
    python src/download_podcasts.py

    # Transcribe an episode
    python src/transcribe_assemblyai.py "podcast_downloads/EpisodeName.mp3"
    ```

## Project Structure

-   `src/`: Source code (downloaders, transcribers)
-   `podcast_downloads/`: Raw audio files
-   `transcripts/`: Processed JSON transcripts with speaker diarization

## Architecture

The system aims to use a "Tri-Store" approach:
1.  **Document Store**: Metadata & raw transcripts
2.  **Vector Database**: Semantic search
3.  **Knowledge Graph**: Entity relationships
