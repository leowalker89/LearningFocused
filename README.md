# Future of Education Knowledge Engine

A queryable "Second Brain" for the Future of Education podcast archive (and related Substack articles), enabling deep exploration of educational concepts through semantic search and knowledge graphs.

## Architecture: The "Tri-Store" Approach

This project implements a robust "Tri-Store" architecture to handle different aspects of knowledge retrieval:

1.  **Data Lake (Filesystem)**:
    -   Stores raw MP3s, AssemblyAI transcripts, and intermediate processed JSONs.
    -   Serves as the "Ground Truth" archive for all data.

2.  **Semantic Store (ChromaDB)**:
    -   Stores vector embeddings for **Topic Segments**, **Combined Summaries**, and **Substack article summaries/text**.
    -   Powered by OpenAI `text-embedding-3-small` for high-precision semantic search.
    -   Enables natural language queries to find specific discussion points.

3.  **Knowledge Graph (Neo4j)**:
    -   Stores structured entities (People, Concepts, Organizations) and their relationships (e.g., `ADVOCATES_FOR`, `CRITICIZES`).
    -   Extracted using Gemini (via LangChain graph transformers) for structural context.

## Quick Start

### 1. Prerequisites

-   **Python 3.12+**
-   [uv](https://github.com/astral-sh/uv) for modern Python package management.
-   [Docker](https://www.docker.com/) (optional, if running Neo4j locally).

### 2. Environment Setup

Create a `.env` file in the project root with the following keys:

```bash
# Transcription Service
ASSEMBLYAI_API_KEY=your_key_here

# LLM & Embeddings
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key

# Knowledge Graph (Neo4j)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 3. Installation

Install dependencies using `uv`:

```bash
uv sync
```

### 4. Running the Pipeline

The system is designed as a modular pipeline. You can run the core data processing steps with one command:

```bash
uv run python -m src.pipeline.audio.run
```

This script orchestrates:
1.  **Download**: Fetches new episodes from the RSS feed.
2.  **Transcribe**: Converts audio to text using AssemblyAI (with speaker diarization).
3.  **Identify**: Maps generic "Speaker A" labels to real names (e.g., "MacKenzie Price") using Gemini.
4.  **Segment**: Breaks transcripts into coherent "Topic Chunks" using Gemini.
5.  **Summarize**: Generates comprehensive summaries for single episodes or multi-part series.
6.  **Index (optional)**: Updates Chroma (embeddings) and Neo4j (knowledge graph), unless skipped via flags.

You can also ingest Substack posts (text-first pipeline):

```bash
uv run python -m src.pipeline.substack.run -- --mode daily --ingest-limit 10
```

You can also run individual audio steps directly:

```bash
uv run python -m src.pipeline.audio.download
uv run python -m src.pipeline.audio.transcribe /path/to/episode.mp3
uv run python -m src.pipeline.audio.identify_speakers /path/to/transcript.json
uv run python -m src.pipeline.audio.segment_topics
uv run python -m src.pipeline.audio.generate_summaries
```
### 5. Building the Knowledge Base

Once the raw data is processed, you populate the searchable databases with these commands:

**Step 1: Create Vector Embeddings (ChromaDB)**
```bash
uv run python -m src.database.chroma_manager
```
*Collects Documents from pipeline-owned indexers (audio + substack), generates embeddings, and stores them in `chroma_db/`.*

**Step 2: Build Knowledge Graph (Neo4j)**
```bash
uv run python -m src.database.neo4j_manager
```
*Extracts entities + relationships from audio segments and Substack articles and pushes them to your Neo4j instance.*

### 6. Inspecting the Data

You can explore the processed data using the analysis tools:

**Inspect ChromaDB (Vector Store)**
```bash
uv run python -m src.analysis.inspect_chroma
# Optional: Test semantic search
uv run python -m src.analysis.inspect_chroma --query "parent burnout"
```

**Inspect Neo4j (Knowledge Graph)**
```bash
uv run python -m src.analysis.inspect_graph
```

**Investigate Specific Entity in Graph**
```bash
uv run python -m src.analysis.investigate_entity "MacKenzie Price"
```

## Project Structure & Components

| File/Directory | Description |
| :--- | :--- |
| `src/pipeline/` | Processing pipelines (separated by source type). |
| `src/pipeline/audio/` | Audio/podcast pipeline steps. `run.py` is the canonical runner; `process_all.py` is the thin processing-only runner. |
| `src/pipeline/substack/` | Substack/article pipeline. `run.py` orchestrates ingest+summaries and can update Chroma; `process_all.py` is processing-only. |
| `src/database/` | Database managers (thin adapters/orchestrators): `chroma_manager.py` (Vectors), `neo4j_manager.py` (Graph). Pipeline-owned indexers live under `src/pipeline/*/index_*.py`. |
| `src/analysis/` | Tools for inspecting data: `inspect_chroma.py`, `inspect_graph.py`, `investigate_entity.py`. |
| `podcast_downloads/` | Storage for raw audio files. |
| `transcripts/` | Raw JSON transcripts with speaker diarization. |
| `segmented_transcripts/` | Transcripts split by topic (ready for RAG). |
| `combined_summaries/` | Series-level summaries and grouping logic. |
| `chroma_db/` | Local persistence for the vector database. |
