## Pipelines

This package contains the processing pipelines for different source types.

- **Audio/Podcast pipeline**: `src/pipeline/audio/`
- **Substack pipeline**: `src/pipeline/substack/`

### Recommended entrypoints

- **Run the full audio pipeline**:

```bash
uv run python -m src.pipeline.audio.run
```

- **Run the processing-only audio orchestrator** (download → transcribe → identify → segment):

```bash
uv run python -m src.pipeline.audio.process_all
```

- **Run the full Substack pipeline** (ingest + summarize; optional indexing via flags):

```bash
uv run python -m src.pipeline.substack.run -- --mode daily --ingest-limit 10
```

### Notes

- The audio pipeline is the current production path.
- The Substack pipeline is intentionally separate to keep ingestion concerns isolated.
