"""Audio source runner.

This is the canonical orchestration entrypoint for the audio/podcast source.

It composes:
- file production (download/transcribe/identify/segment): `src.pipeline.audio.process_all`
- summary generation: `src.pipeline.audio.generate_summaries`
- optional indexing:
  - Chroma embeddings: `src.database.chroma_manager.update_chroma_db`
  - Neo4j graph extraction: `src.database.neo4j_manager.update_knowledge_graph`

Design:
- Keep `process_all` thin and incremental/backfill-friendly.
- Keep indexing optional and explicit via flags.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from dotenv import load_dotenv

from src.config import COMBINED_DIR, METADATA_DIR, TRANSCRIPTS_DIR
from src.database.chroma_manager import update_chroma_db
from src.database.neo4j_manager import update_knowledge_graph
from src.pipeline.audio import process_all
from src.pipeline.audio.generate_summaries import process_combined_summaries


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the audio pipeline (processing + optional indexing).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip download/transcribe/identify/segment (does not skip summaries/indexing).",
    )
    p.add_argument(
        "--skip-summaries",
        action="store_true",
        help="Skip combined summary generation.",
    )
    p.add_argument(
        "--skip-chroma",
        action="store_true",
        help="Skip updating the Chroma vector DB.",
    )
    p.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip updating the Neo4j knowledge graph.",
    )
    p.add_argument(
        "--reset-chroma",
        action="store_true",
        help="Reset the Chroma collection before indexing (as supported by update_chroma_db).",
    )

    p.add_argument(
        "process_all_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments forwarded to `src.pipeline.audio.process_all`. "
            "Example: -- --mode backfill --skip-download --skip-transcribe --segment-limit 25"
        ),
    )

    return p


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)

    # argparse.REMAINDER includes the leading "--" (if present). Drop it.
    forwarded = list(args.process_all_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    if not args.skip_processing:
        process_all.main(forwarded)

    if not args.skip_summaries:
        process_combined_summaries(TRANSCRIPTS_DIR, METADATA_DIR, COMBINED_DIR)

    if not args.skip_chroma:
        update_chroma_db(reset=args.reset_chroma)

    if not args.skip_neo4j:
        update_knowledge_graph()


if __name__ == "__main__":
    main()
