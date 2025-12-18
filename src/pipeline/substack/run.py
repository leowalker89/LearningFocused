"""Substack source runner.

This is the canonical orchestration entrypoint for ingesting Substack posts and producing:
- persisted article artifacts (metadata/html/text)
- doc-level structured summaries + tags
- optional indexing into Chroma (so `search_knowledge_base` can retrieve them)
"""

from __future__ import annotations

import argparse
from typing import Sequence

from dotenv import load_dotenv

from src.config import ensure_data_dirs
from src.database.chroma_manager import update_chroma_db
from src.pipeline.substack import process_all

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the Substack pipeline (processing + optional indexing).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--skip-processing", action="store_true", help="Skip ingest + summaries.")
    p.add_argument("--skip-chroma", action="store_true", help="Skip updating the Chroma vector DB.")
    p.add_argument("--reset-chroma", action="store_true", help="Reset the Chroma collection before indexing.")
    p.add_argument(
        "process_all_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to `src.pipeline.substack.process_all`. Example: -- --mode daily --ingest-limit 10",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    ensure_data_dirs()
    args = build_parser().parse_args(argv)

    forwarded = list(args.process_all_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    if not args.skip_processing:
        process_all.main(forwarded)

    if not args.skip_chroma:
        update_chroma_db(reset=bool(args.reset_chroma))


if __name__ == "__main__":
    main()
