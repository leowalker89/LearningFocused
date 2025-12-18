"""Substack pipeline runner (incremental ingest + summarization).

This mirrors the orchestration approach used in `src.pipeline.audio.process_all`,
but focuses on text-only artifacts:
- ingest RSS items -> persist metadata/html/text
- generate a doc-level structured summary + tags
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from src.config import ARTICLE_SUMMARIES_DIR, SUBSTACK_FEED_URL, SUBSTACK_METADATA_DIR, ensure_data_dirs
from src.pipeline.substack.download_articles import ingest_substack_feed
from src.pipeline.substack.summarize_articles import summarize_article


@dataclass(frozen=True)
class Stats:
    processed: int
    skipped: int
    failed: int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Substack pipeline runner")
    p.add_argument("--mode", choices=["daily", "backfill"], default="daily")
    p.add_argument("--feed-url", default=SUBSTACK_FEED_URL)
    p.add_argument("--ingest-limit", type=int, default=10)
    p.add_argument("--summarize-limit", type=int, default=None)
    p.add_argument("--allow-updates", action="store_true", help="Update if content changed.")
    p.add_argument("--force-summaries", action="store_true", help="Regenerate summaries even if present.")
    p.add_argument("--skip-ingest", action="store_true")
    p.add_argument("--skip-summaries", action="store_true")
    return p


def _summarize_missing(*, limit: int | None, force: bool) -> Stats:
    summary_paths = sorted(ARTICLE_SUMMARIES_DIR.glob("*_summary.json"))
    summarized_ids = {p.name.replace("_summary.json", "") for p in summary_paths}

    candidates = sorted(SUBSTACK_METADATA_DIR.glob("*.json"))
    doc_ids = [p.stem for p in candidates if p.is_file()]

    to_run = [d for d in doc_ids if force or d not in summarized_ids]
    if limit is not None:
        to_run = to_run[: max(0, limit)]

    processed = skipped = failed = 0
    for doc_id in to_run:
        res = summarize_article(doc_id=doc_id, force=force)
        if res.status == "created":
            processed += 1
        elif res.status == "skipped":
            skipped += 1
        else:
            failed += 1
            print(f"Summarize failed for {doc_id}: {res.reason}")

    return Stats(processed=processed, skipped=skipped, failed=failed)


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    ensure_data_dirs()
    args = build_parser().parse_args(argv)

    if not args.skip_ingest:
        print("\n=== INGEST RSS ===")
        results = ingest_substack_feed(
            feed_url=args.feed_url,
            limit=args.ingest_limit if args.mode == "daily" else None,
            allow_updates=bool(args.allow_updates),
        )
        created = len([r for r in results if r.status == "created"])
        updated = len([r for r in results if r.status == "updated"])
        skipped = len([r for r in results if r.status == "skipped"])
        failed = len([r for r in results if r.status == "failed"])
        print(f"Ingest summary: created={created} updated={updated} skipped={skipped} failed={failed}")

    if not args.skip_summaries:
        print("\n=== SUMMARIZE ARTICLES ===")
        stats = _summarize_missing(limit=args.summarize_limit, force=bool(args.force_summaries))
        print(f"Summaries: processed={stats.processed} skipped={stats.skipped} failed={stats.failed}")

    print("\nDone.")


if __name__ == "__main__":
    main()


