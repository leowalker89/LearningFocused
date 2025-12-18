"""Inspect pipeline state across filesystem + Chroma + Neo4j.

This is a lightweight operational tool for sanity-checking:
- how many Substack artifacts exist on disk
- what’s currently stored in Chroma (overall + by type)
- what’s currently stored in Neo4j (nodes/relationships + ingest registry stats)
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.config import (
    ARTICLE_SUMMARIES_DIR,
    CHROMA_DIR,
    SUBSTACK_HTML_DIR,
    SUBSTACK_METADATA_DIR,
    SUBSTACK_TEXT_DIR,
)
from src.database.neo4j_manager import get_graph

COLLECTION_NAME = "education_knowledge_engine"


def _count_files(d: Path, pattern: str) -> int:
    if not d.exists():
        return 0
    return sum(1 for _ in d.glob(pattern))


def _inspect_folders() -> None:
    print("\n=== Filesystem (Substack artifacts) ===")
    print(f"substack_articles/metadata/*.json: {_count_files(SUBSTACK_METADATA_DIR, '*.json')}")
    print(f"substack_articles/html/*.html:    {_count_files(SUBSTACK_HTML_DIR, '*.html')}")
    print(f"substack_articles/text/*.md:      {_count_files(SUBSTACK_TEXT_DIR, '*.md')}")
    print(f"article_summaries/*_summary.json: {_count_files(ARTICLE_SUMMARIES_DIR, '*_summary.json')}")


def _iter_chroma_ids_and_metadatas(collection: Any, *, page_size: int = 2000):
    """Iterate (id, metadata) from Chroma by paging through the collection."""
    offset = 0
    while True:
        data = collection.get(include=["metadatas"], limit=page_size, offset=offset)
        ids = data.get("ids") or []
        metas = data.get("metadatas") or []
        if not ids:
            break
        for _id, meta in zip(ids, metas):
            yield str(_id), meta if isinstance(meta, dict) else {}
        offset += len(ids)


def _inspect_chroma(*, show_type_breakdown: bool, check_dupes: bool) -> None:
    print("\n=== Chroma ===")
    print(f"persist dir: {CHROMA_DIR}")
    if not CHROMA_DIR.exists():
        print("Chroma dir not found.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    try:
        total = store._collection.count()  # type: ignore[attr-defined]
        print(f"total docs: {total}")
    except Exception as e:
        print(f"total docs: (unable to count) {e}")
        total = None

    if not show_type_breakdown and not check_dupes:
        return

    # Accurate counts by paging metadatas (works regardless of backend filter support).
    type_counts: Counter[str] = Counter()
    total_seen = 0
    for _id, meta in _iter_chroma_ids_and_metadatas(store._collection):  # type: ignore[attr-defined]
        total_seen += 1
        t = meta.get("type")
        if isinstance(t, str):
            type_counts[t] += 1
        else:
            type_counts["(missing type)"] += 1

    print("\n--- counts by type (full scan) ---")
    for t, c in type_counts.most_common():
        print(f"{t}: {c}")
    if total is not None and total_seen != total:
        print(f"(note) scanned {total_seen} docs but collection.count() reported {total}")

    if not check_dupes:
        return

    # Duplicate heuristics
    print("\n--- duplicate checks (heuristics) ---")
    
    # 1) Substack: duplicates by (type, doc_id)
    substack_key_counts: Counter[tuple[str, str]] = Counter()
    # 2) Audio Segments: duplicates by (episode_id, start_time, topic)
    seg_key_counts: Counter[tuple[str, str, str]] = Counter()
    # 3) Audio Summaries (takeaways, overview, motivation): duplicates by (episode_id, type)
    audio_sum_counts: Counter[tuple[str, str]] = Counter()

    for _id, meta in _iter_chroma_ids_and_metadatas(store._collection):  # type: ignore[attr-defined]
        t = meta.get("type")
        
        # Substack
        if isinstance(t, str) and t.startswith("article_"):
            doc_id = meta.get("doc_id")
            if isinstance(doc_id, str):
                substack_key_counts[(t, doc_id)] += 1
        
        # Audio Segments
        elif t == "transcript_segment":
            episode_id = meta.get("episode_id")
            start_time = meta.get("start_time")
            topic = meta.get("topic")
            if isinstance(episode_id, str) and isinstance(topic, str):
                seg_key_counts[(episode_id, str(start_time), topic)] += 1
        
        # Audio Summaries
        elif t in {"key_takeaway", "series_overview", "series_motivation"}:
            episode_id = meta.get("episode_id")
            if isinstance(episode_id, str) and isinstance(t, str):
                audio_sum_counts[(episode_id, t)] += 1

    dup_substack = [(k, v) for k, v in substack_key_counts.items() if v > 1]
    dup_segments = [(k, v) for k, v in seg_key_counts.items() if v > 1]
    dup_audio_sum = [(k, v) for k, v in audio_sum_counts.items() if v > 1]

    print(f"Substack duplicates (type, doc_id): {len(dup_substack)} keys")
    for (t, doc_id), c in sorted(dup_substack, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {t} {doc_id}: {c}")

    print(f"Audio Segment duplicates (ep_id, start, topic): {len(dup_segments)} keys")
    for (episode_id, start_time, topic), c in sorted(dup_segments, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {episode_id} {start_time} {topic}: {c}")

    print(f"Audio Summary duplicates (ep_id, type): {len(dup_audio_sum)} keys")
    for (episode_id, t), c in sorted(dup_audio_sum, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {episode_id} {t}: {c}")


def _inspect_neo4j(*, check_dupes: bool) -> None:
    print("\n=== Neo4j ===")
    graph = get_graph()

    counts = graph.query(
        """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->()
        RETURN count(distinct n) AS nodes, count(r) AS relationships
        """
    )
    if counts:
        print(f"nodes: {counts[0].get('nodes')}")
        print(f"relationships: {counts[0].get('relationships')}")

    ingest_counts = graph.query(
        """
        MATCH (d:IngestedDoc)
        RETURN d.status AS status, count(*) AS cnt
        ORDER BY cnt DESC
        """
    )
    if ingest_counts:
        print("\n--- IngestedDoc by status ---")
        for row in ingest_counts:
            print(f"{row.get('status')}: {row.get('cnt')}")

    substack_ingest = graph.query(
        """
        MATCH (d:IngestedDoc)
        WHERE d.source_key STARTS WITH 'substack:'
        RETURN d.status AS status, count(*) AS cnt
        ORDER BY cnt DESC
        """
    )
    if substack_ingest:
        print("\n--- IngestedDoc (substack) by status ---")
        for row in substack_ingest:
            print(f"{row.get('status')}: {row.get('cnt')}")

    audio_ingest = graph.query(
        """
        MATCH (d:IngestedDoc)
        WHERE d.source_key STARTS WITH 'audio:'
        RETURN d.status AS status, count(*) AS cnt
        ORDER BY cnt DESC
        """
    )
    if audio_ingest:
        print("\n--- IngestedDoc (audio) by status ---")
        for row in audio_ingest:
            print(f"{row.get('status')}: {row.get('cnt')}")

    if not check_dupes:
        return

    print("\n--- duplicate checks (heuristics) ---")
    # 1) Ingest registry should be unique by (source_key, content_hash)
    ing_dup = graph.query(
        """
        MATCH (d:IngestedDoc)
        WITH d.source_key AS source_key, d.content_hash AS content_hash, count(*) AS c
        WHERE c > 1
        RETURN source_key, content_hash, c
        ORDER BY c DESC
        LIMIT 20
        """
    )
    print(f"IngestedDoc duplicates (source_key, content_hash): {len(ing_dup)}")
    for row in ing_dup:
        print(f"  {row.get('source_key')} {row.get('content_hash')}: {row.get('c')}")

    # 2) Nodes: duplicates by (primary label, id) when `n.id` exists (common in LangChain graph docs)
    node_dupes = graph.query(
        """
        MATCH (n)
        WHERE n.id IS NOT NULL
        WITH labels(n)[0] AS label, n.id AS id, count(*) AS c
        WHERE c > 1
        RETURN label, id, c
        ORDER BY c DESC
        LIMIT 20
        """
    )
    print(f"node duplicates by (label, id): {len(node_dupes)}")
    for row in node_dupes:
        print(f"  {row.get('label')} {row.get('id')}: {row.get('c')}")

    # 3) Relationships: duplicates by (src.id, type, dst.id) (heuristic)
    rel_dupes = graph.query(
        """
        MATCH (a)-[r]->(b)
        WHERE a.id IS NOT NULL AND b.id IS NOT NULL
        WITH a.id AS a_id, type(r) AS rel_type, b.id AS b_id, count(*) AS c
        WHERE c > 1
        RETURN a_id, rel_type, b_id, c
        ORDER BY c DESC
        LIMIT 20
        """
    )
    print(f"relationship duplicates by (a.id, type, b.id): {len(rel_dupes)}")
    for row in rel_dupes:
        print(f"  {row.get('a_id')} -[{row.get('rel_type')}]-> {row.get('b_id')}: {row.get('c')}")


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(description="Inspect filesystem + Chroma + Neo4j state.")
    p.add_argument("--no-chroma", action="store_true", help="Skip Chroma inspection.")
    p.add_argument("--no-neo4j", action="store_true", help="Skip Neo4j inspection.")
    p.add_argument("--type-breakdown", action="store_true", help="Show Chroma counts by metadata.type.")
    p.add_argument("--check-dupes", action="store_true", help="Run duplicate checks (may take longer).")
    args = p.parse_args()

    _inspect_folders()
    if not args.no_chroma:
        _inspect_chroma(show_type_breakdown=bool(args.type_breakdown), check_dupes=bool(args.check_dupes))
    if not args.no_neo4j:
        _inspect_neo4j(check_dupes=bool(args.check_dupes))


if __name__ == "__main__":
    main()


