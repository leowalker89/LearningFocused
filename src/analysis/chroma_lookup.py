"""Lookup and explore Chroma records by ID (and find neighbors) without OpenAI calls.

This pairs nicely with `src.analysis.visualize_chroma`:
- Hover a point to copy its `id`
- Run `uv run python -m src.analysis.chroma_lookup get <id>`
- Or: `neighbors <id>` to see nearest vectors via stored embeddings
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from dotenv import load_dotenv

from src.config import CHROMA_DIR

load_dotenv()

COLLECTION_NAME = "education_knowledge_engine"


def _connect_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def cmd_get(doc_id: str, max_chars: int) -> None:
    """Print metadata + document text for a specific Chroma ID."""
    c = _connect_collection()
    res = c.get(ids=[doc_id], include=["metadatas", "documents"])
    if not res.get("ids"):
        print(f"No record found for id={doc_id!r}")
        return
    meta = (res.get("metadatas") or [{}])[0] or {}
    doc = (res.get("documents") or [""])[0] or ""
    print(f"id: {doc_id}")
    print("metadata:")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("\ndocument:")
    print(doc[:max_chars])


def cmd_neighbors(doc_id: str, k: int, max_chars: int) -> None:
    """Show nearest neighbors to a record by using its stored embedding."""
    c = _connect_collection()
    base = c.get(ids=[doc_id], include=["embeddings"])
    if not base.get("ids"):
        print(f"No record found for id={doc_id!r}")
        return
    emb = (base.get("embeddings") or [None])[0]
    if emb is None:
        print(f"No embedding stored for id={doc_id!r}")
        return

    # Query by embedding directly (no embedding model / API needed).
    q = c.query(
        query_embeddings=[emb],
        n_results=k,
        include=["metadatas", "documents", "distances"],
    )

    ids = (q.get("ids") or [[]])[0]
    metas = (q.get("metadatas") or [[]])[0]
    docs = (q.get("documents") or [[]])[0]
    dists = (q.get("distances") or [[]])[0]

    print(f"Nearest neighbors for id={doc_id!r} (k={k})")
    for i in range(len(ids)):
        rid = ids[i]
        dist = dists[i] if i < len(dists) else None
        meta = metas[i] if i < len(metas) else {}
        doc = docs[i] if i < len(docs) else ""
        print("\n---")
        print(f"id: {rid}")
        if dist is not None:
            print(f"distance: {dist}")
        # show a couple common metadata fields if present
        for key in ("type", "source", "topic", "episode", "title"):
            if isinstance(meta, dict) and key in meta:
                print(f"{key}: {meta.get(key)}")
        print("preview:")
        print(str(doc)[:max_chars].replace("\n", " "))


def cmd_search_contains(text: str, limit: int) -> None:
    """Find records whose document text contains a substring (server-side where_document)."""
    c = _connect_collection()
    res = c.get(
        where_document={"$contains": text},
        limit=limit,
        include=["metadatas"],
    )
    ids = res.get("ids") or []
    metas = res.get("metadatas") or []
    print(f"Found {len(ids)} records where document contains {text!r} (showing up to {limit})")
    for i, rid in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        print(f"- {rid}  meta={meta}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lookup Chroma docs by id and explore neighbors.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_get = sub.add_parser("get", help="Get a document+metadata by id")
    p_get.add_argument("id", help="Chroma record id")
    p_get.add_argument("--max-chars", type=int, default=4000, help="Max characters of text to print")

    p_n = sub.add_parser("neighbors", help="Nearest neighbors for an id (uses stored embeddings)")
    p_n.add_argument("id", help="Chroma record id")
    p_n.add_argument("--k", type=int, default=10, help="Number of neighbors to show")
    p_n.add_argument("--max-chars", type=int, default=300, help="Preview length for each neighbor")

    p_s = sub.add_parser("contains", help="Search documents that contain a substring")
    p_s.add_argument("text", help="Substring to search for")
    p_s.add_argument("--limit", type=int, default=20, help="Max results")

    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.cmd == "get":
        cmd_get(args.id, max_chars=args.max_chars)
    elif args.cmd == "neighbors":
        cmd_neighbors(args.id, k=args.k, max_chars=args.max_chars)
    elif args.cmd == "contains":
        cmd_search_contains(args.text, limit=args.limit)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()


