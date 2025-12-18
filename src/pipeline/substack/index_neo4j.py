"""Substack pipeline -> Neo4j indexing helpers.

Pipeline-owned logic:
- how we choose/construct Documents from Substack artifacts for graph extraction
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def collect_substack_graph_documents(
    text_dir: Path,
    metadata_dir: Path,
    *,
    limit: int | None = None,
) -> List[Document]:
    """Load Substack article texts and convert them to Documents for graph extraction."""
    documents: List[Document] = []
    text_files = sorted(text_dir.glob("*.md"))
    if not text_files:
        logger.info("No Substack article texts found in %s", text_dir)
        return []

    for text_path in text_files:
        meta_path = metadata_dir / f"{text_path.stem}.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta: Dict[str, Any] = json.load(f)

            text = text_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            title = meta.get("title") or text_path.stem
            canonical_url = meta.get("canonical_url")

            header = f"Article: {title}\n"
            if canonical_url:
                header += f"URL: {canonical_url}\n"
            header += "\n"

            documents.append(
                Document(
                    page_content=header + text,
                    metadata={
                        "source": str(text_path),
                        "doc_id": meta.get("doc_id", text_path.stem),
                        "canonical_url": canonical_url,
                        "title": title,
                        "published_at": meta.get("published_at"),
                        "author": meta.get("author"),
                        "source_type": meta.get("source_type"),
                    },
                )
            )
            if limit is not None and len(documents) >= limit:
                return documents
        except Exception as e:
            logger.exception("Error reading Substack article %s: %s", text_path, e)

    return documents


