"""Audio pipeline -> Neo4j indexing helpers.

Pipeline-owned logic:
- how we choose/construct Documents from audio artifacts for graph extraction
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def collect_audio_graph_documents(segmented_dir: Path) -> List[Document]:
    """Load segmented transcripts and convert them to Documents for graph extraction."""
    documents: List[Document] = []
    files = list(segmented_dir.glob("*_segmented.json"))

    if not files:
        logger.info("No segmented transcripts found in %s", segmented_dir)
        return []

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            episode_id = data.get("episode_id", "Unknown")
            title = data.get("title", "Unknown")

            for segment in data.get("segments", []) or []:
                text_content = segment.get("content", "") or ""
                if not text_content:
                    continue
                topic = segment.get("topic")

                # Provide enough context to make relationship extraction meaningful.
                # This steers the graph transformer without hard-coding prompts here.
                header = f"Episode: {title}\n"
                if topic:
                    header += f"Topic: {topic}\n"
                header += "\n"

                documents.append(
                    Document(
                        page_content=header + text_content,
                        metadata={
                            "source": str(file_path),
                            "episode_id": episode_id,
                            "title": title,
                            "topic": topic,
                            "start_time": segment.get("start_time"),
                            "end_time": segment.get("end_time"),
                            "speakers": segment.get("speakers", []),
                        },
                    )
                )
        except Exception as e:
            logger.exception("Error reading segmented transcript %s: %s", file_path, e)

    return documents


