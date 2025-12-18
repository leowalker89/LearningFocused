"""Substack article summarization + tags extraction.

Produces an embed-friendly summary object with:
- overview/themes/key_takeaways/value_proposition (aligned with podcast combined summaries)
- why_reference: why you'd look this up later / what it helps with
- learning_hooks: optional hooks for learning-focused audiences (e.g., Alpha School / 2-hour learning)
- tags/keywords/people/orgs/content_type and reading-time estimate

Important: references/studies should only be included if explicitly mentioned in the article text
to avoid hallucination.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import ARTICLE_SUMMARIES_DIR, SUBSTACK_METADATA_DIR, SUBSTACK_TEXT_DIR
from src.pipeline.substack.llm_config import get_article_summary_llm

load_dotenv()


class ArticleSummary(BaseModel):
    """Structured summary payload for a single article."""

    thesis: str = Field(description="1–3 sentence thesis capturing the core claim.")
    overview: str = Field(description="Cohesive overview summary suitable for embedding.")
    themes: list[str] = Field(description="5–12 themes/topics.")
    key_takeaways: list[str] = Field(description="5–12 concrete takeaways.")
    value_proposition: str = Field(description="Why this article is worth reading.")
    why_reference: str = Field(
        description="When/why you would look this article up later (reference use)."
    )
    learning_hooks: list[str] = Field(
        description=(
            "Optional: 2–6 hooks connecting this article to learning-focused interests "
            "(e.g., Alpha School, 2-hour learning, mastery learning). Keep factual and grounded in the text."
        )
    )

    content_type: str = Field(
        description="One of: essay, announcement, roundup, interview, research_note, other."
    )
    keywords: list[str] = Field(description="10–25 keywords for retrieval.")
    people: list[str] = Field(description="People explicitly mentioned (names only).")
    orgs: list[str] = Field(description="Organizations explicitly mentioned (names only).")

    studies_and_sources: list[str] = Field(
        description=(
            "List of studies/books/reports explicitly mentioned in the article text. "
            "If none are mentioned, return an empty list."
        )
    )


@dataclass(frozen=True)
class SummarizeResult:
    status: str  # created|skipped|failed
    doc_id: str
    reason: str | None = None


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _reading_time_minutes(word_count: int, wpm: int = 220) -> int:
    if word_count <= 0:
        return 0
    return max(1, (word_count + wpm - 1) // wpm)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_article(
    *,
    doc_id: str,
    metadata_dir: Path = SUBSTACK_METADATA_DIR,
    text_dir: Path = SUBSTACK_TEXT_DIR,
    output_dir: Path = ARTICLE_SUMMARIES_DIR,
    model: str = "gemini-flash-latest",
    temperature: float = 0.1,
    force: bool = False,
) -> SummarizeResult:
    """Generate and persist a structured summary for a given article doc_id."""
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = metadata_dir / f"{doc_id}.json"
    text_path = text_dir / f"{doc_id}.md"
    out_path = output_dir / f"{doc_id}_summary.json"

    if out_path.exists() and not force:
        return SummarizeResult(status="skipped", doc_id=doc_id, reason="Summary exists")

    if not meta_path.exists():
        return SummarizeResult(status="failed", doc_id=doc_id, reason="Missing metadata")
    if not text_path.exists():
        return SummarizeResult(status="failed", doc_id=doc_id, reason="Missing text")

    meta = _load_json(meta_path)
    text = text_path.read_text(encoding="utf-8").strip()
    if not text:
        return SummarizeResult(status="failed", doc_id=doc_id, reason="Empty text")

    wc = _word_count(text)
    reading_time = _reading_time_minutes(wc)

    llm = get_article_summary_llm(model=model, temperature=temperature)
    parser = PydanticOutputParser(pydantic_object=ArticleSummary)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert education content curator building a searchable knowledge base.

You will be given the FULL TEXT of a Substack article (usually short).

Goals:
- Produce an embed-friendly summary with strong retrieval tags.
- Be specific and concrete. Prefer short, information-dense sentences.
- DO NOT invent studies, statistics, citations, quotes, or named entities.
- For studies_and_sources: ONLY include sources explicitly mentioned in the text. If none, return [].
- For people/orgs: ONLY include names explicitly present in the text.
- For learning_hooks: connect to learning-focused interests ONLY when the article supports it.

Output MUST follow this schema:
{format_instructions}
""",
            ),
            (
                "user",
                "Article metadata:\n{meta_json}\n\nArticle text:\n{text}\n",
            ),
        ]
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke(
            {
                "meta_json": json.dumps(
                    {
                        "title": meta.get("title"),
                        "canonical_url": meta.get("canonical_url"),
                        "published_at": meta.get("published_at"),
                        "author": meta.get("author"),
                        "source": meta.get("source"),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "text": text,
                "format_instructions": parser.get_format_instructions(),
            }
        )
    except Exception as e:
        return SummarizeResult(status="failed", doc_id=doc_id, reason=str(e))

    summary_payload: dict[str, Any] = {
        "doc_id": doc_id,
        "type": "substack_article",
        "source": meta.get("source"),
        "source_type": meta.get("source_type"),
        "title": meta.get("title"),
        "canonical_url": meta.get("canonical_url"),
        "published_at": meta.get("published_at"),
        "author": meta.get("author"),
        "reading_time_minutes": reading_time,
        "word_count": wc,
        "generated_content": result.model_dump(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "content_hash": meta.get("content_hash"),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    return SummarizeResult(status="created", doc_id=doc_id)


