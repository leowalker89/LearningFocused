"""Substack article ingestion (RSS -> persisted HTML/text/metadata).

This is the text-first analogue to the podcast download/transcribe steps.
It is designed to be:
- idempotent (unique key: canonical_url)
- update-aware (reprocess if content hash changes)
- reproducible (store raw HTML + normalized text + metadata)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import feedparser

from src.config import SUBSTACK_FEED_URL, SUBSTACK_HTML_DIR, SUBSTACK_METADATA_DIR, SUBSTACK_TEXT_DIR


def canonicalize_url(url: str) -> str:
    """Canonicalize common tracking variants so idempotency is stable."""
    parsed = urlparse(url.strip())
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)

    def _keep(k: str) -> bool:
        kl = k.lower()
        if kl.startswith("utm_"):
            return False
        if kl in {"ref", "fbclid", "gclid", "igshid"}:
            return False
        return True

    kept = [(k, v) for (k, v) in query_pairs if _keep(k)]
    new_query = urlencode(kept, doseq=True)

    # Normalize: drop fragment, trim trailing ? and /
    normalized = parsed._replace(query=new_query, fragment="")
    rebuilt = urlunparse(normalized)
    rebuilt = rebuilt.rstrip("?")
    # Keep scheme+netloc, but normalize path trailing slash
    if rebuilt.endswith("/") and parsed.path not in {"/"}:
        rebuilt = rebuilt.rstrip("/")
    return rebuilt


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "article"


def doc_id_from_canonical_url(canonical_url: str) -> str:
    """Create a stable filesystem-safe doc id derived from the canonical URL."""
    parsed = urlparse(canonical_url)
    slug = _slugify(Path(parsed.path).name or "article")
    short_hash = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{short_hash}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _best_effort_iso_utc(published: str | None) -> str | None:
    """Convert feedparser-ish timestamps to ISO 8601 UTC when possible."""
    if not published:
        return None
    # feedparser provides parsed tuples on entry.published_parsed; parse here is best-effort only
    # as some feeds include non-RFC822 formats.
    try:
        dt = datetime.fromisoformat(published)
    except Exception:
        dt = None
    if dt is None:
        return published
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def html_to_markdownish_text(html: str) -> str:
    """Convert HTML to readable plaintext/markdown-ish text without extra deps.

    This is deliberately conservative: preserve paragraph breaks, headings, and links
    (anchor text + URL).
    """
    # Remove script/style blocks
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", html)

    # Links: <a href="...">Text</a> -> Text (url)
    def _link_repl(m: re.Match[str]) -> str:
        href = m.group("href").strip()
        text = re.sub(r"(?is)<[^>]+>", "", m.group("text") or "").strip()
        if not text:
            return href
        return f"{text} ({href})"

    html = re.sub(
        r'(?is)<a[^>]+href=["\'](?P<href>[^"\']+)["\'][^>]*>(?P<text>.*?)</a>',
        _link_repl,
        html,
    )

    # Headings -> blank lines + prefix
    html = re.sub(r"(?is)<h1[^>]*>(.*?)</h1>", r"\n\n# \1\n\n", html)
    html = re.sub(r"(?is)<h2[^>]*>(.*?)</h2>", r"\n\n## \1\n\n", html)
    html = re.sub(r"(?is)<h3[^>]*>(.*?)</h3>", r"\n\n### \1\n\n", html)

    # Paragraphs / breaks / list items
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</p\s*>", "\n\n", html)
    html = re.sub(r"(?is)<p[^>]*>", "", html)
    html = re.sub(r"(?is)<li[^>]*>", "- ", html)
    html = re.sub(r"(?is)</li\s*>", "\n", html)
    html = re.sub(r"(?is)</(ul|ol)\s*>", "\n", html)
    html = re.sub(r"(?is)<hr\s*/?>", "\n\n---\n\n", html)

    # Strip remaining tags
    html = re.sub(r"(?is)<[^>]+>", "", html)

    # Decode entities and normalize whitespace
    html = html.replace("\u00a0", " ")
    html = re.sub(r"[ \t]+\n", "\n", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


@dataclass(frozen=True)
class IngestResult:
    """Outcome of a single RSS item ingest attempt."""

    status: str  # created|updated|skipped|failed
    doc_id: str | None
    canonical_url: str | None
    reason: str | None = None


def _load_existing_index(metadata_dir: Path) -> dict[str, dict[str, Any]]:
    """Load existing metadata keyed by canonical_url."""
    index: dict[str, dict[str, Any]] = {}
    for path in metadata_dir.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            canonical_url = data.get("canonical_url")
            if canonical_url:
                index[str(canonical_url)] = data
        except Exception:
            continue
    return index


def _extract_entry_html(entry: Any) -> str:
    # Prefer content:encoded (feedparser maps to entry.content[*].value)
    if hasattr(entry, "content") and entry.content:
        c0 = entry.content[0]
        if isinstance(c0, dict) and c0.get("value"):
            return str(c0["value"])
        if getattr(c0, "value", None):
            return str(c0.value)
    # Fallback to description/summary
    return str(entry.get("summary") or entry.get("description") or "")


def ingest_substack_feed(
    *,
    feed_url: str = SUBSTACK_FEED_URL,
    source_name: str = "future_of_education_substack",
    source_type: str = "substack_article",
    limit: int | None = None,
    target_new: int | None = None,
    allow_updates: bool = True,
    metadata_dir: Path = SUBSTACK_METADATA_DIR,
    html_dir: Path = SUBSTACK_HTML_DIR,
    text_dir: Path = SUBSTACK_TEXT_DIR,
) -> list[IngestResult]:
    """Ingest a Substack RSS feed and persist raw+normalized artifacts.

    Args:
        feed_url: RSS feed URL.
        source_name: Stored in metadata for provenance.
        source_type: Stored in metadata for provenance.
        limit: Max number of feed entries to *scan* (not "new items"). If None, scans all entries.
        target_new: Stop once this many items are created/updated (best-effort; limited by feed size).
        allow_updates: If False, never overwrite existing artifacts even if content changes.
        metadata_dir/html_dir/text_dir: Output artifact directories.
    """
    metadata_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    feed = feedparser.parse(feed_url)
    entries = list(feed.entries)
    if limit is not None:
        entries = entries[: max(0, limit)]

    existing = _load_existing_index(metadata_dir)
    results: list[IngestResult] = []
    new_count = 0

    for entry in entries:
        try:
            title = str(entry.get("title") or "").strip() or "Untitled"
            url = str(entry.get("link") or "").strip()
            if not url:
                results.append(IngestResult(status="failed", doc_id=None, canonical_url=None, reason="Missing <link>"))
                continue
            canonical_url = canonicalize_url(url)
            doc_id = doc_id_from_canonical_url(canonical_url)

            raw_html = _extract_entry_html(entry)
            text = html_to_markdownish_text(raw_html)
            content_hash = _sha256_text(text)

            prev = existing.get(canonical_url)
            if prev and prev.get("content_hash") == content_hash:
                results.append(IngestResult(status="skipped", doc_id=doc_id, canonical_url=canonical_url, reason="Unchanged"))
                continue
            if prev and not allow_updates:
                results.append(IngestResult(status="skipped", doc_id=doc_id, canonical_url=canonical_url, reason="Updates disabled"))
                continue

            published = entry.get("published")
            published_iso = _best_effort_iso_utc(str(published)) if published else None
            author = entry.get("author") or entry.get("dc_creator") or entry.get("creator")
            guid = entry.get("id") or entry.get("guid")
            summary_html = entry.get("summary") or entry.get("description")

            meta: dict[str, Any] = {
                "doc_id": doc_id,
                "source": source_name,
                "source_type": source_type,
                "type": source_type,
                "title": title,
                "url": url,
                "canonical_url": canonical_url,
                "published": str(published) if published else None,
                "published_at": published_iso,
                "author": str(author) if author else None,
                "guid": str(guid) if guid else None,
                "summary_html": str(summary_html) if summary_html else None,
                "content_hash": content_hash,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }

            # Persist artifacts
            meta_path = metadata_dir / f"{doc_id}.json"
            html_path = html_dir / f"{doc_id}.html"
            text_path = text_dir / f"{doc_id}.md"

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(raw_html)
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")

            status = "updated" if prev else "created"
            results.append(IngestResult(status=status, doc_id=doc_id, canonical_url=canonical_url))
            new_count += 1

            if target_new is not None and new_count >= max(0, target_new):
                break
        except Exception as e:
            results.append(IngestResult(status="failed", doc_id=None, canonical_url=None, reason=str(e)))

    return results


