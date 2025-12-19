"""Visualize and summarize the Chroma vector database.

This module is intentionally read-only: it connects to the persisted Chroma DB,
pulls stored embeddings + metadata, reduces embeddings to 2D, and writes a
scatter-friendly output (CSV) and optionally an interactive HTML plot.

Why this exists:
- `src.analysis.inspect_chroma` is great for quick sanity checks and a search.
- This tool helps you *see clusters* and inspect trends across metadata.

Dependencies:
- Works in "summary/CSV-only" mode with existing project deps.
- For best results, install one of:
  - UMAP for nonlinear projection: `uv add umap-learn`
  - Or sklearn for PCA: `uv add scikit-learn`
  - For interactive plot: `uv add plotly`
  - For static PNG plot: `uv add pandas matplotlib seaborn`
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dotenv import load_dotenv

from src.config import CHROMA_DIR

load_dotenv()

COLLECTION_NAME = "education_knowledge_engine"


@dataclass(frozen=True)
class ChromaPoint:
    """One vector record projected into 2D."""

    id: str
    x: float
    y: float
    metadata: Mapping[str, Any]
    document_preview: str


def _safe_preview(text: str | None, limit: int = 200) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.split())
    return cleaned[:limit]

def _safe_excerpt(text: str | None, limit: int = 2000) -> str:
    """Longer excerpt suitable for JSONL export (still bounded)."""
    if not text:
        return ""
    cleaned = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return cleaned[:limit]


def _try_import_umap() -> Any | None:
    try:
        import umap  # type: ignore

        return umap
    except Exception:
        return None


def _try_import_sklearn() -> Any | None:
    try:
        from sklearn.decomposition import PCA  # type: ignore

        return PCA
    except Exception:
        return None


def _try_import_plotly() -> Any | None:
    try:
        import plotly.express as px  # type: ignore

        return px
    except Exception:
        return None


def _try_import_matplotlib() -> Any | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _try_import_pandas() -> Any | None:
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception:
        return None


def _try_import_seaborn() -> Any | None:
    try:
        import seaborn as sns  # type: ignore

        return sns
    except Exception:
        return None


def _connect_collection(persist_dir: Path, name: str):
    """Connect to a persisted Chroma collection via the native client."""
    import chromadb  # chromadb is already a project dependency

    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(name=name)


def _iter_batches(
    collection: Any,
    limit: int,
    batch_size: int,
) -> Iterable[dict[str, Any]]:
    """Yield collection.get() batches with embeddings included.

    Chroma's API supports `limit` + `offset`. We keep it simple and deterministic.
    """
    if limit <= 0:
        return

    remaining = limit
    offset = 0
    while remaining > 0:
        take = min(batch_size, remaining)
        batch = collection.get(
            limit=take,
            offset=offset,
            include=["embeddings", "metadatas", "documents"],
        )
        yield batch
        got = len(batch.get("ids") or [])
        if got <= 0:
            return
        remaining -= got
        offset += got

def _none_to_empty(value: Any) -> Any:
    """Avoid `x or []` with numpy arrays (truthiness is ambiguous)."""
    return [] if value is None else value


def _to_py_list(value: Any) -> list[Any]:
    """Convert numpy arrays (or tuples) to plain Python lists."""
    try:
        # numpy.ndarray has .tolist()
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return list(tolist())
    except Exception:
        pass
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except Exception:
        return [value]


def _reduce_to_2d(
    embeddings: Sequence[Sequence[float]],
    method: str,
    random_state: int,
) -> list[tuple[float, float]]:
    """Reduce embeddings to 2D via UMAP or PCA (depending on availability)."""
    import numpy as np

    arr = np.asarray(embeddings, dtype="float32")
    if arr.ndim != 2 or arr.shape[0] < 2:
        # Degenerate output; avoid crashing in reducers.
        return [(0.0, 0.0) for _ in range(int(arr.shape[0]))]

    def numpy_pca_2d(a: "np.ndarray") -> "np.ndarray":
        """PCA to 2D using only numpy (SVD on centered data)."""
        centered = a - a.mean(axis=0, keepdims=True)
        # Vt rows are principal directions; project onto first 2.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2].T  # (dims, 2)
        return centered @ components  # (n, 2)

    method_norm = method.lower().strip()
    if method_norm == "umap":
        umap = _try_import_umap()
        if umap is None:
            # Fall back gracefully so this script works out-of-the-box.
            print("UMAP not installed; falling back to numpy PCA. (Install: `uv add umap-learn`)")
            coords = numpy_pca_2d(arr)
        else:
            reducer = umap.UMAP(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(arr)
    elif method_norm == "pca":
        PCA = _try_import_sklearn()
        if PCA is None:
            coords = numpy_pca_2d(arr)
        else:
            reducer = PCA(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(arr)
    else:
        raise ValueError("Unknown --method. Use 'umap' or 'pca'.")

    out: list[tuple[float, float]] = []
    for row in coords:
        x = float(row[0])
        y = float(row[1])
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
        else:
            out.append((0.0, 0.0))
    return out


def _write_csv(points: Sequence[ChromaPoint], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "x", "y", "metadata_json", "document_preview"],
        )
        writer.writeheader()
        for p in points:
            writer.writerow(
                {
                    "id": p.id,
                    "x": f"{p.x:.6f}",
                    "y": f"{p.y:.6f}",
                    "metadata_json": json.dumps(p.metadata, ensure_ascii=False),
                    "document_preview": p.document_preview,
                }
            )


def _write_html_plot(points: Sequence[ChromaPoint], out_html: Path) -> None:
    px = _try_import_plotly()
    if px is None:
        raise RuntimeError("Plotly not installed. Install with: `uv add plotly`")

    # Plotly wants columnar data.
    rows: list[dict[str, Any]] = []
    for p in points:
        row = {
            "id": p.id,
            "x": p.x,
            "y": p.y,
            "preview": p.document_preview,
            "metadata_json": json.dumps(p.metadata, ensure_ascii=False),
        }
        # Flatten common metadata keys if present (helps coloring/filtering).
        for k in ("source", "type", "topic", "episode", "title"):
            if k in p.metadata:
                row[k] = p.metadata.get(k)
        rows.append(row)

    def choose_color_key(color_by: str | None) -> str | None:
        if not color_by:
            return None
        # Only color if at least one row has a non-null value for this key.
        for r in rows:
            if r.get(color_by) is not None:
                return color_by
        return None

    fig = px.scatter(
        rows,
        x="x",
        y="y",
        hover_name="id",
        hover_data=["preview", "source", "type", "topic", "episode", "title"],
        color=choose_color_key(None),
        title=f"Chroma Embeddings Projection ({COLLECTION_NAME})",
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")

def _write_html_plot_with_color(points: Sequence[ChromaPoint], out_html: Path, color_by: str | None) -> None:
    """HTML plot that respects a user-provided `--color-by` when possible."""
    # Use graph_objects for better hover control + click handlers.
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as e:
        raise RuntimeError("Plotly not installed. Install with: `uv add plotly`") from e

    # Build flat rows and a short, readable hover payload.
    rows: list[dict[str, Any]] = []
    for p in points:
        meta = dict(p.metadata or {})
        row: dict[str, Any] = {
            "id": p.id,
            "x": p.x,
            "y": p.y,
            "type": meta.get("type"),
            "source": meta.get("source"),
            "source_type": meta.get("source_type"),
            "title": meta.get("title"),
            "doc_id": meta.get("doc_id"),
            "canonical_url": meta.get("canonical_url"),
            "topic": meta.get("topic"),
            "author": meta.get("author"),
            "published_at": meta.get("published_at"),
            "snippet": p.document_preview,
        }
        # Allow arbitrary color key (only if present).
        if color_by:
            row[color_by] = meta.get(color_by)
        rows.append(row)

    def _safe(v: Any) -> str:
        if v is None:
            return ""
        return html.escape(str(v))

    def _build_hover(row: dict[str, Any]) -> str:
        url = row.get("canonical_url")
        url_line = f"<a href='{_safe(url)}' target='_blank'>open source</a><br>" if url else ""
        return (
            f"<b>{_safe(row.get('title') or row.get('doc_id') or row.get('id'))}</b><br>"
            f"Type: {_safe(row.get('type'))}<br>"
            f"Source: {_safe(row.get('source'))} {_safe(row.get('source_type'))}<br>"
            f"doc_id: <span style='font-family: monospace'>{_safe(row.get('doc_id'))}</span><br>"
            f"id: <span style='font-family: monospace'>{_safe(row.get('id'))}</span><br>"
            f"{url_line}"
            f"<span style='color:#666'>{_safe(row.get('snippet'))}</span>"
            "<br><i>(click a point to copy its id)</i>"
        )

    # Determine coloring.
    color_key = None
    if color_by and any(r.get(color_by) is not None for r in rows):
        color_key = color_by
    elif any(r.get("type") is not None for r in rows):
        color_key = "type"

    # Group traces by color key so legend toggles are useful.
    groups: dict[str, list[dict[str, Any]]] = {}
    if color_key:
        for r in rows:
            groups.setdefault(str(r.get(color_key) or "unknown"), []).append(r)
    else:
        groups["all"] = rows

    fig = go.Figure()
    for group_name, items in groups.items():
        fig.add_trace(
            go.Scattergl(
                x=[i["x"] for i in items],
                y=[i["y"] for i in items],
                mode="markers",
                name=group_name,
                marker=dict(size=7, opacity=0.7),
                # customdata[0] is the id we copy on click
                customdata=[[i["id"]] for i in items],
                hovertemplate="%{text}<extra></extra>",
                text=[_build_hover(i) for i in items],
            )
        )

    fig.update_layout(
        title=(
            f"Chroma Embeddings Projection ({COLLECTION_NAME}) — "
            "zoom/pan, lasso select, legend-click to filter"
        ),
        hovermode="closest",
        legend_title_text=color_key or "group",
        margin=dict(l=40, r=20, t=70, b=40),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
    )
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y")

    # Click-to-copy for the Chroma id.
    post_script = """
const plot = document.getElementsByClassName('plotly-graph-div')[0];
if (plot) {
  plot.on('plotly_click', function(data) {
    try {
      const id = data.points && data.points[0] && data.points[0].customdata && data.points[0].customdata[0];
      if (!id) return;
      navigator.clipboard.writeText(id);
    } catch (e) {}
  });
}
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out_html),
        include_plotlyjs="cdn",
        post_script=post_script,
        config={"displayModeBar": True, "scrollZoom": True},
    )


def _write_jsonl_export(
    ids: Sequence[str],
    coords: Sequence[tuple[float, float]],
    metadatas: Sequence[Mapping[str, Any]],
    documents: Sequence[str],
    out_jsonl: Path,
    doc_max_chars: int,
) -> None:
    """Write a JSONL lookup file so you can grep/inspect points outside the plot."""
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, doc_id in enumerate(ids):
            x, y = coords[i]
            payload = {
                "id": doc_id,
                "x": float(x),
                "y": float(y),
                "metadata": metadatas[i],
                "document_excerpt": _safe_excerpt(documents[i], limit=doc_max_chars),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_png_plot(points: Sequence[ChromaPoint], out_png: Path, color_by: str | None) -> None:
    """Write a static scatter plot (PNG) using matplotlib (and seaborn if available)."""
    plt = _try_import_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib not installed. Install with: `uv add matplotlib`")

    pd = _try_import_pandas()
    sns = _try_import_seaborn()

    # Convert to a dataframe-like structure if pandas exists; otherwise plot raw.
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if pd is not None:
        rows: list[dict[str, Any]] = []
        for p in points:
            row: dict[str, Any] = {"x": p.x, "y": p.y, "id": p.id, "preview": p.document_preview}
            if color_by:
                row[color_by] = p.metadata.get(color_by)
            rows.append(row)

        df = pd.DataFrame(rows)
        plt.figure(figsize=(10, 8))
        if sns is not None and color_by:
            # seaborn handles categorical palettes nicely
            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue=color_by,
                s=12,
                alpha=0.65,
                linewidth=0,
            )
            plt.legend(loc="best", fontsize="small", frameon=True)
        else:
            plt.scatter(df["x"], df["y"], s=10, alpha=0.65)
        plt.title(f"Chroma 2D Projection ({COLLECTION_NAME})")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        return

    # Fallback without pandas
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    plt.figure(figsize=(10, 8))
    plt.scatter(xs, ys, s=10, alpha=0.65)
    plt.title(f"Chroma 2D Projection ({COLLECTION_NAME})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def visualize_chroma(
    limit: int,
    batch_size: int,
    method: str,
    random_state: int,
    out_csv: Path,
    out_html: Path | None,
    out_png: Path | None,
    color_by: str | None,
    out_jsonl: Path | None,
    doc_max_chars: int,
) -> None:
    """Generate a 2D projection of stored embeddings and write outputs."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma directory not found: {CHROMA_DIR}. Have you run indexing?"
        )

    collection = _connect_collection(CHROMA_DIR, COLLECTION_NAME)
    total = collection.count()
    print(f"Chroma collection '{COLLECTION_NAME}' docs: {total}")

    embeddings: list[list[float]] = []
    ids: list[str] = []
    metadatas: list[Mapping[str, Any]] = []
    documents: list[str] = []

    for batch in _iter_batches(collection, limit=limit, batch_size=batch_size):
        batch_ids = _to_py_list(_none_to_empty(batch.get("ids")))
        batch_embeddings = _to_py_list(_none_to_empty(batch.get("embeddings")))
        batch_metadatas = _to_py_list(_none_to_empty(batch.get("metadatas")))
        batch_docs = _to_py_list(_none_to_empty(batch.get("documents")))

        # Chroma types are a bit loose; coerce defensively.
        for i in range(len(batch_ids)):
            ids.append(str(batch_ids[i]))
            metadatas.append((batch_metadatas[i] or {}) if i < len(batch_metadatas) else {})
            documents.append(str(batch_docs[i]) if i < len(batch_docs) else "")
            # Each embedding row might itself be a numpy array.
            if i < len(batch_embeddings):
                embeddings.append([float(x) for x in _to_py_list(batch_embeddings[i])])
            else:
                embeddings.append([])

    if not ids:
        raise RuntimeError("No records read from Chroma (collection may be empty).")

    print(f"Read {len(ids)} records for visualization.")
    coords = _reduce_to_2d(embeddings, method=method, random_state=random_state)

    points: list[ChromaPoint] = []
    for idx, (x, y) in enumerate(coords):
        points.append(
            ChromaPoint(
                id=ids[idx],
                x=x,
                y=y,
                metadata=metadatas[idx],
                document_preview=_safe_preview(documents[idx]),
            )
        )

    _write_csv(points, out_csv)
    print(f"Wrote CSV: {out_csv}")

    if out_jsonl is not None:
        try:
            _write_jsonl_export(
                ids=ids,
                coords=coords,
                metadatas=metadatas,
                documents=documents,
                out_jsonl=out_jsonl,
                doc_max_chars=doc_max_chars,
            )
            print(f"Wrote JSONL: {out_jsonl}")
        except Exception as e:
            print(f"Skipping JSONL export: {e}")

    if out_html is not None:
        try:
            _write_html_plot_with_color(points, out_html, color_by=color_by)
            print(f"Wrote HTML: {out_html}")
        except Exception as e:
            print(f"Skipping HTML plot: {e}")

    if out_png is not None:
        try:
            _write_png_plot(points, out_png, color_by=color_by)
            print(f"Wrote PNG: {out_png}")
        except Exception as e:
            print(f"Skipping PNG plot: {e}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize Chroma embeddings as a 2D projection.")
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Max number of records to load from Chroma for visualization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for Chroma reads.",
    )
    parser.add_argument(
        "--method",
        choices=["umap", "pca"],
        default="umap",
        help="2D projection method.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the reducer.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("analysis_outputs/chroma_projection.csv"),
        help="Where to write the CSV output.",
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("analysis_outputs/chroma_projection.html"),
        help="Where to write an interactive HTML plot (requires plotly).",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip writing the HTML plot (still writes CSV).",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("analysis_outputs/chroma_projection.png"),
        help="Where to write a static PNG scatter plot (requires matplotlib; better with seaborn/pandas).",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip writing the PNG plot.",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default=None,
        help="Metadata key to color points by in the PNG/HTML plot (e.g. 'type', 'topic', 'source').",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("analysis_outputs/chroma_projection.jsonl"),
        help="Write a JSONL lookup file (id/x/y/metadata/document excerpt).",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Skip writing the JSONL export.",
    )
    parser.add_argument(
        "--doc-max-chars",
        type=int,
        default=2000,
        help="Max chars of document text to include in JSONL export (per record).",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    visualize_chroma(
        limit=args.limit,
        batch_size=args.batch_size,
        method=args.method,
        random_state=args.random_state,
        out_csv=args.out_csv,
        out_html=None if args.no_html else args.out_html,
        out_png=None if args.no_png else args.out_png,
        color_by=args.color_by,
        out_jsonl=None if args.no_jsonl else args.out_jsonl,
        doc_max_chars=args.doc_max_chars,
    )


