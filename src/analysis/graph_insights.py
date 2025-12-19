"""Inspect Neo4j for "big nodes", distributions, and (optional) communities.

This complements:
- `src.analysis.inspect_graph` (quick schema + top connected nodes)
- `src.analysis.investigate_entity` (local neighborhood + docs)

It focuses on:
- Label/relationship distributions (what's growing?)
- Top-degree nodes by label (what are the biggest hubs?)
- Relationship patterns (what kinds of edges dominate?)
- Degree distributions (is it hub-and-spoke?)
- Co-mention networks (what concepts/people co-occur?)
- Optional: Graph Data Science (GDS) calls if the plugin is installed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

from src.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("analysis_outputs/neo4j")


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


def _try_import_networkx() -> Any | None:
    try:
        import networkx as nx  # type: ignore

        return nx
    except Exception:
        return None


def _require_neo4j_config() -> None:
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        raise RuntimeError(
            "Missing Neo4j configuration. Ensure NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD are set."
        )


def _connect_graph() -> Neo4jGraph:
    _require_neo4j_config()
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    graph.refresh_schema()
    return graph


def _print_table(title: str, rows: list[dict[str, Any]], limit: int = 25) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(no results)")
        return
    for row in rows[:limit]:
        print(row)


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    pd = _try_import_pandas()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return
    # Minimal fallback without pandas
    import csv

    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_bar(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    out_png: Path,
    max_bars: int = 30,
) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib not installed. Install with: `uv add matplotlib`")
    sns = _try_import_seaborn()

    sliced = rows[:max_bars]
    xs = [str(r.get(x_key, "")) for r in sliced]
    ys = [float(r.get(y_key, 0)) for r in sliced]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    if sns is not None:
        sns.barplot(x=xs, y=ys)
    else:
        plt.bar(xs, ys)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_degree_hist(degrees: list[int], title: str, out_png: Path) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib not installed. Install with: `uv add matplotlib`")
    sns = _try_import_seaborn()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    if sns is not None:
        sns.histplot(degrees, bins=50, log_scale=(False, True))
    else:
        plt.hist(degrees, bins=50)
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("degree")
    plt.ylabel("count (log scale)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_comention_network(edges: list[dict[str, Any]], out_png: Path) -> None:
    """Plot a small co-mention network (requires networkx + matplotlib)."""
    nx = _try_import_networkx()
    plt = _try_import_matplotlib()
    if nx is None or plt is None:
        raise RuntimeError("Install networkx + matplotlib for network plots: `uv add networkx matplotlib`")

    g = nx.Graph()
    for e in edges:
        a = str(e.get("a", ""))
        b = str(e.get("b", ""))
        w = float(e.get("co_mentions", 1))
        if a and b and a != b:
            g.add_edge(a, b, weight=w)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(g, seed=42, k=0.6)
    weights = [g[u][v].get("weight", 1.0) for u, v in g.edges()]
    # Scale edge width gently
    widths = [max(0.5, min(6.0, w / 3.0)) for w in weights]
    nx.draw_networkx_edges(g, pos, alpha=0.35, width=widths)
    nx.draw_networkx_nodes(g, pos, node_size=120, alpha=0.9)
    nx.draw_networkx_labels(g, pos, font_size=8)
    plt.title("Top co-mentions network (entity co-occurrence via documents)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def graph_insights(
    top_k: int,
    out_dir: Path,
    write_csv: bool,
    plots: bool,
    plot_network: bool,
    explain: bool,
) -> None:
    """Run a set of read-only queries that reveal graph size + structure."""
    graph = _connect_graph()

    counts = graph.query(
        """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->()
        RETURN count(distinct n) as nodes, count(r) as relationships
        """
    )
    _print_table("Totals", counts, limit=1)
    if explain and counts:
        c0 = counts[0]
        print(
            "\nHow to read this:\n"
            f"- nodes={c0.get('nodes')}: total unique nodes in the graph\n"
            f"- relationships={c0.get('relationships')}: total edges\n"
            "- If relationships >> nodes, your graph is richly connected; if nodes >> relationships, it’s sparse.\n"
        )

    label_counts = graph.query(
        """
        MATCH (n)
        UNWIND labels(n) AS label
        RETURN label, count(*) AS count
        ORDER BY count DESC
        """
    )
    _print_table("Node counts by label", label_counts)
    if explain and label_counts:
        print(
            "\nHow to read this:\n"
            "- This tells you *what kinds of things* dominate your graph.\n"
            "- If `Document` is large, you’ve ingested lots of sources; if `Concept`/`Person` is large, extraction is productive.\n"
        )

    rel_counts = graph.query(
        """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(*) AS count
        ORDER BY count DESC
        """
    )
    _print_table("Relationship counts by type", rel_counts)
    if explain and rel_counts:
        print(
            "\nHow to read this:\n"
            "- This is your “verb distribution”. It’s the fastest way to see what your extractor is *doing*.\n"
            "- Example: lots of `MENTIONS` means broad entity linking; more `CAUSES`/`CONTRADICTS` means richer reasoning edges.\n"
        )

    if write_csv:
        _write_csv(counts, out_dir / "totals.csv")
        _write_csv(label_counts, out_dir / "node_counts_by_label.csv")
        _write_csv(rel_counts, out_dir / "relationship_counts_by_type.csv")

    if plots:
        try:
            _plot_bar(
                label_counts,
                x_key="label",
                y_key="count",
                title="Node counts by label",
                out_png=out_dir / "node_counts_by_label.png",
            )
            _plot_bar(
                rel_counts,
                x_key="rel_type",
                y_key="count",
                title="Relationship counts by type",
                out_png=out_dir / "relationship_counts_by_type.png",
            )
        except Exception as e:
            logger.info(f"(plots skipped): {e}")

    # Highest-degree nodes overall (undirected degree)
    top_degree = graph.query(
        """
        MATCH (n)-[r]-()
        RETURN coalesce(n.id, n.title, elementId(n)) AS id,
               labels(n) AS labels,
               count(r) AS degree
        ORDER BY degree DESC
        LIMIT $k
        """,
        params={"k": top_k},
    )
    _print_table(f"Top {top_k} nodes by degree", top_degree, limit=top_k)
    if explain and top_degree:
        print(
            "\nHow to read this:\n"
            "- Degree = how many edges touch the node.\n"
            "- The top nodes are your “hubs” (frequent guests, recurring concepts, or noisy entities like Speaker A/B).\n"
            "- If you see lots of Speaker labels, that’s a good cleanup target (normalize diarization entities).\n"
        )

    if write_csv:
        _write_csv(top_degree, out_dir / "top_nodes_by_degree.csv")

    # Highest-degree per label (helps avoid Document nodes drowning everything).
    per_label = graph.query(
        """
        MATCH (n)-[r]-()
        WITH labels(n)[0] AS label,
             coalesce(n.id, n.title, elementId(n)) AS id,
             count(r) AS degree
        ORDER BY label, degree DESC
        WITH label, collect({id: id, degree: degree})[0..$k] AS top
        RETURN label, top
        ORDER BY label ASC
        """,
        params={"k": min(top_k, 20)},
    )
    _print_table("Top nodes per label", per_label, limit=1000)
    if explain and per_label:
        print(
            "\nHow to read this:\n"
            "- This prevents `Document` nodes from hiding the real hubs.\n"
            "- Look specifically at the `Concept` and `Person` label rows: those are usually your most useful entry points.\n"
        )

    if write_csv:
        _write_csv(per_label, out_dir / "top_nodes_per_label.csv")

    # Documents with most mentions (if that schema exists)
    docs_by_mentions = graph.query(
        """
        MATCH (d:Document)-[:MENTIONS]->(n)
        RETURN coalesce(d.title, d.id, elementId(d)) AS doc,
               count(distinct n) AS unique_mentions
        ORDER BY unique_mentions DESC
        LIMIT $k
        """,
        params={"k": top_k},
    )
    _print_table(f"Top {top_k} documents by unique mentions", docs_by_mentions, limit=top_k)
    if explain and docs_by_mentions:
        print(
            "\nHow to read this:\n"
            "- Documents with high `unique_mentions` are “dense” episodes/posts (tons of distinct entities).\n"
            "- These are great candidates for: series summaries, curated reading lists, and graph neighborhood exploration.\n"
        )

    if write_csv:
        _write_csv(docs_by_mentions, out_dir / "top_documents_by_unique_mentions.csv")

    # Degree distribution (for trend/hubiness)
    degree_rows = graph.query(
        """
        MATCH (n)
        RETURN COUNT { (n)--() } AS degree
        """
    )
    degrees = [int(r.get("degree", 0)) for r in degree_rows if r.get("degree") is not None]
    print(f"\nDegree distribution: n={len(degrees)}, max={max(degrees) if degrees else 0}")
    if explain and degrees:
        print(
            "\nHow to read this:\n"
            "- If the histogram has a long tail (few nodes with huge degree), it’s hub-and-spoke.\n"
            "- That’s normal for “mentions” graphs, but you’ll want community detection / co-mention subgraphs to see clusters.\n"
        )
    if write_csv:
        _write_csv([{"degree": d} for d in degrees], out_dir / "degree_distribution.csv")
    if plots:
        try:
            _plot_degree_hist(degrees, "Degree distribution (log y)", out_dir / "degree_distribution.png")
        except Exception as e:
            logger.info(f"(degree histogram skipped): {e}")

    # Co-mention edges (Document -> Entity -> Document induces entity co-occurrence)
    comention_edges = graph.query(
        """
        MATCH (d:Document)-[:MENTIONS]->(e)
        WITH d, collect(distinct e) AS ents
        UNWIND ents AS e1
        UNWIND ents AS e2
        WITH e1, e2
        WHERE elementId(e1) < elementId(e2)
        RETURN coalesce(e1.id, e1.title, elementId(e1)) AS a,
               coalesce(e2.id, e2.title, elementId(e2)) AS b,
               count(*) AS co_mentions
        ORDER BY co_mentions DESC
        LIMIT $k
        """,
        params={"k": top_k},
    )
    _print_table(f"Top {top_k} entity co-mentions (edge list)", comention_edges, limit=top_k)
    if explain and comention_edges:
        print(
            "\nHow to read this:\n"
            "- These pairs co-occur in the same documents most often.\n"
            "- Treat them as “topic glue”: they often indicate a stable theme (e.g., a founder + school model + AI).\n"
        )
    if write_csv:
        _write_csv(comention_edges, out_dir / "top_entity_comentions.csv")
    if plots and plot_network:
        try:
            _plot_comention_network(comention_edges, out_dir / "top_entity_comentions_network.png")
        except Exception as e:
            logger.info(f"(network plot skipped): {e}")

    # Optional: check for GDS plugin and show a suggested community workflow.
    try:
        gds_version = graph.query("CALL gds.version() YIELD version RETURN version LIMIT 1")
        _print_table("GDS version (plugin detected)", gds_version, limit=1)

        print(
            "\n--- Suggested next step (community detection with GDS) ---\n"
            "If you want clustering/communities, a common pattern is:\n"
            "1) Project an in-memory graph (pick labels/relationships)\n"
            "2) Run WCC/Louvain/Leiden\n"
            "3) Write back `community` as a node property for visualization in Neo4j Browser/Bloom\n"
        )
        print(
            "Example (edit labels/rels to match your schema):\n"
            "CALL gds.graph.project('kg', ['Person','Concept','Organization'],\n"
            "  {MENTIONS: {orientation:'UNDIRECTED'}, RELATED_TO:{orientation:'UNDIRECTED'}});\n"
            "CALL gds.louvain.write('kg', {writeProperty:'community'});\n"
            "CALL gds.graph.drop('kg');\n"
        )
    except Exception as e:
        logger.info(f"\n(GDS not available or not configured): {e}")
        logger.info(
            "To do clustering visually without GDS: run focused Cypher queries that return a subgraph\n"
            "and use Neo4j Browser's graph view (or Neo4j Bloom)."
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Richer Neo4j inspection: biggest nodes + distributions.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="How many top results to print for rank-based tables.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write CSVs/PNGs into.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write CSV outputs.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write PNG plots (requires matplotlib; better with seaborn).",
    )
    parser.add_argument(
        "--plot-network",
        action="store_true",
        help="Also render a small co-mention network PNG (requires networkx).",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print short plain-English hints for interpreting each section.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    graph_insights(
        top_k=args.top_k,
        out_dir=args.out_dir,
        write_csv=not args.no_csv,
        plots=args.plots,
        plot_network=args.plot_network,
        explain=args.explain,
    )


