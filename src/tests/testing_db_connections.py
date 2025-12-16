"""Quick connectivity checks for Chroma and Neo4j data sources."""

import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

try:
    from src.database.chroma_manager import get_vector_store, query_segments
    from src.database.neo4j_manager import get_graph
    from src.config import CHROMA_DIR, SEGMENTED_DIR
except Exception as import_err:
    print(f"Import error: {import_err}")
    sys.exit(1)


def check_chroma(sample_query: str = "alpha school") -> bool:
    """Validate Chroma is reachable and returns results for a sample query."""
    try:
        vector_store = get_vector_store()
        results = query_segments(sample_query, k=1)
        print(f"Chroma OK: returned {len(results)} results for '{sample_query}'.")
        # Confirm the collection is readable
        try:
            stats: Optional[dict] = getattr(vector_store, "_collection", None)
            if stats and hasattr(stats, "count"):
                print(f"Chroma collection count: {stats.count()}")
        except Exception:
            # Collection stats are best-effort; ignore if unsupported
            pass
        return True
    except Exception as exc:  # pragma: no cover - best-effort runtime check
        print(f"Chroma check failed: {exc}")
        return False


def check_neo4j() -> bool:
    """Validate Neo4j is reachable and can execute a trivial query."""
    try:
        graph = get_graph()
        res = graph.query("MATCH (n) RETURN count(n) AS cnt LIMIT 1")
        count = res[0].get("cnt", 0) if res else 0
        print(f"Neo4j OK: node count sample={count}")
        return True
    except Exception as exc:  # pragma: no cover - best-effort runtime check
        print(f"Neo4j check failed: {exc}")
        return False


def check_filesystem() -> bool:
    """Ensure expected data directories exist and are readable."""
    ok = True
    if not CHROMA_DIR.exists():
        print(f"Chroma dir missing: {CHROMA_DIR}")
        ok = False
    else:
        print(f"Chroma dir present: {CHROMA_DIR}")

    if not SEGMENTED_DIR.exists():
        print(f"Segmented transcripts dir missing: {SEGMENTED_DIR}")
        ok = False
    else:
        sample_files = list(Path(SEGMENTED_DIR).glob("*.json"))[:3]
        print(f"Segmented transcripts present: {len(sample_files)} sample(s) {sample_files}")
    return ok


def main() -> int:
    """Run connectivity checks; exit 0 on success, 1 on any failure."""
    load_dotenv()
    fs_ok = check_filesystem()
    chroma_ok = check_chroma()
    neo4j_ok = check_neo4j()
    return 0 if fs_ok and chroma_ok and neo4j_ok else 1


if __name__ == "__main__":
    sys.exit(main())

