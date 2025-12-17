"""Substack source runner (scaffold).

This will become the canonical orchestration entrypoint for ingesting Substack posts,
processing them into segments/summaries, and optionally indexing into Chroma + Neo4j.

See: planning/substack-plan.md
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Substack pipeline (not implemented yet).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.parse_args()

    raise NotImplementedError(
        "Substack runner is not implemented yet. See planning/substack-plan.md for the planned pipeline."
    )


if __name__ == "__main__":
    main()
