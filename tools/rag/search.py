#!/usr/bin/env python3
"""Search the indexed wiki content using ChromaDB vector similarity."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _common import output, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Search query")
    parser.add_argument("--n-results", type=int, default=3,
                        help="Number of results to return (default: 3)")
    args = parser.parse_args()

    try:
        from rag_system import RAGSystem
    except ImportError as e:
        error(f"Cannot import RAGSystem: {e}")

    rag = RAGSystem()
    results = rag.search(args.query, n_results=args.n_results)

    output({
        "query": args.query,
        "results": [
            {
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0),
            }
            for r in results
        ],
    })


if __name__ == "__main__":
    main()
