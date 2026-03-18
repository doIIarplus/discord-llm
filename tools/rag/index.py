#!/usr/bin/env python3
"""Index a MediaWiki XML dump into the ChromaDB vector store."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _common import output, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-dump", default="maplestorywikinet.xml",
                        help="Path to MediaWiki XML dump")
    parser.add_argument("--clear-existing", action="store_true",
                        help="Clear the existing index before re-indexing")
    args = parser.parse_args()

    if not os.path.exists(args.wiki_dump):
        error(f"Wiki dump file not found: {args.wiki_dump}")

    try:
        from rag_system import RAGSystem
    except ImportError as e:
        error(f"Cannot import RAGSystem: {e}")

    rag = RAGSystem()

    if args.clear_existing:
        rag.clear_collection()

    rag.index_wiki_dump(args.wiki_dump)
    stats = rag.get_stats()

    output({
        "indexed": True,
        "wiki_dump": args.wiki_dump,
        "cleared_existing": args.clear_existing,
        "total_chunks": stats.get("total_chunks", 0),
        "collection_name": stats.get("collection_name", ""),
    })


if __name__ == "__main__":
    main()
