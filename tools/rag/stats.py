#!/usr/bin/env python3
"""Show statistics about the indexed wiki content in ChromaDB."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _common import output, error


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()

    try:
        from rag_system import RAGSystem
    except ImportError as e:
        error(f"Cannot import RAGSystem: {e}")

    rag = RAGSystem()
    stats = rag.get_stats()

    output(stats)


if __name__ == "__main__":
    main()
