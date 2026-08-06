#!/usr/bin/env python3
"""Show statistics about the indexed wiki content in ChromaDB."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _common import output, error
import os as _ga_os, sys as _ga_sys
_ga_sys.path.insert(0, _ga_os.path.join(_ga_os.path.dirname(_ga_os.path.abspath(__file__)), '..'))
from _guild_access import require_integration


def main():
    # Per-guild tool gating (tools/_guild_access.py). The guild id comes
    # from the trusted env var the bot injects, never from arguments.
    require_integration('rag')
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
