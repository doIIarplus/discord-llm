#!/usr/bin/env python3
"""Search the web using Tavily API and return structured results.

Returns raw search results as JSON — no LLM summarization.
Claude handles summarization and interpretation itself.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=5,
                        help="Maximum number of results (default: 5)")
    args = parser.parse_args()

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        error("TAVILY_API_KEY not set in environment")

    try:
        from tavily import TavilyClient
    except ImportError:
        error("tavily-python not installed. Run: pip install tavily-python")

    client = TavilyClient(api_key=api_key)
    response = client.search(args.query, max_results=args.max_results)

    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        })

    output({"query": args.query, "results": results})


if __name__ == "__main__":
    main()
