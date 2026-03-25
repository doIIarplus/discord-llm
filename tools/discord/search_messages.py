#!/usr/bin/env python3
"""Search messages in a Discord guild (server).

Uses Discord's message search API (available to bot accounts with the
MESSAGE_CONTENT intent). Returns matching messages with context.

Examples:
  # Search for a keyword in the whole server
  search_messages.py --guild-id 363154169294618625 --query "deployment"

  # Search in a specific channel
  search_messages.py --guild-id 363154169294618625 --query "bug fix" --channel-id 123456789

  # Search messages from a specific user
  search_messages.py --guild-id 363154169294618625 --query "reminder" --author-id 118567805678256128

  # Limit results
  search_messages.py --guild-id 363154169294618625 --query "error" --max-results 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error
from discord._client import DiscordClient


def _format_message(msg):
    """Extract key fields from a Discord message object."""
    return {
        "id": msg["id"],
        "author": msg["author"]["username"],
        "author_id": msg["author"]["id"],
        "content": msg["content"],
        "channel_id": msg["channel_id"],
        "timestamp": msg["timestamp"],
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--guild-id", required=True,
                        help="Discord guild (server) ID")
    parser.add_argument("--query", required=True,
                        help="Search query string")
    parser.add_argument("--channel-id", default=None,
                        help="Limit search to this channel")
    parser.add_argument("--author-id", default=None,
                        help="Limit search to this author")
    parser.add_argument("--max-results", type=int, default=10,
                        help="Maximum results to return (default 10, max 25)")
    args = parser.parse_args()

    client = DiscordClient()

    params = {"content": args.query}
    if args.channel_id:
        params["channel_id"] = args.channel_id
    if args.author_id:
        params["author_id"] = args.author_id

    data = client.get(f"/guilds/{args.guild_id}/messages/search", params)

    # Discord returns messages grouped in arrays (with context messages)
    results = []
    for group in data.get("messages", []):
        # The matched message is the one with hit=true, typically the middle one
        for msg in group:
            if msg.get("hit"):
                results.append(_format_message(msg))

    results = results[:args.max_results]

    output({
        "query": args.query,
        "total_results": data.get("total_results", 0),
        "returned": len(results),
        "messages": results,
    })


if __name__ == "__main__":
    main()
