#!/usr/bin/env python3
"""Summarize recent chat history into user profiles and server events.

Reads new messages from chat_history.db since the last summarization run,
sends them to Claude for analysis, and stores the resulting user profiles
and server events back into the database.

Self-gates on two conditions:
  1. There must be new messages since the last summary.
  2. The most recent message must be older than IDLE_MINUTES (default 60),
     meaning the server has gone quiet — avoids summarizing mid-conversation.

Intended to be called by the scheduler every 5 minutes:
  python tools/memory/summarize.py --guild-id 363154169294618625

The scheduler's cron runs this frequently, but most invocations exit
immediately (no new messages, or server still active).
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

# Add project root to path so we can import chat_history and config
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, PROJECT_DIR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error

from chat_history import (
    get_summarizer_state,
    get_messages_since,
    get_latest_message_time,
    get_latest_message_id,
    has_new_messages_since,
    get_user_profiles,
    get_channel_summaries,
    upsert_user_profile,
    upsert_channel_summary,
    insert_server_event,
    update_summarizer_state,
)
from config import MEMORY_SUMMARIZE_BATCH_SIZE, MEMORY_IDLE_MINUTES

# Claude CLI path
CLAUDE_CLI = "claude"


def _should_run(guild_id: str) -> dict:
    """Check if we should run summarization. Returns status dict."""
    state = get_summarizer_state(guild_id)

    # Check for new messages
    if state:
        last_id = state["last_processed_message_id"]
        if not has_new_messages_since(guild_id, last_id):
            return {"should_run": False, "reason": "no new messages since last summary"}

    # Check idle time — don't summarize mid-conversation
    latest_time_str = get_latest_message_time(guild_id)
    if not latest_time_str:
        return {"should_run": False, "reason": "no messages in database"}

    latest_time = datetime.fromisoformat(latest_time_str)
    if latest_time.tzinfo is None:
        latest_time = latest_time.replace(tzinfo=timezone.utc)

    idle_threshold = datetime.now(timezone.utc) - timedelta(minutes=MEMORY_IDLE_MINUTES)
    if latest_time > idle_threshold:
        minutes_ago = (datetime.now(timezone.utc) - latest_time).total_seconds() / 60
        return {
            "should_run": False,
            "reason": f"server still active (last message {minutes_ago:.0f}m ago, need {MEMORY_IDLE_MINUTES}m idle)",
        }

    return {"should_run": True}


def _build_summarization_prompt(messages: list, existing_profiles: list, existing_channels: list) -> str:
    """Build the prompt for Claude to analyze messages and update profiles/channel summaries."""
    # Format existing profiles
    profiles_section = "None yet — create initial profiles from the messages below."
    if existing_profiles:
        profile_lines = []
        for p in existing_profiles:
            profile_lines.append({
                "user_id": p["user_id"],
                "user_name": p["user_name"],
                "profile_summary": p["profile_summary"],
                "likes": p["likes"],
                "dislikes": p["dislikes"],
                "personality_traits": p["personality_traits"],
            })
        profiles_section = json.dumps(profile_lines, indent=2)

    # Format existing channel summaries
    channels_section = "None yet — create initial summaries from the messages below."
    if existing_channels:
        channel_lines = []
        for c in existing_channels:
            channel_lines.append({
                "channel_id": c["channel_id"],
                "channel_name": c["channel_name"],
                "summary": c["summary"],
                "active_topics": c["active_topics"],
            })
        channels_section = json.dumps(channel_lines, indent=2)

    # Format messages (grouped by channel for readability)
    channels_in_messages = {}
    for m in messages:
        cid = m["channel_id"]
        if cid not in channels_in_messages:
            channels_in_messages[cid] = []
        ts = m["created_at"][:16].replace("T", " ")
        reply = ""
        if m.get("reply_to_message_id"):
            reply = f" (replying to msg {m['reply_to_message_id']})"
        channels_in_messages[cid].append(
            f"[{ts}] {m['author_name']} (id={m['author_id']}){reply}: {m['content']}"
        )

    msg_parts = []
    for cid, lines in channels_in_messages.items():
        msg_parts.append(f"### Channel {cid}")
        msg_parts.extend(lines)
    messages_section = "\n".join(msg_parts)

    return f"""You are analyzing Discord chat history to maintain persistent memory across three dimensions: per-user profiles, per-channel summaries, and notable server events.

## Existing User Profiles
{profiles_section}

## Existing Channel Summaries
{channels_section}

## New Messages Since Last Analysis
{messages_section}

## Instructions
1. UPDATE each user's profile based on new evidence. Merge with existing data — do not discard prior observations unless contradicted. For new users, create a profile.
2. UPDATE each channel's summary based on new messages in that channel. Describe what the channel is typically used for, its tone, and current active topics. For new channels, create a summary.
3. Identify any notable EVENTS (group decisions, arguments, celebrations, inside jokes, plans, interesting discussions). Do NOT create events for mundane conversation.
4. Keep everything concise but informative.
5. Do NOT include the bot's own profile.
6. For channel_name, use a descriptive name if you can infer it from context, otherwise use the channel_id.

Output ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "user_profiles": [
    {{
      "user_id": "123",
      "user_name": "display_name",
      "profile_summary": "Brief personality and behavior summary",
      "likes": "things they enjoy, comma-separated",
      "dislikes": "things they dislike, comma-separated",
      "personality_traits": "key traits, comma-separated"
    }}
  ],
  "channel_summaries": [
    {{
      "channel_id": "456",
      "channel_name": "general",
      "summary": "What this channel is used for and its general vibe",
      "active_topics": "current/recent discussion topics, comma-separated"
    }}
  ],
  "events": [
    {{
      "event_summary": "What happened",
      "participants": ["user_id1", "user_id2"],
      "occurred_at": "ISO timestamp of when it happened",
      "source_message_ids": ["msg_id1", "msg_id2"]
    }}
  ]
}}

If there are no notable events, return an empty events array.
Include ALL users and channels that appeared in the messages."""


async def _call_claude(prompt: str) -> str:
    """Call Claude CLI for summarization (no tools needed, just text analysis)."""
    cmd = [
        CLAUDE_CLI,
        "-p",
        "--output-format", "json",
        "--verbose",
        "--model", "sonnet",
        "--no-session-persistence",
        "--tools", "",
    ]

    env = os.environ.copy()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    stdout, stderr = await asyncio.wait_for(
        proc.communicate(input=prompt.encode("utf-8")),
        timeout=300,
    )

    raw_out = stdout.decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        raw_err = stderr.decode("utf-8", errors="replace").strip()
        error(f"Claude CLI failed (rc={proc.returncode}): {raw_err[:500]}")

    # Parse verbose JSON output to extract the result text
    try:
        events = json.loads(raw_out)
        # Find the result entry
        for event in events:
            if isinstance(event, dict) and event.get("type") == "result":
                result = event.get("result", "")
                # Result might be a string or a list of content blocks
                if isinstance(result, str):
                    return result
                if isinstance(result, list):
                    texts = []
                    for block in result:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
                    return "\n".join(texts)
        # Fallback: if no result event found, return raw
        return raw_out
    except json.JSONDecodeError:
        return raw_out


def _parse_summary_response(response: str) -> dict:
    """Parse the JSON response from Claude, handling markdown fences."""
    text = response.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to find JSON within the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        error(f"Failed to parse Claude response as JSON: {e}\nResponse: {text[:500]}")


async def _run_summarization(guild_id: str, dry_run: bool = False):
    """Main summarization pipeline."""
    # Get watermark
    state = get_summarizer_state(guild_id)
    since_id = state["last_processed_message_id"] if state else None

    # Fetch new messages
    messages = get_messages_since(guild_id, since_message_id=since_id, limit=MEMORY_SUMMARIZE_BATCH_SIZE)
    if not messages:
        output({"status": "no_messages", "guild_id": guild_id})
        return

    print(f"[summarize] Processing {len(messages)} new messages for guild {guild_id}", file=sys.stderr)

    if dry_run:
        output({
            "status": "dry_run",
            "guild_id": guild_id,
            "message_count": len(messages),
            "first_message": messages[0]["created_at"],
            "last_message": messages[-1]["created_at"],
        })
        return

    # Get existing profiles and channel summaries
    existing_profiles = get_user_profiles(guild_id)
    existing_channels = get_channel_summaries(guild_id)

    # Build prompt and call Claude
    prompt = _build_summarization_prompt(messages, existing_profiles, existing_channels)
    print(f"[summarize] Calling Claude (prompt: {len(prompt)} chars)...", file=sys.stderr)
    start = time.perf_counter()
    response = await _call_claude(prompt)
    duration = time.perf_counter() - start
    print(f"[summarize] Claude responded in {duration:.1f}s", file=sys.stderr)

    # Parse response
    data = _parse_summary_response(response)

    # Upsert profiles
    profiles_updated = 0
    for profile in data.get("user_profiles", []):
        upsert_user_profile(
            guild_id=guild_id,
            user_id=profile["user_id"],
            user_name=profile["user_name"],
            profile_summary=profile.get("profile_summary", ""),
            likes=profile.get("likes", ""),
            dislikes=profile.get("dislikes", ""),
            personality_traits=profile.get("personality_traits", ""),
        )
        profiles_updated += 1

    # Upsert channel summaries
    channels_updated = 0
    for ch in data.get("channel_summaries", []):
        upsert_channel_summary(
            guild_id=guild_id,
            channel_id=ch["channel_id"],
            channel_name=ch.get("channel_name", ch["channel_id"]),
            summary=ch.get("summary", ""),
            active_topics=ch.get("active_topics", ""),
        )
        channels_updated += 1

    # Insert events
    events_created = 0
    for event in data.get("events", []):
        insert_server_event(
            guild_id=guild_id,
            event_summary=event["event_summary"],
            participants=event.get("participants"),
            occurred_at=event.get("occurred_at"),
            source_message_ids=event.get("source_message_ids"),
        )
        events_created += 1

    # Update watermark to the last message we processed
    last_msg_id = messages[-1]["message_id"]
    update_summarizer_state(guild_id, last_msg_id)

    output({
        "status": "completed",
        "guild_id": guild_id,
        "messages_processed": len(messages),
        "profiles_updated": profiles_updated,
        "channels_updated": channels_updated,
        "events_created": events_created,
        "duration_seconds": round(duration, 1),
    })


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--guild-id", required=True,
                        help="Discord guild (server) ID to summarize")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without calling Claude")
    parser.add_argument("--force", action="store_true",
                        help="Skip idle-time check (still requires new messages)")
    args = parser.parse_args()

    # Gate checks
    if not args.force:
        check = _should_run(args.guild_id)
        if not check["should_run"]:
            # Silent exit — this is the normal case for most cron invocations
            output({"status": "skipped", "reason": check["reason"]})
            return

    asyncio.run(_run_summarization(args.guild_id, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
