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
from typing import Optional

# Add project root to path so we can import chat_history and config
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, PROJECT_DIR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error

from chat_history import (
    DM_GUILD_SENTINEL,
    get_summarizer_state,
    get_messages_since,
    get_latest_message_time,
    get_latest_message_id,
    has_new_messages_since,
    get_user_profiles,
    get_user_aliases,
    get_channel_summaries,
    get_channel_name,
    get_dm_channel_id_for_user,
    get_dm_messages_since,
    get_dm_latest_message_time,
    dm_has_new_messages_since,
    upsert_user_profile,
    upsert_channel_summary,
    insert_server_event,
    update_summarizer_state,
    update_friendliness_score,
)
from config import (
    MEMORY_SUMMARIZE_BATCH_SIZE, MEMORY_IDLE_MINUTES,
    MEMORY_MAX_PROFILE_CHARS, MEMORY_MAX_CHANNEL_CHARS, MEMORY_MAX_EVENTS,
    DM_PROFILE_USER_IDS,
)

# Claude CLI path — use absolute path for cron compatibility
CLAUDE_CLI = os.path.expanduser("~/.local/bin/claude")


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
    aliases = get_user_aliases()

    # Format known aliases section
    aliases_section = ""
    if aliases:
        alias_lines = []
        for uid, info in aliases.items():
            real = info.get("real", "") if isinstance(info, dict) else info
            preferred = info.get("preferred", "") if isinstance(info, dict) else ""
            if preferred and preferred != real:
                alias_lines.append(f"- User ID {uid}: real name is {real}, prefers to be called {preferred}")
            else:
                alias_lines.append(f"- User ID {uid}: real name is {real}")
        aliases_section = "\n## Known Real Names\nUse real names in profiles for identification, but note their preferred name.\n" + "\n".join(alias_lines) + "\n"

    # Format existing profiles
    profiles_section = "None yet — create initial profiles from the messages below."
    if existing_profiles:
        profile_lines = []
        for p in existing_profiles:
            profile_lines.append({
                "user_id": p["user_id"],
                "user_name": p["user_name"],
                "profile": p["profile"],
                "current_friendliness_score": float(p.get("friendliness_score") or 0.0),
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
        line = f"[{ts}] {m['author_name']} (id={m['author_id']}){reply}: {m['content']}"
        if m.get("image_summary"):
            line += f" [attached image: {m['image_summary']}]"
        channels_in_messages[cid].append(line)

    msg_parts = []
    for cid, lines in channels_in_messages.items():
        name = get_channel_name(messages[0]["guild_id"], cid) or cid
        msg_parts.append(f"### #{name} (channel_id={cid})")
        msg_parts.extend(lines)
    messages_section = "\n".join(msg_parts)

    return f"""Analyze Discord chat history. Update user profiles and channel summaries. Identify notable events.
{aliases_section}
## Existing User Profiles
{profiles_section}

## Existing Channel Summaries
{channels_section}

## New Messages
{messages_section}

## Instructions
- Merge new observations into existing profiles. Preserve what isn't contradicted.
- Cover: personality, communication style, interests, relationships with others, notable behavior.
- Keep profiles 2-4 paragraphs. Be specific (names, details) not generic.
- Update channel summaries with recent topics and tone.
- Log notable events (decisions, debates, milestones, inside jokes). Skip mundane chat.
- Do NOT include the bot's own profile.
- For each user, provide a friendliness_adjustment (-2.0 to +2.0) based on their interactions with the bot (jaspt) in these messages. Consider: were they polite, playful, hostile, appreciative, dismissive? Normal conversation = 0. Being nice/funny toward the bot = +0.5 to +1.0. Being mean/hostile toward the bot = -0.5 to -1.0. Only use extremes for truly exceptional behavior. If a user didn't interact with the bot at all, use 0.

Output ONLY valid JSON:
{{
  "user_profiles": [
    {{
      "user_id": "123",
      "user_name": "display_name",
      "profile": "Profile text..."
    }}
  ],
  "channel_summaries": [
    {{
      "channel_id": "456",
      "channel_name": "general",
      "summary": "Channel summary..."
    }}
  ],
  "events": [
    {{
      "event_summary": "Detailed description of what happened and why it matters",
      "participants": ["user_id1", "user_id2"],
      "occurred_at": "ISO timestamp",
      "source_message_ids": ["msg_id1", "msg_id2"]
    }}
  ],
  "friendliness_adjustments": {{
    "user_id_1": 0.5,
    "user_id_2": -1.0
  }}
}}

Include ALL users and channels that appeared in the messages."""


async def _call_claude(prompt: str) -> str:
    """Call Claude CLI for summarization (no tools needed, just text analysis)."""
    cmd = [
        CLAUDE_CLI,
        "-p",
        "--output-format", "json",
        "--verbose",
        "--model", "sonnet",
        "--effort", "medium",
        "--no-session-persistence",
        "--tools", "",
    ]

    env = os.environ.copy()
    # Ensure HOME is set (needed by claude CLI for config)
    env.setdefault("HOME", os.path.expanduser("~"))
    # Remove ANTHROPIC_API_KEY so claude CLI uses Max subscription auth
    # instead of the API key (which may have insufficient quota)
    env.pop("ANTHROPIC_API_KEY", None)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    stdout, stderr = await asyncio.wait_for(
        proc.communicate(input=prompt.encode("utf-8")),
        timeout=600,
    )

    raw_out = stdout.decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        raw_err = stderr.decode("utf-8", errors="replace").strip()
        detail = raw_err or raw_out
        error(f"Claude CLI failed (rc={proc.returncode})", details={"stderr": raw_err[:1000], "stdout": raw_out[:1000]})

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


async def _compact_oversized(guild_id: str) -> int:
    """Compact profiles and channel summaries that exceed size limits.

    Sends oversized text to Claude with instructions to condense while
    preserving key facts. Returns count of items compacted.
    """
    compacted = 0

    # Check profiles
    all_profiles = get_user_profiles(guild_id)
    for p in all_profiles:
        if len(p.get("profile", "")) > MEMORY_MAX_PROFILE_CHARS:
            print(f"[summarize] Compacting profile for {p['user_name']} "
                  f"({len(p['profile'])} > {MEMORY_MAX_PROFILE_CHARS} chars)", file=sys.stderr)
            prompt = (
                f"Condense this user profile to under {MEMORY_MAX_PROFILE_CHARS} characters. "
                f"Keep the most important facts about personality, interests, and relationships. "
                f"Drop redundant details and older events. Output ONLY the condensed profile text, nothing else.\n\n"
                f"{p['profile']}"
            )
            condensed = await _call_claude(prompt)
            condensed = condensed.strip().strip('"')
            if condensed and len(condensed) < len(p["profile"]):
                upsert_user_profile(guild_id, p["user_id"], p["user_name"], condensed)
                compacted += 1
                print(f"[summarize]   {len(p['profile'])} -> {len(condensed)} chars", file=sys.stderr)

    # Check channel summaries
    all_channels = get_channel_summaries(guild_id)
    for c in all_channels:
        if len(c.get("summary", "")) > MEMORY_MAX_CHANNEL_CHARS:
            print(f"[summarize] Compacting channel #{c['channel_name']} "
                  f"({len(c['summary'])} > {MEMORY_MAX_CHANNEL_CHARS} chars)", file=sys.stderr)
            prompt = (
                f"Condense this channel summary to under {MEMORY_MAX_CHANNEL_CHARS} characters. "
                f"Keep the channel's purpose, tone, and most recent topics. "
                f"Drop older topics. Output ONLY the condensed summary text, nothing else.\n\n"
                f"{c['summary']}"
            )
            condensed = await _call_claude(prompt)
            condensed = condensed.strip().strip('"')
            if condensed and len(condensed) < len(c["summary"]):
                upsert_channel_summary(guild_id, c["channel_id"], c["channel_name"], condensed)
                compacted += 1
                print(f"[summarize]   {len(c['summary'])} -> {len(condensed)} chars", file=sys.stderr)

    return compacted


def _prune_old_events(guild_id: str) -> int:
    """Delete old events beyond the retention limit. Returns count pruned."""
    from chat_history import _get_conn
    conn = _get_conn()
    count = conn.execute(
        "SELECT COUNT(*) as cnt FROM server_events WHERE guild_id = ?", (guild_id,)
    ).fetchone()["cnt"]

    if count <= MEMORY_MAX_EVENTS:
        return 0

    to_delete = count - MEMORY_MAX_EVENTS
    conn.execute(
        """DELETE FROM server_events WHERE id IN (
            SELECT id FROM server_events WHERE guild_id = ?
            ORDER BY occurred_at ASC LIMIT ?
        )""",
        (guild_id, to_delete),
    )
    conn.commit()
    print(f"[summarize] Pruned {to_delete} old events (kept {MEMORY_MAX_EVENTS})", file=sys.stderr)
    return to_delete


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

    # Only include profiles/channels for users and channels that appear in new messages
    active_user_ids = {m["author_id"] for m in messages}
    active_channel_ids = {m["channel_id"] for m in messages}

    all_profiles = get_user_profiles(guild_id)
    existing_profiles = [p for p in all_profiles if p["user_id"] in active_user_ids]

    all_channels = get_channel_summaries(guild_id)
    existing_channels = [c for c in all_channels if c["channel_id"] in active_channel_ids]

    print(f"[summarize] Active users: {len(existing_profiles)}/{len(all_profiles)}, "
          f"channels: {len(existing_channels)}/{len(all_channels)}", file=sys.stderr)

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
            profile=profile.get("profile", ""),
        )
        profiles_updated += 1

    # Upsert channel summaries — use real channel name from DB, not AI-generated
    channels_updated = 0
    for ch in data.get("channel_summaries", []):
        real_name = get_channel_name(guild_id, ch["channel_id"]) or ch.get("channel_name", ch["channel_id"])
        upsert_channel_summary(
            guild_id=guild_id,
            channel_id=ch["channel_id"],
            channel_name=real_name,
            summary=ch.get("summary", ""),
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

    # Apply friendliness score adjustments
    friendliness_updates = {}
    for user_id, adj in data.get("friendliness_adjustments", {}).items():
        try:
            adj_float = float(adj)
            if adj_float != 0:
                new_score = update_friendliness_score(guild_id, user_id, adj_float)
                friendliness_updates[user_id] = {"adjustment": adj_float, "new_score": new_score}
                print(f"[summarize] Friendliness {user_id}: {adj_float:+.1f} -> {new_score:.1f}", file=sys.stderr)
        except (ValueError, TypeError):
            pass

    # Compact oversized profiles and channel summaries
    compacted = await _compact_oversized(guild_id)

    # Prune old events
    events_pruned = _prune_old_events(guild_id)

    # Update watermark to the last message we processed
    last_msg_id = messages[-1]["message_id"]
    update_summarizer_state(guild_id, last_msg_id)

    result = {
        "status": "completed",
        "guild_id": guild_id,
        "messages_processed": len(messages),
        "profiles_updated": profiles_updated,
        "channels_updated": channels_updated,
        "events_created": events_created,
        "duration_seconds": round(duration, 1),
    }
    if compacted:
        result["compacted"] = compacted
    if events_pruned:
        result["events_pruned"] = events_pruned
    if friendliness_updates:
        result["friendliness_updates"] = friendliness_updates
    output(result)


# ---------------------------------------------------------------------------
# DM mode — per-user long-term profile built from a single DM channel.
# ---------------------------------------------------------------------------

def _dm_watermark_key(user_id: str) -> str:
    """Synthetic key used to store DM watermark separately from guild watermarks."""
    return f"dm:{user_id}"


def _should_run_dm(user_id: str) -> dict:
    """Self-gate for DM mode — needs new messages and an idle window."""
    state = get_summarizer_state(_dm_watermark_key(user_id))

    if state:
        last_id = state["last_processed_message_id"]
        if not dm_has_new_messages_since(user_id, last_id):
            return {"should_run": False, "reason": "no new DM messages since last summary"}

    latest_time_str = get_dm_latest_message_time(user_id)
    if not latest_time_str:
        return {"should_run": False, "reason": "no DM messages in database"}

    latest_time = datetime.fromisoformat(latest_time_str)
    if latest_time.tzinfo is None:
        latest_time = latest_time.replace(tzinfo=timezone.utc)

    idle_threshold = datetime.now(timezone.utc) - timedelta(minutes=MEMORY_IDLE_MINUTES)
    if latest_time > idle_threshold:
        minutes_ago = (datetime.now(timezone.utc) - latest_time).total_seconds() / 60
        return {
            "should_run": False,
            "reason": f"DM still active (last message {minutes_ago:.0f}m ago, need {MEMORY_IDLE_MINUTES}m idle)",
        }

    return {"should_run": True}


def _build_dm_summarization_prompt(user_id: str, messages: list, existing_profile: Optional[dict]) -> str:
    """Prompt Claude to maintain a single-user DM profile.

    DM-tuned: focus on personal preferences, ongoing projects, and recurring topics
    the user discusses ONE-ON-ONE with the bot. Skip social dynamics with other users
    (irrelevant in DMs), don't create events, don't summarize channels.
    """
    aliases = get_user_aliases()
    real_name = ""
    if user_id in aliases:
        info = aliases[user_id]
        real_name = info.get("real", "") if isinstance(info, dict) else info

    existing_section = "None yet — create the initial profile from the messages below."
    if existing_profile:
        existing_section = json.dumps({
            "user_id": existing_profile["user_id"],
            "user_name": existing_profile["user_name"],
            "profile": existing_profile["profile"],
        }, indent=2)

    msg_lines = []
    for m in messages:
        ts = m["created_at"][:16].replace("T", " ")
        reply = ""
        if m.get("reply_to_message_id"):
            reply = f" (replying to msg {m['reply_to_message_id']})"
        line = f"[{ts}] {m['author_name']} (id={m['author_id']}){reply}: {m['content']}"
        if m.get("image_summary"):
            line += f" [attached image: {m['image_summary']}]"
        msg_lines.append(line)
    messages_section = "\n".join(msg_lines)

    name_hint = f" (real name: {real_name})" if real_name else ""

    return f"""Maintain a long-term DM profile for user_id={user_id}{name_hint}, based on their 1-on-1 conversation with you (jaspt, a Discord bot).

## Existing Profile
{existing_section}

## New DM Messages
{messages_section}

## Instructions
- Update or create the profile for user_id={user_id} ONLY. Do not create profiles for other users mentioned.
- Focus on what's relevant for a 1-on-1 assistant relationship:
  - Personal preferences, opinions, taste
  - Ongoing projects, side projects, technical interests
  - Recurring topics they bring up in DMs
  - Things they've explicitly shared (background, role, location, schedule, gear, etc.)
  - Communication style they use with you specifically (tone, brevity, slang)
- Merge new observations into the existing profile. Preserve what isn't contradicted.
- Skip social dynamics, gossip, or anything they said about other users — DMs are private.
- Do NOT include the bot's own information.
- Keep the profile 2-5 paragraphs. Be specific (names, projects, preferences), not generic.

Output ONLY valid JSON:
{{
  "profile": "Profile text..."
}}"""


def _parse_dm_response(response: str) -> dict:
    """Parse the DM summarizer's JSON response, tolerating markdown fences."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        error(f"Failed to parse Claude DM response as JSON.\nResponse: {text[:500]}")


async def _run_dm_summarization(user_id: str, dry_run: bool = False):
    """DM-mode pipeline: maintain a single-user profile from their DM channel."""
    if user_id not in DM_PROFILE_USER_IDS:
        error(f"User {user_id} is not in DM_PROFILE_USER_IDS allowlist")

    watermark_key = _dm_watermark_key(user_id)
    state = get_summarizer_state(watermark_key)
    since_id = state["last_processed_message_id"] if state else None

    messages = get_dm_messages_since(user_id, since_message_id=since_id, limit=MEMORY_SUMMARIZE_BATCH_SIZE)
    if not messages:
        output({"status": "no_messages", "dm_user_id": user_id})
        return

    print(f"[summarize-dm] Processing {len(messages)} new DM messages for user {user_id}", file=sys.stderr)

    if dry_run:
        output({
            "status": "dry_run",
            "dm_user_id": user_id,
            "message_count": len(messages),
            "first_message": messages[0]["created_at"],
            "last_message": messages[-1]["created_at"],
        })
        return

    # Fetch any existing DM profile for this user. DM profiles are stored under
    # guild_id=DM_GUILD_SENTINEL, keyed by user_id within that scope.
    all_dm_profiles = get_user_profiles(DM_GUILD_SENTINEL)
    existing = next((p for p in all_dm_profiles if p["user_id"] == user_id), None)

    user_name = messages[-1]["author_name"] if messages[-1]["author_id"] == user_id else (
        next((m["author_name"] for m in messages if m["author_id"] == user_id), user_id)
    )

    prompt = _build_dm_summarization_prompt(user_id, messages, existing)
    print(f"[summarize-dm] Calling Claude (prompt: {len(prompt)} chars)...", file=sys.stderr)
    start = time.perf_counter()
    response = await _call_claude(prompt)
    duration = time.perf_counter() - start
    print(f"[summarize-dm] Claude responded in {duration:.1f}s", file=sys.stderr)

    data = _parse_dm_response(response)
    new_profile = (data or {}).get("profile", "").strip()
    if not new_profile:
        error("DM summarizer returned empty profile")

    upsert_user_profile(
        guild_id=DM_GUILD_SENTINEL,
        user_id=user_id,
        user_name=user_name,
        profile=new_profile,
    )

    last_msg_id = messages[-1]["message_id"]
    update_summarizer_state(watermark_key, last_msg_id)

    output({
        "status": "completed",
        "dm_user_id": user_id,
        "messages_processed": len(messages),
        "profile_length": len(new_profile),
        "duration_seconds": round(duration, 1),
    })


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--guild-id",
                      help="Discord guild (server) ID to summarize")
    mode.add_argument("--dm-user-id",
                      help="Discord user ID — summarize that user's DM channel into a per-user DM profile")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without calling Claude")
    parser.add_argument("--force", action="store_true",
                        help="Skip idle-time check (still requires new messages)")
    args = parser.parse_args()

    if args.dm_user_id:
        if not args.force:
            check = _should_run_dm(args.dm_user_id)
            if not check["should_run"]:
                output({"status": "skipped", "reason": check["reason"]})
                return
        asyncio.run(_run_dm_summarization(args.dm_user_id, dry_run=args.dry_run))
        return

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
