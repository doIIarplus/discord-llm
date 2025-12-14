"""Discord mention extractor for extracting user, channel, and role IDs from messages"""

import re
import logging
from typing import Dict, List, NamedTuple, Optional
from dataclasses import dataclass

import discord

logger = logging.getLogger("MentionExtractor")


@dataclass
class ExtractedMentions:
    """Container for extracted Discord mentions"""
    users: List[Dict[str, str]]  # [{"id": "123", "display_name": "User"}]
    channels: List[Dict[str, str]]  # [{"id": "456", "name": "general"}]
    roles: List[Dict[str, str]]  # [{"id": "789", "name": "Admin"}]


# Regex patterns for Discord mention formats
# User mention: <@USER_ID> or <@!USER_ID> (with nickname)
USER_MENTION_PATTERN = re.compile(r'<@!?(\d+)>')
# Channel mention: <#CHANNEL_ID>
CHANNEL_MENTION_PATTERN = re.compile(r'<#(\d+)>')
# Role mention: <@&ROLE_ID>
ROLE_MENTION_PATTERN = re.compile(r'<@&(\d+)>')


def extract_user_ids(text: str) -> List[str]:
    """Extract user IDs from mention syntax in text"""
    return USER_MENTION_PATTERN.findall(text)


def extract_channel_ids(text: str) -> List[str]:
    """Extract channel IDs from mention syntax in text"""
    return CHANNEL_MENTION_PATTERN.findall(text)


def extract_role_ids(text: str) -> List[str]:
    """Extract role IDs from mention syntax in text"""
    return ROLE_MENTION_PATTERN.findall(text)


def extract_all_mentions(text: str) -> Dict[str, List[str]]:
    """
    Extract all Discord mentions from text.

    Args:
        text: The message text containing potential mentions

    Returns:
        Dictionary with 'users', 'channels', and 'roles' keys containing lists of IDs
    """
    return {
        'users': extract_user_ids(text),
        'channels': extract_channel_ids(text),
        'roles': extract_role_ids(text)
    }


def resolve_mentions(
    text: str,
    guild: Optional[discord.Guild] = None
) -> ExtractedMentions:
    """
    Extract and resolve Discord mentions to their names/details.

    Args:
        text: The message text containing potential mentions
        guild: Optional Discord guild to resolve names from

    Returns:
        ExtractedMentions with resolved user, channel, and role information
    """
    users = []
    channels = []
    roles = []

    # Extract user mentions
    for user_id in extract_user_ids(text):
        user_info = {"id": user_id, "display_name": f"User#{user_id}"}
        if guild:
            member = guild.get_member(int(user_id))
            if member:
                user_info["display_name"] = member.display_name
                user_info["username"] = member.name
        users.append(user_info)

    # Extract channel mentions
    for channel_id in extract_channel_ids(text):
        channel_info = {"id": channel_id, "name": f"channel-{channel_id}"}
        if guild:
            channel = guild.get_channel(int(channel_id))
            if channel:
                channel_info["name"] = channel.name
        channels.append(channel_info)

    # Extract role mentions
    for role_id in extract_role_ids(text):
        role_info = {"id": role_id, "name": f"Role#{role_id}"}
        if guild:
            role = guild.get_role(int(role_id))
            if role:
                role_info["name"] = role.name
        roles.append(role_info)

    return ExtractedMentions(users=users, channels=channels, roles=roles)


def format_mentions_context(mentions: ExtractedMentions) -> str:
    """
    Format extracted mentions into a context string for the LLM.

    Args:
        mentions: ExtractedMentions object with resolved mentions

    Returns:
        Formatted string describing the mentions, or empty string if no mentions
    """
    if not mentions.users and not mentions.channels and not mentions.roles:
        return ""

    parts = ["[Discord Mentions in this message:]"]

    if mentions.users:
        user_parts = []
        for user in mentions.users:
            display = user.get('display_name', user['id'])
            user_parts.append(f"@{display} (user_id: {user['id']})")
        parts.append(f"- Users mentioned: {', '.join(user_parts)}")

    if mentions.channels:
        channel_parts = []
        for channel in mentions.channels:
            name = channel.get('name', channel['id'])
            channel_parts.append(f"#{name} (channel_id: {channel['id']})")
        parts.append(f"- Channels mentioned: {', '.join(channel_parts)}")

    if mentions.roles:
        role_parts = []
        for role in mentions.roles:
            name = role.get('name', role['id'])
            role_parts.append(f"@{name} (role_id: {role['id']})")
        parts.append(f"- Roles mentioned: {', '.join(role_parts)}")

    parts.append("[Use these IDs when calling tools that require user_id, channel_id, or role parameters]")

    return "\n".join(parts)


def extract_mention_context(text: str, guild: Optional[discord.Guild] = None) -> str:
    """
    Main entry point: Extract mentions from text and return formatted context.

    Args:
        text: Input text that may contain Discord mentions
        guild: Optional Discord guild to resolve names

    Returns:
        Formatted string with mention context, or empty string if no mentions
    """
    mentions = resolve_mentions(text, guild)
    return format_mentions_context(mentions)
