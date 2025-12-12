"""Discord-specific tools for interacting with the Discord server"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import discord

from .base import Tool, ToolParameter, ParameterType, ToolResult, registry

logger = logging.getLogger("DiscordTools")


@registry.register
class GetChannelMessagesTool(Tool):
    """Tool to read recent messages from the current or specified channel"""

    name = "get_channel_messages"
    description = "Retrieve recent messages from a Discord channel. Useful for getting context about recent conversations."
    category = "discord"
    requires_discord_context = True
    parameters = [
        ToolParameter(
            name="limit",
            description="Number of messages to retrieve (max 50)",
            param_type=ParameterType.INTEGER,
            required=False,
            default=10
        ),
        ToolParameter(
            name="channel_id",
            description="Channel ID to read from. If not provided, uses current channel.",
            param_type=ParameterType.STRING,
            required=False
        ),
        ToolParameter(
            name="before_message_id",
            description="Get messages before this message ID (for pagination)",
            param_type=ParameterType.STRING,
            required=False
        ),
    ]

    async def execute(self, **kwargs) -> ToolResult:
        ctx = kwargs.get("_discord_context")
        if not ctx or not ctx.bot:
            return ToolResult(success=False, output=None, error="Discord context required")

        limit = min(kwargs.get("limit", 10), 50)
        channel_id = kwargs.get("channel_id")
        before_id = kwargs.get("before_message_id")

        try:
            # Get channel
            if channel_id:
                channel = ctx.bot.get_channel(int(channel_id))
                if not channel:
                    return ToolResult(success=False, output=None, error=f"Channel {channel_id} not found")
            else:
                channel = ctx.channel

            # Get messages
            before = None
            if before_id:
                before = discord.Object(id=int(before_id))

            messages = []
            async for msg in channel.history(limit=limit, before=before):
                messages.append({
                    "id": str(msg.id),
                    "author": msg.author.display_name,
                    "author_id": str(msg.author.id),
                    "content": msg.content[:500],  # Truncate long messages
                    "timestamp": msg.created_at.isoformat(),
                    "has_attachments": len(msg.attachments) > 0,
                    "has_embeds": len(msg.embeds) > 0,
                })

            return ToolResult(
                success=True,
                output={
                    "channel": channel.name,
                    "message_count": len(messages),
                    "messages": messages
                }
            )

        except discord.Forbidden:
            return ToolResult(success=False, output=None, error="No permission to read this channel")
        except Exception as e:
            logger.error(f"Error getting messages: {e}", exc_info=True)
            return ToolResult(success=False, output=None, error=str(e))


@registry.register
class GetUserInfoTool(Tool):
    """Tool to get information about a Discord user"""

    name = "get_user_info"
    description = "Get information about a Discord user including their name, roles, join date, and status."
    category = "discord"
    requires_discord_context = True
    parameters = [
        ToolParameter(
            name="user_id",
            description="The Discord user ID. If not provided, gets info about the message author.",
            param_type=ParameterType.STRING,
            required=False
        ),
        ToolParameter(
            name="username",
            description="The username to search for (partial match). Alternative to user_id.",
            param_type=ParameterType.STRING,
            required=False
        ),
    ]

    async def execute(self, **kwargs) -> ToolResult:
        ctx = kwargs.get("_discord_context")
        if not ctx or not ctx.guild:
            return ToolResult(success=False, output=None, error="Discord context required")

        user_id = kwargs.get("user_id")
        username = kwargs.get("username")

        try:
            member = None

            if user_id:
                member = ctx.guild.get_member(int(user_id))
            elif username:
                # Search by username (partial match)
                username_lower = username.lower()
                for m in ctx.guild.members:
                    if username_lower in m.name.lower() or username_lower in m.display_name.lower():
                        member = m
                        break
            else:
                # Default to message author
                member = ctx.author

            if not member:
                return ToolResult(
                    success=False,
                    output=None,
                    error="User not found"
                )

            # Build user info
            roles = [role.name for role in member.roles if role.name != "@everyone"]

            info = {
                "id": str(member.id),
                "username": member.name,
                "display_name": member.display_name,
                "discriminator": member.discriminator,
                "bot": member.bot,
                "roles": roles,
                "joined_server": member.joined_at.isoformat() if member.joined_at else None,
                "account_created": member.created_at.isoformat(),
                "status": str(member.status),
                "is_on_mobile": member.is_on_mobile(),
            }

            # Add activity if present
            if member.activity:
                info["activity"] = {
                    "type": str(member.activity.type),
                    "name": member.activity.name
                }

            return ToolResult(success=True, output=info)

        except Exception as e:
            logger.error(f"Error getting user info: {e}", exc_info=True)
            return ToolResult(success=False, output=None, error=str(e))


@registry.register
class GetChannelInfoTool(Tool):
    """Tool to get information about a Discord channel"""

    name = "get_channel_info"
    description = "Get information about a Discord channel including its name, topic, and settings."
    category = "discord"
    requires_discord_context = True
    parameters = [
        ToolParameter(
            name="channel_id",
            description="The channel ID. If not provided, uses current channel.",
            param_type=ParameterType.STRING,
            required=False
        ),
        ToolParameter(
            name="channel_name",
            description="The channel name to search for. Alternative to channel_id.",
            param_type=ParameterType.STRING,
            required=False
        ),
    ]

    async def execute(self, **kwargs) -> ToolResult:
        ctx = kwargs.get("_discord_context")
        if not ctx or not ctx.guild:
            return ToolResult(success=False, output=None, error="Discord context required")

        channel_id = kwargs.get("channel_id")
        channel_name = kwargs.get("channel_name")

        try:
            channel = None

            if channel_id:
                channel = ctx.guild.get_channel(int(channel_id))
            elif channel_name:
                # Search by name
                channel_name_lower = channel_name.lower()
                for ch in ctx.guild.channels:
                    if channel_name_lower in ch.name.lower():
                        channel = ch
                        break
            else:
                channel = ctx.channel

            if not channel:
                return ToolResult(success=False, output=None, error="Channel not found")

            info = {
                "id": str(channel.id),
                "name": channel.name,
                "type": str(channel.type),
                "position": channel.position,
                "created_at": channel.created_at.isoformat(),
            }

            # Add type-specific info
            if isinstance(channel, discord.TextChannel):
                info.update({
                    "topic": channel.topic,
                    "nsfw": channel.is_nsfw(),
                    "slowmode_delay": channel.slowmode_delay,
                    "category": channel.category.name if channel.category else None,
                })
            elif isinstance(channel, discord.VoiceChannel):
                info.update({
                    "bitrate": channel.bitrate,
                    "user_limit": channel.user_limit,
                    "connected_users": len(channel.members),
                })

            return ToolResult(success=True, output=info)

        except Exception as e:
            logger.error(f"Error getting channel info: {e}", exc_info=True)
            return ToolResult(success=False, output=None, error=str(e))


@registry.register
class ListChannelsTool(Tool):
    """Tool to list all channels in the server"""

    name = "list_channels"
    description = "List all channels in the Discord server, organized by category."
    category = "discord"
    requires_discord_context = True
    parameters = [
        ToolParameter(
            name="channel_type",
            description="Filter by channel type: 'text', 'voice', or 'all'",
            param_type=ParameterType.STRING,
            required=False,
            default="all",
            enum=["text", "voice", "all"]
        ),
    ]

    async def execute(self, **kwargs) -> ToolResult:
        ctx = kwargs.get("_discord_context")
        if not ctx or not ctx.guild:
            return ToolResult(success=False, output=None, error="Discord context required")

        channel_type = kwargs.get("channel_type", "all")

        try:
            channels_by_category = {}

            for channel in ctx.guild.channels:
                # Filter by type
                if channel_type == "text" and not isinstance(channel, discord.TextChannel):
                    continue
                if channel_type == "voice" and not isinstance(channel, discord.VoiceChannel):
                    continue

                # Skip categories themselves
                if isinstance(channel, discord.CategoryChannel):
                    continue

                category_name = channel.category.name if channel.category else "No Category"

                if category_name not in channels_by_category:
                    channels_by_category[category_name] = []

                channels_by_category[category_name].append({
                    "id": str(channel.id),
                    "name": channel.name,
                    "type": str(channel.type),
                })

            return ToolResult(
                success=True,
                output={
                    "server": ctx.guild.name,
                    "channel_count": sum(len(ch) for ch in channels_by_category.values()),
                    "channels": channels_by_category
                }
            )

        except Exception as e:
            logger.error(f"Error listing channels: {e}", exc_info=True)
            return ToolResult(success=False, output=None, error=str(e))


@registry.register
class GetServerInfoTool(Tool):
    """Tool to get information about the Discord server"""

    name = "get_server_info"
    description = "Get information about the current Discord server including member count, roles, and settings."
    category = "discord"
    requires_discord_context = True
    parameters = []

    async def execute(self, **kwargs) -> ToolResult:
        ctx = kwargs.get("_discord_context")
        if not ctx or not ctx.guild:
            return ToolResult(success=False, output=None, error="Discord context required")

        try:
            guild = ctx.guild

            info = {
                "id": str(guild.id),
                "name": guild.name,
                "description": guild.description,
                "owner": guild.owner.display_name if guild.owner else None,
                "created_at": guild.created_at.isoformat(),
                "member_count": guild.member_count,
                "channel_count": len(guild.channels),
                "role_count": len(guild.roles),
                "emoji_count": len(guild.emojis),
                "boost_level": guild.premium_tier,
                "boost_count": guild.premium_subscription_count,
                "roles": [r.name for r in guild.roles if r.name != "@everyone"][:20],  # Limit roles shown
                "features": list(guild.features)[:10],
            }

            return ToolResult(success=True, output=info)

        except Exception as e:
            logger.error(f"Error getting server info: {e}", exc_info=True)
            return ToolResult(success=False, output=None, error=str(e))


@registry.register
class SearchMessagesTool(Tool):
    """Tool to search for messages containing specific text"""

    name = "search_messages"
    description = "Search for messages in the current channel containing specific text."
    category = "discord"
    requires_discord_context = True
    parameters = [
        ToolParameter(
            name="query",
            description="The text to search for in messages",
            param_type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="limit",
            description="Maximum number of messages to search through (max 100)",
            param_type=ParameterType.INTEGER,
            required=False,
            default=50
        ),
        ToolParameter(
            name="author_id",
            description="Only search messages from this user ID",
            param_type=ParameterType.STRING,
            required=False
        ),
    ]

    async def execute(self, **kwargs) -> ToolResult:
        ctx = kwargs.get("_discord_context")
        if not ctx or not ctx.channel:
            return ToolResult(success=False, output=None, error="Discord context required")

        query = kwargs.get("query", "").lower()
        limit = min(kwargs.get("limit", 50), 100)
        author_id = kwargs.get("author_id")

        if not query:
            return ToolResult(success=False, output=None, error="Query is required")

        try:
            matches = []
            async for msg in ctx.channel.history(limit=limit):
                # Filter by author if specified
                if author_id and str(msg.author.id) != author_id:
                    continue

                # Search in content
                if query in msg.content.lower():
                    matches.append({
                        "id": str(msg.id),
                        "author": msg.author.display_name,
                        "content": msg.content[:300],
                        "timestamp": msg.created_at.isoformat(),
                    })

            return ToolResult(
                success=True,
                output={
                    "query": query,
                    "matches_found": len(matches),
                    "messages": matches[:20]  # Limit results
                }
            )

        except Exception as e:
            logger.error(f"Error searching messages: {e}", exc_info=True)
            return ToolResult(success=False, output=None, error=str(e))


@registry.register
class GetUserPresenceTool(Tool):
    """Tool to check who is online in the server"""

    name = "get_online_members"
    description = "Get a list of members currently online in the server."
    category = "discord"
    requires_discord_context = True
    parameters = [
        ToolParameter(
            name="status_filter",
            description="Filter by status: 'online', 'idle', 'dnd', or 'all'",
            param_type=ParameterType.STRING,
            required=False,
            default="all",
            enum=["online", "idle", "dnd", "all"]
        ),
        ToolParameter(
            name="limit",
            description="Maximum number of members to return",
            param_type=ParameterType.INTEGER,
            required=False,
            default=20
        ),
    ]

    async def execute(self, **kwargs) -> ToolResult:
        ctx = kwargs.get("_discord_context")
        if not ctx or not ctx.guild:
            return ToolResult(success=False, output=None, error="Discord context required")

        status_filter = kwargs.get("status_filter", "all")
        limit = kwargs.get("limit", 20)

        try:
            members = []

            for member in ctx.guild.members:
                if member.bot:
                    continue

                status = str(member.status)

                # Apply filter
                if status_filter != "all" and status != status_filter:
                    continue

                if status == "offline":
                    continue

                members.append({
                    "name": member.display_name,
                    "status": status,
                    "activity": member.activity.name if member.activity else None,
                })

                if len(members) >= limit:
                    break

            return ToolResult(
                success=True,
                output={
                    "online_count": len(members),
                    "members": members
                }
            )

        except Exception as e:
            logger.error(f"Error getting online members: {e}", exc_info=True)
            return ToolResult(success=False, output=None, error=str(e))
