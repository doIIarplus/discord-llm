"""Requesting-user permission enforcement for Discord CLI tools.

Every Discord tool that performs (or reads) a privileged action calls
``require_permission()`` so the action is gated by the *permissions of the
Discord user who asked for it* — NOT by the LLM's judgement. If donairplus
asks the bot to "delete every channel" but lacks Manage Channels, the tool
hard-rejects before touching the Discord API.

How the requesting user's identity reaches the tool
---------------------------------------------------
bot.py knows who triggered the bot (the message author). When it spawns the
Claude Code process it injects that identity into the process environment as
``DISCORD_REQUESTING_USER_ID`` / ``DISCORD_REQUESTING_GUILD_ID``. Those are
inherited by the Bash tool subprocesses Claude spawns, so a tool can read the
*real* requester regardless of what ``--user-id`` the model passes on the
command line.

For the persistent-tmux (PTY) mode, whose environment is fixed at session
start, bot.py instead writes the per-message identity to ``.request_context``
next to this file just before sending the prompt; the tool falls back to it.

Trust boundary (read this)
--------------------------
This stops the *normal* failure mode: the model relaying a destructive request
from a user who is not authorised for it. It is NOT a sandbox. A model with
unrestricted Bash could in principle override the env var (``VAR=x python ...``)
or call the Discord API directly. For a hard guarantee the enforcement must
move to a Claude Code PreToolUse hook or out of the model's reach entirely.
The check FAILS CLOSED: if the requester cannot be determined, the action is
denied.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import error
from discord._client import DiscordClient

# Discord permission bit flags (the subset the tools enforce).
# https://discord.com/developers/docs/topics/permissions
PERMISSIONS = {
    "KICK_MEMBERS": 1 << 1,
    "BAN_MEMBERS": 1 << 2,
    "ADMINISTRATOR": 1 << 3,
    "MANAGE_CHANNELS": 1 << 4,
    "MANAGE_GUILD": 1 << 5,
    "ADD_REACTIONS": 1 << 6,
    "VIEW_CHANNEL": 1 << 10,
    "SEND_MESSAGES": 1 << 11,
    "MANAGE_MESSAGES": 1 << 13,
    "CHANGE_NICKNAME": 1 << 26,
    "MANAGE_NICKNAMES": 1 << 27,
    "MANAGE_ROLES": 1 << 28,
    "MANAGE_WEBHOOKS": 1 << 29,
    # Manage Expressions — the emoji/sticker/soundboard permission. Discord
    # renamed MANAGE_EMOJIS_AND_STICKERS to MANAGE_GUILD_EXPRESSIONS without
    # changing the bit, so the same flag covers both names (discord.py's
    # manage_expressions / manage_emojis alias the same value).
    "MANAGE_EXPRESSIONS": 1 << 30,
    "CREATE_PUBLIC_THREADS": 1 << 34,
    "MODERATE_MEMBERS": 1 << 40,
}

ADMINISTRATOR = PERMISSIONS["ADMINISTRATOR"]

_CONTEXT_FILE = os.path.join(os.path.dirname(__file__), ".request_context")


def get_requester():
    """Return (user_id, guild_id) of the Discord user who triggered the bot.

    Prefers the per-process env vars set by bot.py; falls back to the context
    file written by the PTY path. Values are strings, or None if unavailable.
    """
    uid = os.environ.get("DISCORD_REQUESTING_USER_ID") or None
    gid = os.environ.get("DISCORD_REQUESTING_GUILD_ID") or None
    if uid:
        return uid, gid
    try:
        with open(_CONTEXT_FILE) as f:
            data = json.load(f)
        uid = str(data.get("user_id") or "") or None
        gid = str(data.get("guild_id") or "") or None
        return uid, gid
    except (OSError, ValueError):
        return None, None


def is_system_context():
    """Return True if this invocation is a trusted scheduler/system context.

    The scheduler (tools/scheduler/run_due.py) sets ``DISCORD_SYSTEM_CONTEXT=1``
    in the environment of the subprocesses it launches for due tasks. When there
    is no requesting user (no ``DISCORD_REQUESTING_USER_ID`` env var AND no
    ``.request_context`` file), a scheduler-launched tool is running on behalf of
    the system (e.g. a cron reminder), not a Discord user, so permission checks
    that would otherwise fail closed are allowed to proceed.

    This bypass is gated behind the explicit env flag so that *arbitrary*
    no-context invocations still fail closed — only tasks the scheduler itself
    started (which set the flag) get the system pass. If a real requesting user
    identity IS present, this returns False and normal enforcement applies.
    """
    if os.environ.get("DISCORD_SYSTEM_CONTEXT") != "1":
        return False
    uid, _ = get_requester()
    return uid is None


def _base_permissions(guild, member, requester_id):
    """Compute a member's guild-level permission bitfield.

    Returns None to signal "has every permission" (owner or Administrator).
    """
    if str(guild.get("owner_id")) == str(requester_id):
        return None  # owner has all permissions
    roles_by_id = {r["id"]: int(r.get("permissions", 0)) for r in guild.get("roles", [])}
    # @everyone role shares the guild's ID and always applies.
    perms = roles_by_id.get(str(guild["id"]), 0)
    for rid in member.get("roles", []):
        perms |= roles_by_id.get(rid, 0)
    if perms & ADMINISTRATOR:
        return None
    return perms


def _apply_channel_overwrites(perms, channel, member, requester_id, guild_id):
    """Layer a channel's permission overwrites onto guild-level ``perms``.

    Implements Discord's overwrite precedence: @everyone, then the union of the
    member's role overwrites, then the member-specific overwrite.
    """
    overwrites = {o["id"]: o for o in channel.get("permission_overwrites", [])}

    everyone = overwrites.get(str(guild_id))
    if everyone:
        perms &= ~int(everyone.get("deny", 0))
        perms |= int(everyone.get("allow", 0))

    allow = deny = 0
    for rid in member.get("roles", []):
        ow = overwrites.get(rid)
        if ow and int(ow.get("type", 0)) == 0:  # role overwrite
            allow |= int(ow.get("allow", 0))
            deny |= int(ow.get("deny", 0))
    perms &= ~deny
    perms |= allow

    member_ow = overwrites.get(str(requester_id))
    if member_ow and int(member_ow.get("type", 1)) == 1:  # member overwrite
        perms &= ~int(member_ow.get("deny", 0))
        perms |= int(member_ow.get("allow", 0))

    return perms


def require_permission(perm_name, guild_id=None, channel_id=None):
    """Abort the tool unless the requesting user holds ``perm_name``.

    ``perm_name`` must be a key of PERMISSIONS. Pass ``guild_id`` and/or
    ``channel_id`` for context; either can be omitted and will be resolved from
    the requester context / the channel object. Channel-level overwrites are
    applied when ``channel_id`` is given.

    Returns silently if allowed; calls ``error()`` (exit 1) if denied or if the
    requester cannot be verified (fail closed).
    """
    if perm_name not in PERMISSIONS:
        error(f"Internal: unknown permission '{perm_name}'")
    required = PERMISSIONS[perm_name]

    requester_id, requester_guild = get_requester()
    if not requester_id:
        # No Discord user context. If the scheduler launched this tool (it sets
        # DISCORD_SYSTEM_CONTEXT=1), treat it as a trusted system context and
        # allow the action instead of hard-rejecting. Any other no-context
        # invocation still fails closed.
        if is_system_context():
            return
        error(
            "Permission check failed: could not determine which Discord user "
            "requested this action, so it was blocked. (The bot injects the "
            "requesting user's identity; it is missing here.)",
            details={"required_permission": perm_name},
        )

    client = DiscordClient()

    gid = str(guild_id or requester_guild or "")
    if not gid and channel_id:
        ch = client.get(f"/channels/{channel_id}")
        gid = str(ch.get("guild_id") or "")
    if not gid:
        error(
            "Permission check failed: no guild context available to verify "
            "the requesting user's permissions.",
            details={"required_permission": perm_name, "requesting_user_id": requester_id},
        )

    guild = client.get_or_none(f"/guilds/{gid}")
    member = client.get_or_none(f"/guilds/{gid}/members/{requester_id}")
    if not guild or not member:
        error(
            f"Permission denied: the requesting user (discord_id={requester_id}) "
            f"is not a resolvable member of guild {gid}, so the action was blocked.",
            details={"required_permission": perm_name, "requesting_user_id": requester_id, "guild_id": gid},
        )

    perms = _base_permissions(guild, member, requester_id)
    if perms is None:
        return  # owner / administrator — allowed

    if channel_id:
        channel = client.get(f"/channels/{channel_id}")
        perms = _apply_channel_overwrites(perms, channel, member, requester_id, gid)

    if not (perms & required):
        error(
            f"Permission denied: the requesting user (discord_id={requester_id}) "
            f"does not have the '{perm_name}' permission required for this action. "
            f"This was rejected based on their Discord roles, not the model's choice.",
            details={
                "required_permission": perm_name,
                "requesting_user_id": requester_id,
                "guild_id": gid,
            },
        )
