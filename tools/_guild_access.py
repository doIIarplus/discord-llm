"""Per-guild tool gating, enforced inside the tools themselves.

Why this exists
---------------
Claude runs with `--allowedTools Bash,WebSearch,WebFetch`, and Bash can reach
every script under tools/. The per-guild allowlist in config.py decides which
integrations a given Discord server may use; this module is what actually
enforces it at the point of use.

Trust boundary
--------------
The guild id comes from `DISCORD_REQUESTING_GUILD_ID`, injected into the tool
subprocess env by the bot (see claude_code_client._build_env) — NOT from
anything the model passes as an argument. So a model that is talked into
calling a tool on behalf of the wrong server still gets rejected.

This stops the realistic failure mode (the model relaying a request it should
not honor). It is NOT a sandbox: a guild that has Bash at all could in
principle script around these tools. Guilds whose allowlist is empty are
therefore denied Bash entirely by bot.py, which is the only hard boundary here.

Usage — one call at the top of a tool's main():

    from _guild_access import require_integration
    require_integration("scheduler")
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Same fallback the Discord permission layer uses: the persistent PTY session's
# env is fixed at startup, so per-message identity is written to a file instead.
_CONTEXT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "discord", ".request_context"
)


def requesting_guild_id():
    """The trusted guild id for this invocation, or None."""
    gid = os.environ.get("DISCORD_REQUESTING_GUILD_ID") or None
    if gid:
        return gid
    try:
        with open(_CONTEXT_FILE) as f:
            return (json.load(f).get("guild_id") or None)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def is_system_context() -> bool:
    """True when the scheduler launched this tool, not a Discord user.

    Mirrors tools/discord/_permissions.is_system_context. run_due.py sets
    DISCORD_SYSTEM_CONTEXT=1 for the tasks it starts; a cron reminder calling
    send_message.py has no requesting guild and must not be denied. The flag is
    required, so an arbitrary no-context invocation still fails closed.
    """
    if os.environ.get("DISCORD_SYSTEM_CONTEXT") != "1":
        return False
    return requesting_guild_id() is None


def integration_allowed(integration: str, guild_id=None) -> bool:
    """Whether `integration` is allowed for the requesting guild."""
    from config import tools_allowed_for

    gid = guild_id or requesting_guild_id()
    if not gid:
        # Fail closed on guild-scoped checks: an unverifiable requester gets
        # nothing. (Scheduler-launched tasks set the context explicitly.)
        return False
    return integration in tools_allowed_for(gid)


def require_integration(integration: str) -> None:
    """Exit with a permission error unless this guild may use `integration`."""
    if is_system_context():
        return  # scheduler-launched (cron reminder, summarizer) — not a guild request
    gid = requesting_guild_id()
    if gid is None:
        print(
            "error: cannot verify which Discord server this request came from, "
            "so the tool is denied (fail-closed).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not integration_allowed(integration, gid):
        from config import tools_allowed_for

        allowed = sorted(tools_allowed_for(gid))
        print(
            f"error: the '{integration}' tools are not enabled for this server "
            f"(guild {gid}). Allowed here: {', '.join(allowed) if allowed else 'none'}. "
            "Relay this to the user plainly — do not retry or work around it.",
            file=sys.stderr,
        )
        sys.exit(1)
