"""Owner-only access control for Splitwise CLI tools.

The Splitwise tools act on dollarplus's personal Splitwise account, so they must
only run when the *requesting* Discord user is the account owner. bot.py injects
the triggering user's identity as DISCORD_REQUESTING_USER_ID (see
tools/discord/_permissions.py for the full mechanism); this enforces it in code
rather than relying on the model to honour a prompt instruction.

Fails closed: if the requester can't be determined, access is denied.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import error

# The Discord user allowed to use Splitwise tools (dollarplus). Overridable via
# env for other deployments.
OWNER_DISCORD_ID = os.environ.get("SPLITWISE_OWNER_DISCORD_ID", "118567805678256128")


def require_owner():
    """Abort unless the requesting Discord user is the Splitwise account owner."""
    requester = os.environ.get("DISCORD_REQUESTING_USER_ID") or None
    if not requester:
        error(
            "Splitwise tools are restricted to the account owner, but the "
            "requesting user's identity could not be verified, so access was "
            "denied."
        )
    if str(requester) != str(OWNER_DISCORD_ID):
        error(
            "Access denied: Splitwise tools are tied to dollarplus's personal "
            f"account and can only be used by that user (requesting "
            f"discord_id={requester}).",
            details={"requesting_user_id": requester},
        )
