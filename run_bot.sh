#!/usr/bin/env bash
# Wrapper script that restarts the bot when it exits with code 42.
# Exit code 42 = "restart requested" (e.g., after self-modification).
# Any other exit code stops the loop.
#
# The bot's own file access is sandboxed at the Python level (sandbox.py).
# The Claude Code subprocess is sandboxed via .claude/settings.json
# (permission deny rules + sandbox config) and --allowedTools restrictions.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source venv/bin/activate

RESTART_EXIT_CODE=42

while true; do
    echo "[$(date)] Starting bot..."
    python bot.py
    EXIT_CODE=$?
    echo "[$(date)] Bot exited with code $EXIT_CODE"

    if [ "$EXIT_CODE" -ne "$RESTART_EXIT_CODE" ]; then
        echo "Not a restart request. Exiting wrapper."
        exit $EXIT_CODE
    fi

    echo "Restart requested. Restarting in 2 seconds..."
    sleep 2
done
