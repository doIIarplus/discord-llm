#!/usr/bin/env bash
# Wrapper script that restarts the bot when it exits with code 42.
# Exit code 42 = "restart requested" (e.g., after self-modification).
# Any other exit code stops the loop.
#
# When bubblewrap (bwrap) is installed, the bot runs inside an OS-level
# sandbox that restricts filesystem access to the project directory.
# Install: sudo apt-get install -y bubblewrap socat

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source venv/bin/activate

RESTART_EXIT_CODE=42

# Build the bwrap command if available
build_sandbox_cmd() {
    if ! command -v bwrap &>/dev/null; then
        echo "python" "bot.py"
        return
    fi

    echo "bwrap" \
        "--die-with-parent" \
        "--ro-bind" "/" "/" \
        "--dev" "/dev" \
        "--proc" "/proc" \
        "--tmpfs" "/tmp" \
        "--bind" "$SCRIPT_DIR" "$SCRIPT_DIR" \
        "--unsetenv" "HOME" \
        "--setenv" "HOME" "$SCRIPT_DIR" \
        "--" \
        "python" "bot.py"
}

while true; do
    echo "[$(date)] Starting bot..."
    if command -v bwrap &>/dev/null; then
        echo "[sandbox] Running with bubblewrap filesystem isolation"
        bwrap \
            --die-with-parent \
            --ro-bind / / \
            --dev /dev \
            --proc /proc \
            --tmpfs /tmp \
            --bind "$SCRIPT_DIR" "$SCRIPT_DIR" \
            -- \
            python bot.py
    else
        echo "[sandbox] WARNING: bubblewrap not installed — running without OS-level sandbox"
        echo "[sandbox] Install with: sudo apt-get install -y bubblewrap socat"
        python bot.py
    fi
    EXIT_CODE=$?
    echo "[$(date)] Bot exited with code $EXIT_CODE"

    if [ "$EXIT_CODE" -ne "$RESTART_EXIT_CODE" ]; then
        echo "Not a restart request. Exiting wrapper."
        exit $EXIT_CODE
    fi

    echo "Restart requested. Restarting in 2 seconds..."
    sleep 2
done
