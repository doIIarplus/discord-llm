#!/usr/bin/env bash
# Wrapper script that restarts the bot when it exits with code 42.
# Exit code 42 = "restart requested" (e.g., after self-modification).
# Non-zero exits (crashes) also trigger a restart so the bot can self-diagnose.
# Exit code 0 = clean shutdown (Ctrl+C, etc.) — stops the loop.
#
# OS-level sandboxing via bubblewrap (bwrap):
#   - Filesystem is read-only by default (--ro-bind / /)
#   - Project dir is writable (for code edits, state, attachments, images)
#   - Claude Code dirs are writable (CLI needs internal state)
#   - Sensitive dirs are masked with empty tmpfs (/mnt, ~/.ssh, ~/.aws, etc.)
#   - Network is NOT restricted (bot needs Discord, Ollama, SD, Tavily, Anthropic)
#
# Install: sudo apt-get install -y bubblewrap

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source venv/bin/activate

# Load CLAUDE_CODE_OAUTH_TOKEN from .env if set (long-lived OAuth token for Claude CLI)
OAUTH_TOKEN=$(grep -oP '^CLAUDE_CODE_OAUTH_TOKEN=\K.*' "$SCRIPT_DIR/.env" 2>/dev/null)
if [ -n "$OAUTH_TOKEN" ]; then
    export CLAUDE_CODE_OAUTH_TOKEN="$OAUTH_TOKEN"
    echo "[auth] Using long-lived OAuth token from .env"
fi

RESTART_EXIT_CODE=42
HOME_DIR="$HOME"
LOG_FILE="$SCRIPT_DIR/bot.log"

# Ensure Claude Code state dirs exist (bwrap --bind requires them)
mkdir -p "$HOME_DIR/.claude"
mkdir -p "$HOME_DIR/.local/share/claude"

run_bot() {
    # Trim log file to last 200 lines before each start
    if [ -f "$LOG_FILE" ]; then
        tail -n 200 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
    fi

    if ! command -v bwrap &>/dev/null; then
        echo "[sandbox] WARNING: bubblewrap not installed — no OS-level sandbox"
        echo "[sandbox] Install with: sudo apt-get install -y bubblewrap"
        python bot.py 2>&1 | tee -a "$LOG_FILE"
        return "${PIPESTATUS[0]}"
    fi

    # Resolve the real path of resolv.conf (on WSL2 it's a symlink to /mnt/wsl/resolv.conf)
    local RESOLV_REAL
    RESOLV_REAL="$(realpath /etc/resolv.conf 2>/dev/null)"

    # Build bwrap args — only mask dirs that actually exist
    local BWRAP_ARGS=(
        --ro-bind / /
        --dev /dev
        --proc /proc
        --tmpfs /tmp
        --bind "$SCRIPT_DIR" "$SCRIPT_DIR"
        --bind "$HOME_DIR/.claude" "$HOME_DIR/.claude"
        --bind "$HOME_DIR/.local/share/claude" "$HOME_DIR/.local/share/claude"
        --tmpfs /mnt
    )

    # On WSL2, /etc/resolv.conf -> /mnt/wsl/resolv.conf. Since we mask /mnt,
    # we must bind the real file back in so DNS works inside the sandbox.
    if [[ "$RESOLV_REAL" == /mnt/* ]] && [ -f "$RESOLV_REAL" ]; then
        local RESOLV_DIR
        RESOLV_DIR="$(dirname "$RESOLV_REAL")"
        BWRAP_ARGS+=(--ro-bind "$RESOLV_DIR" "$RESOLV_DIR")
    fi

    # Mask sensitive dirs only if they exist as real directories
    # (bwrap --tmpfs needs the mount point to exist on the underlying fs)
    for dir in "$HOME_DIR/.ssh" "$HOME_DIR/.gnupg" "$HOME_DIR/.aws"; do
        if [ -d "$dir" ] && [ ! -L "$dir" ]; then
            BWRAP_ARGS+=(--tmpfs "$dir")
        fi
    done

    echo "[sandbox] Running with bubblewrap filesystem isolation"
    bwrap "${BWRAP_ARGS[@]}" -- python bot.py 2>&1 | tee -a "$LOG_FILE"
    return "${PIPESTATUS[0]}"
}

while true; do
    echo "[$(date)] Starting bot..."
    run_bot
    EXIT_CODE=$?
    echo "[$(date)] Bot exited with code $EXIT_CODE" | tee -a "$LOG_FILE"

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "Clean shutdown. Exiting wrapper."
        exit 0
    fi

    if [ "$EXIT_CODE" -eq "$RESTART_EXIT_CODE" ]; then
        echo "Restart requested. Restarting in 2 seconds..."
        sleep 2
        continue
    fi

    # Crash detected — write sentinel and restart so bot can self-diagnose
    echo "$EXIT_CODE" > "$SCRIPT_DIR/crash_exit_code"
    echo "[$(date)] Crash detected (exit code $EXIT_CODE). Restarting for self-diagnosis..." | tee -a "$LOG_FILE"
    sleep 2
done
