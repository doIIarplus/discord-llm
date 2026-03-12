#!/usr/bin/env bash
# Quick test: verify bwrap sandbox works (DNS, filesystem isolation)
# Run manually: bash test_sandbox.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_DIR="$HOME"

RESOLV_REAL="$(realpath /etc/resolv.conf 2>/dev/null)"
echo "resolv.conf -> $RESOLV_REAL"

BWRAP_ARGS=(
    --ro-bind / /
    --dev /dev
    --proc /proc
    --tmpfs /tmp
    --bind "$SCRIPT_DIR" "$SCRIPT_DIR"
    --bind "$HOME_DIR/.claude" "$HOME_DIR/.claude"
    --bind "$HOME_DIR/.local/share/claude" "$HOME_DIR/.local/share/claude"
    --tmpfs /mnt
)

# On WSL2, /etc/resolv.conf symlinks to /mnt/wsl/resolv.conf.
# Bind the real file back so DNS works after masking /mnt.
if [[ "$RESOLV_REAL" == /mnt/* ]] && [ -f "$RESOLV_REAL" ]; then
    RESOLV_DIR="$(dirname "$RESOLV_REAL")"
    BWRAP_ARGS+=(--ro-bind "$RESOLV_DIR" "$RESOLV_DIR")
    echo "Binding $RESOLV_DIR for DNS"
fi

for dir in "$HOME_DIR/.ssh" "$HOME_DIR/.gnupg" "$HOME_DIR/.aws"; do
    [ -d "$dir" ] && BWRAP_ARGS+=(--tmpfs "$dir")
done

echo ""
echo "Running tests inside bwrap sandbox..."
echo ""

bwrap "${BWRAP_ARGS[@]}" -- python3 -c "
import os, socket

# 1. DNS
try:
    r = socket.getaddrinfo('google.com', 80, socket.AF_INET, socket.SOCK_STREAM)
    print(f'PASS: DNS works ({r[0][4][0]})')
except Exception as e:
    print(f'FAIL: DNS broken ({e})')

# 2. /mnt/c blocked
try:
    os.listdir('/mnt/c/Users/Daniel/Documents')
    print('FAIL: /mnt/c/Users/Daniel/Documents is accessible!')
except (FileNotFoundError, OSError) as e:
    print(f'PASS: /mnt/c blocked ({type(e).__name__})')

# 3. Project dir writable
try:
    f = '$SCRIPT_DIR/_sandbox_test'
    open(f, 'w').write('ok')
    os.remove(f)
    print('PASS: project dir writable')
except Exception as e:
    print(f'FAIL: project dir not writable ({e})')

# 4. /etc read-only
try:
    open('/etc/_test', 'w')
    print('FAIL: /etc is writable!')
except OSError:
    print('PASS: /etc read-only')

# 5. Home dir read-only (outside project)
try:
    open('$HOME_DIR/_test', 'w')
    print('FAIL: home dir is writable!')
except OSError:
    print('PASS: home dir read-only')

print()
print('All tests done.')
"
