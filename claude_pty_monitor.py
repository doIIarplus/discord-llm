#!/usr/bin/env python3
"""Web-based live monitor for Claude Code tmux sessions.

Opens a websocket server that streams the tmux pane content to a browser.
Access from Windows at http://localhost:8765

Usage:
    python claude_pty_monitor.py [--session claude_bot] [--port 8765]
"""

import argparse
import asyncio
import html
import subprocess

import websockets

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Claude Code Monitor</title>
<style>
  body {{ background: #1a1a2e; margin: 0; padding: 20px; font-family: monospace; }}
  h1 {{ color: #e0e0e0; font-size: 14px; margin-bottom: 10px; }}
  #status {{ color: #888; font-size: 12px; margin-bottom: 10px; }}
  #terminal {{
    background: #0d1117; color: #e6edf3; padding: 16px;
    border-radius: 8px; border: 1px solid #30363d;
    white-space: pre; font-size: 13px; line-height: 1.4;
    overflow-x: auto; min-height: 400px;
  }}
</style>
</head>
<body>
<h1>Claude Code Monitor — session: {session}</h1>
<div id="status">connecting...</div>
<div id="terminal"></div>
<script>
const ws = new WebSocket("ws://localhost:{port}/ws");
const term = document.getElementById("terminal");
const status = document.getElementById("status");
ws.onopen = () => {{ status.textContent = "connected — refreshing every 500ms"; }};
ws.onmessage = (e) => {{ term.textContent = e.data; }};
ws.onclose = () => {{ status.textContent = "disconnected"; }};
ws.onerror = () => {{ status.textContent = "error"; }};
</script>
</body>
</html>"""


async def capture_pane(session_name: str) -> str:
    """Capture tmux pane content."""
    result = await asyncio.to_thread(
        subprocess.run,
        ["tmux", "capture-pane", "-t", session_name, "-p"],
        capture_output=True, text=True, timeout=5,
    )
    return result.stdout if result.returncode == 0 else "(session not found)"


async def ws_handler(websocket, session_name: str):
    """Stream tmux pane content to a websocket client."""
    try:
        while True:
            screen = await capture_pane(session_name)
            await websocket.send(screen)
            await asyncio.sleep(0.5)
    except websockets.ConnectionClosed:
        pass


async def http_handler(path, request_headers):
    """Serve the HTML page on regular HTTP requests."""
    if path == "/ws":
        return None  # Let websocket handler take over
    # Return HTML page
    return (200, [("Content-Type", "text/html")], HTML_PAGE.encode())


async def main(session_name: str, port: int):
    page = HTML_PAGE.replace("{session}", session_name).replace("{port}", str(port))

    async def handler(websocket):
        # Check if this is a websocket upgrade or regular HTTP
        await ws_handler(websocket, session_name)

    async def process_request(path, headers):
        if path != "/ws":
            return (200, [("Content-Type", "text/html")], page.encode())

    async with websockets.serve(handler, "0.0.0.0", port, process_request=process_request):
        print(f"[monitor] Serving at http://localhost:{port} — watching tmux session '{session_name}'")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="claude_bot", help="tmux session name")
    parser.add_argument("--port", type=int, default=8765, help="HTTP/WS port")
    args = parser.parse_args()
    asyncio.run(main(args.session, args.port))
