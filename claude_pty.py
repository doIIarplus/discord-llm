"""Tmux-based Claude Code interactive session manager.

Runs Claude Code in a persistent tmux session, sends prompts via `send-keys`,
and reads responses via `capture-pane`. This looks like a legitimate interactive
terminal session to Claude Code.

Live monitoring: attach from any terminal with `tmux attach -t <session_name>`
"""

import asyncio
import re
import subprocess
import time
import os
from dataclasses import dataclass, field
from typing import Optional

# Claude Code CLI path
CLAUDE_CLI = os.path.expanduser("~/.local/bin/claude")

# The prompt indicator Claude Code shows when ready for input
READY_PROMPT = "❯"

# Markers in the TUI output
RESPONSE_MARKER = "●"  # Claude's response starts with this bullet


@dataclass
class ClaudeTmuxSession:
    """Manages a single Claude Code interactive session inside tmux."""

    session_name: str = "claude_bot"
    model: str = "opus"
    width: int = 200
    height: int = 50
    _alive: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def start(self) -> None:
        """Spawn Claude Code in a new tmux session, or reuse an existing one."""
        # Check if a session already exists and is ready
        if self.is_alive:
            screen = await self._capture_pane()
            if READY_PROMPT in screen:
                self._alive = True
                print(f"[claude_pty] Reusing existing session '{self.session_name}'")
                return

        # Kill broken session if any
        await self._tmux("kill-session", "-t", self.session_name, check=False)
        await asyncio.sleep(0.5)

        # Start new session with claude
        cmd = f"{CLAUDE_CLI} --dangerously-skip-permissions --model {self.model}"
        await self._tmux(
            "new-session", "-d",
            "-s", self.session_name,
            "-x", str(self.width),
            "-y", str(self.height),
            cmd,
        )

        # Wait for the ready prompt
        if not await self._wait_for_ready(timeout=30):
            raise RuntimeError("Claude Code failed to reach ready state within 30s")

        self._alive = True
        print(f"[claude_pty] Session '{self.session_name}' started")

    async def send_prompt(self, prompt: str, timeout: float = 600) -> str:
        """Send a prompt and return the response text.

        Args:
            prompt: The prompt to send.
            timeout: Max seconds to wait for a response.

        Returns:
            The response text (cleaned).
        """
        async with self._lock:
            if not self._alive:
                await self.start()

            # Capture screen before sending to know where the new response starts
            before = await self._capture_pane(scrollback=True)

            # Send the prompt — collapse to single line to avoid newlines
            # being interpreted as Enter keypresses by the TUI
            single_line = prompt.replace("\n", " ").replace("\r", "")
            await self._send_keys(single_line)
            # Brief pause for the TUI to process the paste before submitting
            await asyncio.sleep(0.5)
            await self._send_keys("Enter", literal=False)

            # Wait for response to complete (new ready prompt appears)
            response = await self._wait_for_response(before, timeout)
            return response

    async def get_screen(self) -> str:
        """Get the current screen content (for monitoring)."""
        return await self._capture_pane()

    async def get_full_history(self) -> str:
        """Get the full scrollback history."""
        return await self._capture_pane(scrollback=True)

    async def stop(self) -> None:
        """Kill the tmux session."""
        await self._tmux("kill-session", "-t", self.session_name, check=False)
        self._alive = False
        print(f"[claude_pty] Session '{self.session_name}' stopped")

    @property
    def is_alive(self) -> bool:
        """Check if the tmux session exists."""
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", self.session_name],
                capture_output=True, timeout=5,
            )
            self._alive = result.returncode == 0
            return self._alive
        except Exception:
            self._alive = False
            return False

    # -- Internal methods --

    async def _wait_for_ready(self, timeout: float = 30) -> bool:
        """Wait until Claude Code shows its input prompt."""
        start = time.time()
        while time.time() - start < timeout:
            screen = await self._capture_pane()
            lines = screen.strip().split("\n")

            # Check for the permission prompt and accept it
            for line in lines:
                if "No, exit" in line:
                    # Navigate to "Yes" and accept
                    await self._send_keys("Down", literal=False)
                    await asyncio.sleep(0.3)
                    await self._send_keys("Enter", literal=False)
                    await asyncio.sleep(2)
                    continue

            # Look for the ready prompt (❯ on its own line, indicating input area)
            for line in lines:
                stripped = line.strip()
                if stripped == READY_PROMPT or stripped == f"{READY_PROMPT} ":
                    return True

            await asyncio.sleep(1)
        return False

    async def _wait_for_response(self, before_text: str, timeout: float) -> str:
        """Wait for Claude to finish responding and extract the response.

        Detects completion by watching for a new ready prompt (❯) after
        the response content. Uses output quiescence as a secondary signal.
        """
        start = time.time()
        last_change_time = time.time()
        last_screen = ""
        prompt_seen_in_response = False

        while time.time() - start < timeout:
            # Fast crash detection — if session died, bail immediately
            if not self.is_alive:
                self._alive = False
                raise RuntimeError("Claude Code tmux session died mid-response")

            screen = await self._capture_pane(scrollback=True)

            if screen != last_screen:
                last_change_time = time.time()
                last_screen = screen

            # Parse the response from the screen
            response = self._extract_response(screen, before_text)

            # Check if we see a new ready prompt AFTER response content
            # The pattern is: response text, then ❯ on a blank line
            lines = screen.strip().split("\n")
            found_response_content = False
            found_ready_after = False

            for line in reversed(lines):
                stripped = line.strip()
                if stripped == READY_PROMPT or stripped == f"{READY_PROMPT} ":
                    if found_response_content:
                        found_ready_after = True
                        break
                elif stripped and stripped != "─" * len(stripped) and "bypass permissions" not in stripped and "shift+tab" not in stripped:
                    # Found non-empty, non-decoration line
                    found_response_content = True

            if found_ready_after and response:
                return response

            # Quiescence fallback: if screen hasn't changed for 5s and we have content
            if response and (time.time() - last_change_time > 5):
                return response

            await asyncio.sleep(0.5)

        # Timeout — return whatever we have
        screen = await self._capture_pane(scrollback=True)
        response = self._extract_response(screen, before_text)
        if response:
            return response
        raise TimeoutError(f"No response from Claude after {timeout}s")

    def _extract_response(self, screen: str, before_text: str) -> str:
        """Extract Claude's response from the screen output.

        Only considers content that is NEW (not in before_text).
        Finds the last ● response marker in new content and collects
        everything between it and the next ready prompt (❯).
        """
        # Only look at lines that are new since we sent the prompt
        before_lines = set(before_text.strip().split("\n"))
        screen_lines = screen.strip().split("\n")
        new_lines = [l for l in screen_lines if l not in before_lines]

        # Find the LAST ● marker in new content (handles multi-turn tool use)
        last_marker_idx = -1
        for i, line in enumerate(new_lines):
            if line.strip().startswith(RESPONSE_MARKER):
                last_marker_idx = i

        if last_marker_idx == -1:
            return ""

        # Collect from the last ● marker to the end (or next ❯)
        response_lines = []
        first_line = new_lines[last_marker_idx].strip()
        after_marker = first_line[len(RESPONSE_MARKER):].strip()
        if after_marker:
            response_lines.append(after_marker)

        for line in new_lines[last_marker_idx + 1:]:
            stripped = line.strip()
            if stripped == READY_PROMPT or stripped == f"{READY_PROMPT} ":
                break
            if "─" * 20 in stripped:
                continue
            if "bypass permissions" in stripped or "shift+tab" in stripped:
                continue
            if stripped.startswith("/") and len(stripped) < 20:
                continue
            response_lines.append(line)

        # Clean up
        text = "\n".join(response_lines).strip()
        text = re.sub(r"\n*─+\s*$", "", text).strip()

        return text

    async def _capture_pane(self, scrollback: bool = False) -> str:
        """Capture the tmux pane content."""
        args = ["tmux", "capture-pane", "-t", self.session_name, "-p"]
        if scrollback:
            args.extend(["-S", "-1000"])  # Last 1000 lines of scrollback

        result = await asyncio.to_thread(
            subprocess.run, args,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout if result.returncode == 0 else ""

    async def _send_keys(self, keys: str, literal: bool = True) -> None:
        """Send keys to the tmux session."""
        args = ["tmux", "send-keys", "-t", self.session_name]
        if literal:
            args.extend(["-l", keys])
        else:
            args.append(keys)

        await asyncio.to_thread(
            subprocess.run, args,
            capture_output=True, timeout=5,
        )

    async def _tmux(self, *args, check: bool = True) -> subprocess.CompletedProcess:
        """Run a tmux command."""
        cmd = ["tmux"] + list(args)
        result = await asyncio.to_thread(
            subprocess.run, cmd,
            capture_output=True, text=True, timeout=10,
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"tmux command failed: {' '.join(cmd)}\n{result.stderr}")
        return result
