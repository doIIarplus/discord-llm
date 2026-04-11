"""Claude Code client using persistent tmux-based interactive sessions.

Drop-in replacement for ClaudeCodeClient that runs Claude Code in interactive
mode inside tmux instead of spawning one-shot `claude -p` processes.

For general queries (generate, generate_with_tools), uses the persistent
tmux session. For code edits (run_code_edit, run_plugin_edit), falls back
to one-shot `claude -p` processes since those need specific tool restrictions.
"""

import asyncio
import time
from typing import List, Optional, Tuple

from claude_code_client import (
    ClaudeCodeClient, RateLimitError, TestResult,
    _RATE_LIMIT_KEYWORDS, _log_request,
)
from claude_pty import ClaudeTmuxSession


class ClaudeCodeClientPTY(ClaudeCodeClient):
    """Claude Code client that uses a persistent tmux interactive session.

    Inherits from ClaudeCodeClient and overrides the generation methods to
    route through a persistent tmux session. Code edit methods are inherited
    as-is (they use one-shot processes with restricted tools).
    """

    def __init__(self, session_name: str = "claude_bot", model: str = "opus"):
        super().__init__()
        self._session = ClaudeTmuxSession(
            session_name=session_name,
            model=model,
        )
        self._started = False
        self._system_prompt_sent = False
        self._session_model = model

    async def ensure_session(self):
        """Start the tmux session if not already running."""
        if not self._started or not self._session.is_alive:
            await self._session.start()
            self._started = True
            self._system_prompt_sent = False

    async def set_system_prompt(self, system_prompt: str):
        """Send the system prompt to the session (once, or when it changes)."""
        await self.ensure_session()
        if not self._system_prompt_sent:
            await self._session.send_prompt(
                f"For this entire session, adopt this persona and follow these rules:\n\n{system_prompt}\n\n"
                f"Acknowledge with just 'ok' and wait for messages."
            )
            self._system_prompt_sent = True
            print("[pty] System prompt sent to tmux session")

    def _check_pty_rate_limit(self, response: str):
        """Check if a PTY response contains rate limit indicators."""
        lower = response.lower()
        if any(kw in lower for kw in _RATE_LIMIT_KEYWORDS):
            # Set rate limit state (default 30 min cooldown)
            self._rate_limited_until = time.time() + 1800
            raise RateLimitError(
                "Claude Code rate limited (detected in PTY response)",
                resets_at=self._rate_limited_until,
            )

    async def _pty_generate(self, prompt: str, method: str) -> str:
        """Core PTY generation with rate limit detection and logging."""
        if self.is_rate_limited:
            raise RateLimitError(
                f"Claude Code rate limited, resets at {self.rate_limit_resets_at}",
                resets_at=self._rate_limited_until,
            )

        await self.ensure_session()

        start_time = time.perf_counter()
        response = await self._session.send_prompt(prompt)
        duration = time.perf_counter() - start_time

        # Check for rate limits in the response
        self._check_pty_rate_limit(response)

        # Log to SQLite (same DB as CLI client, limited metadata)
        _log_request(
            method=f"pty_{method}",
            model=self._session_model,
            prompt=prompt,
            raw_json="",
            raw_stderr="",
            exit_code=0,
            duration_ms=duration * 1000,
            thinking=None,
            response=response,
            cost_usd=None,
            input_tokens=None,
            output_tokens=None,
            cache_read_tokens=None,
            cache_creation_tokens=None,
        )

        return response

    async def generate(
        self,
        prompt: str,
        model: str = "sonnet",
        images: Optional[List[str]] = None,
    ) -> str:
        """Generate a response via the persistent tmux session."""
        if images:
            print("  [warning: images not supported via Claude Code PTY, ignoring]")
        return await self._pty_generate(prompt, "generate")

    async def generate_with_search(
        self,
        prompt: str,
        model: str = "sonnet",
        images: Optional[List[str]] = None,
    ) -> Tuple[str, List[dict]]:
        """Generate with search — interactive session has all tools available."""
        if images:
            print("  [warning: images not supported via Claude Code PTY, ignoring]")
        text = await self._pty_generate(prompt, "generate_with_search")
        return text, []

    async def generate_with_tools(
        self,
        prompt: str,
        model: str = "sonnet",
        images: Optional[List[str]] = None,
    ) -> Tuple[str, List[dict]]:
        """Generate with tools — interactive session has all tools available."""
        if images:
            print("  [warning: images not supported via Claude Code PTY, ignoring]")
        text = await self._pty_generate(prompt, "generate_with_tools")
        return text, []

    # run_code_edit, run_plugin_edit, run_tests — inherited from ClaudeCodeClient
    # These use one-shot `claude -p` processes with specific tool restrictions,
    # which is fine since they're infrequent and need isolation.

    async def get_session_screen(self) -> str:
        """Get the current tmux screen content (for monitoring)."""
        return await self._session.get_screen()

    async def shutdown(self):
        """Stop the tmux session."""
        await self._session.stop()
        self._started = False
