"""Claude Code CLI client for Discord LLM Bot.

Uses the `claude` CLI in non-interactive print mode (-p) to generate responses.
This routes through your Claude Max/Pro subscription instead of API credits.
"""

import asyncio
import json
import os
import time
import traceback
from datetime import datetime, timezone
from typing import List, Optional, Tuple

CLAUDE_CLI = "claude"

# Keywords in error messages that indicate rate limiting
_RATE_LIMIT_KEYWORDS = ["rate limit", "rate_limit", "usage limit", "capacity", "overloaded", "too many"]


class RateLimitError(Exception):
    """Raised when Claude Code CLI hits a rate limit."""

    def __init__(self, message: str, resets_at: Optional[float] = None):
        super().__init__(message)
        self.resets_at = resets_at


class ClaudeCodeClient:
    """Client that shells out to the Claude Code CLI."""

    def __init__(self):
        # Track rate limit state so we skip CLI calls until reset
        self._rate_limited_until: Optional[float] = None

    @property
    def is_rate_limited(self) -> bool:
        if self._rate_limited_until is None:
            return False
        if time.time() >= self._rate_limited_until:
            self._rate_limited_until = None
            return False
        return True

    @property
    def rate_limit_resets_at(self) -> Optional[str]:
        """Human-readable reset time, or None if not rate limited."""
        if not self.is_rate_limited:
            return None
        dt = datetime.fromtimestamp(self._rate_limited_until, tz=timezone.utc)
        return dt.astimezone().strftime("%I:%M %p %Z")

    def _format_reset_time(self, timestamp: float) -> str:
        """Format a unix timestamp into a human-readable local time string."""
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        local = dt.astimezone()
        # Show relative time too
        delta = timestamp - time.time()
        if delta <= 0:
            return local.strftime("%I:%M %p %Z")
        hours = int(delta // 3600)
        mins = int((delta % 3600) // 60)
        if hours > 0:
            relative = f"{hours}h {mins}m"
        else:
            relative = f"{mins}m"
        return f"{local.strftime('%I:%M %p %Z')} ({relative} from now)"

    async def generate(
        self,
        prompt: str,
        model: str = "sonnet",
        images: Optional[List[str]] = None,
    ) -> str:
        """Generate a response via Claude Code CLI."""
        if images:
            print("  [warning: images not supported via Claude Code CLI, ignoring]")

        return await self._run_cli(prompt, model)

    async def generate_with_search(
        self,
        prompt: str,
        model: str = "sonnet",
        images: Optional[List[str]] = None,
    ) -> Tuple[str, List[dict]]:
        """Generate a response with web search tools enabled.

        Returns:
            (response_text, sources) — sources is always empty for CLI mode
            since we can't easily extract citations from CLI output.
        """
        if images:
            print("  [warning: images not supported via Claude Code CLI, ignoring]")

        text = await self._run_cli(prompt, model, enable_search=True)
        return text, []

    async def run_code_edit(
        self,
        instruction: str,
        model: str = "opus",
        timeout: float = 300.0,
    ) -> Tuple[str, int]:
        """Run Claude Code CLI with file editing permissions, scoped to project dir.

        Returns:
            (response_text, exit_code)
        """
        if self.is_rate_limited:
            reset = self.rate_limit_resets_at
            raise RateLimitError(
                f"Claude Code rate limited, resets at {reset}",
                resets_at=self._rate_limited_until,
            )

        PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

        _ALIAS_MAP = {
            "claude-code": "sonnet",
            "claude-code-opus": "opus",
        }
        model_alias = _ALIAS_MAP.get(model, model)
        if "sonnet" in model_alias:
            model_alias = "sonnet"
        elif "opus" in model_alias:
            model_alias = "opus"

        cmd = [
            CLAUDE_CLI,
            "-p",
            "--output-format", "json",
            "--model", model_alias,
            "--no-session-persistence",
            "--dangerously-skip-permissions",
            "--allowedTools", "Read,Write,Edit,Glob,Grep",
        ]

        full_prompt = (
            f"You are modifying a Discord bot project at {PROJECT_DIR}. "
            f"IMPORTANT: Only modify files within this directory. "
            f"Do NOT create files outside this directory. "
            f"Do NOT install system packages or modify .env or credential files. "
            f"After making changes, briefly describe what you changed.\n\n"
            f"User request: {instruction}"
        )

        _STRIP_VARS = {"CLAUDECODE", "ANTHROPIC_API_KEY"}
        env = {k: v for k, v in os.environ.items() if k not in _STRIP_VARS}

        try:
            start_time = time.perf_counter()

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=PROJECT_DIR,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=full_prompt.encode("utf-8")),
                timeout=timeout,
            )

            duration = time.perf_counter() - start_time
            print(f"Claude Code edit took {duration:.2f}s")

            raw_out = stdout.decode("utf-8", errors="replace").strip()
            raw_err = stderr.decode("utf-8", errors="replace").strip()

            data = None
            try:
                data = json.loads(raw_out)
            except json.JSONDecodeError:
                pass

            if isinstance(data, dict):
                self._check_rate_limit(data)

            if proc.returncode != 0:
                detail = raw_err or raw_out
                if self._is_rate_limit_error(detail, data):
                    resets_at = self._extract_reset_time(data)
                    if resets_at:
                        self._rate_limited_until = resets_at
                    else:
                        self._rate_limited_until = time.time() + 1800
                    raise RateLimitError(
                        "Claude Code rate limited during code edit",
                        resets_at=self._rate_limited_until,
                    )

            if isinstance(data, dict) and "result" in data:
                return data["result"] or "No response.", proc.returncode

            return raw_out or "No response.", proc.returncode

        except asyncio.TimeoutError:
            print(f"Claude Code edit timed out after {timeout}s")
            proc.kill()
            return f"Timed out after {timeout}s", -1
        except RateLimitError:
            raise
        except Exception as e:
            print(f"Error in Claude Code edit: {e}")
            print(traceback.format_exc())
            return str(e), -1

    async def _run_cli(
        self,
        prompt: str,
        model: str,
        enable_search: bool = False,
        timeout: float = 120.0,
    ) -> str:
        """Run the claude CLI and return the response text."""
        # Check if we're currently rate limited
        if self.is_rate_limited:
            reset = self.rate_limit_resets_at
            raise RateLimitError(
                f"Claude Code rate limited, resets at {reset}",
                resets_at=self._rate_limited_until,
            )

        # Map model values to CLI aliases
        _ALIAS_MAP = {
            "claude-code": "sonnet",
            "claude-code-opus": "opus",
        }
        model_alias = _ALIAS_MAP.get(model, model)
        if "sonnet" in model_alias:
            model_alias = "sonnet"
        elif "opus" in model_alias:
            model_alias = "opus"
        elif "haiku" in model_alias:
            model_alias = "haiku"

        cmd = [
            CLAUDE_CLI,
            "-p",
            "--output-format", "json",
            "--model", model_alias,
            "--no-session-persistence",
        ]

        # Only allow web tools — no file editing, no bash
        if enable_search:
            cmd.extend(["--allowedTools", "WebSearch,WebFetch"])
        else:
            cmd.extend(["--tools", ""])

        try:
            start_time = time.perf_counter()

            # Strip env vars that interfere with Claude Code CLI:
            # - CLAUDECODE: prevents "cannot launch inside another session" error
            # - ANTHROPIC_API_KEY: forces CLI to use subscription auth instead of a (possibly invalid) API key
            _STRIP_VARS = {"CLAUDECODE", "ANTHROPIC_API_KEY"}
            env = {k: v for k, v in os.environ.items() if k not in _STRIP_VARS}

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=timeout,
            )

            duration = time.perf_counter() - start_time
            print(f"Claude Code CLI took {duration:.2f}s")

            raw_out = stdout.decode("utf-8", errors="replace").strip()
            raw_err = stderr.decode("utf-8", errors="replace").strip()

            # Try to parse JSON (works for both success and error responses)
            data = None
            try:
                data = json.loads(raw_out)
            except json.JSONDecodeError:
                pass

            # Check for rate limit in the JSON response
            if isinstance(data, dict):
                self._check_rate_limit(data)

            if proc.returncode != 0:
                detail = raw_err or raw_out
                print(f"Claude Code CLI error (rc={proc.returncode}): {detail}")

                # Check if the error is a rate limit
                if self._is_rate_limit_error(detail, data):
                    resets_at = self._extract_reset_time(data)
                    if resets_at:
                        self._rate_limited_until = resets_at
                        reset_str = self._format_reset_time(resets_at)
                        raise RateLimitError(
                            f"Claude Code rate limited, resets at {reset_str}",
                            resets_at=resets_at,
                        )
                    else:
                        # Rate limited but no reset time — default to 30 min
                        self._rate_limited_until = time.time() + 1800
                        raise RateLimitError(
                            "Claude Code rate limited (resets in ~30 min)",
                            resets_at=self._rate_limited_until,
                        )

                raise RuntimeError(f"Claude Code CLI exited with code {proc.returncode}: {detail}")

            # Parse successful JSON response
            if isinstance(data, dict) and "result" in data:
                return data["result"] or "No response from Claude Code."

            # Fallback: treat as plain text
            return raw_out if raw_out else "No response from Claude Code."

        except asyncio.TimeoutError:
            print(f"Claude Code CLI timed out after {timeout}s")
            proc.kill()
            raise RuntimeError(f"Claude Code CLI timed out after {timeout}s")
        except (RateLimitError, RuntimeError):
            raise
        except Exception as e:
            print(f"Error in Claude Code CLI: {e}")
            print(traceback.format_exc())
            raise

    def _check_rate_limit(self, data: dict):
        """Check JSON response for rate limit info and update state."""
        # The result JSON may contain usage info with rate limit details
        usage = data.get("usage", {})

        # Check if there's a rate limit event embedded in the response
        # (visible in verbose mode, but the result object may also signal it)
        if data.get("is_error") and data.get("result", ""):
            result_text = data["result"].lower()
            if any(kw in result_text for kw in _RATE_LIMIT_KEYWORDS):
                resets_at = self._extract_reset_time(data)
                if resets_at:
                    self._rate_limited_until = resets_at

    def _is_rate_limit_error(self, detail: str, data: Optional[dict]) -> bool:
        """Check if an error response indicates rate limiting."""
        detail_lower = detail.lower()
        if any(kw in detail_lower for kw in _RATE_LIMIT_KEYWORDS):
            return True

        # Check JSON is_error field with rate limit content
        if isinstance(data, dict) and data.get("is_error"):
            result_text = (data.get("result") or "").lower()
            if any(kw in result_text for kw in _RATE_LIMIT_KEYWORDS):
                return True

        return False

    def _extract_reset_time(self, data: Optional[dict]) -> Optional[float]:
        """Try to extract a reset timestamp from the JSON response."""
        if not isinstance(data, dict):
            return None

        # Check for rate_limit_info in the response
        rate_info = data.get("rate_limit_info", {})
        if rate_info:
            resets_at = rate_info.get("resetsAt")
            if resets_at:
                return float(resets_at)

        # Check result text for timestamp patterns
        result_text = data.get("result", "")
        if "resets" in result_text.lower():
            # Try to find a unix timestamp in the text
            import re
            ts_match = re.search(r'(\d{10,13})', result_text)
            if ts_match:
                ts = int(ts_match.group(1))
                if ts > 1e12:  # milliseconds
                    ts = ts / 1000
                return float(ts)

        return None
