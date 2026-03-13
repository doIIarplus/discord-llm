"""Claude Code CLI client for Discord LLM Bot.

Uses the `claude` CLI in non-interactive print mode (-p) to generate responses.
This routes through your Claude Max/Pro subscription instead of API credits.
"""

import asyncio
import json
import os
import re
import sqlite3
import time
import traceback
from datetime import datetime, timezone
from typing import List, Optional, Tuple

CLAUDE_CLI = "claude"

# Ollama configuration for Claude Code integration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Claude model aliases — anything not in this set is treated as an Ollama model
_CLAUDE_ALIASES = {"sonnet", "opus", "haiku", "claude-code", "claude-code-opus"}

# Map cc- prefixed model values to their actual Ollama model names
_OLLAMA_MODEL_MAP = {
    "cc-qwen3.5:35b-a3b-q8_0": "qwen3.5:35b-a3b-q8_0",
}

# SQLite log database path
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_code_logs.db")


def _init_db():
    """Create the logging database and table if they don't exist."""
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            method TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt TEXT NOT NULL,
            thinking TEXT,
            response TEXT,
            raw_json TEXT NOT NULL,
            raw_stderr TEXT,
            exit_code INTEGER,
            duration_ms REAL,
            cost_usd REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_read_tokens INTEGER,
            cache_creation_tokens INTEGER
        )
    """)
    conn.commit()
    conn.close()


def _log_request(
    method: str,
    model: str,
    prompt: str,
    raw_json: str,
    raw_stderr: str = "",
    exit_code: int = 0,
    duration_ms: float = 0,
    thinking: str = None,
    response: str = None,
    cost_usd: float = None,
    input_tokens: int = None,
    output_tokens: int = None,
    cache_read_tokens: int = None,
    cache_creation_tokens: int = None,
):
    """Log a request/response to the SQLite database."""
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.execute(
            """INSERT INTO requests
               (timestamp, method, model, prompt, thinking, response, raw_json,
                raw_stderr, exit_code, duration_ms, cost_usd, input_tokens,
                output_tokens, cache_read_tokens, cache_creation_tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                method,
                model,
                prompt,
                thinking,
                response,
                raw_json,
                raw_stderr,
                exit_code,
                duration_ms,
                cost_usd,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_creation_tokens,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  [warning: failed to log request to DB: {e}]")


def _parse_verbose_output(raw_out: str) -> dict:
    """Parse verbose JSON array output from claude CLI.

    Returns a dict with keys:
        result_text, thinking, cost_usd, input_tokens, output_tokens,
        cache_read_tokens, cache_creation_tokens, duration_ms,
        is_error, result_data (the full result entry dict)
    """
    parsed = {
        "result_text": None,
        "thinking": None,
        "cost_usd": None,
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
        "cache_creation_tokens": None,
        "duration_ms": None,
        "is_error": False,
        "result_data": None,
    }

    try:
        data = json.loads(raw_out)
    except json.JSONDecodeError:
        return parsed

    # Verbose mode returns a JSON array of events
    if isinstance(data, list):
        thinking_parts = []
        for entry in data:
            entry_type = entry.get("type")

            # Extract thinking from assistant messages
            if entry_type == "assistant":
                for block in entry.get("message", {}).get("content", []):
                    if block.get("type") == "thinking":
                        thinking_parts.append(block.get("thinking", ""))

            # Extract result
            if entry_type == "result":
                parsed["result_data"] = entry
                parsed["result_text"] = entry.get("result")
                parsed["is_error"] = entry.get("is_error", False)
                parsed["duration_ms"] = entry.get("duration_ms")
                parsed["cost_usd"] = entry.get("total_cost_usd")

                usage = entry.get("usage", {})
                parsed["input_tokens"] = usage.get("input_tokens")
                parsed["output_tokens"] = usage.get("output_tokens")
                parsed["cache_read_tokens"] = usage.get("cache_read_input_tokens")
                parsed["cache_creation_tokens"] = usage.get("cache_creation_input_tokens")

        if thinking_parts:
            parsed["thinking"] = "\n---\n".join(thinking_parts)

    # Non-verbose fallback: single JSON object (e.g. error before verbose kicks in)
    elif isinstance(data, dict):
        parsed["result_data"] = data
        parsed["result_text"] = data.get("result")
        parsed["is_error"] = data.get("is_error", False)
        parsed["duration_ms"] = data.get("duration_ms")
        parsed["cost_usd"] = data.get("total_cost_usd")

        usage = data.get("usage", {})
        parsed["input_tokens"] = usage.get("input_tokens")
        parsed["output_tokens"] = usage.get("output_tokens")
        parsed["cache_read_tokens"] = usage.get("cache_read_input_tokens")
        parsed["cache_creation_tokens"] = usage.get("cache_creation_input_tokens")

    return parsed


# Initialize the database on import
_init_db()

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

    @staticmethod
    def _is_ollama_model(model: str) -> bool:
        """Check if a model should be routed through Ollama instead of Anthropic."""
        return model in _OLLAMA_MODEL_MAP or model not in _CLAUDE_ALIASES

    @staticmethod
    def _build_env(model: str) -> dict:
        """Build environment variables for the CLI process.

        For Ollama models, sets ANTHROPIC_BASE_URL and auth vars.
        For Claude models, strips vars that interfere with subscription auth.
        """
        _STRIP_VARS = {"CLAUDECODE", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"}
        env = {k: v for k, v in os.environ.items() if k not in _STRIP_VARS}

        if ClaudeCodeClient._is_ollama_model(model):
            env["ANTHROPIC_BASE_URL"] = OLLAMA_BASE_URL
            env["ANTHROPIC_AUTH_TOKEN"] = "ollama"
            env["ANTHROPIC_API_KEY"] = ""
        return env

    @staticmethod
    def _resolve_model(model: str) -> str:
        """Resolve a model value to the CLI --model argument."""
        # Check Ollama model map first (cc- prefixed models)
        if model in _OLLAMA_MODEL_MAP:
            return _OLLAMA_MODEL_MAP[model]

        _ALIAS_MAP = {
            "claude-code": "sonnet",
            "claude-code-opus": "opus",
        }
        resolved = _ALIAS_MAP.get(model, model)
        # For Claude aliases, normalize
        if "sonnet" in resolved:
            return "sonnet"
        if "opus" in resolved:
            return "opus"
        if "haiku" in resolved:
            return "haiku"
        # For other Ollama models, pass through as-is
        return resolved

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

        return await self._run_cli(prompt, model, method="generate")

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

        text = await self._run_cli(prompt, model, enable_search=True, method="generate_with_search")
        return text, []

    async def run_code_edit(
        self,
        instruction: str,
        model: str = "opus",
        timeout: float = 600.0,
        log_context: str = "",
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

        model_alias = self._resolve_model(model)

        cmd = [
            CLAUDE_CLI,
            "-p",
            "--output-format", "json",
            "--verbose",
            "--model", model_alias,
            "--no-session-persistence",
            "--dangerously-skip-permissions",
            "--allowedTools", "Read,Write,Edit,Glob,Grep",
        ]

        log_section = ""
        if log_context:
            log_section = (
                f"\n\nRECENT BOT LOGS (use these to diagnose the issue):\n"
                f"```\n{log_context}\n```\n"
            )

        full_prompt = (
            f"You are modifying a Discord bot project at {PROJECT_DIR}. "
            f"IMPORTANT RULES:\n"
            f"- Only modify files within this directory.\n"
            f"- Do NOT create files outside this directory.\n"
            f"- Do NOT install system packages or modify .env or credential files.\n"
            f"- Do NOT write code that accesses files outside {PROJECT_DIR}. "
            f"No hardcoded paths to /mnt/, /etc/, /home/, or any directory outside the project.\n"
            f"- Any file I/O in generated code MUST use safe_path() from sandbox.py to validate paths. "
            f"Import it with: from sandbox import safe_path\n"
            f"- After making changes, briefly describe what you changed.\n"
            f"{log_section}\n"
            f"User request: {instruction}"
        )

        env = self._build_env(model)

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

            # Parse verbose JSON array output
            parsed = _parse_verbose_output(raw_out)
            result_data = parsed["result_data"]

            if isinstance(result_data, dict):
                self._check_rate_limit(result_data)

            # Log to SQLite
            _log_request(
                method="run_code_edit",
                model=model_alias,
                prompt=full_prompt,
                raw_json=raw_out,
                raw_stderr=raw_err,
                exit_code=proc.returncode,
                duration_ms=parsed["duration_ms"] or (duration * 1000),
                thinking=parsed["thinking"],
                response=parsed["result_text"],
                cost_usd=parsed["cost_usd"],
                input_tokens=parsed["input_tokens"],
                output_tokens=parsed["output_tokens"],
                cache_read_tokens=parsed["cache_read_tokens"],
                cache_creation_tokens=parsed["cache_creation_tokens"],
            )

            if proc.returncode != 0:
                detail = raw_err or raw_out
                if self._is_rate_limit_error(detail, result_data):
                    resets_at = self._extract_reset_time(result_data)
                    if resets_at:
                        self._rate_limited_until = resets_at
                    else:
                        self._rate_limited_until = time.time() + 1800
                    raise RateLimitError(
                        "Claude Code rate limited during code edit",
                        resets_at=self._rate_limited_until,
                    )

            response_text = parsed["result_text"] or raw_out or "No response."
            return response_text, proc.returncode

        except asyncio.TimeoutError:
            print(f"Claude Code edit timed out after {timeout}s")
            proc.kill()
            _log_request(
                method="run_code_edit", model=model_alias, prompt=full_prompt,
                raw_json="", raw_stderr="timeout", exit_code=-1,
                duration_ms=timeout * 1000,
            )
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
        timeout: float = 600.0,
        method: str = "generate",
    ) -> str:
        """Run the claude CLI and return the response text."""
        # Check if we're currently rate limited
        if self.is_rate_limited:
            reset = self.rate_limit_resets_at
            raise RateLimitError(
                f"Claude Code rate limited, resets at {reset}",
                resets_at=self._rate_limited_until,
            )

        model_alias = self._resolve_model(model)

        cmd = [
            CLAUDE_CLI,
            "-p",
            "--output-format", "json",
            "--verbose",
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

            env = self._build_env(model)

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

            # Parse verbose JSON array output
            parsed = _parse_verbose_output(raw_out)
            result_data = parsed["result_data"]

            # Check for rate limit in the result entry
            if isinstance(result_data, dict):
                self._check_rate_limit(result_data)

            # Log to SQLite
            _log_request(
                method=method,
                model=model_alias,
                prompt=prompt,
                raw_json=raw_out,
                raw_stderr=raw_err,
                exit_code=proc.returncode,
                duration_ms=parsed["duration_ms"] or (duration * 1000),
                thinking=parsed["thinking"],
                response=parsed["result_text"],
                cost_usd=parsed["cost_usd"],
                input_tokens=parsed["input_tokens"],
                output_tokens=parsed["output_tokens"],
                cache_read_tokens=parsed["cache_read_tokens"],
                cache_creation_tokens=parsed["cache_creation_tokens"],
            )

            if proc.returncode != 0:
                detail = raw_err or raw_out
                print(f"Claude Code CLI error (rc={proc.returncode}): {detail}")

                # Check if the error is a rate limit
                if self._is_rate_limit_error(detail, result_data):
                    resets_at = self._extract_reset_time(result_data)
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

            # Return the result text
            if parsed["result_text"]:
                return parsed["result_text"]

            return "No response from Claude Code."

        except asyncio.TimeoutError:
            print(f"Claude Code CLI timed out after {timeout}s")
            proc.kill()
            # Log the timeout
            _log_request(
                method=method, model=model_alias, prompt=prompt,
                raw_json="", raw_stderr="timeout", exit_code=-1,
                duration_ms=timeout * 1000,
            )
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
            ts_match = re.search(r'(\d{10,13})', result_text)
            if ts_match:
                ts = int(ts_match.group(1))
                if ts > 1e12:  # milliseconds
                    ts = ts / 1000
                return float(ts)

        return None
