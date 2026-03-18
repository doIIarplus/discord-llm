"""Claude Code CLI client for Discord LLM Bot.

Uses the `claude` CLI in non-interactive print mode (-p) to generate responses.
This routes through your Claude Max/Pro subscription instead of API credits.
"""

import asyncio
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass, field
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


@dataclass
class TestResult:
    """Results from the pre-apply test step."""

    passed: bool
    tier1_report: str
    tier2_report: str = ""
    tier1_passed: bool = True
    tier2_passed: bool = True
    tier2_skipped: bool = False

    @property
    def summary(self) -> str:
        parts = []
        if not self.tier1_passed:
            parts.append("Import check failed")
        if not self.tier2_passed:
            parts.append("Validation tests failed")
        if self.tier2_skipped:
            parts.append("Deep validation skipped")
        return "; ".join(parts) if parts else "All tests passed"

    @property
    def full_report(self) -> str:
        lines = ["**Tier 1 (Import Check):**"]
        lines.append(f"```\n{self.tier1_report[:500]}\n```")
        if self.tier2_report:
            lines.append("**Tier 2 (Validation Tests):**")
            lines.append(f"```\n{self.tier2_report[:800]}\n```")
        elif self.tier2_skipped:
            lines.append("*Tier 2 skipped (rate limited or Tier 1 failed)*")
        return "\n".join(lines)


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

    async def generate_with_tools(
        self,
        prompt: str,
        model: str = "sonnet",
        images: Optional[List[str]] = None,
    ) -> Tuple[str, List[dict]]:
        """Generate a response with Bash + web tools enabled.

        Claude can call CLI tools in tools/ via Bash and search the web.
        Tool documentation is picked up automatically from CLAUDE.md.

        Returns:
            (response_text, sources) — same interface as generate_with_search.
        """
        if images:
            print("  [warning: images not supported via Claude Code CLI, ignoring]")

        text = await self._run_cli(prompt, model, enable_tools=True, method="generate_with_tools")
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

    async def run_plugin_edit(
        self,
        instruction: str,
        model: str = "opus",
        timeout: float = 600.0,
        log_context: str = "",
        existing_plugins: List[str] = None,
    ) -> Tuple[str, int]:
        """Run Claude Code CLI scoped to plugin files only.

        Same mechanism as run_code_edit but with a plugin-focused prompt
        that restricts edits to the plugins/ directory.

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

        plugin_list = "\n".join(
            f"  - plugins/{p}.py" for p in (existing_plugins or [])
        )

        log_section = ""
        if log_context:
            log_section = (
                f"\n\nRECENT BOT LOGS (use these to diagnose issues):\n"
                f"```\n{log_context}\n```\n"
            )

        full_prompt = (
            f"You are modifying a Discord bot's PLUGIN system at {PROJECT_DIR}.\n\n"
            f"PLUGIN SYSTEM RULES:\n"
            f"- FIRST read plugin_base.py to understand the BasePlugin interface.\n"
            f"- THEN read plugins/_example.py for a complete example of how to write a plugin.\n"
            f"- All new features MUST be implemented as plugins in the plugins/ directory.\n"
            f"- Each plugin is a single .py file with a class inheriting from BasePlugin.\n"
            f"- Use self.ctx to access bot state (see PluginBotContext in plugin_base.py).\n"
            f"- Do NOT modify core files: bot.py, plugin_manager.py, plugin_base.py, config.py, commands.py.\n"
            f"- Do NOT modify other plugin files unless the instruction explicitly asks for it.\n"
            f"- Any file I/O in generated code MUST use safe_path() from sandbox.py.\n"
            f"- After creating/modifying a plugin, it will be hot-reloaded (no restart needed).\n"
            f"- After making changes, briefly describe what you changed.\n"
            f"\n"
            f"EXISTING PLUGINS:\n{plugin_list or '  (none yet)'}\n"
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
            print(f"Claude Code plugin edit took {duration:.2f}s")

            raw_out = stdout.decode("utf-8", errors="replace").strip()
            raw_err = stderr.decode("utf-8", errors="replace").strip()

            parsed = _parse_verbose_output(raw_out)
            result_data = parsed["result_data"]

            if isinstance(result_data, dict):
                self._check_rate_limit(result_data)

            _log_request(
                method="run_plugin_edit",
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
                        "Claude Code rate limited during plugin edit",
                        resets_at=self._rate_limited_until,
                    )

            response_text = parsed["result_text"] or raw_out or "No response."
            return response_text, proc.returncode

        except asyncio.TimeoutError:
            print(f"Claude Code plugin edit timed out after {timeout}s")
            proc.kill()
            _log_request(
                method="run_plugin_edit", model=model_alias, prompt=full_prompt,
                raw_json="", raw_stderr="timeout", exit_code=-1,
                duration_ms=timeout * 1000,
            )
            return f"Timed out after {timeout}s", -1
        except RateLimitError:
            raise
        except Exception as e:
            print(f"Error in Claude Code plugin edit: {e}")
            print(traceback.format_exc())
            return str(e), -1

    async def run_tests(
        self,
        diff: str,
        change_type: str = "core",
        plugin_names: List[str] = None,
        model: str = "sonnet",
        timeout: float = 120.0,
    ) -> TestResult:
        """Run validation tests on pending code changes.

        Tier 1: Direct import check via subprocess (fast, free).
        Tier 2: Claude Code writes and runs tests based on the diff (if Tier 1 passes).

        Returns:
            TestResult with pass/fail status and reports.
        """
        PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

        # ── Tier 1: Import checks ──────────────────────────────────────
        tier1_results = []
        tier1_passed = True

        if change_type == "plugin" and plugin_names:
            for name in plugin_names:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-c", f"import plugins.{name}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=PROJECT_DIR,
                )
                try:
                    _, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=15.0
                    )
                    if proc.returncode == 0:
                        tier1_results.append(f"PASS: import plugins.{name}")
                    else:
                        tier1_passed = False
                        err = stderr.decode("utf-8", errors="replace").strip()
                        # Show last 3 lines of traceback for brevity
                        err_lines = err.splitlines()
                        short_err = "\n".join(err_lines[-3:]) if len(err_lines) > 3 else err
                        tier1_results.append(f"FAIL: import plugins.{name}\n  {short_err}")
                except asyncio.TimeoutError:
                    tier1_passed = False
                    tier1_results.append(f"FAIL: import plugins.{name} (timed out)")
                    proc.kill()
        else:
            # Core change: verify main module imports
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", "import bot",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=PROJECT_DIR,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=30.0
                )
                if proc.returncode == 0:
                    tier1_results.append("PASS: import bot")
                else:
                    tier1_passed = False
                    err = stderr.decode("utf-8", errors="replace").strip()
                    err_lines = err.splitlines()
                    short_err = "\n".join(err_lines[-3:]) if len(err_lines) > 3 else err
                    tier1_results.append(f"FAIL: import bot\n  {short_err}")
            except asyncio.TimeoutError:
                tier1_passed = False
                tier1_results.append("FAIL: import bot (timed out)")
                proc.kill()

        tier1_report = "\n".join(tier1_results)

        # If Tier 1 failed, skip Tier 2
        if not tier1_passed:
            return TestResult(
                passed=False,
                tier1_report=tier1_report,
                tier1_passed=False,
                tier2_skipped=True,
            )

        # ── Tier 2: Claude Code validation tests ──────────────────────
        # Skip if rate limited
        if self.is_rate_limited:
            return TestResult(
                passed=True,  # Tier 1 passed, so cautiously optimistic
                tier1_report=tier1_report,
                tier1_passed=True,
                tier2_skipped=True,
            )

        model_alias = self._resolve_model(model)

        cmd = [
            CLAUDE_CLI,
            "-p",
            "--output-format", "json",
            "--verbose",
            "--model", model_alias,
            "--no-session-persistence",
            "--dangerously-skip-permissions",
            "--allowedTools", "Read,Bash,Glob,Grep",
        ]

        # Truncate diff for prompt
        diff_for_prompt = diff[:8000]
        if len(diff) > 8000:
            diff_for_prompt += "\n... (diff truncated, read changed files for full context)"

        test_prompt = (
            f"You are a test engineer validating changes to a Discord bot at {PROJECT_DIR}.\n\n"
            f"Here is the diff of changes that were just made:\n```diff\n{diff_for_prompt}\n```\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Read the changed files to understand the full context.\n"
            f"2. Write and run a short Python test script using Bash that validates:\n"
            f"   - The changed code is syntactically valid\n"
            f"   - Key functions/classes can be imported and instantiated where possible\n"
            f"   - Any obvious logic errors (wrong arg counts, missing attributes, type mismatches)\n"
            f"   - For plugins: instantiate the plugin class with a mock context if feasible\n"
            f"3. STRICT RULES:\n"
            f"   - Do NOT start the bot or connect to Discord/Ollama/any external service.\n"
            f"   - Do NOT make any network requests (no HTTP, no API calls, no sockets).\n"
            f"   - Do NOT use .env variables, API keys, or tokens.\n"
            f"   - Do NOT modify any source files. You are READ-ONLY except for Bash.\n"
            f"   - Do NOT install any packages.\n"
            f"   - Keep tests focused and fast (under 10 seconds total).\n"
            f"4. End your response with exactly one of:\n"
            f"   TEST_RESULT: PASS\n"
            f"   TEST_RESULT: FAIL\n"
            f"   followed by a one-line explanation.\n"
        )

        env = self._build_env(model)
        tier2_report = ""
        tier2_passed = True

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
                proc.communicate(input=test_prompt.encode("utf-8")),
                timeout=timeout,
            )

            duration = time.perf_counter() - start_time
            print(f"Claude Code test validation took {duration:.2f}s")

            raw_out = stdout.decode("utf-8", errors="replace").strip()
            raw_err = stderr.decode("utf-8", errors="replace").strip()

            parsed = _parse_verbose_output(raw_out)
            result_data = parsed["result_data"]

            if isinstance(result_data, dict):
                self._check_rate_limit(result_data)

            _log_request(
                method="run_tests",
                model=model_alias,
                prompt=test_prompt,
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

            tier2_report = parsed["result_text"] or raw_out or "No test output."

            # Check for TEST_RESULT marker
            if "TEST_RESULT: FAIL" in tier2_report:
                tier2_passed = False
            elif "TEST_RESULT: PASS" in tier2_report:
                tier2_passed = True
            elif proc.returncode != 0:
                # CLI error but no explicit marker — check for rate limit
                if self._is_rate_limit_error(raw_err or raw_out, result_data):
                    resets_at = self._extract_reset_time(result_data)
                    if resets_at:
                        self._rate_limited_until = resets_at
                    else:
                        self._rate_limited_until = time.time() + 1800
                    # Treat as skipped, not failed
                    return TestResult(
                        passed=True,
                        tier1_report=tier1_report,
                        tier1_passed=True,
                        tier2_skipped=True,
                        tier2_report="Tier 2 skipped: rate limited",
                    )
                tier2_passed = False
                tier2_report = f"Test runner exited with code {proc.returncode}:\n{tier2_report}"

        except asyncio.TimeoutError:
            print(f"Claude Code test validation timed out after {timeout}s")
            proc.kill()
            tier2_report = f"Test validation timed out after {timeout}s"
            tier2_passed = False
        except Exception as e:
            print(f"Error in Claude Code test validation: {e}")
            print(traceback.format_exc())
            tier2_report = f"Test runner error: {e}"
            tier2_passed = False

        return TestResult(
            passed=tier1_passed and tier2_passed,
            tier1_report=tier1_report,
            tier2_report=tier2_report,
            tier1_passed=tier1_passed,
            tier2_passed=tier2_passed,
        )

    async def _run_cli(
        self,
        prompt: str,
        model: str,
        enable_search: bool = False,
        enable_tools: bool = False,
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

        # Tool access modes (mutually exclusive):
        #   enable_tools: Bash + web (for CLI tool calling)
        #   enable_search: web only
        #   default: no tools
        if enable_tools:
            cmd.extend(["--allowedTools", "Bash,WebSearch,WebFetch"])
        elif enable_search:
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
