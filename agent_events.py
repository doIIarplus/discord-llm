"""Turn Claude Code stream-json events into something renderable in Discord.

The CLI emits newline-delimited JSON while it works. Verified sequence:
    system/init  ->  rate_limit_event  ->  assistant (content blocks)  ->  result
`assistant` events carry the interesting part: `tool_use` blocks naming what the
agent is doing, and `text` blocks with what it's saying. `user` events carry
`tool_result`.

This module is pure (no I/O, no Discord), so the renderer can be unit-tested
against recorded streams.
"""

from typing import Optional

# Tool name -> how to describe it in one line. Anything unknown falls back to
# the raw tool name, which is fine — better a slightly odd label than a crash.
_VERBS = {
    "Read": "reading", "Write": "writing", "Edit": "editing",
    "Glob": "searching", "Grep": "searching", "Bash": "running",
    "Task": "delegating to a subagent", "WebSearch": "searching the web",
    "WebFetch": "fetching", "NotebookEdit": "editing notebook",
}
# Which input field names the thing being acted on.
_TARGET_KEYS = ("file_path", "path", "pattern", "command", "description",
                "prompt", "url", "query")


def _target(tool: str, inp: dict) -> str:
    for k in _TARGET_KEYS:
        v = inp.get(k)
        if isinstance(v, str) and v.strip():
            v = v.strip().splitlines()[0]
            return v[:90]
    return ""


def describe_tool_use(block: dict) -> dict:
    """One tool_use block -> {tool, verb, target, label, file}."""
    tool = block.get("name") or "tool"
    inp = block.get("input") or {}
    verb = _VERBS.get(tool, tool)
    target = _target(tool, inp)
    file_path = inp.get("file_path") or inp.get("path")
    label = f"{verb} {target}".strip() if target else verb
    return {"tool": tool, "verb": verb, "target": target,
            "label": label, "file": file_path}


def parse_event(event: dict) -> Optional[dict]:
    """Normalize one stream event, or None if it isn't worth showing.

    Returned kinds: init, tool, text, result, error.
    """
    if not isinstance(event, dict):
        return None
    etype = event.get("type")

    if etype == "system" and event.get("subtype") == "init":
        return {"kind": "init",
                "session_id": event.get("session_id"),
                "tools": event.get("tools") or [],
                "model": event.get("model")}

    if etype == "assistant":
        msg = event.get("message") or {}
        out = []
        for block in msg.get("content") or []:
            btype = block.get("type")
            if btype == "tool_use":
                d = describe_tool_use(block)
                d["kind"] = "tool"
                out.append(d)
            elif btype == "text":
                text = (block.get("text") or "").strip()
                if text:
                    out.append({"kind": "text", "text": text})
        if not out:
            return None
        # Prefer reporting the tool call; text is usually narration around it.
        return next((o for o in out if o["kind"] == "tool"), out[0])

    if etype == "result":
        return {"kind": "result",
                "result": event.get("result"),
                "is_error": bool(event.get("is_error")),
                "num_turns": event.get("num_turns"),
                "cost_usd": event.get("total_cost_usd"),
                "session_id": event.get("session_id")}

    if etype == "system" and event.get("subtype") == "api_retry":
        return {"kind": "error",
                "text": f"api retry {event.get('attempt')}/{event.get('max_retries')}"}

    return None


class ProgressState:
    """Accumulates parsed events into a compact live summary."""

    MAX_RECENT = 5

    def __init__(self, task: str = "", repo: str = ""):
        self.task = task
        self.repo = repo
        self.session_id = None
        self.recent = []          # last few human-readable actions
        self.files = []           # files touched, in order, deduped
        self.tool_counts = {}
        self.subagents = 0
        self.finished = False
        self.failed = False
        self.result = None
        self.cost_usd = None
        self.num_turns = None

    def apply(self, event: dict) -> bool:
        """Fold one raw stream event in. Returns True if the view changed."""
        parsed = parse_event(event)
        if parsed is None:
            return False
        kind = parsed["kind"]

        if kind == "init":
            self.session_id = parsed.get("session_id") or self.session_id
            return False          # nothing user-visible yet

        if kind == "tool":
            self.tool_counts[parsed["tool"]] = self.tool_counts.get(parsed["tool"], 0) + 1
            if parsed["tool"] == "Task":
                self.subagents += 1
            f = parsed.get("file")
            if f and f not in self.files:
                self.files.append(f)
            self.recent.append(parsed["label"])
            del self.recent[:-self.MAX_RECENT]
            return True

        if kind == "text":
            self.recent.append(parsed["text"].splitlines()[0][:90])
            del self.recent[:-self.MAX_RECENT]
            return True

        if kind == "result":
            self.finished = True
            self.failed = parsed["is_error"]
            self.result = parsed["result"]
            self.cost_usd = parsed["cost_usd"]
            self.num_turns = parsed["num_turns"]
            self.session_id = parsed.get("session_id") or self.session_id
            return True

        if kind == "error":
            self.recent.append(parsed["text"])
            del self.recent[:-self.MAX_RECENT]
            return True
        return False

    def render(self, elapsed_s: float, status: str = "running") -> str:
        """A compact Discord message body. Kept well under 2000 chars."""
        mins, secs = divmod(int(elapsed_s), 60)
        clock = f"{mins}m{secs:02d}s" if mins else f"{secs}s"

        head = {
            "running": "🔨 working",
            "awaiting_push": "✅ done",
            "failed": "❌ failed",
            "cancelled": "🛑 cancelled",
        }.get(status, status)

        lines = [f"{head} · `{self.repo}` · {clock}"]
        if self.task:
            lines.append(f"> {self.task[:160]}")

        if status == "running":
            for r in self.recent[-3:]:
                lines.append(f"  ▸ {r}")
        if self.files:
            shown = ", ".join(f"`{f.rsplit('/', 1)[-1]}`" for f in self.files[:6])
            more = f" +{len(self.files) - 6}" if len(self.files) > 6 else ""
            lines.append(f"  files: {shown}{more}")
        if self.subagents:
            lines.append(f"  subagents: {self.subagents}")

        if self.finished and self.result:
            lines.append("")
            lines.append(self.result[:900])
        if self.cost_usd:
            lines.append(f"-# {self.num_turns} turns · ${self.cost_usd:.2f}")

        body = "\n".join(lines)
        return body[:1900]
