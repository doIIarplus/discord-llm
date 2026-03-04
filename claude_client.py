"""Anthropic Claude API client for Discord LLM Bot"""

import re
import time
import traceback
from typing import List, Optional, Tuple

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MONTHLY_BUDGET
from usage_tracker import UsageTracker


class ClaudeClient:
    """Client for interacting with the Anthropic Claude API"""

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self.usage_tracker = UsageTracker(monthly_budget=CLAUDE_MONTHLY_BUDGET)

    def _parse_prompt(self, prompt: str) -> Tuple[str, List[dict]]:
        """Parse a flat prompt string into (system, messages) for Claude's API.

        Expected format:
            System: <system prompt>
            User (<name>): <content>
            Assistant: <content>
            ...
            Assistant:
        """
        system = ""
        messages = []

        # Extract system prompt
        system_match = re.match(r"System:\s*(.*?)(?=\nUser |\nAssistant:)", prompt, re.DOTALL)
        if system_match:
            system = system_match.group(1).strip()

        # Extract conversation turns
        turn_pattern = re.compile(
            r"(User\s*(?:\([^)]*\))?|Assistant)\s*:\s*(.*?)(?=\nUser\s*(?:\([^)]*\))?\s*:|\nAssistant\s*:|$)",
            re.DOTALL,
        )

        for match in turn_pattern.finditer(prompt):
            role_raw = match.group(1).strip()
            content = match.group(2).strip()
            if not content:
                continue

            role = "user" if role_raw.startswith("User") else "assistant"

            # Merge consecutive same-role messages (Claude API requires alternating)
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n" + content
            else:
                messages.append({"role": role, "content": content})

        # Ensure messages start with user and alternate properly
        if not messages:
            messages = [{"role": "user", "content": prompt}]

        return system, messages

    def _attach_images(self, messages: List[dict], images: List[str]):
        """Attach base64 images to the last user message."""
        if not images or not messages:
            return

        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return

        text_content = messages[last_user_idx]["content"]
        content_blocks = []
        for img_b64 in images:
            # Detect media type from base64 header
            media_type = "image/jpeg"
            if img_b64.startswith("iVBOR"):
                media_type = "image/png"
            elif img_b64.startswith("R0lGOD"):
                media_type = "image/gif"
            elif img_b64.startswith("UklGR"):
                media_type = "image/webp"

            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_b64,
                },
            })
        content_blocks.append({"type": "text", "text": text_content})
        messages[last_user_idx]["content"] = content_blocks

    def _record_usage(self, model: str, response):
        """Extract and record token usage from an API response."""
        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        web_searches = 0
        server_tool_use = getattr(usage, "server_tool_use", None)
        if server_tool_use:
            web_searches = getattr(server_tool_use, "web_search_requests", 0)
        self.usage_tracker.record_usage(model, input_tokens, output_tokens, web_searches)

    async def generate(
        self,
        prompt: str,
        model: str,
        images: Optional[List[str]] = None,
    ) -> str:
        """Generate a response from Claude.

        Args:
            prompt: Formatted prompt string (System: ... User: ... Assistant:)
            model: Claude model ID
            images: Optional list of base64-encoded images
        """
        try:
            system, messages = self._parse_prompt(prompt)
            if images:
                self._attach_images(messages, images)

            start_time = time.perf_counter()

            kwargs = {
                "model": model,
                "max_tokens": 4096,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system

            response = await self.client.messages.create(**kwargs)

            duration = time.perf_counter() - start_time
            print(f"Claude generation took {duration:.2f}s")

            self._record_usage(model, response)

            # Extract text from response content blocks
            text_parts = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)

            return "\n".join(text_parts) if text_parts else "No response from Claude."

        except Exception as e:
            print(f"Error in Claude generate: {e}")
            print(traceback.format_exc())
            raise

    async def generate_with_search(
        self,
        prompt: str,
        model: str,
        images: Optional[List[str]] = None,
    ) -> Tuple[str, List[dict]]:
        """Generate a response from Claude with web search enabled.

        Returns:
            (response_text, sources) where sources is a list of
            {"url": ..., "title": ...} dicts extracted from citations.
        """
        try:
            system, messages = self._parse_prompt(prompt)
            if images:
                self._attach_images(messages, images)

            start_time = time.perf_counter()

            kwargs = {
                "model": model,
                "max_tokens": 4096,
                "messages": messages,
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 5,
                    }
                ],
            }
            if system:
                kwargs["system"] = system

            response = await self.client.messages.create(**kwargs)

            # Handle pause_turn: keep sending back until done
            while response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": "Please continue."})
                kwargs["messages"] = messages
                response = await self.client.messages.create(**kwargs)

            duration = time.perf_counter() - start_time
            print(f"Claude generation (with search) took {duration:.2f}s")

            self._record_usage(model, response)

            # Extract text and sources from response
            text_parts = []
            sources = []
            seen_urls = set()

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    # Extract citations if present
                    if hasattr(block, "citations") and block.citations:
                        for citation in block.citations:
                            if hasattr(citation, "url") and citation.url not in seen_urls:
                                seen_urls.add(citation.url)
                                sources.append({
                                    "url": citation.url,
                                    "title": getattr(citation, "title", citation.url),
                                })

            response_text = "\n".join(text_parts) if text_parts else "No response from Claude."
            return response_text, sources

        except Exception as e:
            print(f"Error in Claude generate_with_search: {e}")
            print(traceback.format_exc())
            raise
