"""Tool executor for handling LLM tool calls"""

import json
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import discord

from .base import Tool, ToolRegistry, ToolResult, registry

logger = logging.getLogger("ToolExecutor")


@dataclass
class ToolCall:
    """Represents a parsed tool call from LLM output"""
    name: str
    arguments: Dict[str, Any]
    raw_text: str = ""


@dataclass
class DiscordContext:
    """Context from Discord for tool execution"""
    message: Optional[discord.Message] = None
    channel: Optional[discord.TextChannel] = None
    guild: Optional[discord.Guild] = None
    author: Optional[discord.Member] = None
    bot: Any = None  # Reference to the bot instance


class ToolExecutor:
    """
    Executes tools called by the LLM.

    Handles parsing tool calls from LLM output, executing tools,
    and formatting results for the LLM.
    """

    # Pattern to match tool calls in LLM output
    # Format: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
    TOOL_CALL_PATTERN = re.compile(
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
        re.DOTALL
    )

    # Alternative format: ```tool\n{...}\n```
    ALT_TOOL_PATTERN = re.compile(
        r'```tool\s*\n(\{.*?\})\s*\n```',
        re.DOTALL
    )

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        """
        Initialize the executor.

        Args:
            tool_registry: Optional registry to use. Defaults to global registry.
        """
        self.registry = tool_registry or registry
        self.max_tool_calls_per_turn = 5  # Prevent infinite loops
        logger.info("ToolExecutor initialized")

    def parse_tool_calls(self, text: str) -> List[ToolCall]:
        """
        Parse tool calls from LLM output text.

        Args:
            text: The LLM output text

        Returns:
            List of parsed ToolCall objects
        """
        tool_calls = []

        # Try primary pattern
        matches = self.TOOL_CALL_PATTERN.findall(text)
        if not matches:
            # Try alternative pattern
            matches = self.ALT_TOOL_PATTERN.findall(text)

        for match in matches:
            try:
                data = json.loads(match)
                if "name" in data:
                    tool_calls.append(ToolCall(
                        name=data["name"],
                        arguments=data.get("arguments", data.get("params", {})),
                        raw_text=match
                    ))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")
                continue

        return tool_calls

    def has_tool_calls(self, text: str) -> bool:
        """Check if text contains any tool calls"""
        return bool(self.TOOL_CALL_PATTERN.search(text) or
                    self.ALT_TOOL_PATTERN.search(text))

    def remove_tool_calls(self, text: str) -> str:
        """Remove tool call tags from text, leaving only the response"""
        text = self.TOOL_CALL_PATTERN.sub('', text)
        text = self.ALT_TOOL_PATTERN.sub('', text)
        return text.strip()

    async def execute_tool(
        self,
        tool_call: ToolCall,
        context: Optional[DiscordContext] = None
    ) -> ToolResult:
        """
        Execute a single tool call.

        Args:
            tool_call: The tool call to execute
            context: Optional Discord context for tools that need it

        Returns:
            ToolResult with success status and output
        """
        tool = self.registry.get(tool_call.name)

        if not tool:
            logger.warning(f"Tool not found: {tool_call.name}")
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {tool_call.name}"
            )

        # Validate parameters
        is_valid, error = tool.validate_params(tool_call.arguments)
        if not is_valid:
            logger.warning(f"Invalid parameters for {tool_call.name}: {error}")
            return ToolResult(success=False, output=None, error=error)

        # Add Discord context if required
        kwargs = dict(tool_call.arguments)
        if tool.requires_discord_context:
            if not context:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Tool {tool_call.name} requires Discord context"
                )
            kwargs["_discord_context"] = context

        try:
            logger.info(f"Executing tool: {tool_call.name}")
            result = await tool.execute(**kwargs)
            logger.info(f"Tool {tool_call.name} completed: success={result.success}")
            return result
        except Exception as e:
            logger.error(f"Tool execution error for {tool_call.name}: {e}", exc_info=True)
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool execution failed: {str(e)}"
            )

    async def execute_all(
        self,
        text: str,
        context: Optional[DiscordContext] = None
    ) -> Tuple[List[ToolResult], str]:
        """
        Parse and execute all tool calls in text.

        Args:
            text: LLM output text potentially containing tool calls
            context: Optional Discord context

        Returns:
            Tuple of (list of results, cleaned text without tool calls)
        """
        tool_calls = self.parse_tool_calls(text)
        results = []

        # Limit tool calls to prevent runaway execution
        for tool_call in tool_calls[:self.max_tool_calls_per_turn]:
            result = await self.execute_tool(tool_call, context)
            results.append(result)

        cleaned_text = self.remove_tool_calls(text)
        return results, cleaned_text

    def format_results_for_llm(self, results: List[ToolResult]) -> str:
        """
        Format tool results for feeding back to the LLM.

        Args:
            results: List of tool execution results

        Returns:
            Formatted string for the LLM
        """
        if not results:
            return ""

        parts = ["Tool Results:"]
        for i, result in enumerate(results, 1):
            if result.success:
                parts.append(f"[{i}] Success: {result.to_string()}")
            else:
                parts.append(f"[{i}] Failed: {result.error}")

        return "\n".join(parts)

    def get_system_prompt_addition(self) -> str:
        """
        Get the tool-related addition to the system prompt.

        Returns:
            String to append to the system prompt explaining tool usage.
        """
        tools = self.registry.get_all()
        if not tools:
            return ""

        lines = [
            "\n\n## Tools",
            "You have access to the following tools. To use a tool, include a tool call in your response using this format:",
            "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}</tool_call>",
            "",
            "You can call multiple tools in one response if needed. After tool results are returned, continue your response.",
            "",
            "Available tools:",
        ]

        for tool in tools:
            params_desc = []
            for p in tool.parameters:
                req = "required" if p.required else "optional"
                params_desc.append(f"    - {p.name} ({p.param_type.value}, {req}): {p.description}")

            lines.append(f"\n### {tool.name}")
            lines.append(f"{tool.description}")
            if params_desc:
                lines.append("Parameters:")
                lines.extend(params_desc)

        return "\n".join(lines)
