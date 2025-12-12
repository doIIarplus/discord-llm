"""Base classes for the tool system"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type
import logging

logger = logging.getLogger("Tools")


class ParameterType(Enum):
    """Supported parameter types for tools"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    description: str
    param_type: ParameterType
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None  # For parameters with fixed choices

    def to_schema(self) -> Dict[str, Any]:
        """Convert parameter to JSON schema format"""
        schema = {
            "type": self.param_type.value,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        return schema


@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    output: Any
    error: Optional[str] = None

    def to_string(self) -> str:
        """Convert result to string for LLM consumption"""
        if not self.success:
            return f"Error: {self.error}"
        if isinstance(self.output, str):
            return self.output
        return str(self.output)


class Tool(ABC):
    """
    Abstract base class for all tools.

    To create a new tool:
    1. Subclass Tool
    2. Define name, description, and parameters
    3. Implement the execute() method

    Example:
        class MyTool(Tool):
            name = "my_tool"
            description = "Does something useful"
            parameters = [
                ToolParameter(
                    name="input",
                    description="The input to process",
                    param_type=ParameterType.STRING
                )
            ]

            async def execute(self, input: str, **kwargs) -> ToolResult:
                result = do_something(input)
                return ToolResult(success=True, output=result)
    """

    # Tool metadata - override these in subclasses
    name: str = ""
    description: str = ""
    parameters: List[ToolParameter] = field(default_factory=list)

    # Optional: category for organizing tools
    category: str = "general"

    # Whether this tool requires Discord context (message, channel, guild)
    requires_discord_context: bool = False

    def __init__(self):
        if not self.name:
            raise ValueError(f"Tool {self.__class__.__name__} must define a name")
        if not self.description:
            raise ValueError(f"Tool {self.__class__.__name__} must define a description")

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with success status and output
        """
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling"""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate that required parameters are provided"""
        for param in self.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.name}"

            if param.name in params:
                value = params[param.name]
                # Type validation
                if param.param_type == ParameterType.INTEGER and not isinstance(value, int):
                    try:
                        params[param.name] = int(value)
                    except (ValueError, TypeError):
                        return False, f"Parameter {param.name} must be an integer"

                elif param.param_type == ParameterType.FLOAT and not isinstance(value, (int, float)):
                    try:
                        params[param.name] = float(value)
                    except (ValueError, TypeError):
                        return False, f"Parameter {param.name} must be a number"

                elif param.param_type == ParameterType.BOOLEAN and not isinstance(value, bool):
                    if isinstance(value, str):
                        params[param.name] = value.lower() in ("true", "1", "yes")
                    else:
                        return False, f"Parameter {param.name} must be a boolean"

                # Enum validation
                if param.enum and value not in param.enum:
                    return False, f"Parameter {param.name} must be one of: {param.enum}"

        return True, None


class ToolRegistry:
    """Registry for managing available tools"""

    _instance: Optional["ToolRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, Tool] = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._tools: Dict[str, Tool] = {}
            self._initialized = True
            logger.info("ToolRegistry initialized")

    def register(self, tool_class: Type[Tool]) -> Type[Tool]:
        """
        Register a tool class. Can be used as a decorator.

        Example:
            @registry.register
            class MyTool(Tool):
                ...
        """
        tool_instance = tool_class()
        self._tools[tool_instance.name] = tool_instance
        logger.info(f"Registered tool: {tool_instance.name}")
        return tool_class

    def register_instance(self, tool: Tool) -> None:
        """Register an existing tool instance"""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool instance: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def get_all(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self._tools.values())

    def get_by_category(self, category: str) -> List[Tool]:
        """Get all tools in a category"""
        return [t for t in self._tools.values() if t.category == category]

    def get_schemas(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get schemas for tools (for LLM function calling).

        Args:
            tool_names: Optional list of tool names to include. If None, includes all.
        """
        tools = self._tools.values()
        if tool_names:
            tools = [t for t in tools if t.name in tool_names]
        return [t.get_schema() for t in tools]

    def get_tool_descriptions(self) -> str:
        """Get a formatted string of all tool descriptions for the LLM"""
        lines = ["Available tools:"]
        for tool in self._tools.values():
            params_str = ", ".join(
                f"{p.name}: {p.param_type.value}" + (" (optional)" if not p.required else "")
                for p in tool.parameters
            )
            lines.append(f"- {tool.name}({params_str}): {tool.description}")
        return "\n".join(lines)

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name"""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def clear(self) -> None:
        """Clear all registered tools"""
        self._tools.clear()
        logger.info("Cleared all tools from registry")


# Global registry instance
registry = ToolRegistry()
