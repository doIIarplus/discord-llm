"""Utility tools for general-purpose functionality"""

import logging
import re
import math
from datetime import datetime
from typing import Any, Dict

from .base import Tool, ToolParameter, ParameterType, ToolResult, registry

logger = logging.getLogger("UtilityTools")


@registry.register
class CalculatorTool(Tool):
    """Tool for mathematical calculations"""

    name = "calculator"
    description = "Perform mathematical calculations. Supports basic arithmetic, powers, roots, and common math functions."
    category = "utility"
    requires_discord_context = False
    parameters = [
        ToolParameter(
            name="expression",
            description="The mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(3.14)')",
            param_type=ParameterType.STRING,
            required=True
        ),
    ]

    # Safe math functions to allow
    SAFE_FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'exp': math.exp,
        'floor': math.floor,
        'ceil': math.ceil,
        'factorial': math.factorial,
        'pi': math.pi,
        'e': math.e,
    }

    async def execute(self, **kwargs) -> ToolResult:
        expression = kwargs.get("expression", "")

        if not expression:
            return ToolResult(success=False, output=None, error="Expression is required")

        try:
            # Sanitize expression - only allow safe characters
            sanitized = re.sub(r'[^0-9+\-*/().,%^ a-zA-Z_]', '', expression)

            # Replace common notation
            sanitized = sanitized.replace('^', '**')

            # Evaluate in restricted namespace
            result = eval(sanitized, {"__builtins__": {}}, self.SAFE_FUNCTIONS)

            return ToolResult(
                success=True,
                output={
                    "expression": expression,
                    "result": result
                }
            )

        except ZeroDivisionError:
            return ToolResult(success=False, output=None, error="Division by zero")
        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return ToolResult(success=False, output=None, error=f"Calculation failed: {str(e)}")


@registry.register
class CurrentTimeTool(Tool):
    """Tool to get the current date and time"""

    name = "get_current_time"
    description = "Get the current date and time. Useful for time-sensitive questions."
    category = "utility"
    requires_discord_context = False
    parameters = [
        ToolParameter(
            name="format",
            description="Output format: 'full', 'date', 'time', or 'unix'",
            param_type=ParameterType.STRING,
            required=False,
            default="full",
            enum=["full", "date", "time", "unix"]
        ),
    ]

    async def execute(self, **kwargs) -> ToolResult:
        fmt = kwargs.get("format", "full")
        now = datetime.now()

        try:
            if fmt == "full":
                result = now.strftime("%Y-%m-%d %H:%M:%S")
            elif fmt == "date":
                result = now.strftime("%Y-%m-%d")
            elif fmt == "time":
                result = now.strftime("%H:%M:%S")
            elif fmt == "unix":
                result = int(now.timestamp())
            else:
                result = now.isoformat()

            return ToolResult(
                success=True,
                output={
                    "current_time": result,
                    "format": fmt,
                    "timezone": "local"
                }
            )

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


@registry.register
class TextAnalysisTool(Tool):
    """Tool to analyze text (word count, character count, etc.)"""

    name = "analyze_text"
    description = "Analyze text to get statistics like word count, character count, sentence count, etc."
    category = "utility"
    requires_discord_context = False
    parameters = [
        ToolParameter(
            name="text",
            description="The text to analyze",
            param_type=ParameterType.STRING,
            required=True
        ),
    ]

    async def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text", "")

        if not text:
            return ToolResult(success=False, output=None, error="Text is required")

        try:
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            analysis = {
                "character_count": len(text),
                "character_count_no_spaces": len(text.replace(" ", "")),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "paragraph_count": len(text.split("\n\n")),
                "average_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 2),
                "unique_words": len(set(w.lower() for w in words)),
            }

            return ToolResult(success=True, output=analysis)

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


@registry.register
class RandomNumberTool(Tool):
    """Tool to generate random numbers"""

    name = "random_number"
    description = "Generate a random number within a specified range."
    category = "utility"
    requires_discord_context = False
    parameters = [
        ToolParameter(
            name="min_value",
            description="Minimum value (inclusive)",
            param_type=ParameterType.INTEGER,
            required=False,
            default=1
        ),
        ToolParameter(
            name="max_value",
            description="Maximum value (inclusive)",
            param_type=ParameterType.INTEGER,
            required=False,
            default=100
        ),
        ToolParameter(
            name="count",
            description="Number of random numbers to generate",
            param_type=ParameterType.INTEGER,
            required=False,
            default=1
        ),
    ]

    async def execute(self, **kwargs) -> ToolResult:
        import random

        min_val = kwargs.get("min_value", 1)
        max_val = kwargs.get("max_value", 100)
        count = min(kwargs.get("count", 1), 100)  # Cap at 100

        if min_val > max_val:
            return ToolResult(success=False, output=None, error="min_value must be <= max_value")

        try:
            numbers = [random.randint(min_val, max_val) for _ in range(count)]

            return ToolResult(
                success=True,
                output={
                    "numbers": numbers if count > 1 else numbers[0],
                    "range": f"{min_val}-{max_val}",
                    "count": count
                }
            )

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


@registry.register
class UnitConverterTool(Tool):
    """Tool for unit conversions"""

    name = "convert_units"
    description = "Convert between common units of measurement (length, weight, temperature, etc.)"
    category = "utility"
    requires_discord_context = False
    parameters = [
        ToolParameter(
            name="value",
            description="The numeric value to convert",
            param_type=ParameterType.FLOAT,
            required=True
        ),
        ToolParameter(
            name="from_unit",
            description="The source unit (e.g., 'km', 'miles', 'celsius', 'kg')",
            param_type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="to_unit",
            description="The target unit (e.g., 'miles', 'km', 'fahrenheit', 'lbs')",
            param_type=ParameterType.STRING,
            required=True
        ),
    ]

    # Conversion factors to base units
    CONVERSIONS = {
        # Length (base: meters)
        'm': ('length', 1),
        'meters': ('length', 1),
        'km': ('length', 1000),
        'kilometers': ('length', 1000),
        'cm': ('length', 0.01),
        'centimeters': ('length', 0.01),
        'mm': ('length', 0.001),
        'millimeters': ('length', 0.001),
        'miles': ('length', 1609.34),
        'mi': ('length', 1609.34),
        'feet': ('length', 0.3048),
        'ft': ('length', 0.3048),
        'inches': ('length', 0.0254),
        'in': ('length', 0.0254),
        'yards': ('length', 0.9144),
        'yd': ('length', 0.9144),

        # Weight (base: kilograms)
        'kg': ('weight', 1),
        'kilograms': ('weight', 1),
        'g': ('weight', 0.001),
        'grams': ('weight', 0.001),
        'mg': ('weight', 0.000001),
        'milligrams': ('weight', 0.000001),
        'lbs': ('weight', 0.453592),
        'pounds': ('weight', 0.453592),
        'oz': ('weight', 0.0283495),
        'ounces': ('weight', 0.0283495),

        # Temperature (special handling)
        'celsius': ('temperature', 'c'),
        'c': ('temperature', 'c'),
        'fahrenheit': ('temperature', 'f'),
        'f': ('temperature', 'f'),
        'kelvin': ('temperature', 'k'),
        'k': ('temperature', 'k'),
    }

    def convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        # First convert to Celsius
        if from_unit == 'f':
            celsius = (value - 32) * 5 / 9
        elif from_unit == 'k':
            celsius = value - 273.15
        else:
            celsius = value

        # Then convert to target
        if to_unit == 'f':
            return celsius * 9 / 5 + 32
        elif to_unit == 'k':
            return celsius + 273.15
        else:
            return celsius

    async def execute(self, **kwargs) -> ToolResult:
        value = kwargs.get("value")
        from_unit = kwargs.get("from_unit", "").lower()
        to_unit = kwargs.get("to_unit", "").lower()

        if value is None:
            return ToolResult(success=False, output=None, error="Value is required")

        if from_unit not in self.CONVERSIONS:
            return ToolResult(success=False, output=None, error=f"Unknown unit: {from_unit}")
        if to_unit not in self.CONVERSIONS:
            return ToolResult(success=False, output=None, error=f"Unknown unit: {to_unit}")

        from_category, from_factor = self.CONVERSIONS[from_unit]
        to_category, to_factor = self.CONVERSIONS[to_unit]

        if from_category != to_category:
            return ToolResult(
                success=False,
                output=None,
                error=f"Cannot convert {from_category} to {to_category}"
            )

        try:
            if from_category == 'temperature':
                result = self.convert_temperature(value, from_factor, to_factor)
            else:
                # Convert to base unit, then to target
                base_value = value * from_factor
                result = base_value / to_factor

            return ToolResult(
                success=True,
                output={
                    "input": f"{value} {from_unit}",
                    "output": f"{round(result, 6)} {to_unit}",
                    "result": round(result, 6)
                }
            )

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


@registry.register
class DiceRollTool(Tool):
    """Tool for rolling dice"""

    name = "roll_dice"
    description = "Roll dice using standard notation (e.g., '2d6', '1d20+5', '4d6 drop lowest')"
    category = "utility"
    requires_discord_context = False
    parameters = [
        ToolParameter(
            name="notation",
            description="Dice notation (e.g., '2d6', '1d20+5', '3d6')",
            param_type=ParameterType.STRING,
            required=True
        ),
    ]

    async def execute(self, **kwargs) -> ToolResult:
        import random

        notation = kwargs.get("notation", "").lower().strip()

        if not notation:
            return ToolResult(success=False, output=None, error="Dice notation is required")

        try:
            # Parse dice notation: XdY+Z or XdY-Z
            match = re.match(r'(\d+)?d(\d+)([+-]\d+)?', notation)
            if not match:
                return ToolResult(success=False, output=None, error=f"Invalid dice notation: {notation}")

            num_dice = int(match.group(1) or 1)
            die_sides = int(match.group(2))
            modifier = int(match.group(3) or 0)

            if num_dice > 100:
                return ToolResult(success=False, output=None, error="Maximum 100 dice allowed")
            if die_sides > 1000:
                return ToolResult(success=False, output=None, error="Maximum 1000 sides allowed")

            # Roll the dice
            rolls = [random.randint(1, die_sides) for _ in range(num_dice)]
            total = sum(rolls) + modifier

            result = {
                "notation": notation,
                "rolls": rolls,
                "modifier": modifier,
                "total": total,
            }

            return ToolResult(success=True, output=result)

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
