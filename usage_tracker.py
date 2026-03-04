"""Local usage tracker for Claude API spend."""

import json
import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("UsageTracker")

# Pricing per million tokens (USD)
MODEL_PRICING = {
    # model_id: (input_per_mtok, output_per_mtok)
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5-20251101": (5.0, 25.0),
}

# Web search cost: $10 per 1,000 searches
WEB_SEARCH_COST = 0.01  # per search

USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_usage.json")


class UsageTracker:
    """Tracks Claude API usage and estimated spend per calendar month."""

    def __init__(self, monthly_budget: float = 100.0):
        self.monthly_budget = monthly_budget
        self._data = self._load()

    def _current_month_key(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _load(self) -> dict:
        if os.path.exists(USAGE_FILE):
            try:
                with open(USAGE_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt usage file, resetting")
        return {}

    def _save(self):
        try:
            with open(USAGE_FILE, "w") as f:
                json.dump(self._data, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save usage data: {e}")

    def _get_month(self) -> dict:
        key = self._current_month_key()
        if key not in self._data:
            self._data[key] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "web_searches": 0,
                "estimated_cost": 0.0,
                "requests": 0,
            }
        return self._data[key]

    def record_usage(self, model: str, input_tokens: int, output_tokens: int, web_searches: int = 0):
        """Record token usage from an API response and update estimated cost."""
        month = self._get_month()
        month["input_tokens"] += input_tokens
        month["output_tokens"] += output_tokens
        month["web_searches"] += web_searches
        month["requests"] += 1

        # Calculate cost for this request
        input_price, output_price = MODEL_PRICING.get(model, (3.0, 15.0))
        cost = (input_tokens / 1_000_000 * input_price) + (output_tokens / 1_000_000 * output_price)
        cost += web_searches * WEB_SEARCH_COST
        month["estimated_cost"] = round(month["estimated_cost"] + cost, 6)

        self._save()
        logger.info(f"Usage recorded: +{input_tokens}in/{output_tokens}out, ${cost:.4f} this request, ${month['estimated_cost']:.2f} this month")

    @property
    def current_spend(self) -> float:
        """Get estimated spend for the current month."""
        return self._get_month()["estimated_cost"]

    @property
    def budget_remaining(self) -> float:
        """Get remaining budget for the current month."""
        return max(0.0, self.monthly_budget - self.current_spend)

    @property
    def budget_exceeded(self) -> bool:
        """Check if the monthly budget has been exceeded."""
        return self.current_spend >= self.monthly_budget

    def get_stats(self) -> dict:
        """Get usage stats for the current month."""
        month = self._get_month()
        return {
            "month": self._current_month_key(),
            "input_tokens": month["input_tokens"],
            "output_tokens": month["output_tokens"],
            "web_searches": month["web_searches"],
            "requests": month["requests"],
            "estimated_cost": round(month["estimated_cost"], 4),
            "budget": self.monthly_budget,
            "remaining": round(self.budget_remaining, 4),
        }
