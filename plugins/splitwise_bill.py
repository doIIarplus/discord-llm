"""Splitwise bill plugin — create expenses, parse receipts, check balances.

Supports equal splits, ratio-based splits, itemized receipt assignment,
smart friend resolution with disambiguation, and natural language parsing.
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
from typing import Optional

import aiohttp
import discord
from discord import app_commands

from plugin_base import BasePlugin

logger = logging.getLogger("Plugin.splitwise_bill")

# Dedicated file logger for Splitwise API request/response bodies
_api_logger = logging.getLogger("splitwise_api")
_api_logger.setLevel(logging.DEBUG)
_api_logger.propagate = False
_api_fh = logging.FileHandler(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splitwise.log")
)
_api_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
_api_logger.addHandler(_api_fh)

SPLITWISE_BASE_URL = "https://secure.splitwise.com/api/v3.0"
FRIENDS_CACHE_TTL = 300  # 5 minutes


# --- Discord UI Views ---

class ConfirmView(discord.ui.View):
    """Confirm/Cancel buttons. Only the original user can interact."""

    def __init__(self, owner_id: int, timeout: float = 120):
        super().__init__(timeout=timeout)
        self.owner_id = owner_id
        self.result: Optional[bool] = None
        self._event = asyncio.Event()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "Only the person who started this bill can confirm.", ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.green, emoji="✅")
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.result = True
        self._event.set()
        self.stop()
        await interaction.response.defer()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.red, emoji="❌")
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.result = False
        self._event.set()
        self.stop()
        await interaction.response.defer()

    async def wait_for_result(self) -> Optional[bool]:
        try:
            await asyncio.wait_for(self._event.wait(), timeout=self.timeout)
        except asyncio.TimeoutError:
            pass
        return self.result

    async def on_timeout(self):
        self._event.set()
        for child in self.children:
            child.disabled = True


class FriendDisambiguationSelect(discord.ui.Select):
    """Dropdown to pick from ambiguous friend matches."""

    def __init__(self, matches: list, original_name: str):
        options = []
        for f in matches[:25]:
            first = f.get("first_name", "")
            last = f.get("last_name", "")
            email = f.get("email", "")
            label = f"{first} {last}".strip() or email
            desc = email if email else f"ID: {f['id']}"
            options.append(discord.SelectOption(
                label=label[:100], description=desc[:100], value=str(f["id"])
            ))
        super().__init__(
            placeholder=f"Select the right '{original_name}'...",
            options=options,
        )
        self.selected_id: Optional[int] = None

    async def callback(self, interaction: discord.Interaction):
        self.selected_id = int(self.values[0])
        self.view.stop()
        await interaction.response.defer()


class FriendDisambiguationView(discord.ui.View):
    """View wrapping a FriendDisambiguationSelect."""

    def __init__(self, matches: list, original_name: str, owner_id: int, timeout: float = 120):
        super().__init__(timeout=timeout)
        self.owner_id = owner_id
        self.select = FriendDisambiguationSelect(matches, original_name)
        self.add_item(self.select)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "Only the person who started this can select.", ephemeral=True
            )
            return False
        return True

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True


class ItemAssignmentView(discord.ui.View):
    """View for assigning a single receipt item to a person or splitting it."""

    def __init__(self, item: dict, people: list, owner_id: int, timeout: float = 300):
        """
        item: {"name": str, "price": float, "quantity": int}
        people: [{"id": int/str, "name": str}, ...]
        """
        super().__init__(timeout=timeout)
        self.owner_id = owner_id
        self.item = item
        self.assigned_to: Optional[str] = None  # person id or "split"
        self.done_early = False
        self._event = asyncio.Event()

        options = []
        for p in people[:25]:
            options.append(discord.SelectOption(
                label=p["name"][:100], value=str(p["id"])
            ))
        self.select = discord.ui.Select(
            placeholder="Assign this item to...", options=options
        )
        self.select.callback = self._select_callback
        self.add_item(self.select)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "Only the person who started this can assign items.", ephemeral=True
            )
            return False
        return True

    async def _select_callback(self, interaction: discord.Interaction):
        self.assigned_to = self.select.values[0]
        self._event.set()
        self.stop()
        await interaction.response.defer()

    @discord.ui.button(label="Split this item", style=discord.ButtonStyle.blurple)
    async def split_item(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.assigned_to = "split"
        self._event.set()
        self.stop()
        await interaction.response.defer()

    @discord.ui.button(label="Done (split remaining equally)", style=discord.ButtonStyle.grey)
    async def done_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.done_early = True
        self._event.set()
        self.stop()
        await interaction.response.defer()

    async def wait_for_result(self):
        try:
            await asyncio.wait_for(self._event.wait(), timeout=self.timeout)
        except asyncio.TimeoutError:
            pass

    async def on_timeout(self):
        self._event.set()
        for child in self.children:
            child.disabled = True


class SplitwiseBillPlugin(BasePlugin):
    name = "splitwise_bill"
    version = "2.0.0"
    description = "Splitwise integration: bills, receipts, ratios, itemized splits, balances"

    async def on_load(self):
        self._api_key = os.getenv("SPLITWISE_API_KEY", "")

        # Pending receipt states: channel_id -> {...}
        self._pending_receipts = {}

        # Friends cache: {"friends": [...], "fetched_at": float}
        self._friends_cache = None

        self.register_slash_command(
            name="bill",
            description="Add a bill to Splitwise (equal or ratio split)",
            callback=self._bill_command,
        )
        self.register_slash_command(
            name="bill_receipt",
            description="Add a bill by attaching a receipt image",
            callback=self._bill_receipt_command,
        )
        self.register_slash_command(
            name="splitwise_groups",
            description="List your Splitwise groups",
            callback=self._groups_command,
        )
        self.register_slash_command(
            name="splitwise_balance",
            description="Show your current Splitwise balances",
            callback=self._balance_command,
        )

        # Broad trigger — Claude classifies actual intent inside the handler.
        # Matches any message mentioning money, splitting, bills, or splitwise.
        self.register_message_handler(
            pattern=r'(?i)(splitwise|split|bill|expense|\$\d|owe)',
            callback=self._handle_splitwise_message,
            priority=50,
        )

    async def self_test(self) -> bool:
        if not self._api_key:
            self.logger.warning("SPLITWISE_API_KEY not configured — commands will prompt to set it")
        return True

    # --- HTTP helpers ---

    def _headers(self):
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _api_get(self, endpoint: str) -> dict:
        url = f"{SPLITWISE_BASE_URL}{endpoint}"
        _api_logger.debug("GET %s", url)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._headers()) as resp:
                body = await resp.json()
                _api_logger.debug("GET %s -> %s %s", url, resp.status, json.dumps(body, default=str))
                return body

    async def _api_post(self, endpoint: str, data: dict) -> dict:
        url = f"{SPLITWISE_BASE_URL}{endpoint}"
        _api_logger.debug("POST %s body=%s", url, json.dumps(data, default=str))
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=self._headers(), json=data,
            ) as resp:
                body = await resp.json()
                _api_logger.debug("POST %s -> %s %s", url, resp.status, json.dumps(body, default=str))
                return body

    async def _api_post_form(self, endpoint: str, data: dict) -> dict:
        """POST with form-encoded data (used by Splitwise create_expense)."""
        url = f"{SPLITWISE_BASE_URL}{endpoint}"
        _api_logger.debug("POST (form) %s body=%s", url, json.dumps(data, default=str))
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers={"Authorization": f"Bearer {self._api_key}"}, data=data,
            ) as resp:
                body = await resp.json()
                _api_logger.debug("POST (form) %s -> %s %s", url, resp.status, json.dumps(body, default=str))
                return body

    async def _get_current_user(self) -> dict:
        result = await self._api_get("/get_current_user")
        return result.get("user", {})

    async def _get_friends(self) -> list:
        """Fetch friends list with caching (TTL-based)."""
        now = time.time()
        if (
            self._friends_cache
            and now - self._friends_cache["fetched_at"] < FRIENDS_CACHE_TTL
        ):
            return self._friends_cache["friends"]

        result = await self._api_get("/get_friends")
        friends = result.get("friends", [])
        self._friends_cache = {"friends": friends, "fetched_at": now}
        return friends

    def _invalidate_friends_cache(self):
        self._friends_cache = None

    # --- Smart friend resolution ---

    async def _resolve_friends(self, names: list) -> dict:
        """Resolve a list of name strings to Splitwise friends.

        Returns: {
            "resolved": [{"name": str, "friend": dict}, ...],
            "ambiguous": [{"name": str, "matches": [dict, ...]}, ...],
            "not_found": [str, ...],
        }
        """
        friends = await self._get_friends()
        result = {"resolved": [], "ambiguous": [], "not_found": []}

        for name in names:
            name_lower = name.strip().lower()
            if not name_lower:
                continue

            # Pass 1: match on first_name
            matches = [
                f for f in friends
                if (f.get("first_name") or "").lower() == name_lower
            ]
            if len(matches) == 1:
                result["resolved"].append({"name": name, "friend": matches[0]})
                continue
            if len(matches) > 1:
                result["ambiguous"].append({"name": name, "matches": matches})
                continue

            # Pass 2: match on last_name
            matches = [
                f for f in friends
                if (f.get("last_name") or "").lower() == name_lower
            ]
            if len(matches) == 1:
                result["resolved"].append({"name": name, "friend": matches[0]})
                continue
            if len(matches) > 1:
                result["ambiguous"].append({"name": name, "matches": matches})
                continue

            # Pass 3: match on full name
            matches = [
                f for f in friends
                if f"{(f.get('first_name') or '')} {(f.get('last_name') or '')}".strip().lower() == name_lower
            ]
            if len(matches) == 1:
                result["resolved"].append({"name": name, "friend": matches[0]})
                continue
            if len(matches) > 1:
                result["ambiguous"].append({"name": name, "matches": matches})
                continue

            # Pass 4: match on email
            matches = [
                f for f in friends
                if (f.get("email") or "").lower() == name_lower
            ]
            if len(matches) == 1:
                result["resolved"].append({"name": name, "friend": matches[0]})
                continue

            result["not_found"].append(name)

        return result

    async def _resolve_friends_with_disambiguation(
        self, names: list, channel, owner_id: int
    ) -> Optional[list]:
        """Resolve friends, prompting for disambiguation via Discord UI.

        Returns list of friend dicts, or None if resolution failed/cancelled.
        """
        resolution = await self._resolve_friends(names)

        if resolution["not_found"]:
            await channel.send(
                f"Could not find Splitwise friends: {', '.join(resolution['not_found'])}\n"
                "Use their name or email as it appears on Splitwise."
            )
            return None

        resolved_friends = [r["friend"] for r in resolution["resolved"]]

        # Handle ambiguous matches
        for ambig in resolution["ambiguous"]:
            view = FriendDisambiguationView(
                ambig["matches"], ambig["name"], owner_id
            )
            msg = await channel.send(
                f"Multiple matches for **{ambig['name']}** — pick one:",
                view=view,
            )
            await view.wait()
            if view.select.selected_id is None:
                await channel.send("Timed out waiting for selection. Cancelled.")
                return None
            # Find the selected friend
            selected = next(
                (f for f in ambig["matches"] if f["id"] == view.select.selected_id),
                None,
            )
            if selected:
                resolved_friends.append(selected)
            else:
                await channel.send("Selection error. Cancelled.")
                return None

        return resolved_friends

    # --- Expense creation ---

    async def _create_equal_expense(
        self, amount: float, description: str, friend_ids: list, group_id: int = 0
    ) -> dict:
        current_user = await self._get_current_user()
        user_id = current_user["id"]

        num_people = len(friend_ids) + 1
        share = round(amount / num_people, 2)
        paid_share = round(amount, 2)
        # Adjust payer's share to absorb rounding so total matches exactly
        payer_share = round(paid_share - share * len(friend_ids), 2)

        data = {
            "cost": str(paid_share),
            "description": description,
            "currency_code": "USD",
            "split_equally": False,
            "users__0__user_id": user_id,
            "users__0__paid_share": str(paid_share),
            "users__0__owed_share": str(payer_share),
        }
        if group_id:
            data["group_id"] = group_id

        for i, fid in enumerate(friend_ids, start=1):
            data[f"users__{i}__user_id"] = fid
            data[f"users__{i}__paid_share"] = "0.00"
            data[f"users__{i}__owed_share"] = str(share)

        return await self._api_post_form("/create_expense", data)

    async def _create_ratio_expense(
        self, amount: float, description: str, ratios: list,
        user_ids: list, group_id: int = 0
    ) -> dict:
        """Create an expense split by ratios.

        ratios: list of numeric ratios, same length as user_ids.
        user_ids[0] is the current user (payer).
        """
        current_user = await self._get_current_user()
        payer_id = current_user["id"]

        total_ratio = sum(ratios)
        shares = [round(amount * r / total_ratio, 2) for r in ratios]
        # Fix rounding: adjust the largest share
        diff = round(amount - sum(shares), 2)
        if diff != 0:
            max_idx = shares.index(max(shares))
            shares[max_idx] = round(shares[max_idx] + diff, 2)

        data = {
            "cost": str(round(amount, 2)),
            "description": description,
            "currency_code": "USD",
            "split_equally": False,
        }
        if group_id:
            data["group_id"] = group_id

        for i, (uid, share) in enumerate(zip(user_ids, shares)):
            data[f"users__{i}__user_id"] = uid
            data[f"users__{i}__paid_share"] = str(round(amount, 2)) if uid == payer_id else "0.00"
            data[f"users__{i}__owed_share"] = str(share)

        return await self._api_post_form("/create_expense", data)

    async def _create_itemized_expense(
        self, items: list, description: str, total: float,
        user_shares: dict, group_id: int = 0
    ) -> dict:
        """Create an expense with specific share amounts per user.

        user_shares: {user_id: owed_amount} — must include current user.
        """
        current_user = await self._get_current_user()
        user_id = current_user["id"]

        data = {
            "cost": str(round(total, 2)),
            "description": description,
            "currency_code": "USD",
            "split_equally": False,
            "details": json.dumps(items),
        }
        if group_id:
            data["group_id"] = group_id

        data["users__0__user_id"] = user_id
        data["users__0__paid_share"] = str(round(total, 2))
        data["users__0__owed_share"] = str(round(user_shares.get(user_id, 0), 2))

        idx = 1
        for uid, owed in user_shares.items():
            if uid == user_id:
                continue
            data[f"users__{idx}__user_id"] = uid
            data[f"users__{idx}__paid_share"] = "0.00"
            data[f"users__{idx}__owed_share"] = str(round(owed, 2))
            idx += 1

        return await self._api_post_form("/create_expense", data)

    # --- Receipt parsing ---

    async def _parse_receipt_image(self, image_b64: str) -> Optional[dict]:
        """Use the vision model to extract receipt items from a base64 image."""
        from config import IMAGE_RECOGNITION_MODEL

        prompt = (
            "You are a receipt parser. Analyze this receipt image and extract all information "
            "in the following JSON format. Return ONLY valid JSON, no other text:\n"
            "{\n"
            '  "store_name": "Store Name",\n'
            '  "items": [\n'
            '    {"name": "Item 1", "price": 9.99, "quantity": 1},\n'
            '    {"name": "Item 2", "price": 4.50, "quantity": 2}\n'
            "  ],\n"
            '  "subtotal": 18.99,\n'
            '  "tax": 1.52,\n'
            '  "tip": 0.00,\n'
            '  "total": 20.51\n'
            "}\n"
            "If you cannot determine a value, use 0 or null. Extract every line item visible."
        )

        try:
            response = await self.ctx.ollama_client.generate(
                prompt, IMAGE_RECOGNITION_MODEL, images=[image_b64]
            )
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Receipt parsing failed: {e}")
        return None

    def _format_receipt(self, parsed: dict) -> str:
        """Format parsed receipt data into a readable message."""
        lines = []
        store = parsed.get("store_name", "Unknown Store")
        lines.append(f"**Receipt from {store}**\n")

        items = parsed.get("items", [])
        for i, item in enumerate(items, 1):
            name = item.get("name", "???")
            price = item.get("price", 0)
            qty = item.get("quantity", 1)
            if qty > 1:
                lines.append(f"  {i}. {name} x{qty} — ${price:.2f}")
            else:
                lines.append(f"  {i}. {name} — ${price:.2f}")

        lines.append("")
        subtotal = parsed.get("subtotal", 0) or 0
        tax = parsed.get("tax", 0) or 0
        tip = parsed.get("tip", 0) or 0
        total = parsed.get("total", 0) or 0

        lines.append(f"Subtotal: ${subtotal:.2f}")
        if tax:
            lines.append(f"Tax: ${tax:.2f}")
        if tip:
            lines.append(f"Tip: ${tip:.2f}")
        lines.append(f"**Total: ${total:.2f}**")

        return "\n".join(lines)

    # --- Itemized assignment flow ---

    async def _run_item_assignment(
        self, channel, owner_id: int, parsed: dict, people: list
    ) -> Optional[dict]:
        """Run the item-by-item assignment flow with Discord UI.

        people: [{"id": int/str, "name": str}, ...]
        Returns: {person_id: total_owed, ...} or None if cancelled/timed out.
        """
        items = parsed.get("items", [])
        if not items:
            return None

        tallies = {str(p["id"]): 0.0 for p in people}
        num_people = len(people)

        for idx, item in enumerate(items):
            name = item.get("name", "???")
            price = float(item.get("price", 0))
            qty = int(item.get("quantity", 1) or 1)
            item_total = price * qty

            view = ItemAssignmentView(item, people, owner_id)
            msg = await channel.send(
                f"**Item {idx + 1}/{len(items)}:** {name}"
                + (f" x{qty}" if qty > 1 else "")
                + f" — **${item_total:.2f}**\nAssign to someone, split it, or finish early.",
                view=view,
            )
            await view.wait_for_result()

            if view.done_early:
                # Split all remaining items equally
                remaining_items = items[idx:]
                remaining_total = sum(
                    float(it.get("price", 0)) * int(it.get("quantity", 1) or 1)
                    for it in remaining_items
                )
                per_person = remaining_total / num_people
                for pid in tallies:
                    tallies[pid] += per_person
                break
            elif view.assigned_to == "split":
                per_person = item_total / num_people
                for pid in tallies:
                    tallies[pid] += per_person
            elif view.assigned_to is not None:
                tallies[view.assigned_to] = tallies.get(view.assigned_to, 0) + item_total
            else:
                # Timed out — split remaining equally
                remaining_items = items[idx:]
                remaining_total = sum(
                    float(it.get("price", 0)) * int(it.get("quantity", 1) or 1)
                    for it in remaining_items
                )
                per_person = remaining_total / num_people
                for pid in tallies:
                    tallies[pid] += per_person
                await channel.send("Timed out — splitting remaining items equally.")
                break

        # Add tax/tip proportionally
        subtotal = sum(
            float(it.get("price", 0)) * int(it.get("quantity", 1) or 1)
            for it in items
        )
        tax = float(parsed.get("tax", 0) or 0)
        tip = float(parsed.get("tip", 0) or 0)
        extras = tax + tip

        if extras > 0 and subtotal > 0:
            for pid in tallies:
                proportion = tallies[pid] / subtotal if subtotal else 1.0 / num_people
                tallies[pid] += extras * proportion

        # Round
        for pid in tallies:
            tallies[pid] = round(tallies[pid], 2)

        return tallies

    # --- Natural language parsing ---

    async def _classify_intent(self, text: str) -> Optional[dict]:
        """Use Claude to classify message intent and extract structured data.

        Returns: {"action": str, "amount": float|null, "people": [str],
                  "ratio": [int]|null, "description": str|null}
        Actions: "split", "add_bill", "check_balance", "unknown"
        """
        prompt = (
            "You are an intent classifier for a Splitwise bill-splitting bot. "
            "Classify the following message and extract structured data.\n\n"
            "Rules:\n"
            '- "action" must be one of: "split", "add_bill", "check_balance", "unknown"\n'
            '- Use "unknown" for anything that is NOT a request to create/split a bill or check balances. '
            "This includes: bug reports, feature requests, questions about how things work, "
            "complaints about previous bills, requests to modify code, general conversation.\n"
            '- "amount": the total dollar amount as a number, or null\n'
            '- "people": list of people\'s names mentioned to split with (exclude "me"/"I"/the speaker)\n'
            '- "ratio": list of integer ratios if specified (e.g. "1:1:2" -> [1,1,2]), or null\n'
            '- "description": what the bill is for — use the name/label the user gave it, or null. '
            "If the user says 'named X' or 'called X' or 'for X', use X as the description.\n\n"
            "Return ONLY valid JSON, no other text.\n\n"
            f"Message: {text}"
        )
        try:
            response = await self.ctx.claude_client.generate(
                prompt, model="claude-haiku-4-5-20251001"
            )
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Claude intent classification failed: {e}")
        return None

    # --- Confirm/cancel with buttons ---

    async def _confirm_expense(self, channel, owner_id: int, summary: str) -> bool:
        """Show a confirmation view with buttons. Returns True if confirmed."""
        view = ConfirmView(owner_id)
        msg = await channel.send(f"{summary}\n", view=view)
        result = await view.wait_for_result()
        # Disable buttons after interaction
        for child in view.children:
            child.disabled = True
        try:
            await msg.edit(view=view)
        except Exception:
            pass
        if result is True:
            return True
        if result is False:
            await channel.send("Cancelled.")
        else:
            await channel.send("Timed out — cancelled.")
        return False

    # --- Slash commands ---

    async def _bill_command(
        self,
        interaction: discord.Interaction,
        amount: float,
        description: str,
        split_with: str,
        ratios: Optional[str] = None,
    ):
        """Add a bill split equally or by ratio."""
        if not self._api_key:
            await interaction.response.send_message(
                "Splitwise API key not configured. Set SPLITWISE_API_KEY in .env.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)

        names = [n.strip() for n in split_with.split(",") if n.strip()]

        # Resolve friends with disambiguation
        friends = await self._resolve_friends_with_disambiguation(
            names, interaction.channel, interaction.user.id
        )
        if friends is None:
            await interaction.followup.send("Could not resolve all friends. Bill cancelled.")
            return
        if not friends:
            await interaction.followup.send("No valid friends specified to split with.")
            return

        friend_ids = [f["id"] for f in friends]

        # Parse ratios if provided
        if ratios:
            try:
                ratio_list = [int(r.strip()) for r in ratios.split(":")]
            except ValueError:
                await interaction.followup.send(
                    "Invalid ratios format. Use colon-separated integers like `1:1:2`."
                )
                return

            # Ratios must match payer + friends
            expected = len(friend_ids) + 1
            if len(ratio_list) != expected:
                await interaction.followup.send(
                    f"Ratio count ({len(ratio_list)}) doesn't match number of people "
                    f"({expected} including you). Provide {expected} ratios."
                )
                return

            current_user = await self._get_current_user()
            user_ids = [current_user["id"]] + friend_ids

            total_ratio = sum(ratio_list)
            shares = [round(amount * r / total_ratio, 2) for r in ratio_list]
            share_lines = []
            for uid, share, fr in zip(user_ids, shares, ["You"] + [
                f"{f.get('first_name', '')} {f.get('last_name', '')}".strip()
                for f in friends
            ]):
                share_lines.append(f"  {fr}: ${share:.2f}")

            summary = (
                f"**{description}** — ${amount:.2f} (ratio {ratios})\n"
                + "\n".join(share_lines)
            )
            confirmed = await self._confirm_expense(
                interaction.channel, interaction.user.id, summary
            )
            if not confirmed:
                return

            result = await self._create_ratio_expense(
                amount, description, ratio_list, user_ids
            )
        else:
            # Equal split
            num_people = len(friend_ids) + 1
            share = amount / num_people
            summary = (
                f"**{description}** — ${amount:.2f} split {num_people} ways "
                f"(${share:.2f} each)"
            )
            confirmed = await self._confirm_expense(
                interaction.channel, interaction.user.id, summary
            )
            if not confirmed:
                return

            result = await self._create_equal_expense(amount, description, friend_ids)

        if "expenses" in result:
            await interaction.followup.send("Bill added to Splitwise!")
        elif "errors" in result:
            errors = result["errors"]
            err_msg = errors if isinstance(errors, str) else json.dumps(errors)
            await interaction.followup.send(f"Splitwise error: {err_msg}")
        else:
            await interaction.followup.send(f"Unexpected response: {json.dumps(result)[:500]}")

    async def _bill_receipt_command(
        self,
        interaction: discord.Interaction,
        image: discord.Attachment,
        split_with: str,
        description: Optional[str] = None,
        split_mode: Optional[str] = None,
    ):
        """Add a bill by attaching a receipt image.

        split_mode: "equal", "itemized", or "ratio". Defaults to "itemized".
        """
        if not self._api_key:
            await interaction.response.send_message(
                "Splitwise API key not configured. Set SPLITWISE_API_KEY in .env.",
                ephemeral=True,
            )
            return

        if not image.content_type or not image.content_type.startswith("image/"):
            await interaction.response.send_message(
                "Please attach an image file.", ephemeral=True
            )
            return

        mode = (split_mode or "itemized").lower()
        if mode not in ("equal", "itemized", "ratio"):
            await interaction.response.send_message(
                "split_mode must be one of: equal, itemized, ratio", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True)

        # Download and encode image
        try:
            img_bytes = await image.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        except Exception as e:
            await interaction.followup.send(f"Failed to download image: {e}")
            return

        parsed = await self._parse_receipt_image(img_b64)
        if not parsed:
            await interaction.followup.send(
                "Could not parse the receipt. Try a clearer image."
            )
            return

        # Resolve friends
        names = [n.strip() for n in split_with.split(",") if n.strip()]
        friends = await self._resolve_friends_with_disambiguation(
            names, interaction.channel, interaction.user.id
        )
        if friends is None:
            return
        if not friends:
            await interaction.followup.send("No valid friends specified to split with.")
            return

        friend_ids = [f["id"] for f in friends]
        total = float(parsed.get("total", 0) or 0)
        desc = description or parsed.get("store_name", "Receipt")

        receipt_text = self._format_receipt(parsed)
        await interaction.followup.send(receipt_text)

        if mode == "equal":
            num_people = len(friend_ids) + 1
            share = total / num_people
            summary = f"${total:.2f} split {num_people} ways = ${share:.2f} each"
            confirmed = await self._confirm_expense(
                interaction.channel, interaction.user.id, summary
            )
            if not confirmed:
                return
            result = await self._create_equal_expense(total, desc, friend_ids)

        elif mode == "ratio":
            await interaction.channel.send(
                "Enter ratios (colon-separated, e.g. `1:1:2`). "
                f"You need {len(friend_ids) + 1} values (you + {len(friend_ids)} friends)."
            )
            # Store pending for ratio input
            self._pending_receipts[interaction.channel.id] = {
                "parsed": parsed,
                "total": total,
                "description": desc,
                "friend_ids": friend_ids,
                "friends": friends,
                "user_id": interaction.user.id,
                "awaiting": "ratio_input",
            }
            return

        elif mode == "itemized":
            current_user = await self._get_current_user()
            people = [{"id": str(current_user["id"]), "name": "Me (payer)"}]
            for f in friends:
                fname = f"{f.get('first_name', '')} {f.get('last_name', '')}".strip()
                people.append({"id": str(f["id"]), "name": fname})

            tallies = await self._run_item_assignment(
                interaction.channel, interaction.user.id, parsed, people
            )
            if not tallies:
                await interaction.channel.send("Item assignment failed or timed out.")
                return

            # Build summary
            share_lines = []
            for p in people:
                amt = tallies.get(str(p["id"]), 0)
                share_lines.append(f"  {p['name']}: ${amt:.2f}")
            summary = f"**{desc}** — ${total:.2f}\n" + "\n".join(share_lines)

            confirmed = await self._confirm_expense(
                interaction.channel, interaction.user.id, summary
            )
            if not confirmed:
                return

            user_shares = {int(pid): amt for pid, amt in tallies.items()}
            result = await self._create_itemized_expense(
                parsed.get("items", []), desc, total, user_shares
            )

        if "expenses" in result:
            await interaction.channel.send("Bill added to Splitwise!")
        elif "errors" in result:
            errors = result["errors"]
            err_msg = errors if isinstance(errors, str) else json.dumps(errors)
            await interaction.channel.send(f"Splitwise error: {err_msg}")
        else:
            await interaction.channel.send(f"Unexpected response: {json.dumps(result)[:500]}")

    async def _groups_command(self, interaction: discord.Interaction):
        if not self._api_key:
            await interaction.response.send_message(
                "Splitwise API key not configured.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True)

        try:
            result = await self._api_get("/get_groups")
            groups = result.get("groups", [])

            if not groups:
                await interaction.followup.send("No Splitwise groups found.")
                return

            lines = ["**Your Splitwise Groups:**\n"]
            for g in groups:
                name = g.get("name", "???")
                gid = g.get("id", "?")
                members = len(g.get("members", []))
                lines.append(f"  {name} (ID: {gid}, {members} members)")

            await interaction.followup.send("\n".join(lines))
        except Exception as e:
            logger.error(f"Groups fetch failed: {e}")
            await interaction.followup.send(f"Error fetching groups: {e}")

    async def _balance_command(self, interaction: discord.Interaction):
        if not self._api_key:
            await interaction.response.send_message(
                "Splitwise API key not configured.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True)

        try:
            friends = await self._get_friends()
            lines = ["**Splitwise Balances:**\n"]
            has_balances = False

            for f in friends:
                balances = f.get("balance", [])
                for b in balances:
                    amt = float(b.get("amount", 0))
                    if amt == 0:
                        continue
                    has_balances = True
                    currency = b.get("currency_code", "USD")
                    first = f.get("first_name", "")
                    last = f.get("last_name", "")
                    name = f"{first} {last}".strip()
                    if amt > 0:
                        lines.append(f"  **{name}** owes you {currency} {abs(amt):.2f}")
                    else:
                        lines.append(f"  You owe **{name}** {currency} {abs(amt):.2f}")

            if not has_balances:
                lines.append("  All settled up!")

            await interaction.followup.send("\n".join(lines))
        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")
            await interaction.followup.send(f"Error fetching balances: {e}")

    # --- Message handler ---

    async def _handle_splitwise_message(self, message: discord.Message):
        """Handle messages mentioning splitwise/bills when bot is mentioned."""
        bot_user = self.ctx.discord_client.user
        is_mentioned = bot_user in message.mentions
        is_reply = False
        if message.reference and message.reference.resolved:
            is_reply = message.reference.resolved.author.id == bot_user.id

        if not (is_mentioned or is_reply):
            return False

        if not self._api_key:
            await message.channel.send(
                "Splitwise API key not configured. Set SPLITWISE_API_KEY in .env."
            )
            return True

        text_lower = message.content.lower()
        text_clean = re.sub(r'<@!?\d+>', '', text_lower).strip()

        # --- Handle pending states ---
        if message.channel.id in self._pending_receipts:
            pending = self._pending_receipts[message.channel.id]
            if message.author.id != pending["user_id"]:
                return False

            if text_clean in ("cancel", "n", "nah", "no", "nevermind"):
                del self._pending_receipts[message.channel.id]
                await message.channel.send("Cancelled.")
                return True

            # Awaiting friend names
            if pending.get("needs_friends"):
                names_text = re.sub(r'<@!?\d+>', '', message.content).strip()
                names = [n.strip() for n in names_text.split(",") if n.strip()]

                friends = await self._resolve_friends_with_disambiguation(
                    names, message.channel, message.author.id
                )
                if friends is None:
                    return True
                if not friends:
                    await message.channel.send("need at least one person to split with")
                    return True

                friend_ids = [f["id"] for f in friends]
                pending["friend_ids"] = friend_ids
                pending["friends"] = friends
                pending["needs_friends"] = False

                # Default to itemized for receipts
                current_user = await self._get_current_user()
                people = [{"id": str(current_user["id"]), "name": "Me (payer)"}]
                for f in friends:
                    fname = f"{f.get('first_name', '')} {f.get('last_name', '')}".strip()
                    people.append({"id": str(f["id"]), "name": fname})

                tallies = await self._run_item_assignment(
                    message.channel, message.author.id, pending["parsed"], people
                )
                if not tallies:
                    del self._pending_receipts[message.channel.id]
                    await message.channel.send("Item assignment failed or timed out.")
                    return True

                total = pending["total"]
                desc = pending["description"]
                share_lines = []
                for p in people:
                    amt = tallies.get(str(p["id"]), 0)
                    share_lines.append(f"  {p['name']}: ${amt:.2f}")
                summary = f"**{desc}** — ${total:.2f}\n" + "\n".join(share_lines)

                confirmed = await self._confirm_expense(
                    message.channel, message.author.id, summary
                )
                del self._pending_receipts[message.channel.id]
                if not confirmed:
                    return True

                user_shares = {int(pid): amt for pid, amt in tallies.items()}
                try:
                    result = await self._create_itemized_expense(
                        pending["parsed"].get("items", []), desc, total, user_shares
                    )
                    if "expenses" in result:
                        await message.channel.send("Bill added to Splitwise!")
                    elif "errors" in result:
                        err = result["errors"]
                        await message.channel.send(
                            f"Splitwise error: {err if isinstance(err, str) else json.dumps(err)}"
                        )
                    else:
                        await message.channel.send("Unexpected response from Splitwise.")
                except Exception as e:
                    logger.error(f"Expense creation failed: {e}")
                    await message.channel.send(f"Failed to create expense: {e}")
                return True

            # Awaiting ratio input
            if pending.get("awaiting") == "ratio_input":
                try:
                    ratio_list = [int(r.strip()) for r in text_clean.split(":")]
                except ValueError:
                    await message.channel.send(
                        "Invalid format. Use colon-separated integers like `1:1:2`."
                    )
                    return True

                expected = len(pending["friend_ids"]) + 1
                if len(ratio_list) != expected:
                    await message.channel.send(
                        f"Need {expected} ratios (you + {expected - 1} friends), "
                        f"got {len(ratio_list)}."
                    )
                    return True

                current_user = await self._get_current_user()
                user_ids = [current_user["id"]] + pending["friend_ids"]
                amount = pending["total"]
                desc = pending["description"]

                total_ratio = sum(ratio_list)
                shares = [round(amount * r / total_ratio, 2) for r in ratio_list]
                friends = pending.get("friends", [])
                people_names = ["You"] + [
                    f"{f.get('first_name', '')} {f.get('last_name', '')}".strip()
                    for f in friends
                ]
                share_lines = [
                    f"  {name}: ${share:.2f}"
                    for name, share in zip(people_names, shares)
                ]
                summary = (
                    f"**{desc}** — ${amount:.2f} (ratio {text_clean})\n"
                    + "\n".join(share_lines)
                )

                confirmed = await self._confirm_expense(
                    message.channel, message.author.id, summary
                )
                del self._pending_receipts[message.channel.id]
                if not confirmed:
                    return True

                try:
                    result = await self._create_ratio_expense(
                        amount, desc, ratio_list, user_ids
                    )
                    if "expenses" in result:
                        await message.channel.send("Bill added to Splitwise!")
                    elif "errors" in result:
                        err = result["errors"]
                        await message.channel.send(
                            f"Splitwise error: {err if isinstance(err, str) else json.dumps(err)}"
                        )
                    else:
                        await message.channel.send("Unexpected response from Splitwise.")
                except Exception as e:
                    logger.error(f"Expense creation failed: {e}")
                    await message.channel.send(f"Failed to create expense: {e}")
                return True

        # --- Check for attached receipt image ---
        if message.attachments:
            img_attachment = None
            for att in message.attachments:
                if att.content_type and att.content_type.startswith("image/"):
                    img_attachment = att
                    break

            if img_attachment:
                async with message.channel.typing():
                    try:
                        img_bytes = await img_attachment.read()
                        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    except Exception as e:
                        await message.channel.send(f"Failed to download image: {e}")
                        return True

                    parsed = await self._parse_receipt_image(img_b64)
                    if not parsed:
                        await message.channel.send(
                            "couldn't parse the receipt, try a clearer pic"
                        )
                        return True

                    total = parsed.get("total", 0) or 0
                    desc = parsed.get("store_name", "Receipt")

                    self._pending_receipts[message.channel.id] = {
                        "parsed": parsed,
                        "total": total,
                        "description": desc,
                        "friend_ids": [],
                        "user_id": message.author.id,
                        "needs_friends": True,
                    }

                    receipt_text = self._format_receipt(parsed)
                    await message.channel.send(
                        f"{receipt_text}\n\n"
                        f"who do you want to split this with? (comma-separated names/emails)\n"
                        f"or say **cancel** to abort"
                    )
                    return True

        # --- Use Claude to classify intent ---
        async with message.channel.typing():
            intent = await self._classify_intent(text_clean)

        if not intent or intent.get("action") == "unknown":
            return False  # Not a Splitwise action — let the normal bot flow handle it

        action = intent.get("action", "")
        if action == "check_balance":
            # Redirect to balance display
            try:
                friends = await self._get_friends()
                lines = ["**Splitwise Balances:**\n"]
                has_balances = False
                for f in friends:
                    balances = f.get("balance", [])
                    for b in balances:
                        amt = float(b.get("amount", 0))
                        if amt == 0:
                            continue
                        has_balances = True
                        currency = b.get("currency_code", "USD")
                        first = f.get("first_name", "")
                        last = f.get("last_name", "")
                        name = f"{first} {last}".strip()
                        if amt > 0:
                            lines.append(f"  **{name}** owes you {currency} {abs(amt):.2f}")
                        else:
                            lines.append(f"  You owe **{name}** {currency} {abs(amt):.2f}")
                if not has_balances:
                    lines.append("  All settled up!")
                await message.channel.send("\n".join(lines))
            except Exception as e:
                await message.channel.send(f"Error fetching balances: {e}")
            return True

        if action in ("split", "add_bill"):
            amount = intent.get("amount")
            people = intent.get("people", [])
            ratio = intent.get("ratio")
            desc = intent.get("description") or "Bill"

            if not people:
                await message.channel.send(
                    "who do you want to split with? give me some names"
                )
                return True

            if not amount:
                await message.channel.send(
                    "how much is the total? need an amount to split"
                )
                return True

            amount = float(amount)

            # Resolve friends
            friends = await self._resolve_friends_with_disambiguation(
                people, message.channel, message.author.id
            )
            if friends is None:
                return True
            if not friends:
                await message.channel.send("couldn't resolve any friends from those names")
                return True

            friend_ids = [f["id"] for f in friends]

            if ratio:
                ratio_list = [int(r) for r in ratio]
                expected = len(friend_ids) + 1
                if len(ratio_list) != expected:
                    await message.channel.send(
                        f"ratio has {len(ratio_list)} parts but there are {expected} people "
                        f"(including you). fix the ratio and try again"
                    )
                    return True

                current_user = await self._get_current_user()
                user_ids = [current_user["id"]] + friend_ids
                total_ratio = sum(ratio_list)
                shares = [round(amount * r / total_ratio, 2) for r in ratio_list]
                people_names = ["You"] + [
                    f"{f.get('first_name', '')} {f.get('last_name', '')}".strip()
                    for f in friends
                ]
                share_lines = [
                    f"  {name}: ${share:.2f}"
                    for name, share in zip(people_names, shares)
                ]
                summary = (
                    f"**{desc}** — ${amount:.2f} (ratio {':'.join(str(r) for r in ratio_list)})\n"
                    + "\n".join(share_lines)
                )

                confirmed = await self._confirm_expense(
                    message.channel, message.author.id, summary
                )
                if not confirmed:
                    return True

                try:
                    result = await self._create_ratio_expense(
                        amount, desc, ratio_list, user_ids
                    )
                    if result.get("errors"):
                        err = result["errors"]
                        await message.channel.send(
                            f"splitwise error: {err if isinstance(err, str) else json.dumps(err)}"
                        )
                    elif result.get("expenses"):
                        await message.channel.send("added to splitwise!")
                    else:
                        await message.channel.send("unexpected response from splitwise")
                except Exception as e:
                    logger.error(f"Expense creation failed: {e}")
                    await message.channel.send(f"failed to create expense: {e}")
            else:
                num_people = len(friend_ids) + 1
                share = amount / num_people
                summary = (
                    f"**{desc}** — ${amount:.2f} split {num_people} ways "
                    f"(${share:.2f} each)"
                )

                confirmed = await self._confirm_expense(
                    message.channel, message.author.id, summary
                )
                if not confirmed:
                    return True

                try:
                    result = await self._create_equal_expense(
                        amount, desc, friend_ids
                    )
                    if result.get("errors"):
                        err = result["errors"]
                        await message.channel.send(
                            f"splitwise error: {err if isinstance(err, str) else json.dumps(err)}"
                        )
                    elif result.get("expenses"):
                        await message.channel.send("added to splitwise!")
                    else:
                        await message.channel.send("unexpected response from splitwise")
                except Exception as e:
                    logger.error(f"Expense creation failed: {e}")
                    await message.channel.send(f"failed to create expense: {e}")
            return True

        return False

