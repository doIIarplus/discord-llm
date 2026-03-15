"""Splitwise bill plugin — create expenses, parse receipts, check balances."""

import base64
import json
import logging
import os
import re
from typing import Optional

import aiohttp
import discord
from discord import app_commands

from plugin_base import BasePlugin

logger = logging.getLogger("Plugin.splitwise_bill")

SPLITWISE_BASE_URL = "https://secure.splitwise.com/api/v3.0"


class SplitwiseBillPlugin(BasePlugin):
    name = "splitwise_bill"
    version = "1.0.0"
    description = "Splitwise integration: add bills, parse receipts, check balances"

    async def on_load(self):
        self._api_key = os.getenv("SPLITWISE_API_KEY", "")

        # Pending receipt confirmations: channel_id -> {expense_data, parsed_items, user_id}
        self._pending_receipts = {}

        self.register_slash_command(
            name="bill",
            description="Add a bill to Splitwise, split equally",
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

        self.register_message_handler(
            pattern=r'(?i)(splitwise|add.*bill|split.*cost)',
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
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{SPLITWISE_BASE_URL}{endpoint}", headers=self._headers()
            ) as resp:
                return await resp.json()

    async def _api_post(self, endpoint: str, data: dict) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{SPLITWISE_BASE_URL}{endpoint}",
                headers=self._headers(),
                json=data,
            ) as resp:
                return await resp.json()

    async def _get_current_user(self) -> dict:
        result = await self._api_get("/get_current_user")
        return result.get("user", {})

    async def _get_friends(self) -> list:
        result = await self._api_get("/get_friends")
        return result.get("friends", [])

    async def _find_friend(self, identifier: str) -> Optional[dict]:
        """Find a friend by email or name (case-insensitive)."""
        friends = await self._get_friends()
        identifier_lower = identifier.strip().lower()
        for f in friends:
            email = (f.get("email") or "").lower()
            first = (f.get("first_name") or "").lower()
            last = (f.get("last_name") or "").lower()
            full = f"{first} {last}".strip()
            if identifier_lower in (email, first, last, full):
                return f
        return None

    async def _create_equal_expense(
        self, amount: float, description: str, friend_ids: list, group_id: int = 0
    ) -> dict:
        current_user = await self._get_current_user()
        user_id = current_user["id"]

        num_people = len(friend_ids) + 1
        share = round(amount / num_people, 2)
        paid_share = round(amount, 2)

        data = {
            "cost": str(paid_share),
            "description": description,
            "currency_code": "USD",
            "split_equally": False,
            "users__0__user_id": user_id,
            "users__0__paid_share": str(paid_share),
            "users__0__owed_share": str(share),
        }
        if group_id:
            data["group_id"] = group_id

        for i, fid in enumerate(friend_ids, start=1):
            data[f"users__{i}__user_id"] = fid
            data[f"users__{i}__paid_share"] = "0.00"
            data[f"users__{i}__owed_share"] = str(share)

        # Splitwise create_expense uses form-style keys, post as form
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{SPLITWISE_BASE_URL}/create_expense",
                headers={"Authorization": f"Bearer {self._api_key}"},
                data=data,
            ) as resp:
                return await resp.json()

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

        # Current user pays the full amount
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

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{SPLITWISE_BASE_URL}/create_expense",
                headers={"Authorization": f"Bearer {self._api_key}"},
                data=data,
            ) as resp:
                return await resp.json()

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
            # Try to extract JSON from the response
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
        for item in items:
            name = item.get("name", "???")
            price = item.get("price", 0)
            qty = item.get("quantity", 1)
            if qty > 1:
                lines.append(f"  {name} x{qty} — ${price:.2f}")
            else:
                lines.append(f"  {name} — ${price:.2f}")

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

    # --- Slash commands ---

    async def _bill_command(
        self,
        interaction: discord.Interaction,
        amount: float,
        description: str,
        split_with: str,
    ):
        """Add a simple bill split equally."""
        if not self._api_key:
            await interaction.response.send_message(
                "Splitwise API key not configured. Set SPLITWISE_API_KEY in .env.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)

        names = [n.strip() for n in split_with.split(",") if n.strip()]
        friend_ids = []
        not_found = []

        for name in names:
            friend = await self._find_friend(name)
            if friend:
                friend_ids.append(friend["id"])
            else:
                not_found.append(name)

        if not_found:
            await interaction.followup.send(
                f"Could not find Splitwise friends: {', '.join(not_found)}\n"
                "Use their email or name as it appears on Splitwise."
            )
            return

        if not friend_ids:
            await interaction.followup.send("No valid friends specified to split with.")
            return

        result = await self._create_equal_expense(amount, description, friend_ids)

        if "expenses" in result:
            num_people = len(friend_ids) + 1
            share = amount / num_people
            await interaction.followup.send(
                f"Bill added to Splitwise!\n"
                f"**{description}** — ${amount:.2f} split {num_people} ways (${share:.2f} each)"
            )
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
    ):
        """Add a bill by attaching a receipt image."""
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
        friend_ids = []
        not_found = []
        for name in names:
            friend = await self._find_friend(name)
            if friend:
                friend_ids.append(friend["id"])
            else:
                not_found.append(name)

        if not_found:
            await interaction.followup.send(
                f"Could not find Splitwise friends: {', '.join(not_found)}"
            )
            return

        total = parsed.get("total", 0) or 0
        desc = description or parsed.get("store_name", "Receipt")

        # Store pending receipt for confirmation
        self._pending_receipts[interaction.channel_id] = {
            "parsed": parsed,
            "total": total,
            "description": desc,
            "friend_ids": friend_ids,
            "user_id": interaction.user.id,
        }

        receipt_text = self._format_receipt(parsed)
        await interaction.followup.send(
            f"{receipt_text}\n\n"
            f"Split equally with {len(friend_ids)} {'person' if len(friend_ids) == 1 else 'people'}.\n"
            f"Reply **yes** or **confirm** to submit to Splitwise, or **cancel** to abort."
        )

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

        # Check for confirmation of pending receipt
        text_lower = message.content.lower()
        text_clean = re.sub(r'<@!?\d+>', '', text_lower).strip()

        if message.channel.id in self._pending_receipts:
            pending = self._pending_receipts[message.channel.id]
            if message.author.id != pending["user_id"]:
                return False

            if text_clean in ("yes", "confirm", "looks good", "y", "ok"):
                del self._pending_receipts[message.channel.id]
                try:
                    result = await self._create_equal_expense(
                        pending["total"],
                        pending["description"],
                        pending["friend_ids"],
                    )
                    if "expenses" in result:
                        num_people = len(pending["friend_ids"]) + 1
                        share = pending["total"] / num_people
                        await message.channel.send(
                            f"Bill submitted to Splitwise!\n"
                            f"**{pending['description']}** — ${pending['total']:.2f} "
                            f"split {num_people} ways (${share:.2f} each)"
                        )
                    elif "errors" in result:
                        err = result["errors"]
                        await message.channel.send(
                            f"Splitwise error: {err if isinstance(err, str) else json.dumps(err)}"
                        )
                    else:
                        await message.channel.send(f"Unexpected response from Splitwise.")
                except Exception as e:
                    logger.error(f"Expense creation failed: {e}")
                    await message.channel.send(f"Failed to create expense: {e}")
                return True

            elif text_clean in ("no", "cancel", "n", "nah"):
                del self._pending_receipts[message.channel.id]
                await message.channel.send("Receipt cancelled.")
                return True

        # Check for attached receipt image
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

                    # Try to extract who to split with from the message
                    # For now, store as pending and ask
                    self._pending_receipts[message.channel.id] = {
                        "parsed": parsed,
                        "total": total,
                        "description": desc,
                        "friend_ids": [],  # Will be resolved on confirmation
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

        # Handle pending that needs friends list
        if message.channel.id in self._pending_receipts:
            pending = self._pending_receipts[message.channel.id]
            if pending.get("needs_friends") and message.author.id == pending["user_id"]:
                names_text = re.sub(r'<@!?\d+>', '', message.content).strip()
                names = [n.strip() for n in names_text.split(",") if n.strip()]

                friend_ids = []
                not_found = []
                for name in names:
                    friend = await self._find_friend(name)
                    if friend:
                        friend_ids.append(friend["id"])
                    else:
                        not_found.append(name)

                if not_found:
                    await message.channel.send(
                        f"couldn't find: {', '.join(not_found)}. try again or say **cancel**"
                    )
                    return True

                if not friend_ids:
                    await message.channel.send("need at least one person to split with")
                    return True

                pending["friend_ids"] = friend_ids
                pending["needs_friends"] = False
                num_people = len(friend_ids) + 1
                share = pending["total"] / num_people

                await message.channel.send(
                    f"${pending['total']:.2f} split {num_people} ways = ${share:.2f} each\n"
                    f"reply **yes** to confirm or **cancel** to abort"
                )
                return True

        return False
