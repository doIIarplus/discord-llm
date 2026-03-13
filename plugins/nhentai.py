"""nhentai preview plugin — fetches preview images for 6-digit codes."""

import io
import logging
import re

import aiohttp
import discord

from plugin_base import BasePlugin

logger = logging.getLogger("Plugin.nhentai")


class NhentaiPlugin(BasePlugin):
    name = "nhentai"
    version = "1.0.0"
    description = "Fetch nhentai preview for 6-digit codes when bot is mentioned"

    async def on_load(self):
        # Match messages that are ONLY a 6-digit code (after stripping mentions)
        # Priority 5 so it runs early
        self.register_message_handler(
            pattern=r'<@!?\d+>\s*\d{6}\s*$',  # mention + 6 digits
            callback=self._handle_code,
            priority=5,
        )

    async def _handle_code(self, message: discord.Message):
        # Only respond when the bot is mentioned or replied to
        bot_user = self.ctx.discord_client.user
        is_mentioned = bot_user in message.mentions
        is_reply = False
        if message.reference and message.reference.resolved:
            is_reply = message.reference.resolved.author.id == bot_user.id

        if not (is_mentioned or is_reply):
            return False  # Not consumed, let other handlers try

        # Extract the 6-digit code from the message (strip mentions)
        user_text = re.sub(r'<@!?\d+>', '', message.content).strip()
        if not re.fullmatch(r'\d{6}', user_text):
            return False

        code = user_text
        link = f"https://nhentai.net/g/{code}/"
        preview_page = f"https://nhentai.net/g/{code}/3"

        try:
            from bs4 import BeautifulSoup
            from web_extractor import js_renderer

            logger.info(f"Fetching nhentai preview for {code} via Playwright")
            html = await js_renderer.render(preview_page, timeout=20000)
            if html:
                soup = BeautifulSoup(html, "html.parser")
                img_tag = (
                    soup.select_one("div.thumbnail-container a img")
                    or soup.select_one("div#gallery-container img")
                    or soup.select_one("img")
                )
                img_url = img_tag["src"] if img_tag else None
                if img_url:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            img_url,
                            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://nhentai.net/"},
                        ) as img_resp:
                            if img_resp.status == 200:
                                img_data = await img_resp.read()
                                ext = img_url.rsplit(".", 1)[-1].split("?")[0] or "jpg"
                                file = discord.File(io.BytesIO(img_data), filename=f"preview.{ext}")
                                await message.channel.send(link, file=file)
                                return True
            # Fallback
            await message.channel.send(link)
        except Exception as e:
            logger.error(f"Failed to fetch nhentai preview: {e}")
            await message.channel.send(link)
        return True  # Consumed either way
