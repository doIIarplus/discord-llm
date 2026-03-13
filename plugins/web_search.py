"""Web search plugin — /search slash command."""

import logging

import discord
from discord import app_commands

from plugin_base import BasePlugin

logger = logging.getLogger("Plugin.web_search")


class WebSearchPlugin(BasePlugin):
    name = "web_search"
    version = "1.0.0"
    description = "Search the web and get LLM-summarized results via /search"

    async def on_load(self):
        self.register_slash_command(
            name="search",
            description="Search the web and get a summary from the LLM",
            callback=self._search_command,
        )

    async def _search_command(
        self,
        interaction: discord.Interaction,
        query: str,
        results: int = 5,
    ):
        logger.info(f"Search called by {interaction.user.name}: {query[:50]}...")
        await interaction.response.defer(thinking=True)

        from web_extractor import web_search, format_search_results

        search_results = await web_search(query, max_results=results)
        if not search_results:
            await interaction.followup.send("No search results found (is TAVILY_API_KEY set?).")
            return

        raw_context = format_search_results(search_results)
        search_summary = await self.ctx.ollama_client.summarize_search_results(query, raw_context)

        llm_prompt = (
            f"Search Results Summary:\n{search_summary}\n\n"
            f"The user searched for: {query}"
        )

        # Use the bot's query pipeline for consistent context handling
        bot = self.ctx.discord_client
        response = await bot.query_ollama(
            interaction.guild.id,
            interaction.channel.id,
            [{"role": "user", "content": llm_prompt, "images": []}],
        )

        if isinstance(response, list):
            response_text = "\n".join(response)
        else:
            response_text = str(response)

        # Append short domain sources
        seen = set()
        domains = []
        for r in search_results[:3]:
            try:
                domain = r["url"].split("/")[2].removeprefix("www.")
            except (IndexError, AttributeError):
                continue
            if domain not in seen:
                seen.add(domain)
                domains.append(domain)
        if domains:
            response_text += f"\n-# Sources: {', '.join(domains)}"

        if len(response_text) > 2000:
            response_text = response_text[:1997] + "..."

        await interaction.followup.send(response_text)
