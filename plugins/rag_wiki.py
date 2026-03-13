"""RAG wiki plugin — /enable_rag, /disable_rag, /index_wiki, /search_wiki, /rag_stats."""

import asyncio
import logging
import os
import traceback

import discord
from discord import app_commands

from plugin_base import BasePlugin

logger = logging.getLogger("Plugin.rag_wiki")


class RagWikiPlugin(BasePlugin):
    name = "rag_wiki"
    version = "1.0.0"
    description = "RAG system for MediaWiki content search"

    async def on_load(self):
        self.register_slash_command(
            name="enable_rag",
            description="Enable RAG for wiki content",
            callback=self._enable_rag,
        )
        self.register_slash_command(
            name="disable_rag",
            description="Disable RAG for wiki content",
            callback=self._disable_rag,
        )
        self.register_slash_command(
            name="index_wiki",
            description="Index a MediaWiki XML dump for RAG",
            callback=self._index_wiki,
        )
        self.register_slash_command(
            name="search_wiki",
            description="Search the indexed wiki content",
            callback=self._search_wiki,
        )
        self.register_slash_command(
            name="rag_stats",
            description="Get statistics about the indexed wiki content",
            callback=self._rag_stats,
        )

    @property
    def _rag(self):
        return self.ctx.discord_client.rag_system

    async def _enable_rag(self, interaction: discord.Interaction):
        logger.info(f"Enable RAG called by {interaction.user.name}")
        self.ctx.discord_client.rag_enabled = True
        await interaction.response.send_message("RAG enabled. Wiki context will be added to queries.")

    async def _disable_rag(self, interaction: discord.Interaction):
        logger.info(f"Disable RAG called by {interaction.user.name}")
        self.ctx.discord_client.rag_enabled = False
        await interaction.response.send_message("RAG disabled. No wiki context will be added.")

    async def _index_wiki(
        self,
        interaction: discord.Interaction,
        clear_existing: bool = False,
    ):
        logger.info(f"Index wiki called by {interaction.user.name}")
        await interaction.response.defer(thinking=True)

        wiki_path = "maplestorywikinet.xml"
        if not os.path.exists(wiki_path):
            await interaction.followup.send(f"Wiki file not found: {wiki_path}")
            return

        try:
            if clear_existing:
                self._rag.clear_collection()
                await interaction.followup.send("Cleared existing index. Starting indexing...")
            else:
                await interaction.followup.send("Starting wiki indexing. This may take several minutes...")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._rag.index_wiki_dump, wiki_path)

            stats = self._rag.get_stats()
            await interaction.channel.send(
                f"Wiki indexing complete!\nTotal chunks indexed: {stats['total_chunks']}"
            )
        except Exception as e:
            logger.error(f"Error during wiki indexing: {e}", exc_info=True)
            await interaction.channel.send(f"Error during indexing: {e}")

    async def _search_wiki(
        self,
        interaction: discord.Interaction,
        query: str,
        n_results: int = 3,
    ):
        logger.info(f"Search wiki called by {interaction.user.name}: {query[:50]}...")
        await interaction.response.defer(thinking=True)

        try:
            results = self._rag.search(query, n_results=n_results)
            if not results:
                await interaction.followup.send("No results found.")
                return

            response = f"**Search results for:** {query}\n\n"
            for i, result in enumerate(results, 1):
                content = result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"]
                response += f"**{i}. {result['title']}** (Score: {result['score']:.3f})\n{content}\n\n"

            if len(response) > 2000:
                response = response[:1997] + "..."

            await interaction.followup.send(response)
        except Exception as e:
            logger.error(f"Error during wiki search: {e}", exc_info=True)
            await interaction.followup.send(f"Error searching wiki: {e}")

    async def _rag_stats(self, interaction: discord.Interaction):
        logger.info(f"RAG stats called by {interaction.user.name}")
        stats = self._rag.get_stats()
        bot = self.ctx.discord_client
        await interaction.response.send_message(
            f"**RAG System Stats**\n"
            f"Total chunks: {stats['total_chunks']}\n"
            f"Collection: {stats['collection_name']}\n"
            f"RAG enabled: {bot.rag_enabled}"
        )
