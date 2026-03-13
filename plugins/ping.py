"""Ping command plugin — ping Google and show latency."""

import time

import aiohttp

from plugin_base import BasePlugin


class PingPlugin(BasePlugin):
    name = "ping"
    version = "1.0.0"
    description = "Ping Google and show the latency"

    async def on_load(self):
        self.register_slash_command(
            name="ping",
            description="Ping Google and show the latency",
            callback=self._ping_command,
        )

    async def _ping_command(self, interaction):
        await interaction.response.defer(thinking=True)
        try:
            start = time.perf_counter()
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://www.google.com",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    await resp.read()
            latency_ms = (time.perf_counter() - start) * 1000
            await interaction.followup.send(f"Pong! Google responded in **{latency_ms:.1f}ms**")
        except Exception as e:
            self.logger.error(f"Ping failed: {e}", exc_info=True)
            await interaction.followup.send(f"Ping failed: {e}")
