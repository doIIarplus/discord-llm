"""Auto-ping plugin — pings a specific user every 5 minutes."""

import asyncio

from plugin_base import BasePlugin

ALLOWED_USER_ID = 118567805678256128
PING_CHANNEL_ID = 363154169294618625  # Hardcoded channel
PING_INTERVAL = 300  # 5 minutes


class AutoPingPlugin(BasePlugin):
    name = "auto_ping"
    version = "1.0.0"
    description = "Automatically pings a user every 5 minutes"

    async def on_load(self):
        self._pinging = True
        self._task = None

        self.register_slash_command(
            name="start_pinging",
            description="Start the auto-ping loop",
            callback=self._start_command,
        )
        self.register_slash_command(
            name="stop_pinging",
            description="Stop the auto-ping loop",
            callback=self._stop_command,
        )

        # Start the background loop via on_ready hook
        from plugin_base import HookType
        self.register_hook(HookType.ON_READY, self._on_ready)

    async def on_unload(self):
        self._pinging = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _on_ready(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._ping_loop())

    async def _ping_loop(self):
        try:
            while self._pinging:
                await self.ctx.send_message(
                    PING_CHANNEL_ID,
                    f"<@{ALLOWED_USER_ID}> ping",
                )
                await asyncio.sleep(PING_INTERVAL)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Ping loop error: {e}", exc_info=True)

    async def _start_command(self, interaction):
        if interaction.user.id != ALLOWED_USER_ID:
            await interaction.response.send_message(
                "only dollarplus can use this command",
                ephemeral=True,
            )
            return
        self._pinging = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._ping_loop())
        await interaction.response.send_message("auto-ping started", ephemeral=True)

    async def _stop_command(self, interaction):
        if interaction.user.id != ALLOWED_USER_ID:
            await interaction.response.send_message(
                "only dollarplus can use this command",
                ephemeral=True,
            )
            return
        self._pinging = False
        if self._task and not self._task.done():
            self._task.cancel()
        await interaction.response.send_message("auto-ping stopped", ephemeral=True)
