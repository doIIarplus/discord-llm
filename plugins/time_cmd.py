"""!time command plugin — converts human-readable time to Discord timestamps."""

from plugin_base import BasePlugin


class TimeCommandPlugin(BasePlugin):
    name = "time_cmd"
    version = "1.0.0"
    description = "Convert human time to Discord timestamps via !time"

    async def on_load(self):
        self.register_message_handler(
            pattern=r'^!time\b',
            callback=self._handle_time,
            priority=10,
        )

    async def _handle_time(self, message):
        from time_command import handle_time_command
        time_text = message.content.strip()[5:].strip()
        await handle_time_command(message, time_text)
        return True  # Consumed
