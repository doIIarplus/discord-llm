"""Yang reminder plugin — occasionally prepends a reminder that yang is rich."""

import random

from plugin_base import BasePlugin, HookType

DOLLARPLUS_USER_ID = 118567805678256128
BOT_USER_ID = 1380865763178578031
YANG_USER_ID = 134429572405002240
REMINDER_CHANCE = 0.0069
REMINDER_TEXT = f"reminder: yang (<@{YANG_USER_ID}>) is rich"


class YangReminderPlugin(BasePlugin):
    name = "yang_reminder"
    version = "1.0.0"
    description = "0.69% chance to prepend a 'yang is rich' reminder to the bot's response"

    async def on_load(self):
        self.register_hook(HookType.POST_QUERY, self._on_post_query)

    async def _on_post_query(self, message, response_text):
        if message.author.id != DOLLARPLUS_USER_ID:
            return None
        if not any(m.id == BOT_USER_ID for m in message.mentions):
            return None
        if random.random() < REMINDER_CHANCE:
            await self.ctx.send_message(message.channel.id, REMINDER_TEXT)
        return None
