"""Translate plugin — !translate command for English to Chinese translation."""

from plugin_base import BasePlugin


class TranslatePlugin(BasePlugin):
    name = "translate"
    version = "1.0.0"
    description = "Translates English text to Chinese (Simplified) via !translate"

    async def on_load(self):
        self.register_message_handler(
            pattern=r'^!translate\b',
            callback=self._handle_translate,
            priority=50,
        )

    async def _handle_translate(self, message):
        text = message.content[len("!translate"):].strip()
        if not text:
            await message.reply("usage: `!translate <english text>`")
            return True

        prompt = (
            "You are a translator. Translate the following English text to "
            "Chinese (Simplified). Only output the translation, nothing else.\n\n"
            f"{text}"
        )
        translation = await self.ctx.query_llm(prompt)
        await message.reply(translation.strip())
        return True
