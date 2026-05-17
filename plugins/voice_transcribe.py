"""Voice message transcription via local Whisper.

Watches a configured channel for Discord voice messages, transcribes them
with faster-whisper, and feeds the transcription back into the bot's normal
reply pipeline as if the user had typed it with an @mention.
"""

import asyncio
import logging
import os
import tempfile

import discord

import chat_history
from plugin_base import BasePlugin

logger = logging.getLogger("Plugin.voice_transcribe")

VOICE_TRANSCRIBE_CHANNEL_IDS = {1381051356894334999}

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")


class VoiceTranscribePlugin(BasePlugin):
    name = "voice_transcribe"
    version = "1.0.0"
    description = "Transcribe Discord voice messages via local Whisper and feed to the bot"

    async def on_load(self):
        self._whisper = None
        self._whisper_lock = asyncio.Lock()
        # Discord voice messages have empty content, so the regex-based
        # plugin message handlers never fire. discord.Client also lacks
        # add_listener (that's only on commands.Bot), so wrap on_message.
        bot = self.ctx.discord_client
        self._original_on_message = bot.on_message
        original = self._original_on_message

        async def wrapped_on_message(message):
            try:
                if self._is_target_voice(message):
                    await self._handle_voice(message, original)
                    return
            except Exception:
                self.logger.exception(
                    "voice handling failed; falling through to normal on_message"
                )
            await original(message)

        bot.on_message = wrapped_on_message

    async def on_unload(self):
        if hasattr(self, "_original_on_message"):
            self.ctx.discord_client.on_message = self._original_on_message
            del self._original_on_message
        self._whisper = None

    async def _ensure_whisper(self):
        if self._whisper is not None:
            return self._whisper
        async with self._whisper_lock:
            if self._whisper is None:
                from faster_whisper import WhisperModel
                self.logger.info(
                    f"Loading Whisper model={WHISPER_MODEL_SIZE} "
                    f"device={WHISPER_DEVICE} compute_type={WHISPER_COMPUTE_TYPE}"
                )
                self._whisper = await asyncio.to_thread(
                    WhisperModel,
                    WHISPER_MODEL_SIZE,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                )
                self.logger.info("Whisper model loaded")
        return self._whisper

    def _transcribe_sync(self, audio_path: str) -> str:
        segments, _ = self._whisper.transcribe(audio_path, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _is_target_voice(self, message: discord.Message) -> bool:
        if message.author.bot or message.guild is None:
            return False
        if message.channel.id not in VOICE_TRANSCRIBE_CHANNEL_IDS:
            return False
        return any(a.is_voice_message for a in message.attachments)

    async def _handle_voice(self, message: discord.Message, original_on_message):
        voice_att = next(a for a in message.attachments if a.is_voice_message)
        self.logger.info(
            f"Voice message from {message.author.display_name} "
            f"in #{getattr(message.channel, 'name', message.channel.id)} "
            f"({voice_att.duration:.1f}s)"
        )

        ext = os.path.splitext(voice_att.filename)[1] or ".ogg"
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            await voice_att.save(tmp_path)
            await self._ensure_whisper()
            transcription = await asyncio.to_thread(
                self._transcribe_sync, tmp_path
            )
        except Exception as e:
            self.logger.exception(f"Transcription failed: {e}")
            try:
                await message.channel.send(f"couldn't transcribe that voice msg: {e}")
            except Exception:
                pass
            return
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if not transcription:
            self.logger.info("Transcription was empty, skipping")
            return

        self.logger.info(f"Transcription ({len(transcription)} chars): {transcription!r}")

        try:
            await chat_history.update_message_content(message.id, transcription)
        except Exception as e:
            self.logger.warning(f"Failed to update chat_history: {e}")

        bot = self.ctx.discord_client
        message.content = transcription
        if bot.user not in message.mentions:
            message.mentions.append(bot.user)

        await original_on_message(message)
