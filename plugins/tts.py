"""TTS plugin — text-to-speech with voice cloning via Chatterbox."""

import os
import uuid
from pathlib import Path

import discord

from plugin_base import BasePlugin, HookType
from sandbox import safe_path

OUTPUT_DIR = Path(__file__).parent.parent / "api_out" / "tts"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}


class TTSPlugin(BasePlugin):
    name = "tts"
    version = "1.0.0"
    description = "Text-to-speech with voice cloning via Chatterbox"

    async def on_load(self):
        self._engine = None
        self._user_prefs = {}  # user_id -> {"voice": str, "voice_mode": bool}

        self.register_message_handler(
            pattern=r"^!tts\b",
            callback=self._handle_tts,
            priority=20,
        )
        self.register_message_handler(
            pattern=r"^!voice\b",
            callback=self._handle_voice,
            priority=20,
        )
        self.register_message_handler(
            pattern=r"^!voices$",
            callback=self._handle_voices,
            priority=20,
        )
        self.register_hook(HookType.POST_QUERY, self._on_post_query)

    async def on_unload(self):
        if self._engine and self._engine.is_initialized:
            await self._engine.shutdown()

    async def self_test(self) -> bool:
        return True

    # ── Engine ────────────────────────────────────────────────────────

    async def _ensure_engine(self):
        if self._engine is None:
            from tts_engines.chatterbox_engine import ChatterboxEngine
            self._engine = ChatterboxEngine()
        if not self._engine.is_initialized:
            await self._engine.initialize()
        return self._engine

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_user_voice(self, user_id: int) -> str:
        return self._user_prefs.get(user_id, {}).get("voice", "default")

    def _get_voice_mode(self, user_id: int) -> bool:
        return self._user_prefs.get(user_id, {}).get("voice_mode", False)

    def _set_pref(self, user_id: int, **kwargs):
        if user_id not in self._user_prefs:
            self._user_prefs[user_id] = {"voice": "default", "voice_mode": False}
        self._user_prefs[user_id].update(kwargs)

    async def _generate_and_send(self, channel, text: str, user_id: int):
        """Generate TTS audio and send it to the channel."""
        engine = await self._ensure_engine()
        output_path = Path(safe_path(str(OUTPUT_DIR / f"{uuid.uuid4()}.wav")))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        voice = self._get_user_voice(user_id)
        result = await engine.generate(text, voice, output_path)
        try:
            await channel.send(file=discord.File(str(result.audio_path)))
        finally:
            try:
                os.remove(result.audio_path)
            except OSError:
                pass

    # ── Handlers ──────────────────────────────────────────────────────

    async def _handle_tts(self, message: discord.Message):
        text = message.content.partition("!tts")[2].strip()
        if not text:
            await message.channel.send("usage: `!tts <text>`")
            return True

        async with message.channel.typing():
            await self._generate_and_send(
                message.channel, text, message.author.id
            )
        return True

    async def _handle_voice(self, message: discord.Message):
        parts = message.content.partition("!voice")[2].strip().split(None, 1)
        sub = parts[0].lower() if parts else ""

        if sub == "on":
            self._set_pref(message.author.id, voice_mode=True)
            voice = self._get_user_voice(message.author.id)
            await message.channel.send(f"voice mode on (voice: {voice})")
            return True

        if sub == "off":
            self._set_pref(message.author.id, voice_mode=False)
            await message.channel.send("voice mode off")
            return True

        if sub == "set":
            name = parts[1].strip() if len(parts) > 1 else ""
            if not name:
                await message.channel.send("usage: `!voice set <name>`")
                return True
            engine = await self._ensure_engine()
            available = {v.name for v in await engine.list_voices()}
            if name not in available:
                await message.channel.send(
                    f"voice '{name}' not found. use `!voices` to see available voices"
                )
                return True
            self._set_pref(message.author.id, voice=name)
            await message.channel.send(f"voice set to **{name}**")
            return True

        if sub == "clone":
            audio_att = None
            for att in message.attachments:
                ext = os.path.splitext(att.filename)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    audio_att = att
                    break
            if audio_att is None:
                await message.channel.send(
                    "attach an audio file (5+ seconds) to clone a voice.\n"
                    "usage: `!voice clone` with an attached wav/mp3/flac/ogg"
                )
                return True

            stem = os.path.splitext(audio_att.filename)[0]
            ext = os.path.splitext(audio_att.filename)[1]
            tmp_path = Path(safe_path(str(OUTPUT_DIR / f"tmp_clone_{uuid.uuid4()}{ext}")))
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            await audio_att.save(str(tmp_path))

            try:
                engine = await self._ensure_engine()
                voice_info = await engine.clone_voice(stem, tmp_path)
                self._set_pref(message.author.id, voice=voice_info.name)
                await message.channel.send(
                    f"voice **{voice_info.name}** cloned and set as your active voice"
                )
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return True

        # Unknown subcommand
        await message.channel.send(
            "usage: `!voice on|off|set <name>|clone` (attach audio for clone)"
        )
        return True

    async def _handle_voices(self, message: discord.Message):
        engine = await self._ensure_engine()
        voices = await engine.list_voices()
        lines = []
        for v in voices:
            tag = " *(cloned)*" if v.is_cloned else ""
            lines.append(f"• **{v.name}** — {v.description}{tag}")
        await message.channel.send("\n".join(lines) or "no voices available")
        return True

    # ── POST_QUERY hook ───────────────────────────────────────────────

    async def _on_post_query(self, message, response_text, **kwargs):
        if not self._get_voice_mode(message.author.id):
            return None
        engine = await self._ensure_engine()
        output_path = Path(safe_path(str(OUTPUT_DIR / f"{uuid.uuid4()}.wav")))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        voice = self._get_user_voice(message.author.id)
        result = await engine.generate(response_text, voice, output_path)
        return {"audio": str(result.audio_path)}
