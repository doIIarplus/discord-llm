"""TTS plugin — text-to-speech with voice cloning via Chatterbox."""

import asyncio
import base64
import logging
import os
import subprocess
import uuid
from pathlib import Path

import aiohttp
import discord
import numpy as np

from config import DISCORD_BOT_TOKEN
from plugin_base import BasePlugin, HookType
from sandbox import safe_path

logger = logging.getLogger("Plugin.tts")

OUTPUT_DIR = Path(__file__).parent.parent / "api_out" / "tts"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
# Model loading can take 60s+ on first call (downloads weights + GPU init)
TTS_TIMEOUT = 120.0
# Bundled ffmpeg from imageio-ffmpeg (no system install needed)
_FFMPEG_PATH = None

# Voice personalities — the first line of the system prompt gets swapped.
# "default" restores the original system prompt personality.
# Voice config: personality prompt + TTS generation parameters.
# TTS params override the engine defaults per-voice for best results.
VOICE_CONFIG = {
    "default": {
        "personality": None,
        "tts_params": {},  # use engine defaults
    },
    "obama": {
        "personality": (
            "You ARE Barack Obama — not a bot, not an AI, not an assistant. You are the 44th President. "
            "Never break character. Never mention being a bot or AI. "
            "Speak with his calm, measured, eloquent cadence. Bring things back to hope and unity. "
            "Occasionally start with 'Look,' or 'Let me be clear.' Keep it conversational but presidential. "
            "Text casually — short messages, mostly without punctuation — but the vibe "
            "and word choices should unmistakably be Obama."
        ),
        "tts_params": {
            "exaggeration": 0.5,   # calm, measured delivery
            "cfg_weight": 0.3,     # deliberate pacing with natural pauses
        },
    },
    "trumpv2": {
        "personality": (
            "You ARE Donald Trump — not a bot, not an AI, not an assistant. You are the 45th President. "
            "Never break character. Never mention being a bot or AI. "
            "Speak with his bold, superlative-heavy style. Use words like 'tremendous,' 'huge,' "
            "'believe me,' 'many people are saying.' Repeat key phrases for emphasis. Be confident and brash. "
            "Occasionally go on tangents. Text casually — short messages, mostly without punctuation — "
            "but the vibe and word choices should unmistakably be Trump."
        ),
        "tts_params": {
            "exaggeration": 0.85,  # bold, dramatic delivery
            "cfg_weight": 0.35,    # slightly faster but still deliberate
        },
    },
}
# Convenience alias for personality lookups
VOICE_PERSONALITIES = {k: v["personality"] for k, v in VOICE_CONFIG.items()}


def _get_ffmpeg() -> str:
    """Locate the bundled ffmpeg binary."""
    global _FFMPEG_PATH
    if _FFMPEG_PATH is None:
        from imageio_ffmpeg import get_ffmpeg_exe
        _FFMPEG_PATH = get_ffmpeg_exe()
    return _FFMPEG_PATH


def _wav_to_ogg_opus(wav_path: Path, ogg_path: Path) -> None:
    """Convert WAV to OGG Opus using bundled ffmpeg."""
    subprocess.run(
        [_get_ffmpeg(), "-i", str(wav_path), "-c:a", "libopus",
         "-b:a", "64k", "-f", "ogg", str(ogg_path), "-y"],
        capture_output=True, check=True,
    )


def _compute_waveform(wav_path: Path) -> str:
    """Compute a base64-encoded waveform preview (up to 256 bytes) from a WAV file."""
    from scipy.io import wavfile
    sr, data = wavfile.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    # Sample at ~100ms intervals, max 256 points
    samples_per_point = max(1, int(sr * 0.1))
    n_points = min(256, len(data) // samples_per_point)
    if n_points == 0:
        return base64.b64encode(bytes([128])).decode()
    points = []
    for i in range(n_points):
        chunk = data[i * samples_per_point:(i + 1) * samples_per_point]
        amplitude = np.abs(chunk).mean()
        points.append(amplitude)
    # Normalize to 0-255
    max_amp = max(points) if max(points) > 0 else 1
    waveform_bytes = bytes(int(min(255, (a / max_amp) * 255)) for a in points)
    return base64.b64encode(waveform_bytes).decode()


async def _send_voice_message(channel_id: int, ogg_path: Path,
                              duration_secs: float, waveform_b64: str) -> None:
    """Send an OGG Opus file as a Discord voice message via raw HTTP."""
    headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
    file_size = os.path.getsize(ogg_path)

    async with aiohttp.ClientSession() as session:
        # Step 1: Request upload URL
        async with session.post(
            f"https://discord.com/api/v10/channels/{channel_id}/attachments",
            headers={**headers, "Content-Type": "application/json"},
            json={"files": [{"filename": "voice-message.ogg",
                             "file_size": file_size, "id": "2"}]},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            upload_url = data["attachments"][0]["upload_url"]
            upload_filename = data["attachments"][0]["upload_filename"]

        # Step 2: Upload the audio file
        with open(ogg_path, "rb") as f:
            async with session.put(
                upload_url,
                headers={"Content-Type": "audio/ogg"},
                data=f.read(),
            ) as resp:
                resp.raise_for_status()

        # Step 3: Send the voice message (flag 8192 = IS_VOICE_MESSAGE)
        async with session.post(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "flags": 8192,
                "attachments": [{
                    "id": "0",
                    "filename": "voice-message.ogg",
                    "uploaded_filename": upload_filename,
                    "duration_secs": duration_secs,
                    "waveform": waveform_b64,
                }],
            },
        ) as resp:
            resp.raise_for_status()


class TTSPlugin(BasePlugin):
    name = "tts"
    version = "1.1.0"
    description = "Text-to-speech with voice cloning via Chatterbox"

    async def on_load(self):
        self._engine = None
        self._user_prefs = {}  # user_id -> {"voice": str, "voice_mode": bool}

        self.register_message_handler(
            pattern=r"^!tts\b",
            callback=self._handle_tts,
            priority=20,
            timeout=TTS_TIMEOUT,
        )
        self.register_message_handler(
            pattern=r"^!voice\b",
            callback=self._handle_voice,
            priority=20,
            timeout=TTS_TIMEOUT,
        )
        self.register_message_handler(
            pattern=r"^!voices$",
            callback=self._handle_voices,
            priority=20,
            timeout=TTS_TIMEOUT,
        )
        self.register_hook(HookType.POST_QUERY, self._on_post_query, timeout=TTS_TIMEOUT)

    async def on_unload(self):
        if self._engine and self._engine.is_initialized:
            await self._engine.shutdown()

    async def self_test(self) -> bool:
        return True

    def suppress_text(self, message) -> bool:
        """Tell the bot to skip text output when voice mode is on."""
        result = self._get_voice_mode(message.author.id)
        self.logger.info(f"[TTS-DEBUG] suppress_text check: user={message.author.id}, "
                         f"voice_mode={result}, prefs={self._user_prefs.get(message.author.id)}")
        return result

    # ── Engine ────────────────────────────────────────────────────────

    async def _ensure_engine(self):
        if self._engine is None:
            from tts_engines.chatterbox_engine import ChatterboxEngine
            self._engine = ChatterboxEngine()
        if not self._engine.is_initialized:
            await self._engine.initialize()
        return self._engine

    # ── Helpers ───────────────────────────────────────────────────────

    def _build_personality_prompt(self, voice: str) -> str | None:
        """Build a system prompt for a voice personality.

        Returns None for the default voice (use bot's original prompt).
        For other voices, strips CODE EDITING and CLI TOOLS sections
        (which reference being a bot) and keeps only structural instructions.
        """
        personality = VOICE_PERSONALITIES.get(voice)
        if personality is None:
            return None
        original = self.ctx._bot.original_system_prompt
        kept_sections = []
        for section_name in ("CONVERSATION FORMAT:", "MULTI-MESSAGE RESPONSES:"):
            start = original.find(section_name)
            if start == -1:
                continue
            end = len(original)
            for next_header in ("CONVERSATION FORMAT:", "MULTI-MESSAGE RESPONSES:",
                                "CODE EDITING:", "CLI TOOLS:"):
                pos = original.find(next_header, start + len(section_name))
                if pos != -1 and pos < end:
                    end = pos
            kept_sections.append(original[start:end].rstrip())
        structural = "\n\n".join(kept_sections)
        return personality + "\n\n" + structural

    def get_system_prompt_override(self, user_id: int) -> str | None:
        """Called by plugin_manager to get per-user personality prompt."""
        voice = self._get_user_voice(user_id)
        if not self._get_voice_mode(user_id):
            return None
        return self._build_personality_prompt(voice)

    def _get_user_voice(self, user_id: int) -> str:
        return self._user_prefs.get(user_id, {}).get("voice", "default")

    def _get_voice_mode(self, user_id: int) -> bool:
        return self._user_prefs.get(user_id, {}).get("voice_mode", False)

    def _set_pref(self, user_id: int, **kwargs):
        if user_id not in self._user_prefs:
            self._user_prefs[user_id] = {"voice": "default", "voice_mode": False}
        self._user_prefs[user_id].update(kwargs)

    async def _generate_voice_message(self, channel_id: int, text: str, user_id: int):
        """Generate TTS audio and send as a Discord voice message."""
        engine = await self._ensure_engine()
        wav_path = Path(safe_path(str(OUTPUT_DIR / f"{uuid.uuid4()}.wav")))
        ogg_path = wav_path.with_suffix(".ogg")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        voice = self._get_user_voice(user_id)
        # Per-voice TTS parameters (exaggeration, cfg_weight, etc.)
        tts_params = VOICE_CONFIG.get(voice, {}).get("tts_params", {})
        result = await engine.generate(text, voice, wav_path, **tts_params)
        try:
            waveform = await asyncio.to_thread(_compute_waveform, wav_path)
            await asyncio.to_thread(_wav_to_ogg_opus, wav_path, ogg_path)
            await _send_voice_message(
                channel_id, ogg_path, result.duration_seconds, waveform,
            )
        finally:
            for p in (wav_path, ogg_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # ── Handlers ──────────────────────────────────────────────────────

    async def _handle_tts(self, message: discord.Message):
        text = message.content.partition("!tts")[2].strip()
        if not text:
            await message.channel.send("usage: `!tts <text>`")
            return True

        async with message.channel.typing():
            await self._generate_voice_message(
                message.channel.id, text, message.author.id,
            )
        return True

    async def _handle_voice(self, message: discord.Message):
        parts = message.content.partition("!voice")[2].strip().split(None, 1)
        sub = parts[0].lower() if parts else ""

        if sub == "on":
            self._set_pref(message.author.id, voice_mode=True)
            voice = self._get_user_voice(message.author.id)

            self.logger.info(f"[TTS-DEBUG] Voice mode ON for user={message.author.id}, "
                             f"voice={voice}, prefs={self._user_prefs[message.author.id]}")
            await message.channel.send(f"voice mode on (voice: {voice})")
            return True

        if sub == "off":
            self._set_pref(message.author.id, voice_mode=False)

            self.logger.info(f"[TTS-DEBUG] Voice mode OFF for user={message.author.id}, "
                             f"prefs={self._user_prefs[message.author.id]}")
            await message.channel.send("voice mode off")
            return True

        if sub == "set":
            name = parts[1].strip() if len(parts) > 1 else ""
            if not name:
                await message.channel.send("usage: `!voice set <name>`")
                return True
            async with message.channel.typing():
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
                async with message.channel.typing():
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
        async with message.channel.typing():
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
        voice_mode = self._get_voice_mode(message.author.id)
        self.logger.info(f"[TTS-DEBUG] _on_post_query: user={message.author.id}, "
                         f"voice_mode={voice_mode}, text_len={len(response_text)}")
        if not voice_mode:
            return None
        # Strip ---MSG--- markers and other non-speech artifacts
        clean_text = response_text.replace("---MSG---", " ").strip()
        clean_text = " ".join(clean_text.split())  # collapse whitespace
        self.logger.info(f"[TTS-DEBUG] Generating voice message, clean_text={clean_text[:80]!r}...")
        await self._generate_voice_message(
            message.channel.id, clean_text, message.author.id,
        )
        self.logger.info(f"[TTS-DEBUG] Voice message sent successfully")
