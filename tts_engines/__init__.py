"""TTS engine package — generic interface + implementations."""

from tts_engines.base import TTSEngine, TTSResult, VoiceInfo

__all__ = ["TTSEngine", "TTSResult", "VoiceInfo"]
