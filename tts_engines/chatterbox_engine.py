"""Chatterbox TTS engine — zero-shot voice cloning on GPU."""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from tts_engines.base import TTSEngine, TTSResult, VoiceInfo

logger = logging.getLogger("TTSEngine.Chatterbox")

VOICES_DIR = Path(__file__).parent.parent / "voices"
OUTPUT_DIR = Path(__file__).parent.parent / "api_out" / "tts"
DEFAULT_VOICE_NAME = "default"
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


class ChatterboxEngine(TTSEngine):
    """Chatterbox TTS with zero-shot voice cloning."""

    def __init__(self, device: str = "cuda"):
        self._device = device
        self._model = None

    async def initialize(self):
        """Load the Chatterbox model onto the GPU."""
        def _load():
            from chatterbox.tts import ChatterboxTTS
            return ChatterboxTTS.from_pretrained(device=self._device)

        logger.info(f"Loading Chatterbox model on {self._device}...")
        self._model = await asyncio.to_thread(_load)
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Chatterbox model loaded")

    async def generate(
        self,
        text: str,
        voice: str,
        output_path: Path,
        **kwargs,
    ) -> TTSResult:
        """Generate speech. Supports exaggeration and cfg_weight kwargs."""
        import torchaudio

        exaggeration = kwargs.get("exaggeration", 0.5)
        cfg_weight = kwargs.get("cfg_weight", 0.5)
        ref_audio = self._resolve_voice_audio(voice)

        gen_kwargs = {
            "text": text,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }
        if ref_audio is not None:
            gen_kwargs["audio_prompt_path"] = str(ref_audio)

        model = self._model

        def _generate():
            wav = model.generate(**gen_kwargs)
            torchaudio.save(str(output_path), wav.cpu(), model.sr)
            return wav.shape[-1] / model.sr

        duration = await asyncio.to_thread(_generate)
        return TTSResult(
            audio_path=output_path,
            sample_rate=self._model.sr,
            duration_seconds=duration,
        )

    async def list_voices(self) -> List[VoiceInfo]:
        """Return the default voice plus any cloned voices in voices/."""
        voices = [
            VoiceInfo(
                name=DEFAULT_VOICE_NAME,
                description="Built-in default voice",
                is_cloned=False,
            )
        ]
        if VOICES_DIR.exists():
            for f in sorted(VOICES_DIR.iterdir()):
                if f.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                    voices.append(VoiceInfo(
                        name=f.stem,
                        description=f"Cloned voice ({f.suffix})",
                        is_cloned=True,
                        reference_audio=f,
                    ))
        return voices

    async def clone_voice(self, name: str, reference_audio_path: Path) -> VoiceInfo:
        """Copy reference audio into voices/ directory."""
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        dest = VOICES_DIR / f"{name}{reference_audio_path.suffix}"
        await asyncio.to_thread(shutil.copy2, str(reference_audio_path), str(dest))
        return VoiceInfo(
            name=name,
            description=f"Cloned voice ({reference_audio_path.suffix})",
            is_cloned=True,
            reference_audio=dest,
        )

    async def get_default_voice(self) -> str:
        return DEFAULT_VOICE_NAME

    async def shutdown(self):
        """Unload model and free VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("Chatterbox model unloaded")

    @property
    def is_initialized(self) -> bool:
        return self._model is not None

    def _resolve_voice_audio(self, voice: str) -> Optional[Path]:
        """Find reference audio for a voice name. Returns None for default."""
        if voice == DEFAULT_VOICE_NAME:
            return None
        if VOICES_DIR.exists():
            for f in VOICES_DIR.iterdir():
                if f.stem == voice and f.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                    return f
        return None
