"""Chatterbox TTS engine — zero-shot voice cloning on GPU."""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np

from tts_engines.base import TTSEngine, TTSResult, VoiceInfo

logger = logging.getLogger("TTSEngine.Chatterbox")

VOICES_DIR = Path(__file__).parent.parent / "voices"
OUTPUT_DIR = Path(__file__).parent.parent / "api_out" / "tts"
DEFAULT_VOICE_NAME = "default"
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

# Max characters per chunk — keeps each generation short for better prosody.
# Chatterbox works best under ~300 chars (~15s of audio).
MAX_CHUNK_CHARS = 280


def _chunk_text(text: str) -> list[str]:
    """Split text into sentence-boundary chunks, each under MAX_CHUNK_CHARS.

    Generates each chunk with enough context for natural prosody.
    """
    import nltk
    try:
        sentences = nltk.tokenize.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        sentences = nltk.tokenize.sent_tokenize(text)

    chunks = []
    current = ""
    for sent in sentences:
        if current and len(current) + len(sent) + 1 > MAX_CHUNK_CHARS:
            chunks.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}" if current else sent

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text]


class ChatterboxEngine(TTSEngine):
    """Chatterbox TTS with zero-shot voice cloning."""

    # Default generation parameters — tuned for natural, expressive speech.
    # Override per-call via kwargs.
    DEFAULT_PARAMS = {
        "exaggeration": 0.7,     # emotional expressiveness (0.5=neutral, higher=more dramatic)
        "cfg_weight": 0.3,       # pacing/deliberateness (lower=slower, more pauses)
        "temperature": 0.8,      # sampling randomness
        "repetition_penalty": 1.2,
    }

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
        """Generate speech with sentence-level chunking for better prosody.

        kwargs can override: exaggeration, cfg_weight, temperature, repetition_penalty.
        """
        from scipy.io import wavfile

        # Merge defaults with per-call overrides
        params = {**self.DEFAULT_PARAMS, **kwargs}
        ref_audio = self._resolve_voice_audio(voice)

        chunks = _chunk_text(text)
        logger.info(f"Generating {len(chunks)} chunk(s) for {len(text)} chars of text")

        model = self._model
        sr = model.sr

        def _generate_all():
            all_audio = []
            # Small silence gap between sentences (0.15s)
            silence = np.zeros(int(sr * 0.15), dtype=np.float32)

            for i, chunk in enumerate(chunks):
                gen_kwargs = {
                    "text": chunk,
                    "exaggeration": params["exaggeration"],
                    "cfg_weight": params["cfg_weight"],
                    "temperature": params["temperature"],
                    "repetition_penalty": params["repetition_penalty"],
                }
                if ref_audio is not None:
                    gen_kwargs["audio_prompt_path"] = str(ref_audio)

                wav = model.generate(**gen_kwargs)
                audio_np = wav.cpu().squeeze().numpy()
                all_audio.append(audio_np)

                # Add silence between chunks (not after the last one)
                if i < len(chunks) - 1:
                    all_audio.append(silence)

            # Concatenate all chunks
            combined = np.concatenate(all_audio)
            audio_int16 = np.clip(combined * 32767, -32768, 32767).astype(np.int16)
            wavfile.write(str(output_path), sr, audio_int16)
            return combined.shape[0] / sr

        duration = await asyncio.to_thread(_generate_all)
        return TTSResult(
            audio_path=output_path,
            sample_rate=sr,
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
