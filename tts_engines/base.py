"""Abstract base classes for TTS engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TTSResult:
    audio_path: Path
    sample_rate: int
    duration_seconds: float


@dataclass
class VoiceInfo:
    name: str
    description: str
    is_cloned: bool = False
    reference_audio: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TTSEngine(ABC):
    """Abstract base for all TTS backends."""

    @abstractmethod
    async def initialize(self):
        """Load the model into memory."""
        ...

    @abstractmethod
    async def generate(self, text: str, voice: str, output_path: Path, **kwargs) -> TTSResult:
        """Generate speech audio from text."""
        ...

    @abstractmethod
    async def list_voices(self) -> List[VoiceInfo]:
        """Return all available voices."""
        ...

    @abstractmethod
    async def clone_voice(self, name: str, reference_audio_path: Path) -> VoiceInfo:
        """Register a new cloned voice from a reference audio clip."""
        ...

    @abstractmethod
    async def get_default_voice(self) -> str:
        """Return the name of the default voice."""
        ...

    @abstractmethod
    async def shutdown(self):
        """Unload the model and free resources."""
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Whether the engine has been initialized and is ready."""
        ...
