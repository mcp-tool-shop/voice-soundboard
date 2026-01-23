"""
Base TTS Engine Interface.

All TTS backends implement this interface for consistent API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, AsyncGenerator, List, Dict, Any

import numpy as np


@dataclass
class EngineResult:
    """Result from speech synthesis."""

    audio_path: Optional[Path] = None
    samples: Optional[np.ndarray] = None
    sample_rate: int = 24000
    duration_seconds: float = 0.0
    generation_time: float = 0.0
    voice_used: str = ""
    realtime_factor: float = 0.0

    # Engine-specific metadata
    engine_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineCapabilities:
    """Describes what features an engine supports."""

    # Basic features
    supports_streaming: bool = False
    supports_ssml: bool = False

    # Advanced features
    supports_voice_cloning: bool = False
    supports_emotion_control: bool = False
    supports_paralinguistic_tags: bool = False
    supports_emotion_exaggeration: bool = False

    # Paralinguistic tags this engine supports
    paralinguistic_tags: List[str] = field(default_factory=list)

    # Supported languages
    languages: List[str] = field(default_factory=lambda: ["en"])

    # Performance characteristics
    typical_rtf: float = 1.0  # Realtime factor (higher = faster)
    min_latency_ms: float = 200.0


class TTSEngine(ABC):
    """
    Abstract base class for TTS engines.

    All TTS backends must implement this interface to ensure
    consistent API across different models (Kokoro, Chatterbox, etc.)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name identifier."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Return engine capabilities."""
        pass

    @abstractmethod
    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> EngineResult:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Voice identifier (engine-specific)
            speed: Speed multiplier (0.5-2.0)
            **kwargs: Engine-specific parameters

        Returns:
            EngineResult with audio data and metadata
        """
        pass

    @abstractmethod
    def speak_raw(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech and return raw audio samples.

        Args:
            text: Text to synthesize
            voice: Voice identifier
            speed: Speed multiplier
            **kwargs: Engine-specific parameters

        Returns:
            Tuple of (samples array, sample_rate)
        """
        pass

    @abstractmethod
    def list_voices(self) -> List[str]:
        """Get list of available voice IDs."""
        pass

    def get_voice_info(self, voice: str) -> Dict[str, Any]:
        """
        Get metadata about a specific voice.

        Override in subclass for rich metadata.
        """
        return {"id": voice, "name": voice}

    async def stream(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Stream audio generation chunk by chunk.

        Default implementation generates all at once.
        Override for true streaming support.
        """
        samples, sr = self.speak_raw(text, voice, speed, **kwargs)
        yield samples, sr

    def clone_voice(self, audio_path: Path, voice_id: str = "cloned") -> str:
        """
        Clone a voice from an audio sample.

        Args:
            audio_path: Path to reference audio (3-10 seconds recommended)
            voice_id: ID to assign to the cloned voice

        Returns:
            Voice ID that can be used in speak() calls

        Raises:
            NotImplementedError: If engine doesn't support voice cloning
        """
        if not self.capabilities.supports_voice_cloning:
            raise NotImplementedError(f"{self.name} does not support voice cloning")
        raise NotImplementedError("Subclass must implement clone_voice")

    def is_loaded(self) -> bool:
        """Check if the engine model is loaded."""
        return False

    def unload(self) -> None:
        """Unload model from memory (if supported)."""
        pass
