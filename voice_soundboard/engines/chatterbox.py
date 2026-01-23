"""
Chatterbox TTS Engine Backend.

Integrates Resemble AI's Chatterbox model with:
- Paralinguistic tags ([laugh], [cough], [sigh], etc.)
- Emotion exaggeration control (0.0 monotone → 1.0 dramatic)
- Zero-shot voice cloning from 3-10 second samples
- Sub-200ms inference latency

Reference: https://github.com/resemble-ai/chatterbox
"""

from __future__ import annotations

import time
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, AsyncGenerator

import numpy as np
import soundfile as sf

from voice_soundboard.engines.base import TTSEngine, EngineResult, EngineCapabilities
from voice_soundboard.config import Config
from voice_soundboard.security import (
    sanitize_filename,
    safe_join_path,
    validate_speed,
    secure_hash,
)


# Paralinguistic tags supported by Chatterbox Turbo
PARALINGUISTIC_TAGS = [
    "laugh",
    "chuckle",
    "cough",
    "sigh",
    "gasp",
    "groan",
    "sniff",
    "shush",
    "clear throat",
]

# Regex pattern to find tags in text
TAG_PATTERN = re.compile(
    r"\[(" + "|".join(re.escape(tag) for tag in PARALINGUISTIC_TAGS) + r")\]",
    re.IGNORECASE,
)


def validate_paralinguistic_tags(text: str) -> List[str]:
    """
    Extract and validate paralinguistic tags in text.

    Args:
        text: Input text with optional [tag] markers

    Returns:
        List of found tags
    """
    matches = TAG_PATTERN.findall(text)
    return [m.lower() for m in matches]


def has_paralinguistic_tags(text: str) -> bool:
    """Check if text contains any paralinguistic tags."""
    return bool(TAG_PATTERN.search(text))


class ChatterboxEngine(TTSEngine):
    """
    Chatterbox TTS Engine by Resemble AI.

    Features:
    - Paralinguistic tags: [laugh], [cough], [sigh], [gasp], [chuckle], etc.
    - Emotion exaggeration slider (0.0 = monotone, 1.0 = dramatic)
    - Zero-shot voice cloning from short audio samples
    - Sub-200ms latency for real-time applications

    Example:
        engine = ChatterboxEngine()

        # Basic usage with tags
        result = engine.speak(
            "That's hilarious! [laugh] Oh man, [sigh] I needed that.",
            emotion_exaggeration=0.7
        )

        # Voice cloning
        result = engine.speak(
            "Hello, this is my cloned voice!",
            voice="path/to/reference.wav"
        )
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model_variant: str = "turbo",
        device: str = "cuda",
    ):
        """
        Initialize Chatterbox engine.

        Args:
            config: Voice soundboard config
            model_variant: "turbo" (fast, 350M) or "standard" (quality, 500M)
            device: "cuda" or "cpu"
        """
        self.config = config or Config()
        self.model_variant = model_variant
        self.device = device

        self._model = None
        self._model_loaded = False

        # Default parameters
        self.default_exaggeration = 0.5
        self.default_cfg_weight = 0.5

        # Voice library for cloned voices
        self._cloned_voices: Dict[str, Path] = {}

    @property
    def name(self) -> str:
        return f"chatterbox-{self.model_variant}"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_streaming=True,
            supports_ssml=False,  # Uses paralinguistic tags instead
            supports_voice_cloning=True,
            supports_emotion_control=True,
            supports_paralinguistic_tags=True,
            supports_emotion_exaggeration=True,
            paralinguistic_tags=PARALINGUISTIC_TAGS.copy(),
            languages=["en"],  # Turbo is English-only; multilingual variant supports 23+
            typical_rtf=6.0,
            min_latency_ms=180.0,
        )

    def _ensure_model_loaded(self):
        """Lazy-load the Chatterbox model on first use."""
        if self._model_loaded:
            return

        print(f"Loading Chatterbox {self.model_variant} model (device: {self.device})...")
        start = time.time()

        try:
            if self.model_variant == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            else:
                from chatterbox.tts import ChatterboxTTS

                self._model = ChatterboxTTS.from_pretrained(device=self.device)

            self._model_loaded = True
            print(f"Chatterbox model loaded in {time.time() - start:.2f}s")

        except ImportError as e:
            raise ImportError(
                "Chatterbox is not installed. Install with:\n"
                "  pip install chatterbox-tts\n"
                "Or: uv add chatterbox-tts"
            ) from e

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        emotion_exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        save_path: Optional[Path] = None,
        **kwargs,
    ) -> EngineResult:
        """
        Generate speech from text with Chatterbox.

        Args:
            text: Text to synthesize (can include paralinguistic tags like [laugh])
            voice: Path to reference audio for voice cloning (3-10s recommended)
                   or ID of a previously cloned voice
            speed: Speed multiplier (affects cfg_weight; lower = slower)
            emotion_exaggeration: Expressiveness (0.0=monotone, 1.0=dramatic)
            cfg_weight: Reference speaker adherence (0.0-1.0)
            save_path: Where to save audio

        Returns:
            EngineResult with audio and metadata
        """
        self._ensure_model_loaded()

        # Handle emotion exaggeration
        exaggeration = emotion_exaggeration if emotion_exaggeration is not None else self.default_exaggeration
        exaggeration = max(0.0, min(1.0, exaggeration))

        # Handle cfg_weight (affects pacing)
        cfg = cfg_weight if cfg_weight is not None else self.default_cfg_weight
        cfg = max(0.0, min(1.0, cfg))

        # Adjust cfg for speed (lower cfg = slower speech)
        if speed != 1.0:
            speed = validate_speed(speed)
            # Map speed 0.5-2.0 to cfg adjustment
            # Slower speech (speed < 1) needs lower cfg
            speed_factor = (speed - 1.0) * 0.3
            cfg = max(0.0, min(1.0, cfg + speed_factor))

        # Resolve voice reference
        audio_prompt_path = None
        if voice:
            if voice in self._cloned_voices:
                audio_prompt_path = str(self._cloned_voices[voice])
            elif Path(voice).exists():
                audio_prompt_path = voice
            # else: voice is ignored, model uses default

        # Check for paralinguistic tags
        found_tags = validate_paralinguistic_tags(text)

        # Generate speech
        start = time.time()

        wav = self._model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg,
        )

        gen_time = time.time() - start

        # Convert to numpy
        if hasattr(wav, "numpy"):
            samples = wav.squeeze().cpu().numpy()
        else:
            samples = np.array(wav).squeeze()

        # Get sample rate from model
        sample_rate = self._model.sr

        # Calculate metrics
        duration = len(samples) / sample_rate
        rtf = duration / gen_time if gen_time > 0 else 0

        # Determine output path
        if save_path:
            output_path = Path(save_path)
        else:
            text_hash = secure_hash(text, length=8)
            voice_id = Path(voice).stem if voice else "default"
            filename = f"chatterbox_{voice_id}_{text_hash}.wav"
            output_path = safe_join_path(self.config.output_dir, filename)

        # Save audio
        sf.write(str(output_path), samples, sample_rate)

        return EngineResult(
            audio_path=output_path,
            samples=samples,
            sample_rate=sample_rate,
            duration_seconds=duration,
            generation_time=gen_time,
            voice_used=voice or "default",
            realtime_factor=rtf,
            engine_name=self.name,
            metadata={
                "emotion_exaggeration": exaggeration,
                "cfg_weight": cfg,
                "paralinguistic_tags": found_tags,
                "has_voice_reference": audio_prompt_path is not None,
            },
        )

    def speak_raw(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        emotion_exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech and return raw audio samples."""
        self._ensure_model_loaded()

        exaggeration = emotion_exaggeration if emotion_exaggeration is not None else self.default_exaggeration
        cfg = cfg_weight if cfg_weight is not None else self.default_cfg_weight

        audio_prompt_path = None
        if voice:
            if voice in self._cloned_voices:
                audio_prompt_path = str(self._cloned_voices[voice])
            elif Path(voice).exists():
                audio_prompt_path = voice

        wav = self._model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg,
        )

        if hasattr(wav, "numpy"):
            samples = wav.squeeze().cpu().numpy()
        else:
            samples = np.array(wav).squeeze()

        return samples, self._model.sr

    def list_voices(self) -> List[str]:
        """
        Get list of available voices.

        For Chatterbox, this returns cloned voice IDs.
        The model doesn't have preset voices - it clones from reference audio.
        """
        return list(self._cloned_voices.keys())

    def get_voice_info(self, voice: str) -> Dict[str, Any]:
        """Get metadata about a cloned voice."""
        if voice in self._cloned_voices:
            return {
                "id": voice,
                "name": voice,
                "reference_path": str(self._cloned_voices[voice]),
                "type": "cloned",
            }
        return {"id": voice, "name": voice, "type": "unknown"}

    def clone_voice(self, audio_path: Path, voice_id: str = "cloned") -> str:
        """
        Register a voice reference for cloning.

        Args:
            audio_path: Path to reference audio (3-10 seconds recommended)
            voice_id: ID to assign to this voice

        Returns:
            Voice ID that can be used in speak() calls
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        # Validate audio duration (warn if outside optimal range)
        try:
            import soundfile as sf

            info = sf.info(str(audio_path))
            duration = info.duration
            if duration < 3:
                print(f"Warning: Reference audio is only {duration:.1f}s. 3-10s recommended.")
            elif duration > 15:
                print(f"Warning: Reference audio is {duration:.1f}s. 3-10s recommended for best results.")
        except Exception:
            pass  # Don't fail if we can't read duration

        self._cloned_voices[voice_id] = audio_path
        return voice_id

    def list_cloned_voices(self) -> Dict[str, str]:
        """Get dictionary of cloned voice IDs to their reference paths."""
        return {k: str(v) for k, v in self._cloned_voices.items()}

    def remove_cloned_voice(self, voice_id: str) -> bool:
        """Remove a cloned voice from the library."""
        if voice_id in self._cloned_voices:
            del self._cloned_voices[voice_id]
            return True
        return False

    @staticmethod
    def list_paralinguistic_tags() -> List[str]:
        """Get list of supported paralinguistic tags."""
        return PARALINGUISTIC_TAGS.copy()

    @staticmethod
    def format_with_tags(text: str, tags: Dict[str, List[int]]) -> str:
        """
        Insert paralinguistic tags at specified positions.

        Args:
            text: Base text
            tags: Dict mapping tag names to list of word positions

        Returns:
            Text with tags inserted

        Example:
            format_with_tags("Hello how are you", {"laugh": [1], "sigh": [3]})
            # Returns: "Hello [laugh] how are [sigh] you"
        """
        words = text.split()
        for tag, positions in tags.items():
            for pos in sorted(positions, reverse=True):
                if 0 <= pos < len(words):
                    words.insert(pos + 1, f"[{tag}]")
        return " ".join(words)

    def is_loaded(self) -> bool:
        return self._model_loaded

    def unload(self) -> None:
        """Unload model from memory."""
        self._model = None
        self._model_loaded = False

        # Clear CUDA cache if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# Convenience functions for direct use
def speak_chatterbox(
    text: str,
    voice: Optional[str] = None,
    emotion_exaggeration: float = 0.5,
    **kwargs,
) -> Path:
    """
    Quick one-liner to generate speech with Chatterbox.

    Args:
        text: Text to speak (can include [laugh], [sigh], etc.)
        voice: Path to reference audio for voice cloning
        emotion_exaggeration: 0.0 (monotone) to 1.0 (dramatic)

    Returns:
        Path to generated audio file
    """
    engine = ChatterboxEngine()
    result = engine.speak(
        text,
        voice=voice,
        emotion_exaggeration=emotion_exaggeration,
        **kwargs,
    )
    return result.audio_path
