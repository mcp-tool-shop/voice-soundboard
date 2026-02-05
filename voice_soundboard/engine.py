"""
Voice Engine - Core TTS synthesis using Kokoro ONNX.

Provides high-quality text-to-speech with GPU acceleration.

Enable debug logging to see per-stage timing:

    import logging
    logging.basicConfig(level=logging.DEBUG)
"""

from __future__ import annotations

import logging
import os
import time
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from voice_soundboard.config import Config, KOKORO_VOICES, VOICE_PRESETS
from voice_soundboard.security import (
    sanitize_filename, safe_join_path, validate_text_input,
    validate_speed, secure_hash, safe_error_message
)
from voice_soundboard.normalizer import normalize_text as normalize_text_func
from voice_soundboard.exceptions import (
    ModelNotFoundError, VoiceNotFoundError, EngineError,
)

logger = logging.getLogger(__name__)


@dataclass
class SpeechTiming:
    """Per-stage timing breakdown for a speak() call.

    Available on SpeechResult.timing when the engine has debug logging enabled.
    All values in seconds.
    """
    total: float = 0.0
    normalize: float = 0.0
    interpret: float = 0.0
    synthesize: float = 0.0
    save: float = 0.0
    model_load: float = 0.0


@dataclass
class SpeechResult:
    """Result from speech synthesis."""
    audio_path: Path
    duration_seconds: float
    generation_time: float
    voice_used: str
    sample_rate: int
    realtime_factor: float  # How many times faster than realtime
    timing: Optional[SpeechTiming] = field(default=None, repr=False)


class VoiceEngine:
    """
    AI Voice Engine using Kokoro TTS.

    Provides natural speech synthesis with:
    - 50+ voices across multiple languages
    - GPU acceleration via ONNX Runtime
    - Voice presets (assistant, narrator, etc.)
    - Speed control

    Enable debug logging to see per-stage timing:

        import logging
        logging.basicConfig(level=logging.DEBUG)

    Example:
        engine = VoiceEngine()
        result = engine.speak("Hello world!", voice="af_bella")
        print(f"Audio saved to: {result.audio_path}")
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the voice engine.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or Config()
        self._kokoro = None
        self._model_loaded = False

        # Paths to model files (resolved from Config.model_dir)
        self._model_dir = self.config.model_dir
        self._model_path = self._model_dir / "kokoro-v1.0.onnx"
        self._voices_path = self._model_dir / "voices-v1.0.bin"

    def _ensure_model_loaded(self) -> float:
        """Lazy-load the Kokoro model on first use.

        Returns:
            Load time in seconds (0.0 if already loaded).
        """
        if self._model_loaded:
            return 0.0

        from kokoro_onnx import Kokoro

        # Check model files exist
        if not self._model_path.exists():
            raise ModelNotFoundError(str(self._model_path), "Kokoro ONNX")
        if not self._voices_path.exists():
            raise ModelNotFoundError(str(self._voices_path), "Kokoro voices")

        logger.info("Loading Kokoro model (device: %s)...", self.config.device)
        start = time.perf_counter()

        self._kokoro = Kokoro(
            str(self._model_path),
            str(self._voices_path)
        )

        self._model_loaded = True
        elapsed = time.perf_counter() - start
        logger.info("Model loaded in %.2fs", elapsed)
        return elapsed

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        preset: Optional[str] = None,
        speed: Optional[float] = None,
        style: Optional[str] = None,
        emotion: Optional[str] = None,
        save_as: Optional[str] = None,
        normalize: bool = True,
    ) -> SpeechResult:
        """
        Generate speech from text.

        Args:
            text: The text to speak
            voice: Kokoro voice ID (e.g., "af_bella", "am_michael")
            preset: Voice preset name (e.g., "assistant", "narrator")
            speed: Speech speed multiplier (0.5-2.0, default 1.0)
            style: Natural language style hint (e.g., "warmly", "excitedly")
            emotion: Emotion name (e.g., "happy", "calm", "excited").
                    Sets voice and speed from emotion defaults.
            save_as: Optional filename (auto-generated if not provided)
            normalize: Apply text normalization for TTS edge cases (default True).
                      Expands currency ($100 -> one hundred dollars),
                      abbreviations (Dr. -> Doctor), emojis, math symbols, etc.

        Returns:
            SpeechResult with audio path and metadata

        Example:
            result = engine.speak("Hello!", voice="af_bella", speed=1.1)
            result = engine.speak("Hello!", style="warmly and cheerfully")
            result = engine.speak("I'm thrilled!", emotion="excited")
        """
        t_total_start = time.perf_counter()
        timing = SpeechTiming()

        # Model loading (lazy)
        timing.model_load = self._ensure_model_loaded()

        # Stage 1: Normalize
        t0 = time.perf_counter()
        if normalize:
            text = normalize_text_func(text)
        timing.normalize = time.perf_counter() - t0

        # Stage 2: Interpret (emotion -> style -> preset -> defaults)
        t0 = time.perf_counter()

        if emotion:
            from voice_soundboard.emotions import get_emotion_voice_params
            emo_params = get_emotion_voice_params(emotion, voice=voice, speed=speed)
            voice = voice or emo_params.get("voice")
            speed = speed if speed is not None else emo_params.get("speed")

        if style:
            from voice_soundboard.interpreter import apply_style_to_params
            voice, speed, preset = apply_style_to_params(style, voice, speed, preset)

        if preset and preset in VOICE_PRESETS:
            preset_config = VOICE_PRESETS[preset]
            voice = voice or preset_config["voice"]
            speed = speed if speed is not None else preset_config.get("speed", 1.0)

        voice = voice or self.config.default_voice
        speed = speed if speed is not None else self.config.default_speed

        timing.interpret = time.perf_counter() - t0

        # Validate voice
        available_voices = self._kokoro.get_voices()
        if voice not in available_voices:
            raise VoiceNotFoundError(voice, available_voices)

        # Validate and clamp speed
        speed = validate_speed(speed)

        # Stage 3: Synthesize
        t0 = time.perf_counter()
        samples, sample_rate = self._kokoro.create(text, voice=voice, speed=speed)
        timing.synthesize = time.perf_counter() - t0

        # Calculate duration
        duration = len(samples) / sample_rate
        realtime_factor = duration / timing.synthesize if timing.synthesize > 0 else 0

        # Stage 4: Save
        t0 = time.perf_counter()

        if save_as:
            base_name = sanitize_filename(save_as)
            filename = base_name if base_name.endswith('.wav') else f"{base_name}.wav"
        else:
            text_hash = secure_hash(text, length=8)
            filename = f"{voice}_{text_hash}.wav"

        output_path = safe_join_path(self.config.output_dir, filename)
        sf.write(str(output_path), samples, sample_rate)

        timing.save = time.perf_counter() - t0
        timing.total = time.perf_counter() - t_total_start

        logger.debug(
            "speak() timing: total=%.3fs synthesize=%.3fs normalize=%.3fs "
            "interpret=%.3fs save=%.3fs model_load=%.3fs",
            timing.total, timing.synthesize, timing.normalize,
            timing.interpret, timing.save, timing.model_load,
        )

        return SpeechResult(
            audio_path=output_path,
            duration_seconds=duration,
            generation_time=timing.synthesize,
            voice_used=voice,
            sample_rate=sample_rate,
            realtime_factor=realtime_factor,
            timing=timing,
        )

    def speak_raw(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech and return raw audio samples.

        Args:
            text: The text to speak
            voice: Kokoro voice ID
            speed: Speech speed multiplier
            normalize: Apply text normalization for TTS edge cases (default True)

        Returns:
            Tuple of (samples array, sample_rate)
        """
        self._ensure_model_loaded()

        if normalize:
            text = normalize_text_func(text)

        voice = voice or self.config.default_voice
        speed = max(0.5, min(2.0, speed))

        return self._kokoro.create(text, voice=voice, speed=speed)

    def list_voices(self) -> list[str]:
        """Get list of available voice IDs."""
        self._ensure_model_loaded()
        return self._kokoro.get_voices()

    def list_presets(self) -> dict:
        """Get available voice presets with descriptions."""
        return {
            name: {
                "voice": config["voice"],
                "speed": config.get("speed", 1.0),
                "description": config["description"],
            }
            for name, config in VOICE_PRESETS.items()
        }

    def get_voice_info(self, voice: str) -> dict:
        """Get metadata about a specific voice."""
        if voice in KOKORO_VOICES:
            return KOKORO_VOICES[voice]
        return {"name": voice, "gender": "unknown", "accent": "unknown", "style": "unknown"}


def quick_speak(
    text: str,
    voice: str = "af_bella",
    speed: float = 1.0,
    normalize: bool = True
) -> Path:
    """
    Quick one-liner to generate speech.

    Args:
        text: Text to speak
        voice: Voice ID
        speed: Speed multiplier
        normalize: Apply text normalization (default True)

    Returns:
        Path to generated audio file
    """
    engine = VoiceEngine()
    result = engine.speak(text, voice=voice, speed=speed, normalize=normalize)
    return result.audio_path


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.DEBUG, format="%(name)s %(message)s")

    engine = VoiceEngine()

    print("\n--- Testing Voice Engine ---")
    print(f"Available presets: {list(engine.list_presets().keys())}")

    test_texts = [
        ("Hello! I'm your friendly AI assistant.", "assistant"),
        ("And so, our story begins in a distant land.", "storyteller"),
        ("Breaking news: The future has arrived.", "announcer"),
    ]

    for text, preset in test_texts:
        print(f"\n[{preset}] {text[:40]}...")
        result = engine.speak(text, preset=preset)
        print(f"  -> {result.audio_path.name} ({result.duration_seconds:.1f}s, {result.realtime_factor:.1f}x RT)")
