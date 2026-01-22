"""
Voice Engine - Core TTS synthesis using Kokoro ONNX.

Provides high-quality text-to-speech with GPU acceleration.
"""

from __future__ import annotations

import os
import time
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from voice_soundboard.config import Config, KOKORO_VOICES, VOICE_PRESETS
from voice_soundboard.security import (
    sanitize_filename, safe_join_path, validate_text_input,
    validate_speed, secure_hash, safe_error_message
)


@dataclass
class SpeechResult:
    """Result from speech synthesis."""
    audio_path: Path
    duration_seconds: float
    generation_time: float
    voice_used: str
    sample_rate: int
    realtime_factor: float  # How many times faster than realtime


class VoiceEngine:
    """
    AI Voice Engine using Kokoro TTS.

    Provides natural speech synthesis with:
    - 50+ voices across multiple languages
    - GPU acceleration via ONNX Runtime
    - Voice presets (assistant, narrator, etc.)
    - Speed control

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

        # Paths to model files
        self._model_dir = Path("F:/AI/voice-soundboard/models")
        self._model_path = self._model_dir / "kokoro-v1.0.onnx"
        self._voices_path = self._model_dir / "voices-v1.0.bin"

    def _ensure_model_loaded(self):
        """Lazy-load the Kokoro model on first use."""
        if self._model_loaded:
            return

        from kokoro_onnx import Kokoro

        # Check model files exist
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self._model_path}\n"
                "Download from: https://github.com/thewh1teagle/kokoro-onnx/releases"
            )
        if not self._voices_path.exists():
            raise FileNotFoundError(
                f"Voices not found: {self._voices_path}\n"
                "Download from: https://github.com/thewh1teagle/kokoro-onnx/releases"
            )

        print(f"Loading Kokoro model (device: {self.config.device})...")
        start = time.time()

        self._kokoro = Kokoro(
            str(self._model_path),
            str(self._voices_path)
        )

        self._model_loaded = True
        print(f"Model loaded in {time.time() - start:.2f}s")

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        preset: Optional[str] = None,
        speed: Optional[float] = None,
        style: Optional[str] = None,
        save_as: Optional[str] = None,
    ) -> SpeechResult:
        """
        Generate speech from text.

        Args:
            text: The text to speak
            voice: Kokoro voice ID (e.g., "af_bella", "am_michael")
            preset: Voice preset name (e.g., "assistant", "narrator")
            speed: Speech speed multiplier (0.5-2.0, default 1.0)
            style: Natural language style hint (e.g., "warmly", "excitedly")
            save_as: Optional filename (auto-generated if not provided)

        Returns:
            SpeechResult with audio path and metadata

        Example:
            result = engine.speak("Hello!", voice="af_bella", speed=1.1)
            result = engine.speak("Hello!", style="warmly and cheerfully")
        """
        self._ensure_model_loaded()

        # Apply natural language style interpretation
        if style:
            from voice_soundboard.interpreter import apply_style_to_params
            voice, speed, preset = apply_style_to_params(style, voice, speed, preset)

        # Resolve voice from preset if provided
        if preset and preset in VOICE_PRESETS:
            preset_config = VOICE_PRESETS[preset]
            voice = voice or preset_config["voice"]
            speed = speed if speed is not None else preset_config.get("speed", 1.0)

        # Use defaults if not specified
        voice = voice or self.config.default_voice
        speed = speed if speed is not None else self.config.default_speed

        # Validate voice
        available_voices = self._kokoro.get_voices()
        if voice not in available_voices:
            raise ValueError(
                f"Unknown voice: {voice}\n"
                f"Available: {', '.join(sorted(available_voices)[:10])}..."
            )

        # Validate and clamp speed
        speed = validate_speed(speed)

        # Generate speech
        start = time.time()
        samples, sample_rate = self._kokoro.create(text, voice=voice, speed=speed)
        gen_time = time.time() - start

        # Calculate duration
        duration = len(samples) / sample_rate
        realtime_factor = duration / gen_time if gen_time > 0 else 0

        # Generate output filename with security sanitization
        if save_as:
            # SECURITY: Sanitize user-provided filename to prevent path traversal
            base_name = sanitize_filename(save_as)
            filename = base_name if base_name.endswith('.wav') else f"{base_name}.wav"
        else:
            # Create unique filename from text hash (using SHA-256, not MD5)
            text_hash = secure_hash(text, length=8)
            filename = f"{voice}_{text_hash}.wav"

        # SECURITY: Safe path join prevents traversal attacks
        output_path = safe_join_path(self.config.output_dir, filename)

        # Save audio
        sf.write(str(output_path), samples, sample_rate)

        return SpeechResult(
            audio_path=output_path,
            duration_seconds=duration,
            generation_time=gen_time,
            voice_used=voice,
            sample_rate=sample_rate,
            realtime_factor=realtime_factor,
        )

    def speak_raw(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech and return raw audio samples.

        Args:
            text: The text to speak
            voice: Kokoro voice ID
            speed: Speech speed multiplier

        Returns:
            Tuple of (samples array, sample_rate)
        """
        self._ensure_model_loaded()

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


def quick_speak(text: str, voice: str = "af_bella", speed: float = 1.0) -> Path:
    """
    Quick one-liner to generate speech.

    Args:
        text: Text to speak
        voice: Voice ID
        speed: Speed multiplier

    Returns:
        Path to generated audio file
    """
    engine = VoiceEngine()
    result = engine.speak(text, voice=voice, speed=speed)
    return result.audio_path


if __name__ == "__main__":
    # Quick test
    engine = VoiceEngine()

    print("\n--- Testing Voice Engine ---")
    print(f"Available presets: {list(engine.list_presets().keys())}")

    # Test different presets
    test_texts = [
        ("Hello! I'm your friendly AI assistant.", "assistant"),
        ("And so, our story begins in a distant land.", "storyteller"),
        ("Breaking news: The future has arrived.", "announcer"),
    ]

    for text, preset in test_texts:
        print(f"\n[{preset}] {text[:40]}...")
        result = engine.speak(text, preset=preset)
        print(f"  -> {result.audio_path.name} ({result.duration_seconds:.1f}s, {result.realtime_factor:.1f}x RT)")
