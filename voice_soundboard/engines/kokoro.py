"""
Kokoro TTS Engine Backend.

Wraps the Kokoro ONNX model (82M parameters) for fast, lightweight TTS.
This is the original engine from voice-soundboard v0.1.0.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import soundfile as sf

from voice_soundboard.engines.base import TTSEngine, EngineResult, EngineCapabilities
from voice_soundboard.config import Config, KOKORO_VOICES, VOICE_PRESETS
from voice_soundboard.security import (
    sanitize_filename,
    safe_join_path,
    validate_speed,
    secure_hash,
)


class KokoroEngine(TTSEngine):
    """
    Kokoro ONNX TTS Engine.

    A lightweight 82M parameter model providing:
    - 50+ voices across multiple accents
    - GPU acceleration via ONNX Runtime
    - Fast inference (~5x realtime on RTX 5080)

    Example:
        engine = KokoroEngine()
        result = engine.speak("Hello world!", voice="af_bella")
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._kokoro = None
        self._model_loaded = False

        # Paths to model files
        self._model_dir = Path("F:/AI/voice-soundboard/models")
        self._model_path = self._model_dir / "kokoro-v1.0.onnx"
        self._voices_path = self._model_dir / "voices-v1.0.bin"

    @property
    def name(self) -> str:
        return "kokoro"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_streaming=True,
            supports_ssml=True,  # Via our SSML preprocessor
            supports_voice_cloning=False,
            supports_emotion_control=True,  # Via emotion presets
            supports_paralinguistic_tags=False,
            supports_emotion_exaggeration=False,
            paralinguistic_tags=[],
            languages=["en", "ja", "zh"],
            typical_rtf=5.0,
            min_latency_ms=150.0,
        )

    def _ensure_model_loaded(self):
        """Lazy-load the Kokoro model on first use."""
        if self._model_loaded:
            return

        from kokoro_onnx import Kokoro

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

        self._kokoro = Kokoro(str(self._model_path), str(self._voices_path))

        self._model_loaded = True
        print(f"Kokoro model loaded in {time.time() - start:.2f}s")

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        preset: Optional[str] = None,
        style: Optional[str] = None,
        save_path: Optional[Path] = None,
        **kwargs,
    ) -> EngineResult:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Kokoro voice ID (e.g., "af_bella")
            speed: Speed multiplier (0.5-2.0)
            preset: Voice preset name
            style: Natural language style hint
            save_path: Where to save audio (auto-generated if not provided)

        Returns:
            EngineResult with audio path and metadata
        """
        self._ensure_model_loaded()

        # Apply natural language style interpretation
        if style:
            from voice_soundboard.interpreter import apply_style_to_params

            voice, speed, preset = apply_style_to_params(style, voice, speed, preset)

        # Resolve voice from preset
        if preset and preset in VOICE_PRESETS:
            preset_config = VOICE_PRESETS[preset]
            voice = voice or preset_config["voice"]
            speed = preset_config.get("speed", speed)

        # Use defaults
        voice = voice or self.config.default_voice
        speed = validate_speed(speed)

        # Validate voice
        available_voices = self._kokoro.get_voices()
        if voice not in available_voices:
            raise ValueError(
                f"Unknown voice: {voice}\n"
                f"Available: {', '.join(sorted(available_voices)[:10])}..."
            )

        # Generate speech
        start = time.time()
        samples, sample_rate = self._kokoro.create(text, voice=voice, speed=speed)
        gen_time = time.time() - start

        # Calculate metrics
        duration = len(samples) / sample_rate
        rtf = duration / gen_time if gen_time > 0 else 0

        # Determine output path
        if save_path:
            output_path = Path(save_path)
        else:
            text_hash = secure_hash(text, length=8)
            filename = f"{voice}_{text_hash}.wav"
            output_path = safe_join_path(self.config.output_dir, filename)

        # Save audio
        sf.write(str(output_path), samples, sample_rate)

        return EngineResult(
            audio_path=output_path,
            samples=samples,
            sample_rate=sample_rate,
            duration_seconds=duration,
            generation_time=gen_time,
            voice_used=voice,
            realtime_factor=rtf,
            engine_name=self.name,
            metadata={"preset": preset, "speed": speed},
        )

    def speak_raw(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech and return raw audio samples."""
        self._ensure_model_loaded()

        voice = voice or self.config.default_voice
        speed = validate_speed(speed)

        return self._kokoro.create(text, voice=voice, speed=speed)

    def list_voices(self) -> List[str]:
        """Get list of available voice IDs."""
        self._ensure_model_loaded()
        return self._kokoro.get_voices()

    def get_voice_info(self, voice: str) -> Dict[str, Any]:
        """Get metadata about a specific voice."""
        if voice in KOKORO_VOICES:
            info = KOKORO_VOICES[voice].copy()
            info["id"] = voice
            return info
        return {"id": voice, "name": voice, "gender": "unknown", "accent": "unknown"}

    def is_loaded(self) -> bool:
        return self._model_loaded

    def unload(self) -> None:
        """Unload model from memory."""
        self._kokoro = None
        self._model_loaded = False
