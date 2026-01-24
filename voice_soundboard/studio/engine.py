"""
Voice Studio Engine

Core engine for voice preview generation and parameter application.
Integrates TTS synthesis with formant shifting and humanization.
"""

import logging
from typing import Optional, Tuple
from pathlib import Path
import numpy as np

from voice_soundboard.presets.schema import VoicePreset, AcousticParams

logger = logging.getLogger(__name__)


class VoiceStudioEngine:
    """
    Core engine for voice preview generation.

    Provides the synthesis pipeline:
    1. Generate base audio with TTS engine (Kokoro)
    2. Apply formant shifting
    3. Apply humanization (breath, jitter, drift)

    Example:
        >>> engine = VoiceStudioEngine()
        >>> audio, sr = await engine.generate_preview(
        ...     preset=my_preset,
        ...     text="Hello world",
        ...     voice="af_bella"
        ... )
    """

    def __init__(self):
        self._voice_engine = None
        self._formant_shifter = None
        self._humanizer = None

    def _get_voice_engine(self):
        """Lazy-load voice engine."""
        if self._voice_engine is None:
            try:
                from voice_soundboard.engine import VoiceEngine
                self._voice_engine = VoiceEngine()
            except ImportError:
                logger.warning("VoiceEngine not available, using mock")
                self._voice_engine = None
        return self._voice_engine

    def _get_formant_shifter(self):
        """Lazy-load formant shifter."""
        if self._formant_shifter is None:
            try:
                from voice_soundboard.vocology.formants import FormantShifter
                self._formant_shifter = FormantShifter()
            except ImportError:
                logger.warning("FormantShifter not available")
                self._formant_shifter = None
        return self._formant_shifter

    def _get_humanizer(self):
        """Lazy-load humanizer."""
        if self._humanizer is None:
            try:
                from voice_soundboard.vocology.humanize import VoiceHumanizer
                self._humanizer = VoiceHumanizer()
            except ImportError:
                logger.warning("VoiceHumanizer not available")
                self._humanizer = None
        return self._humanizer

    async def generate_preview(
        self,
        preset: VoicePreset,
        text: str,
        voice: str = "af_bella",
        apply_formant: bool = True,
        apply_humanize: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate preview audio with acoustic modifications.

        Pipeline:
        1. Synthesize base audio with Kokoro
        2. Apply formant shifting (if ratio != 1.0)
        3. Apply humanization (breath, jitter, timing)

        Args:
            preset: VoicePreset with acoustic parameters
            text: Text to synthesize
            voice: Kokoro voice ID
            apply_formant: Whether to apply formant shifting
            apply_humanize: Whether to apply humanization

        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        engine = self._get_voice_engine()

        # Step 1: Synthesize base audio
        if engine is not None:
            try:
                # Get speed factor from acoustic params
                speed = 1.0
                if preset.acoustic:
                    speed = preset.acoustic.speed_factor

                audio, sr = await self._async_speak(engine, text, voice, speed)
            except Exception as e:
                logger.error(f"TTS synthesis failed: {e}")
                # Return silence on error
                sr = 24000
                audio = np.zeros(int(sr * 2), dtype=np.float32)
        else:
            # Mock: generate silence
            sr = 24000
            audio = np.zeros(int(sr * 2), dtype=np.float32)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Get acoustic params
        params = preset.acoustic or AcousticParams()

        # Step 2: Apply formant shifting
        if apply_formant and params.formant_ratio != 1.0:
            audio, sr = self._apply_formant_shift(audio, sr, params.formant_ratio)

        # Step 3: Apply humanization
        if apply_humanize:
            audio, sr = self._apply_humanization(audio, sr, params)

        return audio, sr

    async def _async_speak(self, engine, text: str, voice: str, speed: float) -> Tuple[np.ndarray, int]:
        """Async wrapper for speak (may be sync internally)."""
        import asyncio

        # Use speak_raw which returns (samples, sample_rate) directly
        if hasattr(engine, "speak_raw"):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: engine.speak_raw(text, voice=voice, speed=speed)
            )
        elif hasattr(engine, "speak_async"):
            result = await engine.speak_async(text, voice=voice, speed=speed)
            return result.samples, result.sample_rate
        else:
            # Fallback: Run sync speak in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: engine.speak(text, voice=voice, speed=speed)
            )
            # Load from file if no samples
            import soundfile as sf
            audio, sr = sf.read(str(result.audio_path))
            return audio.astype(np.float32), sr

    def _apply_formant_shift(
        self,
        audio: np.ndarray,
        sr: int,
        ratio: float,
    ) -> Tuple[np.ndarray, int]:
        """Apply formant shifting to audio."""
        shifter = self._get_formant_shifter()

        if shifter is not None:
            try:
                return shifter.shift(audio, ratio=ratio, sample_rate=sr)
            except Exception as e:
                logger.error(f"Formant shifting failed: {e}")

        return audio, sr

    def _apply_humanization(
        self,
        audio: np.ndarray,
        sr: int,
        params: AcousticParams,
    ) -> Tuple[np.ndarray, int]:
        """Apply humanization effects to audio."""
        humanizer = self._get_humanizer()

        if humanizer is not None:
            try:
                from voice_soundboard.vocology.humanize import (
                    HumanizeConfig,
                    BreathConfig,
                    PitchHumanizeConfig,
                    TimingHumanizeConfig,
                )

                # Build config from acoustic params
                config = HumanizeConfig(
                    breath=BreathConfig(
                        enabled=params.breath_intensity > 0,
                        intensity=params.breath_intensity,
                        volume_db=params.breath_volume_db,
                    ),
                    pitch=PitchHumanizeConfig(
                        enabled=True,
                        jitter_cents=params.jitter_percent * 5,  # Convert to cents
                        drift_max_cents=params.pitch_drift_cents,
                    ),
                    timing=TimingHumanizeConfig(
                        enabled=True,
                        timing_variation_ms=params.timing_variation_ms,
                    ),
                    intensity=0.5,  # Base intensity
                )

                return humanizer.humanize(audio, config=config, sample_rate=sr)

            except Exception as e:
                logger.error(f"Humanization failed: {e}")

        return audio, sr

    def apply_acoustic_params(
        self,
        audio: np.ndarray,
        params: AcousticParams,
        sample_rate: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Apply acoustic parameters to existing audio.

        Useful for re-processing audio with different settings.

        Args:
            audio: Input audio samples
            params: AcousticParams to apply
            sample_rate: Audio sample rate

        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Apply formant shifting
        if params.formant_ratio != 1.0:
            audio, sample_rate = self._apply_formant_shift(
                audio, sample_rate, params.formant_ratio
            )

        # Apply humanization
        audio, sample_rate = self._apply_humanization(audio, sample_rate, params)

        return audio, sample_rate

    def generate_preview_sync(
        self,
        preset: VoicePreset,
        text: str,
        voice: str = "af_bella",
        apply_formant: bool = True,
        apply_humanize: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Synchronous version of generate_preview.

        For use in non-async contexts.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.generate_preview(
                preset=preset,
                text=text,
                voice=voice,
                apply_formant=apply_formant,
                apply_humanize=apply_humanize,
            )
        )

    def save_preview(
        self,
        audio: np.ndarray,
        sample_rate: int,
        path: Optional[Path] = None,
    ) -> Path:
        """
        Save preview audio to file.

        Args:
            audio: Audio samples
            sample_rate: Sample rate
            path: Output path (auto-generated if None)

        Returns:
            Path to saved file
        """
        import tempfile
        import soundfile as sf

        if path is None:
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="studio_preview_")
            path = Path(path)

        sf.write(str(path), audio, sample_rate)
        return path


# Parameter validation ranges
PARAM_RANGES = {
    "formant_ratio": (0.8, 1.2),
    "breath_intensity": (0.0, 0.5),
    "breath_volume_db": (-40.0, -10.0),
    "jitter_percent": (0.0, 3.0),
    "shimmer_percent": (0.0, 10.0),
    "pitch_drift_cents": (0.0, 20.0),
    "timing_variation_ms": (0.0, 30.0),
    "speed_factor": (0.5, 2.0),
    "pitch_shift_semitones": (-12.0, 12.0),
}


def validate_params(params: dict) -> dict:
    """
    Validate and clamp parameter values to valid ranges.

    Args:
        params: Dictionary of parameter name -> value

    Returns:
        Validated and clamped parameters
    """
    validated = {}

    for key, value in params.items():
        if key in PARAM_RANGES:
            min_val, max_val = PARAM_RANGES[key]
            validated[key] = max(min_val, min(max_val, float(value)))
        else:
            logger.warning(f"Unknown parameter: {key}")

    return validated


def params_to_description(params: AcousticParams) -> str:
    """
    Generate a human-readable description of acoustic parameters.

    Useful for displaying current settings or generating voice prompts.
    """
    parts = []

    # Formant description
    if params.formant_ratio < 0.95:
        depth = "very deep" if params.formant_ratio < 0.9 else "deep"
        parts.append(depth)
    elif params.formant_ratio > 1.05:
        brightness = "very bright" if params.formant_ratio > 1.1 else "bright"
        parts.append(brightness)

    # Breath description
    if params.breath_intensity > 0.25:
        parts.append("breathy")
    elif params.breath_intensity > 0.15:
        parts.append("natural breaths")

    # Jitter/roughness
    if params.jitter_percent > 1.5:
        parts.append("gravelly")
    elif params.jitter_percent > 0.8:
        parts.append("natural texture")

    # Speed
    if params.speed_factor < 0.9:
        parts.append("slow, measured")
    elif params.speed_factor > 1.15:
        parts.append("quick, energetic")

    # Timing variation
    if params.timing_variation_ms > 15:
        parts.append("relaxed timing")
    elif params.timing_variation_ms < 5:
        parts.append("precise timing")

    if not parts:
        return "neutral, balanced voice"

    return ", ".join(parts)
