"""
F5-TTS Engine Backend.

Integrates F5-TTS (Fairytaler that Fakes Fluent and Faithful speech with Flow matching)
with zero-shot voice cloning capabilities using Diffusion Transformer (DiT) architecture.

Features:
- Zero-shot voice cloning from 3-10 second reference audio
- High-quality DiT-based synthesis
- 0.15 RTF (6x faster than real-time)
- Requires reference audio transcription for alignment

Reference: https://github.com/SWivid/F5-TTS
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

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


class F5TTSEngine(TTSEngine):
    """
    F5-TTS Engine using Diffusion Transformer architecture.

    F5-TTS excels at zero-shot voice cloning - it can replicate any voice
    from a short (3-10 second) audio sample with high fidelity.

    Key features:
    - Flow matching with Diffusion Transformer (DiT)
    - Zero-shot voice cloning without fine-tuning
    - ConvNeXt V2 for robust text understanding
    - Sway Sampling for optimized inference

    Example:
        engine = F5TTSEngine()

        # Clone a voice and synthesize
        result = engine.speak(
            "Hello, this is my cloned voice speaking!",
            voice="path/to/reference.wav",
            ref_text="This is what I said in the reference audio.",
        )

        # Use a previously registered voice
        engine.clone_voice("path/to/sample.wav", "my_voice")
        result = engine.speak("Hello again!", voice="my_voice")
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model_variant: str = "F5TTS_v1_Base",
        device: Optional[str] = None,
    ):
        """
        Initialize F5-TTS engine.

        Args:
            config: Voice soundboard config
            model_variant: Model variant to use:
                - "F5TTS_v1_Base": Default F5-TTS model
                - "E2TTS_Base": E2-TTS variant (flat-UNet)
            device: "cuda", "cpu", or None for auto-detection
        """
        self.config = config or Config()
        self.model_variant = model_variant
        self.device = device or self.config.device

        self._model = None
        self._model_loaded = False

        # Default inference parameters
        self.default_cfg_strength = 2.0  # Reference adherence
        self.default_nfe_step = 32  # Inference steps (lower = faster)
        self.default_sway_coef = -1.0  # Sway sampling coefficient

        # Voice library for cloned voices (stores reference audio + transcription)
        self._cloned_voices: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return f"f5-tts-{self.model_variant.lower()}"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_streaming=False,  # Diffusion models generate all at once
            supports_ssml=False,
            supports_voice_cloning=True,  # Core strength
            supports_emotion_control=False,
            supports_paralinguistic_tags=False,
            supports_emotion_exaggeration=False,
            paralinguistic_tags=[],
            languages=["en", "zh"],  # F5-TTS v1 supports English and Chinese
            typical_rtf=6.0,  # ~0.15 RTF = 6x realtime on GPU
            min_latency_ms=500.0,  # Diffusion has higher initial latency
        )

    def _ensure_model_loaded(self):
        """Lazy-load the F5-TTS model on first use."""
        if self._model_loaded:
            return

        print(f"Loading F5-TTS {self.model_variant} model (device: {self.device})...")
        start = time.time()

        try:
            from f5_tts.api import F5TTS

            self._model = F5TTS(
                model=self.model_variant,
                device=self.device,
                use_ema=True,
            )

            self._model_loaded = True
            print(f"F5-TTS model loaded in {time.time() - start:.2f}s")

        except ImportError as e:
            raise ImportError(
                "F5-TTS is not installed. Install with:\n"
                "  pip install f5-tts\n"
                "Or: uv add f5-tts"
            ) from e

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        ref_text: Optional[str] = None,
        cfg_strength: Optional[float] = None,
        nfe_step: Optional[int] = None,
        seed: Optional[int] = None,
        save_path: Optional[Path] = None,
        **kwargs,
    ) -> EngineResult:
        """
        Generate speech from text using F5-TTS.

        Args:
            text: Text to synthesize
            voice: Either:
                - Path to reference audio file (3-10s recommended)
                - ID of a previously cloned voice
            speed: Speed multiplier (0.5-2.0)
            ref_text: Transcription of reference audio (required for new references)
            cfg_strength: CFG strength for reference adherence (0.0-5.0, default 2.0)
            nfe_step: Number of inference steps (16-64, default 32)
            seed: Random seed for reproducibility
            save_path: Where to save audio

        Returns:
            EngineResult with audio and metadata
        """
        self._ensure_model_loaded()

        # Resolve voice reference
        ref_audio_path = None
        ref_transcription = ref_text

        if voice:
            if voice in self._cloned_voices:
                # Use stored voice profile
                profile = self._cloned_voices[voice]
                ref_audio_path = profile["audio_path"]
                ref_transcription = ref_transcription or profile.get("transcription", "")
            elif Path(voice).exists():
                # Direct path to reference audio
                ref_audio_path = voice
            else:
                print(f"Warning: Voice '{voice}' not found, using default synthesis")

        if ref_audio_path and not ref_transcription:
            raise ValueError(
                "ref_text (transcription of reference audio) is required for F5-TTS. "
                "Provide the text that was spoken in the reference audio."
            )

        # Handle parameters
        speed = validate_speed(speed)
        cfg = cfg_strength if cfg_strength is not None else self.default_cfg_strength
        steps = nfe_step if nfe_step is not None else self.default_nfe_step

        # Generate speech
        start = time.time()

        wav, sample_rate, _ = self._model.infer(
            ref_file=ref_audio_path,
            ref_text=ref_transcription or "",
            gen_text=text,
            speed=speed,
            cfg_strength=cfg,
            nfe_step=steps,
            seed=seed,
            remove_silence=False,
        )

        gen_time = time.time() - start

        # Convert to numpy if needed
        if hasattr(wav, "numpy"):
            samples = wav.squeeze().cpu().numpy()
        elif hasattr(wav, "cpu"):
            samples = wav.squeeze().cpu().numpy()
        else:
            samples = np.array(wav).squeeze()

        # Ensure float32
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        # Calculate metrics
        duration = len(samples) / sample_rate
        rtf = duration / gen_time if gen_time > 0 else 0

        # Determine output path
        if save_path:
            output_path = Path(save_path)
        else:
            text_hash = secure_hash(text, length=8)
            voice_id = Path(voice).stem if voice and Path(voice).exists() else (voice or "default")
            filename = f"f5tts_{voice_id}_{text_hash}.wav"
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
                "cfg_strength": cfg,
                "nfe_step": steps,
                "seed": seed,
                "speed": speed,
                "has_reference": ref_audio_path is not None,
                "model_variant": self.model_variant,
            },
        )

    def speak_raw(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        ref_text: Optional[str] = None,
        cfg_strength: Optional[float] = None,
        nfe_step: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech and return raw audio samples."""
        self._ensure_model_loaded()

        # Resolve voice
        ref_audio_path = None
        ref_transcription = ref_text

        if voice:
            if voice in self._cloned_voices:
                profile = self._cloned_voices[voice]
                ref_audio_path = profile["audio_path"]
                ref_transcription = ref_transcription or profile.get("transcription", "")
            elif Path(voice).exists():
                ref_audio_path = voice

        cfg = cfg_strength if cfg_strength is not None else self.default_cfg_strength
        steps = nfe_step if nfe_step is not None else self.default_nfe_step

        wav, sample_rate, _ = self._model.infer(
            ref_file=ref_audio_path,
            ref_text=ref_transcription or "",
            gen_text=text,
            speed=validate_speed(speed),
            cfg_strength=cfg,
            nfe_step=steps,
            seed=seed,
        )

        if hasattr(wav, "numpy"):
            samples = wav.squeeze().cpu().numpy()
        elif hasattr(wav, "cpu"):
            samples = wav.squeeze().cpu().numpy()
        else:
            samples = np.array(wav).squeeze()

        return samples.astype(np.float32), sample_rate

    def list_voices(self) -> List[str]:
        """
        Get list of available voices.

        For F5-TTS, this returns registered cloned voice IDs.
        The model itself uses zero-shot cloning from reference audio.
        """
        return list(self._cloned_voices.keys())

    def get_voice_info(self, voice: str) -> Dict[str, Any]:
        """Get metadata about a cloned voice."""
        if voice in self._cloned_voices:
            profile = self._cloned_voices[voice]
            return {
                "id": voice,
                "name": voice,
                "audio_path": str(profile["audio_path"]),
                "transcription": profile.get("transcription", ""),
                "type": "cloned",
                "engine": "f5-tts",
            }
        return {"id": voice, "name": voice, "type": "unknown"}

    def clone_voice(
        self,
        audio_path: Path,
        voice_id: str = "cloned",
        transcription: Optional[str] = None,
    ) -> str:
        """
        Register a voice reference for cloning.

        F5-TTS requires both the reference audio AND its transcription
        for accurate voice cloning. The transcription helps the model
        understand the phonetic content of the reference.

        Args:
            audio_path: Path to reference audio (3-10 seconds recommended)
            voice_id: ID to assign to this voice
            transcription: What was said in the reference audio (highly recommended)

        Returns:
            Voice ID that can be used in speak() calls
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        # Validate audio duration
        try:
            info = sf.info(str(audio_path))
            duration = info.duration
            if duration < 3:
                print(f"Warning: Reference audio is only {duration:.1f}s. 3-10s recommended.")
            elif duration > 15:
                print(f"Warning: Reference audio is {duration:.1f}s. 3-10s recommended for best results.")
        except Exception:
            pass

        if not transcription:
            print(
                f"Warning: No transcription provided for '{voice_id}'. "
                "F5-TTS works best with accurate transcriptions of reference audio."
            )

        self._cloned_voices[voice_id] = {
            "audio_path": str(audio_path),
            "transcription": transcription or "",
        }

        return voice_id

    def list_cloned_voices(self) -> Dict[str, Dict[str, str]]:
        """Get dictionary of cloned voice IDs to their profiles."""
        return {
            k: {"audio_path": v["audio_path"], "transcription": v.get("transcription", "")}
            for k, v in self._cloned_voices.items()
        }

    def remove_cloned_voice(self, voice_id: str) -> bool:
        """Remove a cloned voice from the library."""
        if voice_id in self._cloned_voices:
            del self._cloned_voices[voice_id]
            return True
        return False

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


# Convenience function for direct use
def speak_f5tts(
    text: str,
    voice: Optional[str] = None,
    ref_text: Optional[str] = None,
    speed: float = 1.0,
    **kwargs,
) -> Path:
    """
    Quick one-liner to generate speech with F5-TTS.

    Args:
        text: Text to speak
        voice: Path to reference audio or registered voice ID
        ref_text: Transcription of reference audio
        speed: Speed multiplier (0.5-2.0)

    Returns:
        Path to generated audio file
    """
    engine = F5TTSEngine()
    result = engine.speak(
        text,
        voice=voice,
        ref_text=ref_text,
        speed=speed,
        **kwargs,
    )
    return result.audio_path
