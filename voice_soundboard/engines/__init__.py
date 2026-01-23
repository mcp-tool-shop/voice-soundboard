"""
Voice Soundboard TTS Engine Backends.

This module provides multiple TTS engine backends with a unified interface:
- KokoroEngine: Original 82M parameter Kokoro ONNX model (fast, lightweight)
- ChatterboxEngine: Resemble AI's Chatterbox with paralinguistic tags and emotion control
- F5TTSEngine: Diffusion Transformer for high-quality zero-shot voice cloning

Example:
    from voice_soundboard.engines import ChatterboxEngine, F5TTSEngine

    # Chatterbox with paralinguistic tags
    engine = ChatterboxEngine()
    result = engine.speak(
        "That's hilarious! [laugh] Oh man, I needed that.",
        emotion_exaggeration=0.7
    )

    # F5-TTS for voice cloning
    f5_engine = F5TTSEngine()
    result = f5_engine.speak(
        "Hello in my cloned voice!",
        voice="reference.wav",
        ref_text="This is the reference audio transcription."
    )
"""

from voice_soundboard.engines.base import TTSEngine, EngineResult, EngineCapabilities
from voice_soundboard.engines.kokoro import KokoroEngine

# Chatterbox is optional - only import if available
try:
    from voice_soundboard.engines.chatterbox import ChatterboxEngine
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    ChatterboxEngine = None  # type: ignore

# F5-TTS is optional - only import if available
try:
    from voice_soundboard.engines.f5tts import F5TTSEngine
    F5TTS_AVAILABLE = True
except ImportError:
    F5TTS_AVAILABLE = False
    F5TTSEngine = None  # type: ignore

__all__ = [
    "TTSEngine",
    "EngineResult",
    "EngineCapabilities",
    "KokoroEngine",
    "ChatterboxEngine",
    "CHATTERBOX_AVAILABLE",
    "F5TTSEngine",
    "F5TTS_AVAILABLE",
]
