"""
Voice Soundboard TTS Engine Backends.

This module provides multiple TTS engine backends with a unified interface:
- KokoroEngine: Original 82M parameter Kokoro ONNX model (fast, lightweight)
- ChatterboxEngine: Resemble AI's Chatterbox with paralinguistic tags and emotion control

Example:
    from voice_soundboard.engines import ChatterboxEngine

    engine = ChatterboxEngine()
    result = engine.speak(
        "That's hilarious! [laugh] Oh man, I needed that.",
        emotion_exaggeration=0.7
    )
"""

from voice_soundboard.engines.base import TTSEngine, EngineResult
from voice_soundboard.engines.kokoro import KokoroEngine

# Chatterbox is optional - only import if available
try:
    from voice_soundboard.engines.chatterbox import ChatterboxEngine
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    ChatterboxEngine = None  # type: ignore

__all__ = [
    "TTSEngine",
    "EngineResult",
    "KokoroEngine",
    "ChatterboxEngine",
    "CHATTERBOX_AVAILABLE",
]
