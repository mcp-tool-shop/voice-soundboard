"""
Voice Cloning Module.

Zero-shot voice cloning from short audio samples with:
- Voice embedding extraction (3-10 seconds of audio)
- Emotion-timbre separation
- Cross-language synthesis
- Voice library management
"""

from voice_soundboard.cloning.extractor import (
    VoiceEmbedding,
    VoiceExtractor,
    ExtractorBackend,
    extract_embedding,
)
from voice_soundboard.cloning.library import (
    VoiceLibrary,
    VoiceProfile,
    get_default_library,
)
from voice_soundboard.cloning.cloner import (
    VoiceCloner,
    CloningResult,
    CloningConfig,
)
from voice_soundboard.cloning.crosslang import (
    CrossLanguageCloner,
    LanguageConfig,
    SUPPORTED_LANGUAGES,
    detect_language,
)
from voice_soundboard.cloning.separation import (
    EmotionTimbreSeparator,
    TimbreEmbedding,
    EmotionEmbedding,
    SeparatedVoice,
    EmotionStyle,
    separate_voice,
    transfer_emotion,
)

__all__ = [
    # Extractor
    "VoiceEmbedding",
    "VoiceExtractor",
    "ExtractorBackend",
    "extract_embedding",
    # Library
    "VoiceLibrary",
    "VoiceProfile",
    "get_default_library",
    # Cloner
    "VoiceCloner",
    "CloningResult",
    "CloningConfig",
    # Cross-language
    "CrossLanguageCloner",
    "LanguageConfig",
    "SUPPORTED_LANGUAGES",
    "detect_language",
    # Emotion-timbre separation
    "EmotionTimbreSeparator",
    "TimbreEmbedding",
    "EmotionEmbedding",
    "SeparatedVoice",
    "EmotionStyle",
    "separate_voice",
    "transfer_emotion",
]
