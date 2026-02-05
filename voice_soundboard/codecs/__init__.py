"""
Neural Audio Codecs Module.

.. warning:: **Experimental.** This module may change or be removed between
   minor releases. See docs/API_STABILITY.md and docs/FEATURE_FLAGS.md.

Provides audio-to-token conversion for LLM integration:
- Mimi: 12.5 Hz codec from Kyutai (CSM/Moshi)
- DualCodec: Semantic-acoustic separation
- Mock: Testing without heavy dependencies
"""

from voice_soundboard.codecs.base import (
    AudioCodec,
    CodecConfig,
    CodecCapabilities,
    CodecType,
    EncodedAudio,
    TokenSequence,
)
from voice_soundboard.codecs.mock import MockCodec
from voice_soundboard.codecs.mimi import MimiCodec, MIMI_AVAILABLE
from voice_soundboard.codecs.dualcodec import DualCodec, DUALCODEC_AVAILABLE
from voice_soundboard.codecs.llm import (
    LLMCodecBridge,
    AudioPrompt,
    VocabularyConfig,
    SpecialTokens,
    TokenType,
    get_codec_vocabulary_info,
    estimate_audio_context_length,
    estimate_audio_duration,
)

__all__ = [
    # Base
    "AudioCodec",
    "CodecConfig",
    "CodecCapabilities",
    "CodecType",
    "EncodedAudio",
    "TokenSequence",
    # Implementations
    "MockCodec",
    "MimiCodec",
    "MIMI_AVAILABLE",
    "DualCodec",
    "DUALCODEC_AVAILABLE",
    # LLM Integration
    "LLMCodecBridge",
    "AudioPrompt",
    "VocabularyConfig",
    "SpecialTokens",
    "TokenType",
    "get_codec_vocabulary_info",
    "estimate_audio_context_length",
    "estimate_audio_duration",
]
