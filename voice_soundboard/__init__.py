"""
Voice Soundboard - AI-powered TTS with natural language control.

A user-friendly voice synthesis system that AI agents can use via MCP
to generate natural speech from text with style hints.

Supports multiple TTS backends:
- Kokoro: Fast, lightweight 82M parameter model (default)
- Chatterbox: Paralinguistic tags, emotion exaggeration, voice cloning

Example:
    from voice_soundboard import VoiceEngine
    engine = VoiceEngine()
    result = engine.speak("Hello world!", preset="assistant")

    # With Chatterbox (requires chatterbox-tts)
    from voice_soundboard.engines import ChatterboxEngine
    cb = ChatterboxEngine()
    result = cb.speak("That's hilarious! [laugh]", emotion_exaggeration=0.7)
"""

__version__ = "1.0.0"

from voice_soundboard.engine import VoiceEngine, SpeechResult, quick_speak
from voice_soundboard.config import Config, KOKORO_VOICES, VOICE_PRESETS
from voice_soundboard.audio import play_audio, stop_playback
from voice_soundboard.interpreter import interpret_style, apply_style_to_params
from voice_soundboard.effects import get_effect, play_effect, list_effects
from voice_soundboard.streaming import (
    StreamingEngine, StreamResult, stream_realtime, RealtimeStreamResult
)
from voice_soundboard.ssml import parse_ssml, ssml_to_text
from voice_soundboard.emotions import (
    get_emotion_params, get_emotion_voice_params, list_emotions, EMOTIONS
)
from voice_soundboard.dialogue import (
    DialogueParser,
    DialogueEngine,
    DialogueResult,
    DialogueLine,
    Speaker,
    ParsedScript,
    VoiceAssigner,
    auto_assign_voices,
)
from voice_soundboard.emotion import (
    EmotionParser,
    EmotionSpan,
    ParsedEmotionText,
    parse_emotion_tags,
    VADPoint,
    emotion_to_vad,
    vad_to_emotion,
    VAD_EMOTIONS,
    blend_emotions,
    EmotionMix,
    EmotionCurve,
    EmotionKeyframe,
)
from voice_soundboard.cloning import (
    VoiceCloner,
    VoiceLibrary,
    VoiceProfile,
    VoiceEmbedding,
    VoiceExtractor,
    ExtractorBackend,
    CloningResult,
    CloningConfig,
    get_default_library,
    extract_embedding,
    CrossLanguageCloner,
    LanguageConfig,
    SUPPORTED_LANGUAGES,
    detect_language,
    EmotionTimbreSeparator,
    TimbreEmbedding,
    EmotionEmbedding,
    SeparatedVoice,
    EmotionStyle,
    separate_voice,
    transfer_emotion,
)

# Engine backends
from voice_soundboard.engines import (
    TTSEngine,
    EngineResult,
    KokoroEngine,
    CHATTERBOX_AVAILABLE,
)

# Optional Chatterbox engine (requires chatterbox-tts)
if CHATTERBOX_AVAILABLE:
    from voice_soundboard.engines import ChatterboxEngine
else:
    ChatterboxEngine = None  # type: ignore

# Neural audio codecs
from voice_soundboard.codecs import (
    AudioCodec,
    CodecConfig,
    CodecCapabilities,
    CodecType,
    EncodedAudio,
    TokenSequence,
    MockCodec,
    MimiCodec,
    MIMI_AVAILABLE,
    DualCodec,
    DUALCODEC_AVAILABLE,
    LLMCodecBridge,
    AudioPrompt,
    VocabularyConfig,
    SpecialTokens,
    TokenType,
    get_codec_vocabulary_info,
    estimate_audio_context_length,
    estimate_audio_duration,
)

# Real-time voice conversion
from voice_soundboard.conversion import (
    VoiceConverter,
    ConversionConfig,
    ConversionResult,
    LatencyMode,
    ConversionState,
    StreamingConverter,
    AudioBuffer,
    ConversionPipeline,
    PipelineStage,
    AudioDevice,
    DeviceType,
    list_audio_devices,
    get_default_input_device,
    get_default_output_device,
    AudioDeviceManager,
    RealtimeConverter,
    RealtimeSession,
    start_realtime_conversion,
    ConversionCallback,
)

# Optional WebSocket server (requires websockets)
try:
    from voice_soundboard.websocket_server import VoiceWebSocketServer, create_server
    _HAS_WEBSOCKET = True
except ImportError:
    _HAS_WEBSOCKET = False

__all__ = [
    # Core
    "VoiceEngine",
    "SpeechResult",
    "Config",
    "KOKORO_VOICES",
    "VOICE_PRESETS",
    "quick_speak",
    # Engine backends
    "TTSEngine",
    "EngineResult",
    "KokoroEngine",
    "ChatterboxEngine",
    "CHATTERBOX_AVAILABLE",
    # Audio
    "play_audio",
    "stop_playback",
    # Style
    "interpret_style",
    "apply_style_to_params",
    # Effects
    "get_effect",
    "play_effect",
    "list_effects",
    # Streaming
    "StreamingEngine",
    "StreamResult",
    "stream_realtime",
    "RealtimeStreamResult",
    # SSML
    "parse_ssml",
    "ssml_to_text",
    # Emotions
    "get_emotion_params",
    "get_emotion_voice_params",
    "list_emotions",
    "EMOTIONS",
    # Dialogue (multi-speaker)
    "DialogueParser",
    "DialogueEngine",
    "DialogueResult",
    "DialogueLine",
    "Speaker",
    "ParsedScript",
    "VoiceAssigner",
    "auto_assign_voices",
    # Advanced Emotion Control
    "EmotionParser",
    "EmotionSpan",
    "ParsedEmotionText",
    "parse_emotion_tags",
    "VADPoint",
    "emotion_to_vad",
    "vad_to_emotion",
    "VAD_EMOTIONS",
    "blend_emotions",
    "EmotionMix",
    "EmotionCurve",
    "EmotionKeyframe",
    # Voice Cloning
    "VoiceCloner",
    "VoiceLibrary",
    "VoiceProfile",
    "VoiceEmbedding",
    "VoiceExtractor",
    "ExtractorBackend",
    "CloningResult",
    "CloningConfig",
    "get_default_library",
    "extract_embedding",
    "CrossLanguageCloner",
    "LanguageConfig",
    "SUPPORTED_LANGUAGES",
    "detect_language",
    "EmotionTimbreSeparator",
    "TimbreEmbedding",
    "EmotionEmbedding",
    "SeparatedVoice",
    "EmotionStyle",
    "separate_voice",
    "transfer_emotion",
    # Neural Audio Codecs
    "AudioCodec",
    "CodecConfig",
    "CodecCapabilities",
    "CodecType",
    "EncodedAudio",
    "TokenSequence",
    "MockCodec",
    "MimiCodec",
    "MIMI_AVAILABLE",
    "DualCodec",
    "DUALCODEC_AVAILABLE",
    "LLMCodecBridge",
    "AudioPrompt",
    "VocabularyConfig",
    "SpecialTokens",
    "TokenType",
    "get_codec_vocabulary_info",
    "estimate_audio_context_length",
    "estimate_audio_duration",
    # Real-time Voice Conversion
    "VoiceConverter",
    "ConversionConfig",
    "ConversionResult",
    "LatencyMode",
    "ConversionState",
    "StreamingConverter",
    "AudioBuffer",
    "ConversionPipeline",
    "PipelineStage",
    "AudioDevice",
    "DeviceType",
    "list_audio_devices",
    "get_default_input_device",
    "get_default_output_device",
    "AudioDeviceManager",
    "RealtimeConverter",
    "RealtimeSession",
    "start_realtime_conversion",
    "ConversionCallback",
    # WebSocket (optional)
    "VoiceWebSocketServer",
    "create_server",
    "_HAS_WEBSOCKET",
]
