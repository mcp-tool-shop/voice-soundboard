"""
MCP Server for Voice Soundboard.

Exposes TTS and sound effects capabilities to AI agents via MCP.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from voice_soundboard.engine import VoiceEngine
from voice_soundboard.config import KOKORO_VOICES, VOICE_PRESETS
from voice_soundboard.audio import play_audio, stop_playback, get_audio_duration
from voice_soundboard.interpreter import interpret_style, apply_style_to_params
from voice_soundboard.effects import get_effect, list_effects, EFFECTS
from voice_soundboard.ssml import parse_ssml
from voice_soundboard.emotions import (
    get_emotion_params, get_emotion_voice_params,
    apply_emotion_to_text, list_emotions, EMOTIONS
)
from voice_soundboard.streaming import stream_realtime, RealtimeStreamResult
from voice_soundboard.dialogue import (
    DialogueParser,
    DialogueEngine,
    auto_assign_voices,
)
from voice_soundboard.emotion import (
    EmotionParser,
    parse_emotion_tags,
    blend_emotions,
    EmotionCurve,
    VAD_EMOTIONS,
    emotion_to_vad,
    list_named_blends,
    get_named_blend,
    list_narrative_curves,
    get_narrative_curve,
)
from voice_soundboard.cloning import (
    VoiceCloner,
    VoiceLibrary,
    get_default_library,
    CrossLanguageCloner,
    SUPPORTED_LANGUAGES,
    EmotionTimbreSeparator,
    detect_language,
)
from voice_soundboard.codecs import (
    MockCodec,
    MimiCodec,
    DualCodec,
    MIMI_AVAILABLE,
    DUALCODEC_AVAILABLE,
    get_codec_vocabulary_info,
    estimate_audio_context_length,
)
from voice_soundboard.conversion import (
    RealtimeConverter,
    VoiceConverter,
    MockVoiceConverter,
    ConversionConfig,
    LatencyMode,
    list_audio_devices,
    DeviceType,
)
from voice_soundboard.llm import (
    SpeechPipeline,
    PipelineConfig,
    StreamingLLMSpeaker,
    StreamConfig,
    ContextAwareSpeaker,
    ContextConfig,
    ConversationManager,
    ConversationConfig,
    create_provider,
    LLMConfig,
    MockLLMProvider,
)
from voice_soundboard.engines import CHATTERBOX_AVAILABLE
if CHATTERBOX_AVAILABLE:
    from voice_soundboard.engines.chatterbox import (
        ChatterboxEngine,
        PARALINGUISTIC_TAGS,
        has_paralinguistic_tags,
    )


# Global engine instances (lazy loaded)
_engine: VoiceEngine | None = None
_chatterbox_engine = None
_f5tts_engine = None
_dialogue_engine: DialogueEngine | None = None
_voice_cloner: VoiceCloner | None = None
_emotion_separator: EmotionTimbreSeparator | None = None
_audio_codec = None
_realtime_converter: RealtimeConverter | None = None
_speech_pipeline: SpeechPipeline | None = None
_conversation_manager: ConversationManager | None = None
_context_speaker: ContextAwareSpeaker | None = None


def get_engine() -> VoiceEngine:
    """Get or create the Kokoro voice engine singleton."""
    global _engine
    if _engine is None:
        try:
            logger.debug("Initializing Kokoro voice engine...")
            _engine = VoiceEngine()
            logger.info("Kokoro voice engine initialized")
        except Exception as e:
            logger.error("Failed to initialize voice engine: %s", e)
            raise RuntimeError(f"Voice engine initialization failed: {e}") from e
    return _engine


def get_chatterbox_engine():
    """Get or create the Chatterbox engine singleton."""
    global _chatterbox_engine
    if _chatterbox_engine is None:
        if not CHATTERBOX_AVAILABLE:
            raise ImportError(
                "Chatterbox is not installed. Install with:\n"
                "  pip install voice-soundboard[chatterbox]\n"
                "Or: pip install chatterbox-tts"
            )
        try:
            logger.debug("Initializing Chatterbox engine...")
            _chatterbox_engine = ChatterboxEngine()
            logger.info("Chatterbox engine initialized")
        except Exception as e:
            logger.error("Failed to initialize Chatterbox engine: %s", e)
            raise RuntimeError(f"Chatterbox engine initialization failed: {e}") from e
    return _chatterbox_engine


def get_f5tts_engine():
    """Get or create the F5-TTS engine singleton."""
    global _f5tts_engine
    if _f5tts_engine is None:
        try:
            from voice_soundboard.engines.f5tts import F5TTSEngine
            logger.debug("Initializing F5-TTS engine...")
            _f5tts_engine = F5TTSEngine()
            logger.info("F5-TTS engine initialized")
        except ImportError as e:
            raise ImportError(
                "F5-TTS is not installed. Install with:\n"
                "  pip install voice-soundboard[f5tts]\n"
                "Or: pip install f5-tts"
            ) from e
        except Exception as e:
            logger.error("Failed to initialize F5-TTS engine: %s", e)
            raise RuntimeError(f"F5-TTS engine initialization failed: {e}") from e
    return _f5tts_engine


def get_dialogue_engine() -> DialogueEngine:
    """Get or create the Dialogue engine singleton."""
    global _dialogue_engine
    if _dialogue_engine is None:
        try:
            logger.debug("Initializing dialogue engine...")
            _dialogue_engine = DialogueEngine(voice_engine=get_engine())
            logger.debug("Dialogue engine initialized")
        except Exception as e:
            logger.error("Failed to initialize dialogue engine: %s", e)
            raise RuntimeError(f"Dialogue engine initialization failed: {e}") from e
    return _dialogue_engine


def get_voice_cloner() -> VoiceCloner:
    """Get or create the VoiceCloner singleton."""
    global _voice_cloner
    if _voice_cloner is None:
        try:
            logger.debug("Initializing voice cloner...")
            _voice_cloner = VoiceCloner()
            logger.debug("Voice cloner initialized")
        except Exception as e:
            logger.error("Failed to initialize voice cloner: %s", e)
            raise RuntimeError(f"Voice cloner initialization failed: {e}") from e
    return _voice_cloner


def get_emotion_separator() -> EmotionTimbreSeparator:
    """Get or create the EmotionTimbreSeparator singleton."""
    global _emotion_separator
    if _emotion_separator is None:
        try:
            logger.debug("Initializing emotion separator...")
            _emotion_separator = EmotionTimbreSeparator()
            logger.debug("Emotion separator initialized")
        except Exception as e:
            logger.error("Failed to initialize emotion separator: %s", e)
            raise RuntimeError(f"Emotion separator initialization failed: {e}") from e
    return _emotion_separator


def get_audio_codec(codec_type: str = "mock"):
    """Get or create an audio codec."""
    global _audio_codec
    if _audio_codec is None or getattr(_audio_codec, "name", "") != codec_type:
        try:
            logger.debug("Initializing %s codec...", codec_type)
            if codec_type == "mimi":
                _audio_codec = MimiCodec()
            elif codec_type == "dualcodec":
                _audio_codec = DualCodec()
            else:
                _audio_codec = MockCodec()
            logger.debug("%s codec initialized", codec_type)
        except Exception as e:
            logger.error("Failed to initialize %s codec: %s", codec_type, e)
            raise RuntimeError(f"Codec initialization failed: {e}") from e
    return _audio_codec


def get_realtime_converter(latency_mode: str = "balanced") -> RealtimeConverter:
    """Get or create the RealtimeConverter singleton."""
    global _realtime_converter
    if _realtime_converter is None:
        try:
            mode_map = {
                "ultra_low": LatencyMode.ULTRA_LOW,
                "low": LatencyMode.LOW,
                "balanced": LatencyMode.BALANCED,
                "high_quality": LatencyMode.HIGH_QUALITY,
            }
            mode = mode_map.get(latency_mode.lower(), LatencyMode.BALANCED)
            config = ConversionConfig(latency_mode=mode)
            logger.debug("Initializing realtime converter (mode=%s)...", latency_mode)
            _realtime_converter = RealtimeConverter(config=config)
            logger.debug("Realtime converter initialized")
        except Exception as e:
            logger.error("Failed to initialize realtime converter: %s", e)
            raise RuntimeError(f"Realtime converter initialization failed: {e}") from e
    return _realtime_converter


def get_speech_pipeline(
    llm_backend: str = "mock",
    llm_model: str = "llama3.2",
    system_prompt: str | None = None,
) -> SpeechPipeline:
    """Get or create the SpeechPipeline singleton."""
    global _speech_pipeline
    if _speech_pipeline is None:
        try:
            config = PipelineConfig(
                llm_backend=llm_backend,
                llm_model=llm_model,
                system_prompt=system_prompt or "You are a helpful voice assistant. Keep responses concise.",
            )
            logger.debug("Initializing speech pipeline (backend=%s)...", llm_backend)
            _speech_pipeline = SpeechPipeline(config=config)
            logger.debug("Speech pipeline initialized")
        except Exception as e:
            logger.error("Failed to initialize speech pipeline: %s", e)
            raise RuntimeError(f"Speech pipeline initialization failed: {e}") from e
    return _speech_pipeline


def get_conversation_manager() -> ConversationManager:
    """Get or create the ConversationManager singleton."""
    global _conversation_manager
    if _conversation_manager is None:
        try:
            config = ConversationConfig()
            logger.debug("Initializing conversation manager...")
            _conversation_manager = ConversationManager(config=config)
            logger.debug("Conversation manager initialized")
        except Exception as e:
            logger.error("Failed to initialize conversation manager: %s", e)
            raise RuntimeError(f"Conversation manager initialization failed: {e}") from e
    return _conversation_manager


def get_context_speaker() -> ContextAwareSpeaker:
    """Get or create the ContextAwareSpeaker singleton."""
    global _context_speaker
    if _context_speaker is None:
        try:
            config = ContextConfig(enable_auto_emotion=True)
            logger.debug("Initializing context speaker...")
            _context_speaker = ContextAwareSpeaker(config=config)
            logger.debug("Context speaker initialized")
        except Exception as e:
            logger.error("Failed to initialize context speaker: %s", e)
            raise RuntimeError(f"Context speaker initialization failed: {e}") from e
    return _context_speaker


# Create MCP server
server = Server("voice-soundboard")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available voice tools."""
    return [
        Tool(
            name="speak",
            description=(
                "Generate natural speech from text. Returns path to audio file. "
                "Use 'style' for natural language hints like 'warmly', 'excitedly', "
                "'like a narrator'. Use 'voice' for specific voice IDs, 'preset' for "
                "predefined personalities (assistant, narrator, announcer, storyteller, whisper)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to speak"
                    },
                    "style": {
                        "type": "string",
                        "description": "Natural language style hint: 'warmly', 'excitedly', 'like a narrator', 'in a british accent'"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Specific voice ID (e.g., 'af_bella', 'bm_george'). Overrides style."
                    },
                    "preset": {
                        "type": "string",
                        "enum": list(VOICE_PRESETS.keys()),
                        "description": "Voice preset: assistant, narrator, announcer, storyteller, whisper"
                    },
                    "speed": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 2.0,
                        "description": "Speech speed multiplier (0.5-2.0, default 1.0)"
                    },
                    "play": {
                        "type": "boolean",
                        "description": "Play audio immediately after generation (default: false)"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="list_voices",
            description="List all available voices with their characteristics (gender, accent, style)",
            inputSchema={
                "type": "object",
                "properties": {
                    "filter_gender": {
                        "type": "string",
                        "enum": ["male", "female"],
                        "description": "Filter by gender"
                    },
                    "filter_accent": {
                        "type": "string",
                        "enum": ["american", "british", "japanese", "mandarin"],
                        "description": "Filter by accent"
                    }
                }
            }
        ),
        Tool(
            name="list_presets",
            description="List available voice presets with descriptions",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="play_audio",
            description="Play an audio file through speakers",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to audio file"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="stop_audio",
            description="Stop any currently playing audio",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="sound_effect",
            description=(
                "Play a sound effect. Available effects: chime, success, error, attention, "
                "click, pop, whoosh, warning, critical, info, rain, white_noise, drone"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "effect": {
                        "type": "string",
                        "description": "Effect name: chime, success, error, attention, click, pop, whoosh, warning, critical, info, rain, white_noise, drone"
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Optional path to save the effect (plays by default)"
                    }
                },
                "required": ["effect"]
            }
        ),
        Tool(
            name="list_effects",
            description="List all available sound effects",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="speak_long",
            description=(
                "Stream speech for long text (paragraphs, articles). "
                "More efficient for text longer than a few sentences. "
                "Saves to file and optionally plays."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Long text to speak"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice ID"
                    },
                    "preset": {
                        "type": "string",
                        "enum": list(VOICE_PRESETS.keys()),
                        "description": "Voice preset"
                    },
                    "speed": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 2.0,
                        "description": "Speed multiplier"
                    },
                    "play": {
                        "type": "boolean",
                        "description": "Play after generation (default: false)"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="speak_ssml",
            description=(
                "Speak text with SSML markup for fine control over pauses, emphasis, "
                "and pronunciation. Supports: <break time='500ms'/>, "
                "<emphasis level='strong'>text</emphasis>, "
                "<say-as interpret-as='date'>2024-01-15</say-as>, "
                "<prosody rate='slow'>text</prosody>, <sub alias='replacement'>abbr</sub>"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ssml": {
                        "type": "string",
                        "description": "SSML-formatted text (with or without <speak> wrapper)"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice ID"
                    },
                    "preset": {
                        "type": "string",
                        "enum": list(VOICE_PRESETS.keys()),
                        "description": "Voice preset"
                    },
                    "play": {
                        "type": "boolean",
                        "description": "Play after generation (default: false)"
                    }
                },
                "required": ["ssml"]
            }
        ),
        Tool(
            name="speak_realtime",
            description=(
                "Stream speech with real-time playback. Audio plays immediately as it generates, "
                "no waiting for the full file. Best for interactive responses."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to speak"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice ID"
                    },
                    "preset": {
                        "type": "string",
                        "enum": list(VOICE_PRESETS.keys()),
                        "description": "Voice preset"
                    },
                    "emotion": {
                        "type": "string",
                        "enum": list(EMOTIONS.keys()),
                        "description": "Emotion to apply (happy, sad, excited, calm, angry, etc.)"
                    },
                    "speed": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 2.0,
                        "description": "Speed multiplier"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="list_emotions",
            description="List all available emotions for speech synthesis with their characteristics",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        # Chatterbox tools (advanced features)
        Tool(
            name="speak_chatterbox",
            description=(
                "Generate expressive multilingual speech with Chatterbox TTS. Supports 23 languages, "
                "paralinguistic tags like [laugh], [sigh], [cough], [gasp], [chuckle] for natural "
                "non-speech sounds, emotion exaggeration control (0.0=monotone to 1.0=dramatic), "
                "and voice cloning from reference audio."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to speak. Can include tags like [laugh], [sigh], [cough], [gasp], [chuckle]"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Path to reference audio (3-10s) for voice cloning, or ID of previously cloned voice"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh. Default: en"
                    },
                    "emotion_exaggeration": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Expressiveness level: 0.0=monotone, 0.5=normal (default), 1.0=dramatic"
                    },
                    "cfg_weight": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Reference speaker adherence. Lower=slower pacing. Default 0.5"
                    },
                    "play": {
                        "type": "boolean",
                        "description": "Play audio immediately after generation (default: false)"
                    }
                },
                "required": ["text"]
            }
        ),
        # F5-TTS tools (high-quality voice cloning)
        Tool(
            name="speak_f5tts",
            description=(
                "Generate high-quality speech with F5-TTS using Diffusion Transformer. "
                "Excels at zero-shot voice cloning from 3-10 second reference audio. "
                "Requires transcription of reference audio for best results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to synthesize"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Path to reference audio (3-10s) for voice cloning, or ID of previously cloned voice"
                    },
                    "ref_text": {
                        "type": "string",
                        "description": "Transcription of the reference audio (required for new references)"
                    },
                    "speed": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 2.0,
                        "description": "Speed multiplier (default: 1.0)"
                    },
                    "cfg_strength": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 5.0,
                        "description": "Reference adherence strength (default: 2.0)"
                    },
                    "nfe_step": {
                        "type": "integer",
                        "minimum": 8,
                        "maximum": 64,
                        "description": "Inference steps. Higher=better quality, slower (default: 32)"
                    },
                    "play": {
                        "type": "boolean",
                        "description": "Play audio immediately after generation (default: false)"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="clone_voice_f5tts",
            description=(
                "Register a voice for F5-TTS cloning. Unlike Chatterbox, F5-TTS requires "
                "transcription of the reference audio for accurate voice cloning."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Path to reference audio file (3-10s recommended)"
                    },
                    "voice_id": {
                        "type": "string",
                        "description": "ID to assign to this voice (default: 'cloned')"
                    },
                    "transcription": {
                        "type": "string",
                        "description": "What is spoken in the reference audio (highly recommended)"
                    }
                },
                "required": ["audio_path"]
            }
        ),
        Tool(
            name="list_chatterbox_languages",
            description="List all 23 languages supported by Chatterbox multilingual model",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="clone_voice",
            description=(
                "Register a voice for cloning from a reference audio sample. "
                "Recommended duration: 3-10 seconds of clear speech. "
                "Returns a voice ID that can be used with speak_chatterbox."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Path to reference audio file (WAV, MP3, etc.)"
                    },
                    "voice_id": {
                        "type": "string",
                        "description": "ID to assign to this voice (default: 'cloned')"
                    }
                },
                "required": ["audio_path"]
            }
        ),
        Tool(
            name="list_cloned_voices",
            description="List all registered cloned voices",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_paralinguistic_tags",
            description="List all supported paralinguistic tags for Chatterbox ([laugh], [sigh], etc.)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        # Dialogue tools (multi-speaker)
        Tool(
            name="speak_dialogue",
            description=(
                "Synthesize multi-speaker dialogue from a script. Supports speaker tags like "
                "[S1:narrator], [S2:alice], stage directions like (whispering), (angrily), "
                "and paralinguistic tags. Automatically assigns distinct voices to each speaker. "
                "Returns a single audio file with all speakers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": (
                            "Dialogue script with speaker tags. Format:\n"
                            "[S1:narrator] The door creaked open.\n"
                            "[S2:alice] (nervously) Hello? [gasp]\n"
                            "[S3:bob] (whispering) Don't go in there..."
                        )
                    },
                    "voices": {
                        "type": "object",
                        "description": (
                            "Optional voice assignments: {speaker_name: voice_id}. "
                            "Unassigned speakers get auto-assigned voices. "
                            "Example: {\"narrator\": \"bm_george\", \"alice\": \"af_bella\"}"
                        ),
                        "additionalProperties": {"type": "string"}
                    },
                    "play": {
                        "type": "boolean",
                        "description": "Play audio immediately after generation (default: false)"
                    }
                },
                "required": ["script"]
            }
        ),
        Tool(
            name="preview_dialogue",
            description=(
                "Preview voice assignments and script info without synthesizing. "
                "Useful for checking speaker detection and auto-assignment before generation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Dialogue script to preview"
                    },
                    "voices": {
                        "type": "object",
                        "description": "Optional voice assignments to preview",
                        "additionalProperties": {"type": "string"}
                    }
                },
                "required": ["script"]
            }
        ),
        # Advanced Emotion Control tools
        Tool(
            name="blend_emotions",
            description=(
                "Blend multiple emotions with weights to create nuanced expressions. "
                "Returns VAD values (Valence-Arousal-Dominance) and closest named emotion. "
                "Example: blend 70% happy + 30% surprised for pleasant surprise."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "emotions": {
                        "type": "array",
                        "description": "List of {emotion, weight} objects. Weights auto-normalize.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "emotion": {"type": "string"},
                                "weight": {"type": "number", "minimum": 0, "maximum": 1}
                            },
                            "required": ["emotion", "weight"]
                        }
                    }
                },
                "required": ["emotions"]
            }
        ),
        Tool(
            name="parse_emotion_text",
            description=(
                "Parse text with inline emotion tags like {happy}text{/happy}. "
                "Returns plain text and emotion spans with positions. "
                "Supports intensity: {excited:0.8}text{/excited}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text with emotion tags: I'm {happy}so glad{/happy} to see you!"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="get_emotion_vad",
            description=(
                "Get VAD (Valence-Arousal-Dominance) values for an emotion. "
                "Valence: -1 (negative) to +1 (positive). "
                "Arousal: 0 (calm) to 1 (excited). "
                "Dominance: 0 (submissive) to 1 (dominant)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "description": "Emotion name (happy, sad, angry, excited, etc.)"
                    }
                },
                "required": ["emotion"]
            }
        ),
        Tool(
            name="list_emotion_blends",
            description="List pre-defined emotion blends like 'bittersweet' (happy+sad), 'nervous_excitement', etc.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_narrative_curves",
            description=(
                "List pre-built emotion curves for narrative patterns like 'tension_build', "
                "'joy_arc', 'suspense', 'revelation', etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="sample_emotion_curve",
            description=(
                "Sample an emotion curve at multiple points. Curves can be named presets "
                "(tension_build, joy_arc) or custom keyframes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "curve_name": {
                        "type": "string",
                        "description": "Named curve preset (tension_build, joy_arc, suspense, etc.)"
                    },
                    "keyframes": {
                        "type": "array",
                        "description": "Custom keyframes as [{position, emotion}] (alternative to curve_name)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "position": {"type": "number", "minimum": 0, "maximum": 1},
                                "emotion": {"type": "string"}
                            },
                            "required": ["position", "emotion"]
                        }
                    },
                    "num_samples": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 20,
                        "description": "Number of sample points (default: 5)"
                    }
                }
            }
        ),
        # Voice Cloning tools
        Tool(
            name="clone_voice_advanced",
            description=(
                "Clone a voice from an audio sample (3-10 seconds). Extracts speaker embedding "
                "and saves to voice library. Requires consent acknowledgment for ethical use. "
                "Returns a voice ID that can be used for synthesis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Path to audio file (WAV, MP3, etc.) with 3-10 seconds of clear speech"
                    },
                    "voice_id": {
                        "type": "string",
                        "description": "Unique ID for this voice (e.g., 'my_voice', 'narrator_clone')"
                    },
                    "name": {
                        "type": "string",
                        "description": "Display name for the voice"
                    },
                    "consent_given": {
                        "type": "boolean",
                        "description": "Acknowledge consent for voice cloning (required)"
                    },
                    "consent_notes": {
                        "type": "string",
                        "description": "Notes about consent (e.g., 'self', 'permission from speaker')"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for organization (e.g., ['narrator', 'male', 'deep'])"
                    },
                    "gender": {
                        "type": "string",
                        "enum": ["male", "female", "neutral"],
                        "description": "Speaker gender (for auto-assignment)"
                    },
                    "language": {
                        "type": "string",
                        "description": "Primary language (e.g., 'en', 'es', 'zh')"
                    }
                },
                "required": ["audio_path", "voice_id", "consent_given"]
            }
        ),
        Tool(
            name="validate_clone_audio",
            description=(
                "Validate an audio file for voice cloning without cloning. "
                "Returns quality metrics and recommendations for better results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Path to audio file to validate"
                    }
                },
                "required": ["audio_path"]
            }
        ),
        Tool(
            name="list_voice_library",
            description=(
                "List all cloned voices in the voice library with metadata. "
                "Supports filtering by tags, gender, language, and quality."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search text in name/description"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags (any match)"
                    },
                    "gender": {
                        "type": "string",
                        "enum": ["male", "female", "neutral"],
                        "description": "Filter by gender"
                    },
                    "language": {
                        "type": "string",
                        "description": "Filter by language"
                    },
                    "min_quality": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Minimum quality rating"
                    }
                }
            }
        ),
        Tool(
            name="get_voice_profile",
            description="Get detailed information about a cloned voice profile.",
            inputSchema={
                "type": "object",
                "properties": {
                    "voice_id": {
                        "type": "string",
                        "description": "Voice ID to get info for"
                    }
                },
                "required": ["voice_id"]
            }
        ),
        Tool(
            name="delete_cloned_voice",
            description="Delete a cloned voice from the library.",
            inputSchema={
                "type": "object",
                "properties": {
                    "voice_id": {
                        "type": "string",
                        "description": "Voice ID to delete"
                    }
                },
                "required": ["voice_id"]
            }
        ),
        Tool(
            name="find_similar_voices",
            description=(
                "Find voices in the library similar to a given audio sample. "
                "Useful for finding existing voices that match a reference."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Path to reference audio"
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum number of similar voices to return (default: 5)"
                    }
                },
                "required": ["audio_path"]
            }
        ),
        Tool(
            name="transfer_voice_emotion",
            description=(
                "Apply a different emotion to a cloned voice. Separates timbre (voice identity) "
                "from emotion and recombines with new emotion."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "voice_id": {
                        "type": "string",
                        "description": "ID of the cloned voice"
                    },
                    "emotion": {
                        "type": "string",
                        "description": "Emotion to apply (happy, sad, angry, excited, calm, etc.)"
                    },
                    "intensity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 2,
                        "description": "Emotion intensity multiplier (default: 1.0)"
                    }
                },
                "required": ["voice_id", "emotion"]
            }
        ),
        Tool(
            name="list_cloning_languages",
            description="List supported languages for cross-language voice cloning.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="check_language_compatibility",
            description=(
                "Check compatibility between source and target languages for voice cloning. "
                "Returns expected quality and recommendations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_language": {
                        "type": "string",
                        "description": "Source language code (e.g., 'en', 'es')"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "Target language code"
                    }
                },
                "required": ["source_language", "target_language"]
            }
        ),
        # Neural Audio Codec tools
        Tool(
            name="encode_audio_tokens",
            description=(
                "Encode audio to discrete tokens for LLM integration. "
                "Converts audio file to token sequence that can be used as LLM input. "
                "Supports Mimi (12.5 Hz), DualCodec (semantic-acoustic), and Mock (testing)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Path to audio file to encode"
                    },
                    "codec": {
                        "type": "string",
                        "enum": ["mock", "mimi", "dualcodec"],
                        "description": "Codec to use (default: mock)"
                    },
                    "include_llm_tokens": {
                        "type": "boolean",
                        "description": "Return LLM-ready tokens with special markers (default: true)"
                    }
                },
                "required": ["audio_path"]
            }
        ),
        Tool(
            name="decode_audio_tokens",
            description=(
                "Decode tokens back to audio. "
                "Converts token sequence (from LLM output) to audio file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tokens": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Token sequence to decode"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save decoded audio (WAV format)"
                    },
                    "codec": {
                        "type": "string",
                        "enum": ["mock", "mimi", "dualcodec"],
                        "description": "Codec to use (must match encoding codec)"
                    }
                },
                "required": ["tokens", "output_path"]
            }
        ),
        Tool(
            name="get_codec_info",
            description=(
                "Get information about available audio codecs. "
                "Returns vocabulary size, frame rate, and capabilities for LLM integration."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "codec": {
                        "type": "string",
                        "enum": ["mock", "mimi", "dualcodec"],
                        "description": "Codec to get info for (default: mock)"
                    }
                }
            }
        ),
        Tool(
            name="estimate_audio_tokens",
            description=(
                "Estimate number of tokens needed for audio of given duration. "
                "Useful for planning LLM context usage."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "duration_seconds": {
                        "type": "number",
                        "description": "Audio duration in seconds"
                    },
                    "codec": {
                        "type": "string",
                        "enum": ["mock", "mimi", "dualcodec"],
                        "description": "Codec to estimate for"
                    }
                },
                "required": ["duration_seconds"]
            }
        ),
        Tool(
            name="voice_convert_dualcodec",
            description=(
                "Convert voice using DualCodec's semantic-acoustic separation. "
                "Takes content from one audio and style/timbre from another."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content_audio": {
                        "type": "string",
                        "description": "Audio file providing content/words"
                    },
                    "style_audio": {
                        "type": "string",
                        "description": "Audio file providing voice style/timbre"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save converted audio"
                    }
                },
                "required": ["content_audio", "style_audio", "output_path"]
            }
        ),
        # Real-time Voice Conversion tools
        Tool(
            name="list_audio_devices",
            description=(
                "List available audio input/output devices for voice conversion. "
                "Returns microphones, speakers, headsets, and other audio devices."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "device_type": {
                        "type": "string",
                        "description": "Filter by type: 'input' (microphones), 'output' (speakers), or 'all'",
                        "enum": ["input", "output", "all"]
                    }
                }
            }
        ),
        Tool(
            name="start_voice_conversion",
            description=(
                "Start real-time voice conversion from microphone to speakers. "
                "Converts your voice to sound like the target voice in real-time. "
                "Call stop_voice_conversion when done."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "target_voice": {
                        "type": "string",
                        "description": "Target voice ID from voice library, or path to reference audio"
                    },
                    "latency_mode": {
                        "type": "string",
                        "description": "Latency/quality trade-off: ultra_low (~60ms), low (~100ms), balanced (~150ms), high_quality (~300ms)",
                        "enum": ["ultra_low", "low", "balanced", "high_quality"]
                    },
                    "input_device": {
                        "type": "string",
                        "description": "Input device name or ID (default: system microphone)"
                    },
                    "output_device": {
                        "type": "string",
                        "description": "Output device name or ID (default: system speakers)"
                    }
                },
                "required": ["target_voice"]
            }
        ),
        Tool(
            name="stop_voice_conversion",
            description=(
                "Stop real-time voice conversion and return session statistics. "
                "Returns latency, quality metrics, and duration information."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_voice_conversion_status",
            description=(
                "Get current status and statistics of the voice conversion session. "
                "Returns whether conversion is running, current latency, and processing stats."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="convert_audio_file",
            description=(
                "Convert an audio file to a different voice (non-realtime). "
                "Processes an entire file and saves the converted audio."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to input audio file"
                    },
                    "target_voice": {
                        "type": "string",
                        "description": "Target voice ID or path to reference audio"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save converted audio"
                    },
                    "preserve_pitch": {
                        "type": "boolean",
                        "description": "Preserve original pitch (default: true)"
                    }
                },
                "required": ["input_path", "target_voice", "output_path"]
            }
        ),
        # LLM Integration tools
        Tool(
            name="speak_with_context",
            description=(
                "Speak text with context-aware prosody. Automatically selects "
                "appropriate emotion based on conversation context and content. "
                "Best for empathetic, natural-sounding responses."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to speak"
                    },
                    "context": {
                        "type": "string",
                        "description": "Conversation context (e.g., 'User expressed frustration')"
                    },
                    "auto_emotion": {
                        "type": "boolean",
                        "description": "Automatically select emotion from context (default: true)"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice ID override"
                    },
                    "preset": {
                        "type": "string",
                        "enum": list(VOICE_PRESETS.keys()),
                        "description": "Voice preset"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="start_conversation",
            description=(
                "Start a new voice conversation session. Sets up the conversation "
                "manager with an optional system prompt and configuration."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt for the conversation"
                    },
                    "llm_backend": {
                        "type": "string",
                        "enum": ["ollama", "openai", "vllm", "mock"],
                        "description": "LLM backend to use (default: mock)"
                    },
                    "llm_model": {
                        "type": "string",
                        "description": "LLM model name (default: llama3.2)"
                    }
                }
            }
        ),
        Tool(
            name="add_conversation_message",
            description=(
                "Add a message to the conversation history. Use this to record "
                "user messages or assistant responses for context tracking."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant"],
                        "description": "Message role"
                    },
                    "content": {
                        "type": "string",
                        "description": "Message content"
                    },
                    "emotion": {
                        "type": "string",
                        "description": "Detected or applied emotion"
                    }
                },
                "required": ["role", "content"]
            }
        ),
        Tool(
            name="get_conversation_context",
            description=(
                "Get the current conversation context for LLM input. Returns "
                "formatted messages suitable for chat completion APIs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "max_messages": {
                        "type": "integer",
                        "description": "Maximum messages to include (default: 10)"
                    }
                }
            }
        ),
        Tool(
            name="get_conversation_stats",
            description="Get statistics about the current conversation.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="end_conversation",
            description=(
                "End the current conversation session and optionally save "
                "the conversation history."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "save_path": {
                        "type": "string",
                        "description": "Path to save conversation history (optional)"
                    }
                }
            }
        ),
        Tool(
            name="detect_user_emotion",
            description=(
                "Detect the user's emotion from their message. Returns sentiment "
                "and emotion for context-aware responses."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "User message to analyze"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="select_response_emotion",
            description=(
                "Select the best emotion for a response based on context and content. "
                "Helps determine how to speak a response naturally."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "response_text": {
                        "type": "string",
                        "description": "The response text to analyze"
                    },
                    "user_emotion": {
                        "type": "string",
                        "description": "The detected user emotion (optional)"
                    }
                },
                "required": ["response_text"]
            }
        ),
        Tool(
            name="list_llm_providers",
            description="List available LLM providers and their status.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "speak":
        return await handle_speak(arguments)
    elif name == "list_voices":
        return await handle_list_voices(arguments)
    elif name == "list_presets":
        return await handle_list_presets(arguments)
    elif name == "play_audio":
        return await handle_play_audio(arguments)
    elif name == "stop_audio":
        return await handle_stop_audio(arguments)
    elif name == "sound_effect":
        return await handle_sound_effect(arguments)
    elif name == "list_effects":
        return await handle_list_effects(arguments)
    elif name == "speak_long":
        return await handle_speak_long(arguments)
    elif name == "speak_ssml":
        return await handle_speak_ssml(arguments)
    elif name == "speak_realtime":
        return await handle_speak_realtime(arguments)
    elif name == "list_emotions":
        return await handle_list_emotions(arguments)
    # Chatterbox tools
    elif name == "speak_chatterbox":
        return await handle_speak_chatterbox(arguments)
    elif name == "clone_voice":
        return await handle_clone_voice(arguments)
    elif name == "list_cloned_voices":
        return await handle_list_cloned_voices(arguments)
    elif name == "list_chatterbox_languages":
        return await handle_list_chatterbox_languages(arguments)
    # F5-TTS tools
    elif name == "speak_f5tts":
        return await handle_speak_f5tts(arguments)
    elif name == "clone_voice_f5tts":
        return await handle_clone_voice_f5tts(arguments)
    elif name == "list_paralinguistic_tags":
        return await handle_list_paralinguistic_tags(arguments)
    # Dialogue tools
    elif name == "speak_dialogue":
        return await handle_speak_dialogue(arguments)
    elif name == "preview_dialogue":
        return await handle_preview_dialogue(arguments)
    # Advanced Emotion Control tools
    elif name == "blend_emotions":
        return await handle_blend_emotions(arguments)
    elif name == "parse_emotion_text":
        return await handle_parse_emotion_text(arguments)
    elif name == "get_emotion_vad":
        return await handle_get_emotion_vad(arguments)
    elif name == "list_emotion_blends":
        return await handle_list_emotion_blends(arguments)
    elif name == "list_narrative_curves":
        return await handle_list_narrative_curves(arguments)
    elif name == "sample_emotion_curve":
        return await handle_sample_emotion_curve(arguments)
    # Voice Cloning tools
    elif name == "clone_voice_advanced":
        return await handle_clone_voice_advanced(arguments)
    elif name == "validate_clone_audio":
        return await handle_validate_clone_audio(arguments)
    elif name == "list_voice_library":
        return await handle_list_voice_library(arguments)
    elif name == "get_voice_profile":
        return await handle_get_voice_profile(arguments)
    elif name == "delete_cloned_voice":
        return await handle_delete_cloned_voice(arguments)
    elif name == "find_similar_voices":
        return await handle_find_similar_voices(arguments)
    elif name == "transfer_voice_emotion":
        return await handle_transfer_voice_emotion(arguments)
    elif name == "list_cloning_languages":
        return await handle_list_cloning_languages(arguments)
    elif name == "check_language_compatibility":
        return await handle_check_language_compatibility(arguments)
    # Neural Audio Codec tools
    elif name == "encode_audio_tokens":
        return await handle_encode_audio_tokens(arguments)
    elif name == "decode_audio_tokens":
        return await handle_decode_audio_tokens(arguments)
    elif name == "get_codec_info":
        return await handle_get_codec_info(arguments)
    elif name == "estimate_audio_tokens":
        return await handle_estimate_audio_tokens(arguments)
    elif name == "voice_convert_dualcodec":
        return await handle_voice_convert_dualcodec(arguments)
    # Real-time Voice Conversion tools
    elif name == "list_audio_devices":
        return await handle_list_audio_devices(arguments)
    elif name == "start_voice_conversion":
        return await handle_start_voice_conversion(arguments)
    elif name == "stop_voice_conversion":
        return await handle_stop_voice_conversion(arguments)
    elif name == "get_voice_conversion_status":
        return await handle_get_voice_conversion_status(arguments)
    elif name == "convert_audio_file":
        return await handle_convert_audio_file(arguments)
    # LLM Integration tools
    elif name == "speak_with_context":
        return await handle_speak_with_context(arguments)
    elif name == "start_conversation":
        return await handle_start_conversation(arguments)
    elif name == "add_conversation_message":
        return await handle_add_conversation_message(arguments)
    elif name == "get_conversation_context":
        return await handle_get_conversation_context(arguments)
    elif name == "get_conversation_stats":
        return await handle_get_conversation_stats(arguments)
    elif name == "end_conversation":
        return await handle_end_conversation(arguments)
    elif name == "detect_user_emotion":
        return await handle_detect_user_emotion(arguments)
    elif name == "select_response_emotion":
        return await handle_select_response_emotion(arguments)
    elif name == "list_llm_providers":
        return await handle_list_llm_providers(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_speak(args: dict[str, Any]) -> list[TextContent]:
    """Generate speech from text."""
    text = args.get("text", "")
    if not text:
        return [TextContent(type="text", text="Error: 'text' is required")]

    style = args.get("style")
    voice = args.get("voice")
    preset = args.get("preset")
    speed = args.get("speed")
    should_play = args.get("play", False)

    # Apply style interpretation
    if style:
        voice, speed, preset = apply_style_to_params(style, voice, speed, preset)

    try:
        engine = get_engine()
        result = engine.speak(
            text=text,
            voice=voice,
            preset=preset,
            speed=speed,
        )

        # Play if requested
        if should_play:
            await asyncio.to_thread(play_audio, result.audio_path)

        response = (
            f"Generated speech:\n"
            f"  File: {result.audio_path}\n"
            f"  Voice: {result.voice_used}\n"
            f"  Duration: {result.duration_seconds:.2f}s\n"
            f"  Speed: {result.realtime_factor:.1f}x realtime"
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating speech: {e}")]


async def handle_list_voices(args: dict[str, Any]) -> list[TextContent]:
    """List available voices with optional filtering."""
    filter_gender = args.get("filter_gender")
    filter_accent = args.get("filter_accent")

    voices = []
    for voice_id, info in sorted(KOKORO_VOICES.items()):
        # Apply filters
        if filter_gender and info.get("gender") != filter_gender:
            continue
        if filter_accent and info.get("accent") != filter_accent:
            continue

        voices.append(
            f"  {voice_id}: {info['name']} ({info['gender']}, {info['accent']}, {info['style']})"
        )

    if not voices:
        return [TextContent(type="text", text="No voices match the filters")]

    response = f"Available voices ({len(voices)}):\n" + "\n".join(voices)
    return [TextContent(type="text", text=response)]


async def handle_list_presets(args: dict[str, Any]) -> list[TextContent]:
    """List voice presets."""
    lines = ["Voice presets:"]
    for name, config in VOICE_PRESETS.items():
        lines.append(
            f"  {name}: {config['description']} (voice: {config['voice']}, speed: {config.get('speed', 1.0)})"
        )
    return [TextContent(type="text", text="\n".join(lines))]


async def handle_play_audio(args: dict[str, Any]) -> list[TextContent]:
    """Play an audio file."""
    path = args.get("path")
    if not path:
        return [TextContent(type="text", text="Error: 'path' is required")]

    path = Path(path)
    if not path.exists():
        return [TextContent(type="text", text=f"Error: File not found: {path}")]

    try:
        duration = get_audio_duration(path)
        await asyncio.to_thread(play_audio, path)
        return [TextContent(type="text", text=f"Played: {path.name} ({duration:.2f}s)")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error playing audio: {e}")]


async def handle_stop_audio(args: dict[str, Any]) -> list[TextContent]:
    """Stop audio playback."""
    try:
        stop_playback()
        return [TextContent(type="text", text="Audio playback stopped")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error stopping audio: {e}")]


async def handle_sound_effect(args: dict[str, Any]) -> list[TextContent]:
    """Play or save a sound effect."""
    effect_name = args.get("effect", "")
    if not effect_name:
        return [TextContent(type="text", text="Error: 'effect' is required")]

    save_path = args.get("save_path")

    try:
        effect = get_effect(effect_name)

        if save_path:
            path = Path(save_path)
            effect.save(path)
            return [TextContent(type="text", text=f"Saved effect '{effect_name}' to: {path}")]
        else:
            # Play the effect
            await asyncio.to_thread(effect.play)
            return [TextContent(type="text", text=f"Played effect: {effect_name} ({effect.duration:.2f}s)")]

    except ValueError as e:
        return [TextContent(type="text", text=str(e))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error with sound effect: {e}")]


async def handle_list_effects(args: dict[str, Any]) -> list[TextContent]:
    """List all available sound effects."""
    effects = list_effects()

    categories = {
        "Chimes": ["chime", "chime_success", "chime_error", "chime_attention"],
        "UI": ["click", "pop", "whoosh"],
        "Alerts": ["alert_warning", "alert_critical", "alert_info"],
        "Ambient": ["rain", "white_noise", "drone"],
    }

    lines = ["Available sound effects:"]
    for category, items in categories.items():
        lines.append(f"\n  {category}:")
        for item in items:
            if item in EFFECTS:
                lines.append(f"    - {item}")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_speak_long(args: dict[str, Any]) -> list[TextContent]:
    """Stream speech for long text."""
    text = args.get("text", "")
    if not text:
        return [TextContent(type="text", text="Error: 'text' is required")]

    voice = args.get("voice")
    preset = args.get("preset")
    speed = args.get("speed", 1.0)
    should_play = args.get("play", False)

    try:
        from voice_soundboard.streaming import StreamingEngine
        import hashlib

        engine = StreamingEngine()

        # Generate output filename
        text_hash = hashlib.md5(text[:100].encode()).hexdigest()[:8]
        output_path = Path("F:/AI/voice-soundboard/output") / f"stream_{text_hash}.wav"

        result = await engine.stream_to_file(
            text=text,
            output_path=output_path,
            voice=voice,
            preset=preset,
            speed=speed,
        )

        if should_play:
            await asyncio.to_thread(play_audio, result.audio_path)

        response = (
            f"Streamed speech:\n"
            f"  File: {result.audio_path}\n"
            f"  Voice: {result.voice_used}\n"
            f"  Duration: {result.total_duration:.2f}s\n"
            f"  Chunks: {result.total_chunks}\n"
            f"  Gen time: {result.generation_time:.2f}s"
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error streaming speech: {e}")]


async def handle_speak_ssml(args: dict[str, Any]) -> list[TextContent]:
    """Speak text with SSML markup."""
    ssml = args.get("ssml", "")
    if not ssml:
        return [TextContent(type="text", text="Error: 'ssml' is required")]

    voice = args.get("voice")
    preset = args.get("preset")
    should_play = args.get("play", False)

    try:
        # Parse SSML to text and extract parameters
        text, ssml_params = parse_ssml(ssml)

        # Use SSML-extracted parameters if not overridden
        if ssml_params.voice and not voice:
            voice = ssml_params.voice
        speed = ssml_params.speed

        engine = get_engine()
        result = engine.speak(
            text=text,
            voice=voice,
            preset=preset,
            speed=speed,
        )

        if should_play:
            await asyncio.to_thread(play_audio, result.audio_path)

        response = (
            f"Generated SSML speech:\n"
            f"  File: {result.audio_path}\n"
            f"  Voice: {result.voice_used}\n"
            f"  Duration: {result.duration_seconds:.2f}s\n"
            f"  Speed: {speed}x\n"
            f"  Processed text: {text[:80]}{'...' if len(text) > 80 else ''}"
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error with SSML speech: {e}")]


async def handle_speak_realtime(args: dict[str, Any]) -> list[TextContent]:
    """Stream speech with real-time playback."""
    text = args.get("text", "")
    if not text:
        return [TextContent(type="text", text="Error: 'text' is required")]

    voice = args.get("voice")
    preset = args.get("preset")
    emotion = args.get("emotion")
    speed = args.get("speed")

    try:
        # Apply emotion if specified
        if emotion:
            emotion_params = get_emotion_voice_params(emotion, voice, speed)
            voice = emotion_params["voice"]
            speed = emotion_params["speed"]
            # Optionally modify text for emotional emphasis
            text = apply_emotion_to_text(text, emotion)

        # Stream with real-time playback
        result = await stream_realtime(
            text=text,
            voice=voice,
            preset=preset,
            speed=speed or 1.0,
        )

        emotion_str = f"\n  Emotion: {emotion}" if emotion else ""
        response = (
            f"Real-time speech completed:\n"
            f"  Duration: {result.total_duration:.2f}s\n"
            f"  Chunks: {result.total_chunks}\n"
            f"  Voice: {result.voice_used}\n"
            f"  Gen time: {result.generation_time:.2f}s{emotion_str}"
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error with realtime speech: {e}")]


async def handle_list_emotions(args: dict[str, Any]) -> list[TextContent]:
    """List available emotions."""
    lines = ["Available emotions:"]

    # Group by category
    categories = {
        "Positive": ["happy", "excited", "joyful"],
        "Calm": ["calm", "peaceful", "neutral"],
        "Negative": ["sad", "melancholy", "angry", "frustrated"],
        "High-energy": ["fearful", "surprised", "urgent"],
        "Professional": ["confident", "serious", "professional"],
        "Storytelling": ["mysterious", "dramatic", "whimsical"],
    }

    for category, emotions in categories.items():
        lines.append(f"\n  {category}:")
        for emotion in emotions:
            if emotion in EMOTIONS:
                params = EMOTIONS[emotion]
                voice = params.voice_preference or "default"
                lines.append(f"    {emotion}: speed={params.speed:.2f}, voice={voice}")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_speak_chatterbox(args: dict[str, Any]) -> list[TextContent]:
    """Generate expressive multilingual speech with Chatterbox."""
    text = args.get("text", "")
    if not text:
        return [TextContent(type="text", text="Error: 'text' is required")]

    voice = args.get("voice")
    language = args.get("language", "en")
    emotion_exaggeration = args.get("emotion_exaggeration")
    cfg_weight = args.get("cfg_weight")
    should_play = args.get("play", False)

    try:
        engine = get_chatterbox_engine()

        result = engine.speak(
            text=text,
            voice=voice,
            language=language,
            emotion_exaggeration=emotion_exaggeration,
            cfg_weight=cfg_weight,
        )

        # Play if requested
        if should_play:
            await asyncio.to_thread(play_audio, result.audio_path)

        # Build response with metadata
        tags_str = ""
        if result.metadata.get("paralinguistic_tags"):
            tags_str = f"\n  Tags used: {result.metadata['paralinguistic_tags']}"

        response = (
            f"Generated Chatterbox speech:\n"
            f"  File: {result.audio_path}\n"
            f"  Duration: {result.duration_seconds:.2f}s\n"
            f"  Speed: {result.realtime_factor:.1f}x realtime\n"
            f"  Language: {result.metadata.get('language', 'en')}\n"
            f"  Emotion exaggeration: {result.metadata.get('emotion_exaggeration', 0.5)}\n"
            f"  CFG weight: {result.metadata.get('cfg_weight', 0.5)}"
            f"{tags_str}"
        )
        return [TextContent(type="text", text=response)]

    except ImportError as e:
        return [TextContent(type="text", text=str(e))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating Chatterbox speech: {e}")]


async def handle_clone_voice(args: dict[str, Any]) -> list[TextContent]:
    """Register a voice for cloning."""
    audio_path = args.get("audio_path")
    if not audio_path:
        return [TextContent(type="text", text="Error: 'audio_path' is required")]

    voice_id = args.get("voice_id", "cloned")

    try:
        engine = get_chatterbox_engine()
        registered_id = engine.clone_voice(Path(audio_path), voice_id)

        return [TextContent(
            type="text",
            text=f"Voice registered successfully!\n"
                 f"  ID: {registered_id}\n"
                 f"  Reference: {audio_path}\n"
                 f"  Use with: speak_chatterbox(text, voice='{registered_id}')"
        )]

    except ImportError as e:
        return [TextContent(type="text", text=str(e))]
    except FileNotFoundError as e:
        return [TextContent(type="text", text=f"Error: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error cloning voice: {e}")]


async def handle_list_cloned_voices(args: dict[str, Any]) -> list[TextContent]:
    """List registered cloned voices."""
    try:
        engine = get_chatterbox_engine()
        voices = engine.list_cloned_voices()

        if not voices:
            return [TextContent(
                type="text",
                text="No cloned voices registered.\n"
                     "Use clone_voice(audio_path) to register a voice."
            )]

        lines = ["Cloned voices:"]
        for voice_id, path in voices.items():
            lines.append(f"  {voice_id}: {path}")

        return [TextContent(type="text", text="\n".join(lines))]

    except ImportError as e:
        return [TextContent(type="text", text=str(e))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error listing voices: {e}")]


async def handle_list_paralinguistic_tags(args: dict[str, Any]) -> list[TextContent]:
    """List supported paralinguistic tags."""
    if not CHATTERBOX_AVAILABLE:
        return [TextContent(
            type="text",
            text="Chatterbox is not installed. Install with:\n"
                 "  pip install voice-soundboard[chatterbox]"
        )]

    lines = [
        "Supported paralinguistic tags for Chatterbox:",
        "",
        "These tags generate natural non-speech sounds in the speaker's voice:",
        ""
    ]

    tag_descriptions = {
        "laugh": "Full laughter",
        "chuckle": "Light, brief laugh",
        "cough": "Clearing throat or coughing",
        "sigh": "Exhale expressing emotion",
        "gasp": "Sharp intake of breath (surprise/shock)",
        "groan": "Sound of displeasure or effort",
        "sniff": "Nasal sound",
        "shush": "Quieting sound",
        "clear throat": "Throat clearing",
    }

    for tag in PARALINGUISTIC_TAGS:
        desc = tag_descriptions.get(tag, "")
        lines.append(f"  [{tag}] - {desc}")

    lines.extend([
        "",
        "Example usage:",
        '  speak_chatterbox("That\'s hilarious! [laugh] Oh man, [sigh] I needed that.")'
    ])

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_list_chatterbox_languages(args: dict[str, Any]) -> list[TextContent]:
    """List all 23 languages supported by Chatterbox multilingual."""
    if not CHATTERBOX_AVAILABLE:
        return [TextContent(
            type="text",
            text="Chatterbox is not installed. Install with:\n"
                 "  pip install voice-soundboard[chatterbox]"
        )]

    from voice_soundboard.engines.chatterbox import CHATTERBOX_LANGUAGES

    language_names = {
        "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
        "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
        "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
        "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
        "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
        "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
    }

    lines = [
        "Chatterbox Multilingual - 23 Supported Languages:",
        "",
    ]

    for code in sorted(CHATTERBOX_LANGUAGES):
        name = language_names.get(code, code)
        lines.append(f"  {code} - {name}")

    lines.extend([
        "",
        "Example usage:",
        '  speak_chatterbox("Bonjour, comment allez-vous?", language="fr")',
        '  speak_chatterbox("Guten Tag!", language="de")',
        '  speak_chatterbox("こんにちは", language="ja")',
    ])

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_speak_f5tts(args: dict[str, Any]) -> list[TextContent]:
    """Generate high-quality speech with F5-TTS voice cloning."""
    text = args.get("text", "")
    if not text:
        return [TextContent(type="text", text="Error: 'text' is required")]

    voice = args.get("voice")
    ref_text = args.get("ref_text")
    speed = args.get("speed", 1.0)
    cfg_strength = args.get("cfg_strength")
    nfe_step = args.get("nfe_step")
    should_play = args.get("play", False)

    try:
        engine = get_f5tts_engine()

        result = engine.speak(
            text=text,
            voice=voice,
            ref_text=ref_text,
            speed=speed,
            cfg_strength=cfg_strength,
            nfe_step=nfe_step,
        )

        # Play if requested
        if should_play:
            await asyncio.to_thread(play_audio, result.audio_path)

        response = (
            f"Generated F5-TTS speech:\n"
            f"  File: {result.audio_path}\n"
            f"  Duration: {result.duration_seconds:.2f}s\n"
            f"  Speed: {result.realtime_factor:.1f}x realtime\n"
            f"  CFG strength: {result.metadata.get('cfg_strength', 2.0)}\n"
            f"  NFE steps: {result.metadata.get('nfe_step', 32)}\n"
            f"  Voice reference: {'Yes' if result.metadata.get('has_reference') else 'No'}"
        )
        return [TextContent(type="text", text=response)]

    except ImportError as e:
        return [TextContent(
            type="text",
            text="F5-TTS is not installed. Install with:\n"
                 "  pip install voice-soundboard[f5tts]\n"
                 f"Error: {e}"
        )]
    except ValueError as e:
        return [TextContent(type="text", text=f"Error: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating F5-TTS speech: {e}")]


async def handle_clone_voice_f5tts(args: dict[str, Any]) -> list[TextContent]:
    """Register a voice for F5-TTS cloning with transcription."""
    audio_path = args.get("audio_path")
    if not audio_path:
        return [TextContent(type="text", text="Error: 'audio_path' is required")]

    voice_id = args.get("voice_id", "cloned")
    transcription = args.get("transcription")

    try:
        engine = get_f5tts_engine()
        registered_id = engine.clone_voice(
            Path(audio_path),
            voice_id,
            transcription=transcription,
        )

        trans_note = ""
        if not transcription:
            trans_note = "\n  Note: No transcription provided. For best results, add transcription."

        return [TextContent(
            type="text",
            text=f"Voice registered for F5-TTS!\n"
                 f"  ID: {registered_id}\n"
                 f"  Reference: {audio_path}\n"
                 f"  Transcription: {transcription or '(not provided)'}\n"
                 f"  Use with: speak_f5tts(text, voice='{registered_id}')"
                 f"{trans_note}"
        )]

    except ImportError as e:
        return [TextContent(
            type="text",
            text="F5-TTS is not installed. Install with:\n"
                 "  pip install voice-soundboard[f5tts]"
        )]
    except FileNotFoundError as e:
        return [TextContent(type="text", text=f"Error: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error cloning voice: {e}")]


async def handle_speak_dialogue(args: dict[str, Any]) -> list[TextContent]:
    """Synthesize multi-speaker dialogue."""
    script = args.get("script", "")
    if not script:
        return [TextContent(type="text", text="Error: 'script' is required")]

    voices = args.get("voices", {})
    should_play = args.get("play", False)

    try:
        import hashlib

        engine = get_dialogue_engine()

        # Generate output filename
        script_hash = hashlib.md5(script[:100].encode()).hexdigest()[:8]
        output_path = Path("F:/AI/voice-soundboard/output") / f"dialogue_{script_hash}.wav"
        output_path.parent.mkdir(exist_ok=True)

        result = engine.synthesize(
            script=script,
            voices=voices,
            output_path=output_path,
        )

        # Play if requested
        if should_play:
            await asyncio.to_thread(play_audio, result.audio_path)

        # Format voice assignments
        voice_lines = []
        for speaker, voice in result.voice_assignments.items():
            voice_lines.append(f"    {speaker}: {voice}")

        # Format per-speaker stats
        speaker_stats = []
        for turn in result.turns:
            if turn.speaker_name not in [s.split(":")[0] for s in speaker_stats]:
                duration = result.get_speaker_duration(turn.speaker_name)
                speaker_stats.append(f"    {turn.speaker_name}: {duration:.2f}s")

        response = (
            f"Generated dialogue:\n"
            f"  File: {result.audio_path}\n"
            f"  Duration: {result.duration_seconds:.2f}s\n"
            f"  Speakers: {result.speaker_count}\n"
            f"  Lines: {result.line_count}\n"
            f"\n  Voice assignments:\n" + "\n".join(voice_lines) +
            f"\n\n  Speaking time:\n" + "\n".join(speaker_stats)
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating dialogue: {e}")]


async def handle_preview_dialogue(args: dict[str, Any]) -> list[TextContent]:
    """Preview dialogue script without synthesizing."""
    script = args.get("script", "")
    if not script:
        return [TextContent(type="text", text="Error: 'script' is required")]

    voices = args.get("voices", {})

    try:
        engine = get_dialogue_engine()

        # Get script info
        info = engine.get_script_info(script)

        # Preview voice assignments
        assignments = engine.preview_assignments(script, voices)

        # Format speaker info
        speaker_lines = []
        for speaker, voice in assignments.items():
            lines_count = info["speaker_lines"].get(speaker, 0)
            speaker_lines.append(f"    {speaker}: {voice} ({lines_count} lines)")

        response = (
            f"Dialogue preview:\n"
            f"  Title: {info.get('title') or '(untitled)'}\n"
            f"  Speakers: {info['speaker_count']}\n"
            f"  Total lines: {info['line_count']}\n"
            f"  Word count: {info['total_words']}\n"
            f"  Est. duration: {info['estimated_duration_seconds']:.1f}s\n"
            f"\n  Voice assignments:\n" + "\n".join(speaker_lines)
        )

        if info.get("metadata"):
            meta_lines = [f"    {k}: {v}" for k, v in info["metadata"].items()]
            response += f"\n\n  Metadata:\n" + "\n".join(meta_lines)

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error previewing dialogue: {e}")]


async def handle_blend_emotions(args: dict[str, Any]) -> list[TextContent]:
    """Blend multiple emotions with weights."""
    emotions_data = args.get("emotions", [])
    if not emotions_data:
        return [TextContent(type="text", text="Error: 'emotions' array is required")]

    try:
        # Convert to list of tuples
        emotion_weights = [
            (e["emotion"], e["weight"])
            for e in emotions_data
        ]

        result = blend_emotions(emotion_weights)

        # Format response
        components = " + ".join(f"{e}:{w:.0%}" for e, w in result.components)
        response = (
            f"Emotion blend result:\n"
            f"  Components: {components}\n"
            f"  Dominant emotion: {result.dominant_emotion}\n"
            f"  Secondary: {', '.join(result.secondary_emotions)}\n"
            f"  Intensity: {result.intensity:.2f}\n"
            f"\n  VAD values:\n"
            f"    Valence: {result.vad.valence:+.2f} (negative to positive)\n"
            f"    Arousal: {result.vad.arousal:.2f} (calm to excited)\n"
            f"    Dominance: {result.vad.dominance:.2f} (submissive to dominant)"
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error blending emotions: {e}")]


async def handle_parse_emotion_text(args: dict[str, Any]) -> list[TextContent]:
    """Parse text with inline emotion tags."""
    text = args.get("text", "")
    if not text:
        return [TextContent(type="text", text="Error: 'text' is required")]

    try:
        result = parse_emotion_tags(text)

        if not result.has_emotion_tags():
            return [TextContent(
                type="text",
                text=f"No emotion tags found.\nPlain text: {result.plain_text}"
            )]

        span_lines = []
        for span in result.spans:
            intensity = f" (intensity: {span.intensity})" if span.intensity != 1.0 else ""
            span_lines.append(
                f"    \"{span.text}\" → {span.emotion}{intensity} "
                f"[chars {span.start_char}-{span.end_char}]"
            )

        response = (
            f"Parsed emotion text:\n"
            f"  Plain text: {result.plain_text}\n"
            f"  Emotions used: {', '.join(result.get_emotions_used())}\n"
            f"\n  Emotion spans:\n" + "\n".join(span_lines)
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error parsing emotion text: {e}")]


async def handle_get_emotion_vad(args: dict[str, Any]) -> list[TextContent]:
    """Get VAD values for an emotion."""
    emotion = args.get("emotion", "")
    if not emotion:
        return [TextContent(type="text", text="Error: 'emotion' is required")]

    try:
        vad = emotion_to_vad(emotion)

        # Describe the emotion placement
        valence_desc = "positive" if vad.valence > 0.2 else "negative" if vad.valence < -0.2 else "neutral"
        arousal_desc = "high-energy" if vad.arousal > 0.6 else "calm" if vad.arousal < 0.4 else "moderate"
        dominance_desc = "dominant" if vad.dominance > 0.6 else "submissive" if vad.dominance < 0.4 else "balanced"

        response = (
            f"VAD values for '{emotion}':\n"
            f"  Valence: {vad.valence:+.2f} ({valence_desc})\n"
            f"  Arousal: {vad.arousal:.2f} ({arousal_desc})\n"
            f"  Dominance: {vad.dominance:.2f} ({dominance_desc})\n"
            f"\n  Character: {valence_desc}, {arousal_desc}, {dominance_desc}"
        )
        return [TextContent(type="text", text=response)]

    except ValueError as e:
        return [TextContent(type="text", text=str(e))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting VAD: {e}")]


async def handle_list_emotion_blends(args: dict[str, Any]) -> list[TextContent]:
    """List pre-defined emotion blends."""
    blends = list_named_blends()

    lines = ["Pre-defined emotion blends:"]
    for name in blends:
        blend = get_named_blend(name)
        if blend:
            components = " + ".join(f"{e}:{w:.0%}" for e, w in blend.components)
            lines.append(f"  {name}: {components} → {blend.dominant_emotion}")

    lines.extend([
        "",
        "Use blend_emotions to create custom blends:",
        "  blend_emotions([{emotion: 'happy', weight: 0.6}, {emotion: 'sad', weight: 0.4}])"
    ])

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_list_narrative_curves(args: dict[str, Any]) -> list[TextContent]:
    """List pre-built narrative emotion curves."""
    curves = list_narrative_curves()

    lines = ["Pre-built narrative emotion curves:"]
    for name in curves:
        curve = get_narrative_curve(name)
        if curve:
            keyframes = " → ".join(kf.emotion for kf in curve.keyframes)
            lines.append(f"  {name}: {keyframes}")

    lines.extend([
        "",
        "Use sample_emotion_curve to see curve values at different positions.",
        "These curves define how emotion changes over the duration of an utterance."
    ])

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_sample_emotion_curve(args: dict[str, Any]) -> list[TextContent]:
    """Sample an emotion curve at multiple points."""
    curve_name = args.get("curve_name")
    keyframes_data = args.get("keyframes")
    num_samples = args.get("num_samples", 5)

    try:
        if curve_name:
            curve = get_narrative_curve(curve_name)
            if not curve:
                available = ", ".join(list_narrative_curves())
                return [TextContent(
                    type="text",
                    text=f"Unknown curve: '{curve_name}'. Available: {available}"
                )]
        elif keyframes_data:
            curve = EmotionCurve()
            for kf in keyframes_data:
                curve.add_point(kf["position"], kf["emotion"])
        else:
            return [TextContent(
                type="text",
                text="Error: Provide either 'curve_name' or 'keyframes'"
            )]

        # Sample the curve
        samples = curve.sample(num_samples)

        lines = [f"Emotion curve samples ({len(curve)} keyframes):"]
        for pos, vad, emotion in samples:
            bar_len = int(vad.arousal * 10)
            bar = "█" * bar_len + "░" * (10 - bar_len)
            lines.append(f"  {pos:.1%}: {emotion:12} [{bar}] A={vad.arousal:.2f}")

        # Add keyframe info
        lines.append("\nKeyframes:")
        for kf in curve.keyframes:
            lines.append(f"  {kf.position:.1%}: {kf.emotion} (easing: {kf.easing})")

        return [TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error sampling curve: {e}")]


async def handle_clone_voice_advanced(args: dict[str, Any]) -> list[TextContent]:
    """Clone a voice from audio with full options."""
    audio_path = args.get("audio_path")
    voice_id = args.get("voice_id")
    consent_given = args.get("consent_given", False)

    if not audio_path:
        return [TextContent(type="text", text="Error: 'audio_path' is required")]
    if not voice_id:
        return [TextContent(type="text", text="Error: 'voice_id' is required")]

    try:
        cloner = get_voice_cloner()

        result = cloner.clone(
            audio=audio_path,
            voice_id=voice_id,
            name=args.get("name"),
            consent_given=consent_given,
            consent_notes=args.get("consent_notes", ""),
            tags=args.get("tags"),
            gender=args.get("gender"),
            language=args.get("language", "en"),
        )

        if not result.success:
            return [TextContent(type="text", text=f"Cloning failed: {result.error}")]

        # Format warnings
        warnings_str = ""
        if result.warnings:
            warnings_str = "\n\n  Warnings:\n    " + "\n    ".join(result.warnings)

        recs_str = ""
        if result.recommendations:
            recs_str = "\n\n  Recommendations:\n    " + "\n    ".join(result.recommendations)

        response = (
            f"Voice cloned successfully!\n"
            f"  Voice ID: {result.voice_id}\n"
            f"  Name: {result.profile.name if result.profile else voice_id}\n"
            f"  Quality score: {result.quality_score:.2f}\n"
            f"  SNR: {result.snr_db:.1f} dB\n"
            f"  Audio duration: {result.audio_duration:.1f}s\n"
            f"  Extraction time: {result.extraction_time:.2f}s"
            f"{warnings_str}{recs_str}"
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error cloning voice: {e}")]


async def handle_validate_clone_audio(args: dict[str, Any]) -> list[TextContent]:
    """Validate audio for voice cloning."""
    audio_path = args.get("audio_path")
    if not audio_path:
        return [TextContent(type="text", text="Error: 'audio_path' is required")]

    try:
        cloner = get_voice_cloner()
        result = cloner.validate_audio(audio_path)

        status = "✓ Valid" if result["is_valid"] else "✗ Invalid"

        issues_str = ""
        if result["issues"]:
            issues_str = "\n\n  Issues:\n    " + "\n    ".join(result["issues"])

        recs_str = ""
        if result["recommendations"]:
            recs_str = "\n\n  Recommendations:\n    " + "\n    ".join(result["recommendations"])

        response = (
            f"Audio validation: {status}\n"
            f"  Duration: {result['duration_seconds']:.1f}s\n"
            f"  Quality score: {result['quality_score']:.2f}\n"
            f"  SNR: {result['snr_db']:.1f} dB"
            f"{issues_str}{recs_str}"
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error validating audio: {e}")]


async def handle_list_voice_library(args: dict[str, Any]) -> list[TextContent]:
    """List voices in the library."""
    try:
        cloner = get_voice_cloner()
        voices = cloner.list_voices(
            query=args.get("query"),
            tags=args.get("tags"),
            gender=args.get("gender"),
            language=args.get("language"),
            min_quality=args.get("min_quality", 0.0),
        )

        if not voices:
            return [TextContent(
                type="text",
                text="No voices in library.\nUse clone_voice_advanced to add voices."
            )]

        lines = [f"Voice library ({len(voices)} voices):"]
        for voice in voices:
            tags_str = f" [{', '.join(voice.tags)}]" if voice.tags else ""
            lines.append(
                f"  {voice.voice_id}: {voice.name} "
                f"({voice.gender or 'unknown'}, {voice.language}){tags_str}"
            )

        return [TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error listing voices: {e}")]


async def handle_get_voice_profile(args: dict[str, Any]) -> list[TextContent]:
    """Get voice profile details."""
    voice_id = args.get("voice_id")
    if not voice_id:
        return [TextContent(type="text", text="Error: 'voice_id' is required")]

    try:
        cloner = get_voice_cloner()
        profile = cloner.get_voice(voice_id)

        if not profile:
            return [TextContent(type="text", text=f"Voice not found: {voice_id}")]

        tags_str = ", ".join(profile.tags) if profile.tags else "(none)"

        response = (
            f"Voice profile: {profile.voice_id}\n"
            f"  Name: {profile.name}\n"
            f"  Description: {profile.description or '(none)'}\n"
            f"  Gender: {profile.gender or 'unknown'}\n"
            f"  Language: {profile.language}\n"
            f"  Tags: {tags_str}\n"
            f"  Quality rating: {profile.quality_rating:.2f}\n"
            f"  Usage count: {profile.usage_count}\n"
            f"  Created: {profile.created_date}\n"
            f"  Consent: {'Yes' if profile.consent_given else 'No'}"
        )

        if profile.embedding:
            response += f"\n  Embedding dim: {profile.embedding.embedding_dim}"

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting profile: {e}")]


async def handle_delete_cloned_voice(args: dict[str, Any]) -> list[TextContent]:
    """Delete a cloned voice."""
    voice_id = args.get("voice_id")
    if not voice_id:
        return [TextContent(type="text", text="Error: 'voice_id' is required")]

    try:
        cloner = get_voice_cloner()
        deleted = cloner.delete_voice(voice_id)

        if deleted:
            return [TextContent(type="text", text=f"Voice '{voice_id}' deleted successfully")]
        else:
            return [TextContent(type="text", text=f"Voice not found: {voice_id}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error deleting voice: {e}")]


async def handle_find_similar_voices(args: dict[str, Any]) -> list[TextContent]:
    """Find similar voices in library."""
    audio_path = args.get("audio_path")
    if not audio_path:
        return [TextContent(type="text", text="Error: 'audio_path' is required")]

    top_k = args.get("top_k", 5)

    try:
        cloner = get_voice_cloner()
        similar = cloner.find_similar(audio_path, top_k=top_k)

        if not similar:
            return [TextContent(
                type="text",
                text="No similar voices found in library."
            )]

        lines = ["Similar voices:"]
        for profile, similarity in similar:
            lines.append(
                f"  {profile.voice_id}: {profile.name} "
                f"(similarity: {similarity:.1%})"
            )

        return [TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error finding similar voices: {e}")]


async def handle_transfer_voice_emotion(args: dict[str, Any]) -> list[TextContent]:
    """Transfer emotion to a cloned voice."""
    voice_id = args.get("voice_id")
    emotion = args.get("emotion")

    if not voice_id:
        return [TextContent(type="text", text="Error: 'voice_id' is required")]
    if not emotion:
        return [TextContent(type="text", text="Error: 'emotion' is required")]

    intensity = args.get("intensity", 1.0)

    try:
        cloner = get_voice_cloner()
        separator = get_emotion_separator()

        profile = cloner.get_voice(voice_id)
        if not profile or not profile.embedding:
            return [TextContent(type="text", text=f"Voice not found: {voice_id}")]

        # Transfer emotion
        combined = separator.transfer_emotion(
            profile.embedding,
            emotion,
            intensity=intensity,
        )

        response = (
            f"Emotion transfer complete:\n"
            f"  Voice: {voice_id}\n"
            f"  Emotion: {emotion}\n"
            f"  Intensity: {intensity:.1f}\n"
            f"  Result embedding shape: {combined.shape}"
        )
        return [TextContent(type="text", text=response)]

    except ValueError as e:
        return [TextContent(type="text", text=str(e))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error transferring emotion: {e}")]


async def handle_list_cloning_languages(args: dict[str, Any]) -> list[TextContent]:
    """List supported languages for cloning."""
    cloner = CrossLanguageCloner()
    languages = cloner.list_supported_languages()

    lines = ["Supported languages for voice cloning:"]
    for lang in languages:
        lines.append(f"  {lang['code']}: {lang['name']} ({lang['native_name']})")

    lines.extend([
        "",
        "Use check_language_compatibility to assess quality between language pairs."
    ])

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_check_language_compatibility(args: dict[str, Any]) -> list[TextContent]:
    """Check language pair compatibility."""
    source = args.get("source_language")
    target = args.get("target_language")

    if not source or not target:
        return [TextContent(
            type="text",
            text="Error: Both 'source_language' and 'target_language' are required"
        )]

    try:
        cloner = CrossLanguageCloner(source_language=source)
        compat = cloner.get_language_pair_compatibility(source, target)

        if not compat["compatible"]:
            return [TextContent(
                type="text",
                text=f"Incompatible: {compat.get('reason', 'Unknown reason')}"
            )]

        issues_str = ""
        if compat["phonetic_issues"]:
            issues_str = "\n\n  Phonetic considerations:\n    " + "\n    ".join(compat["phonetic_issues"])

        recs_str = ""
        if compat["recommendations"]:
            recs_str = "\n\n  Recommendations:\n    " + "\n    ".join(compat["recommendations"])

        family = "Yes" if compat["same_language_family"] else "No"

        response = (
            f"Language compatibility:\n"
            f"  Source: {compat['source']}\n"
            f"  Target: {compat['target']}\n"
            f"  Same language family: {family}\n"
            f"  Expected quality: {compat['expected_quality']:.0%}"
            f"{issues_str}{recs_str}"
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error checking compatibility: {e}")]


# Neural Audio Codec handlers

async def handle_encode_audio_tokens(args: dict[str, Any]) -> list[TextContent]:
    """Encode audio to tokens."""
    audio_path = args.get("audio_path")
    if not audio_path:
        return [TextContent(type="text", text="Error: 'audio_path' is required")]

    codec_type = args.get("codec", "mock")
    include_llm = args.get("include_llm_tokens", True)

    try:
        codec = get_audio_codec(codec_type)
        encoded = codec.encode(audio_path)

        if include_llm:
            tokens = encoded.tokens.to_llm_tokens().tolist()
        else:
            tokens = encoded.tokens.tokens.tolist()

        # Flatten if multi-codebook
        if isinstance(tokens[0], list):
            flat_tokens = [t for cb in tokens for t in cb]
        else:
            flat_tokens = tokens

        response = (
            f"Audio encoded successfully:\n"
            f"  Codec: {codec.name} v{codec.version}\n"
            f"  Duration: {encoded.tokens.source_duration_seconds:.2f}s\n"
            f"  Token count: {len(flat_tokens)}\n"
            f"  Codebooks: {encoded.tokens.num_codebooks}\n"
            f"  Frame rate: {codec.capabilities.frame_rate_hz} Hz\n"
            f"  Quality: {encoded.estimated_quality:.2f}\n\n"
            f"First 20 tokens: {flat_tokens[:20]}"
        )

        if encoded.has_dual_tokens:
            response += (
                f"\n\n  Semantic tokens: {encoded.semantic_tokens.sequence_length}"
                f"\n  Acoustic tokens: {encoded.acoustic_tokens.sequence_length}"
            )

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error encoding audio: {e}")]


async def handle_decode_audio_tokens(args: dict[str, Any]) -> list[TextContent]:
    """Decode tokens to audio."""
    tokens = args.get("tokens")
    output_path = args.get("output_path")

    if not tokens:
        return [TextContent(type="text", text="Error: 'tokens' is required")]
    if not output_path:
        return [TextContent(type="text", text="Error: 'output_path' is required")]

    codec_type = args.get("codec", "mock")

    try:
        import numpy as np

        codec = get_audio_codec(codec_type)
        token_array = np.array(tokens, dtype=np.int64)

        audio, sample_rate = codec.from_llm_tokens(token_array)

        # Save audio
        try:
            import soundfile as sf
            sf.write(output_path, audio, sample_rate)
        except ImportError:
            # Fallback to scipy
            from scipy.io import wavfile
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(output_path, sample_rate, audio_int16)

        duration = len(audio) / sample_rate

        response = (
            f"Audio decoded successfully:\n"
            f"  Output: {output_path}\n"
            f"  Duration: {duration:.2f}s\n"
            f"  Sample rate: {sample_rate} Hz\n"
            f"  Codec: {codec.name}"
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error decoding tokens: {e}")]


async def handle_get_codec_info(args: dict[str, Any]) -> list[TextContent]:
    """Get codec information."""
    codec_type = args.get("codec", "mock")

    try:
        codec = get_audio_codec(codec_type)
        info = get_codec_vocabulary_info(codec)

        caps = codec.capabilities

        lines = [
            f"Codec: {info['codec_name']} v{info['codec_version']}",
            f"",
            f"Token Configuration:",
            f"  Codebooks: {info['num_codebooks']}",
            f"  Codebook size: {info['codebook_size']}",
            f"  Total audio vocab: {info['total_audio_vocab_size']}",
            f"  Tokens per second: {info['tokens_per_second']:.1f}",
            f"",
            f"Audio Configuration:",
            f"  Frame rate: {info['frame_rate_hz']} Hz",
            f"  Sample rate: {info['sample_rate']} Hz",
            f"",
            f"Capabilities:",
            f"  Can encode: {caps.can_encode}",
            f"  Can decode: {caps.can_decode}",
            f"  Can stream: {caps.can_stream}",
            f"  Semantic tokens: {caps.has_semantic_tokens}",
            f"  Acoustic tokens: {caps.has_acoustic_tokens}",
            f"",
            f"Available codecs:",
            f"  - mock: Always available (for testing)",
            f"  - mimi: {'Available' if MIMI_AVAILABLE else 'Not installed'}",
            f"  - dualcodec: {'Available' if DUALCODEC_AVAILABLE else 'Not installed'}",
        ]

        return [TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting codec info: {e}")]


async def handle_estimate_audio_tokens(args: dict[str, Any]) -> list[TextContent]:
    """Estimate tokens for audio duration."""
    duration = args.get("duration_seconds")
    if duration is None:
        return [TextContent(type="text", text="Error: 'duration_seconds' is required")]

    codec_type = args.get("codec", "mock")

    try:
        codec = get_audio_codec(codec_type)
        token_count = estimate_audio_context_length(duration, codec)

        # Also show for other codecs
        mock_tokens = estimate_audio_context_length(duration, MockCodec())

        response = (
            f"Token estimates for {duration:.1f}s of audio:\n"
            f"\n"
            f"  {codec.name}: {token_count} tokens\n"
            f"    ({codec.capabilities.frame_rate_hz} Hz * {codec.capabilities.num_codebooks} codebooks)\n"
            f"\n"
            f"Comparison:\n"
            f"  Mock (50 Hz, 8 CB): {50 * 8 * duration:.0f} tokens\n"
            f"  Mimi (12.5 Hz, 8 CB): {12.5 * 8 * duration:.0f} tokens\n"
            f"  DualCodec (50 Hz, 8 CB): {50 * 8 * duration:.0f} tokens\n"
            f"\n"
            f"Context usage (assuming 128K context):\n"
            f"  {token_count} / 128000 = {token_count / 128000 * 100:.1f}%"
        )

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error estimating tokens: {e}")]


async def handle_voice_convert_dualcodec(args: dict[str, Any]) -> list[TextContent]:
    """Convert voice using DualCodec."""
    content_audio = args.get("content_audio")
    style_audio = args.get("style_audio")
    output_path = args.get("output_path")

    if not content_audio:
        return [TextContent(type="text", text="Error: 'content_audio' is required")]
    if not style_audio:
        return [TextContent(type="text", text="Error: 'style_audio' is required")]
    if not output_path:
        return [TextContent(type="text", text="Error: 'output_path' is required")]

    try:
        import numpy as np

        codec = DualCodec()

        audio, sample_rate = codec.voice_convert(
            content_audio=content_audio,
            style_audio=style_audio,
        )

        # Save audio
        try:
            import soundfile as sf
            sf.write(output_path, audio, sample_rate)
        except ImportError:
            from scipy.io import wavfile
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(output_path, sample_rate, audio_int16)

        duration = len(audio) / sample_rate

        response = (
            f"Voice conversion complete:\n"
            f"  Content from: {content_audio}\n"
            f"  Style from: {style_audio}\n"
            f"  Output: {output_path}\n"
            f"  Duration: {duration:.2f}s\n"
            f"\n"
            f"The output has the content/words from the first audio\n"
            f"spoken in the voice/style of the second audio."
        )

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error converting voice: {e}")]


# =============================================================================
# Real-time Voice Conversion Handlers
# =============================================================================

async def handle_list_audio_devices(args: dict[str, Any]) -> list[TextContent]:
    """List available audio devices."""
    try:
        device_type_str = args.get("device_type", "all")

        device_type = None
        if device_type_str == "input":
            device_type = DeviceType.INPUT
        elif device_type_str == "output":
            device_type = DeviceType.OUTPUT

        devices = list_audio_devices(device_type)

        if not devices:
            return [TextContent(type="text", text="No audio devices found.")]

        lines = [f"Audio Devices ({len(devices)} found):"]
        lines.append("")

        # Group by type
        input_devices = [d for d in devices if d.device_type == DeviceType.INPUT or d.device_type == DeviceType.DUPLEX]
        output_devices = [d for d in devices if d.device_type == DeviceType.OUTPUT or d.device_type == DeviceType.DUPLEX]

        if input_devices and device_type_str in ("input", "all"):
            lines.append("📥 Input Devices (Microphones):")
            for d in input_devices:
                default = " (DEFAULT)" if d.is_default else ""
                lines.append(f"  [{d.id}] {d.name}{default}")
                lines.append(f"      Channels: {d.max_input_channels}, Sample Rate: {d.default_sample_rate:.0f} Hz")
            lines.append("")

        if output_devices and device_type_str in ("output", "all"):
            lines.append("🔊 Output Devices (Speakers):")
            for d in output_devices:
                default = " (DEFAULT)" if d.is_default else ""
                lines.append(f"  [{d.id}] {d.name}{default}")
                lines.append(f"      Channels: {d.max_output_channels}, Sample Rate: {d.default_sample_rate:.0f} Hz")

        return [TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error listing devices: {e}")]


async def handle_start_voice_conversion(args: dict[str, Any]) -> list[TextContent]:
    """Start real-time voice conversion."""
    global _realtime_converter

    target_voice = args.get("target_voice")
    latency_mode = args.get("latency_mode", "balanced")
    input_device = args.get("input_device")
    output_device = args.get("output_device")

    if not target_voice:
        return [TextContent(type="text", text="Error: 'target_voice' is required")]

    try:
        # Stop existing session if running
        if _realtime_converter and _realtime_converter.is_running:
            _realtime_converter.stop()
            _realtime_converter = None

        # Create new converter with specified latency mode
        _realtime_converter = get_realtime_converter(latency_mode)

        # Start conversion
        session = _realtime_converter.start(
            source=input_device,
            target_voice=target_voice,
            output=output_device,
        )

        response = (
            f"🎙️ Voice conversion started!\n"
            f"\n"
            f"Session: {session.session_id}\n"
            f"Target Voice: {target_voice}\n"
            f"Latency Mode: {latency_mode}\n"
            f"Target Latency: ~{_realtime_converter.config.get_latency_ms():.0f}ms\n"
            f"\n"
            f"Input: {session.input_device.name if session.input_device else 'Default'}\n"
            f"Output: {session.output_device.name if session.output_device else 'Default'}\n"
            f"\n"
            f"Speak into your microphone and you'll hear your voice converted!\n"
            f"Use 'stop_voice_conversion' when finished."
        )

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error starting voice conversion: {e}")]


async def handle_stop_voice_conversion(args: dict[str, Any]) -> list[TextContent]:
    """Stop real-time voice conversion."""
    global _realtime_converter

    try:
        if _realtime_converter is None or not _realtime_converter.is_running:
            return [TextContent(type="text", text="No voice conversion session is running.")]

        session = _realtime_converter.stop()

        if session is None:
            return [TextContent(type="text", text="Voice conversion stopped (no session data).")]

        stats = session.stats.to_dict()

        response = (
            f"🛑 Voice conversion stopped.\n"
            f"\n"
            f"Session Summary:\n"
            f"  Session ID: {session.session_id}\n"
            f"  Duration: {session.duration_seconds:.1f}s\n"
            f"  Target Voice: {session.target_voice}\n"
            f"\n"
            f"Performance:\n"
            f"  Chunks Processed: {stats['chunks_processed']}\n"
            f"  Chunks Dropped: {stats['chunks_dropped']}\n"
            f"  Avg Latency: {stats['avg_latency_ms']:.1f}ms\n"
            f"  Min Latency: {stats['min_latency_ms']:.1f}ms\n"
            f"  Max Latency: {stats['max_latency_ms']:.1f}ms\n"
            f"  Realtime Factor: {stats['realtime_factor']:.2f}x"
        )

        if stats['realtime_factor'] > 1.0:
            response += "\n\n⚠️ Processing was slower than real-time."
        else:
            response += "\n\n✅ Processing was fast enough for real-time."

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error stopping voice conversion: {e}")]


async def handle_get_voice_conversion_status(args: dict[str, Any]) -> list[TextContent]:
    """Get voice conversion status."""
    global _realtime_converter

    try:
        if _realtime_converter is None:
            return [TextContent(type="text", text="No voice conversion session initialized.")]

        if not _realtime_converter.is_running:
            return [TextContent(type="text", text="Voice conversion is not currently running.")]

        session = _realtime_converter.current_session
        stats = _realtime_converter.stats.to_dict()
        latency = _realtime_converter.get_latency()

        response = (
            f"🎙️ Voice Conversion Status: RUNNING\n"
            f"\n"
            f"Session: {session.session_id if session else 'Unknown'}\n"
            f"Target Voice: {session.target_voice if session else 'Unknown'}\n"
            f"Duration: {session.duration_seconds:.1f}s\n"
            f"\n"
            f"Live Metrics:\n"
            f"  Current Latency: {latency:.1f}ms\n"
            f"  Chunks Processed: {stats['chunks_processed']}\n"
            f"  Avg Latency: {stats['avg_latency_ms']:.1f}ms\n"
            f"  Realtime Factor: {stats['realtime_factor']:.2f}x"
        )

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting status: {e}")]


async def handle_convert_audio_file(args: dict[str, Any]) -> list[TextContent]:
    """Convert an audio file to a different voice."""
    input_path = args.get("input_path")
    target_voice = args.get("target_voice")
    output_path = args.get("output_path")
    preserve_pitch = args.get("preserve_pitch", True)

    if not input_path:
        return [TextContent(type="text", text="Error: 'input_path' is required")]
    if not target_voice:
        return [TextContent(type="text", text="Error: 'target_voice' is required")]
    if not output_path:
        return [TextContent(type="text", text="Error: 'output_path' is required")]

    try:
        import numpy as np
        from pathlib import Path

        # Load input audio
        try:
            import soundfile as sf
            audio, sample_rate = sf.read(input_path)
            if audio.ndim > 1:
                audio = audio[:, 0]  # Mono
            audio = audio.astype(np.float32)
        except ImportError:
            from scipy.io import wavfile
            sample_rate, audio = wavfile.read(input_path)
            audio = audio.astype(np.float32) / 32768.0

        # Create converter
        config = ConversionConfig(
            preserve_pitch=preserve_pitch,
            sample_rate=sample_rate,
        )
        converter = MockVoiceConverter(config)
        converter.set_target_voice(target_voice)

        # Convert
        result = converter.convert(audio, sample_rate)

        # Save output
        try:
            import soundfile as sf
            sf.write(output_path, result.audio, result.sample_rate)
        except ImportError:
            from scipy.io import wavfile
            audio_int16 = (result.audio * 32767).astype(np.int16)
            wavfile.write(output_path, result.sample_rate, audio_int16)

        response = (
            f"✅ Audio file converted successfully!\n"
            f"\n"
            f"Input: {input_path}\n"
            f"  Duration: {result.input_duration_ms / 1000:.2f}s\n"
            f"\n"
            f"Output: {output_path}\n"
            f"  Duration: {result.output_duration_ms / 1000:.2f}s\n"
            f"  Target Voice: {target_voice}\n"
            f"\n"
            f"Processing:\n"
            f"  Time: {result.processing_time_ms:.0f}ms\n"
            f"  Realtime Factor: {result.realtime_factor:.2f}x\n"
            f"  Similarity Score: {result.similarity_score:.0%}\n"
            f"  Naturalness Score: {result.naturalness_score:.0%}"
        )

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error converting audio file: {e}")]


# =============================================================================
# LLM Integration Tool Handlers
# =============================================================================

async def handle_speak_with_context(args: dict[str, Any]) -> list[TextContent]:
    """Speak text with context-aware prosody."""
    text = args.get("text")
    context = args.get("context")
    auto_emotion = args.get("auto_emotion", True)
    voice = args.get("voice")
    preset = args.get("preset")

    if not text:
        return [TextContent(type="text", text="Error: 'text' is required")]

    try:
        speaker = get_context_speaker()

        # Update context if user message provided
        if context:
            speaker.update_context(context)

        result = speaker.speak(
            text,
            context=context,
            auto_emotion=auto_emotion,
            voice=voice,
            preset=preset,
        )

        response = (
            f"🎤 Context-Aware Speech Generated\n"
            f"\n"
            f"Text: {text[:100]}{'...' if len(text) > 100 else ''}\n"
            f"Emotion: {result['emotion']} (confidence: {result['confidence']:.0%})\n"
            f"Speed Factor: {result['speed_factor']:.2f}x\n"
        )

        if context:
            response += f"Context: {context[:50]}{'...' if len(context) > 50 else ''}\n"

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error speaking with context: {e}")]


async def handle_start_conversation(args: dict[str, Any]) -> list[TextContent]:
    """Start a new conversation session."""
    global _conversation_manager, _speech_pipeline

    system_prompt = args.get("system_prompt")
    llm_backend = args.get("llm_backend", "mock")
    llm_model = args.get("llm_model", "llama3.2")

    try:
        # Create new conversation manager
        config = ConversationConfig(
            system_prompt=system_prompt,
        )
        _conversation_manager = ConversationManager(config=config)
        _conversation_manager.start()

        # Create speech pipeline
        pipeline_config = PipelineConfig(
            llm_backend=llm_backend,
            llm_model=llm_model,
            system_prompt=system_prompt or "You are a helpful voice assistant.",
        )
        _speech_pipeline = SpeechPipeline(config=pipeline_config)

        response = (
            f"🎙️ Conversation Started\n"
            f"\n"
            f"Session ID: {_conversation_manager.id}\n"
            f"LLM Backend: {llm_backend}\n"
            f"LLM Model: {llm_model}\n"
        )

        if system_prompt:
            response += f"System Prompt: {system_prompt[:50]}{'...' if len(system_prompt) > 50 else ''}\n"

        response += "\nReady for conversation!"

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error starting conversation: {e}")]


async def handle_add_conversation_message(args: dict[str, Any]) -> list[TextContent]:
    """Add a message to the conversation."""
    global _conversation_manager

    role = args.get("role")
    content = args.get("content")
    emotion = args.get("emotion")

    if not role or not content:
        return [TextContent(type="text", text="Error: 'role' and 'content' are required")]

    try:
        if _conversation_manager is None:
            _conversation_manager = get_conversation_manager()
            _conversation_manager.start()

        if role == "user":
            msg = _conversation_manager.add_user_message(content, emotion=emotion)
        else:
            msg = _conversation_manager.add_assistant_message(content, emotion=emotion)

        response = (
            f"✅ Message Added\n"
            f"\n"
            f"ID: {msg.id}\n"
            f"Role: {role}\n"
            f"Content: {content[:100]}{'...' if len(content) > 100 else ''}\n"
            f"Turn: {_conversation_manager._turn_count}\n"
        )

        if emotion:
            response += f"Emotion: {emotion}\n"

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error adding message: {e}")]


async def handle_get_conversation_context(args: dict[str, Any]) -> list[TextContent]:
    """Get the conversation context for LLM."""
    global _conversation_manager

    max_messages = args.get("max_messages", 10)

    try:
        if _conversation_manager is None:
            return [TextContent(type="text", text="No active conversation. Use start_conversation first.")]

        context = _conversation_manager.get_llm_context()

        # Limit messages
        if len(context) > max_messages:
            # Keep system message and last N-1 messages
            system_msgs = [m for m in context if m.get("role") == "system"]
            other_msgs = [m for m in context if m.get("role") != "system"]
            context = system_msgs + other_msgs[-(max_messages - len(system_msgs)):]

        response = f"📝 Conversation Context ({len(context)} messages)\n\n"

        for msg in context:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")[:100]
            if len(msg.get("content", "")) > 100:
                content += "..."
            response += f"[{role}] {content}\n\n"

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting context: {e}")]


async def handle_get_conversation_stats(args: dict[str, Any]) -> list[TextContent]:
    """Get conversation statistics."""
    global _conversation_manager

    try:
        if _conversation_manager is None:
            return [TextContent(type="text", text="No active conversation.")]

        stats = _conversation_manager.stats

        response = (
            f"📊 Conversation Statistics\n"
            f"\n"
            f"Session ID: {stats['id']}\n"
            f"State: {stats['state']}\n"
            f"Duration: {stats['duration_seconds']:.1f}s\n"
            f"\n"
            f"Messages:\n"
            f"  Total: {stats['message_count']}\n"
            f"  User: {stats['user_message_count']}\n"
            f"  Assistant: {stats['assistant_message_count']}\n"
            f"\n"
            f"Words:\n"
            f"  User: {stats['user_word_count']}\n"
            f"  Assistant: {stats['assistant_word_count']}\n"
            f"\n"
            f"Turn Count: {stats['turn_count']}\n"
            f"Current Turn: {stats['whose_turn'] or 'None'}"
        )

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting stats: {e}")]


async def handle_end_conversation(args: dict[str, Any]) -> list[TextContent]:
    """End the conversation session."""
    global _conversation_manager, _speech_pipeline

    save_path = args.get("save_path")

    try:
        if _conversation_manager is None:
            return [TextContent(type="text", text="No active conversation to end.")]

        stats = _conversation_manager.stats
        conversation_id = _conversation_manager.id

        # Save if requested
        if save_path:
            _conversation_manager.save(Path(save_path))

        # End the conversation
        _conversation_manager.end()

        # Reset pipeline
        if _speech_pipeline:
            _speech_pipeline.reset()

        response = (
            f"👋 Conversation Ended\n"
            f"\n"
            f"Session ID: {conversation_id}\n"
            f"Duration: {stats['duration_seconds']:.1f}s\n"
            f"Total Turns: {stats['turn_count']}\n"
            f"Messages Exchanged: {stats['message_count']}\n"
        )

        if save_path:
            response += f"\n💾 Saved to: {save_path}"

        # Reset globals
        _conversation_manager = None
        _speech_pipeline = None

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error ending conversation: {e}")]


async def handle_detect_user_emotion(args: dict[str, Any]) -> list[TextContent]:
    """Detect user emotion from their message."""
    message = args.get("message")

    if not message:
        return [TextContent(type="text", text="Error: 'message' is required")]

    try:
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        sentiment, emotion = selector.detect_user_sentiment(message)

        response = (
            f"🔍 User Emotion Analysis\n"
            f"\n"
            f"Message: {message[:100]}{'...' if len(message) > 100 else ''}\n"
            f"\n"
            f"Detected:\n"
            f"  Sentiment: {sentiment}\n"
            f"  Emotion: {emotion}\n"
            f"\n"
            f"Suggested Response Style:\n"
        )

        # Suggest response style
        style_map = {
            "frustrated": "Use sympathetic, calming tone",
            "confused": "Use patient, clear explanations",
            "angry": "Stay calm, acknowledge feelings",
            "sad": "Use sympathetic, supportive tone",
            "happy": "Match their enthusiasm",
            "excited": "Share their excitement",
            "anxious": "Use calm, reassuring tone",
            "neutral": "Use friendly, professional tone",
        }

        response += f"  {style_map.get(emotion, 'Use neutral, helpful tone')}"

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error detecting emotion: {e}")]


async def handle_select_response_emotion(args: dict[str, Any]) -> list[TextContent]:
    """Select the best emotion for a response."""
    response_text = args.get("response_text")
    user_emotion = args.get("user_emotion")

    if not response_text:
        return [TextContent(type="text", text="Error: 'response_text' is required")]

    try:
        from voice_soundboard.llm.context import EmotionSelector, ConversationContext

        selector = EmotionSelector()
        context = ConversationContext()

        if user_emotion:
            context.user_emotion = user_emotion

        emotion, confidence = selector.select_emotion(response_text, context)

        response = (
            f"🎭 Response Emotion Selection\n"
            f"\n"
            f"Response: {response_text[:100]}{'...' if len(response_text) > 100 else ''}\n"
        )

        if user_emotion:
            response += f"User Emotion: {user_emotion}\n"

        response += (
            f"\n"
            f"Selected:\n"
            f"  Emotion: {emotion}\n"
            f"  Confidence: {confidence:.0%}\n"
            f"\n"
            f"Use with speak_realtime or speak_with_context for natural delivery."
        )

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error selecting emotion: {e}")]


async def handle_list_llm_providers(args: dict[str, Any]) -> list[TextContent]:
    """List available LLM providers."""
    try:
        providers = [
            {
                "name": "ollama",
                "description": "Local LLM inference with Ollama",
                "default_url": "http://localhost:11434",
                "requires_key": False,
            },
            {
                "name": "openai",
                "description": "OpenAI API (GPT-4, GPT-3.5)",
                "default_url": "https://api.openai.com/v1",
                "requires_key": True,
            },
            {
                "name": "vllm",
                "description": "High-performance self-hosted inference",
                "default_url": "http://localhost:8000/v1",
                "requires_key": False,
            },
            {
                "name": "mock",
                "description": "Mock provider for testing",
                "default_url": "N/A",
                "requires_key": False,
            },
        ]

        response = "🤖 Available LLM Providers\n\n"

        for p in providers:
            response += (
                f"• {p['name'].upper()}\n"
                f"  {p['description']}\n"
                f"  URL: {p['default_url']}\n"
                f"  API Key: {'Required' if p['requires_key'] else 'Not required'}\n"
                f"\n"
            )

        response += (
            "Use with start_conversation:\n"
            "  start_conversation(llm_backend='ollama', llm_model='llama3.2')"
        )

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error listing providers: {e}")]


async def main():
    """Run the MCP server."""
    logger.info("Starting Voice Soundboard MCP server...")
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error("MCP server error: %s", e)
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
