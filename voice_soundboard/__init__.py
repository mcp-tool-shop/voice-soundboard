"""
Voice Soundboard - AI-powered TTS with natural language control.

A user-friendly voice synthesis system that AI agents can use via MCP
to generate natural speech from text with style hints.
"""

__version__ = "0.1.0"

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
    # WebSocket (optional)
    "VoiceWebSocketServer",
    "create_server",
    "_HAS_WEBSOCKET",
]
