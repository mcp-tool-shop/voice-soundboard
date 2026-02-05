"""
Voice Soundboard - Text-to-speech for AI agents and developers.

Public API (stable):
    VoiceEngine     - Core TTS engine. Call .speak() to generate audio.
    SpeechResult    - Returned by .speak(). Contains .audio_path and metadata.
    Config          - Engine configuration (output dir, device, defaults).
    quick_speak     - One-liner: quick_speak("Hello") -> Path to .wav file.
    play_audio      - Play an audio file through speakers.
    stop_playback   - Stop any playing audio.
    VOICE_PRESETS   - Dict of preset names -> {voice, speed, description}.
    KOKORO_VOICES   - Dict of voice IDs -> {name, gender, accent, style}.

Advanced (stable, import from subpackages):
    voice_soundboard.engines      - TTSEngine base, Kokoro, Chatterbox, F5-TTS
    voice_soundboard.dialogue     - Multi-speaker dialogue synthesis
    voice_soundboard.emotion      - VAD model, emotion blending, curves
    voice_soundboard.cloning      - Voice cloning and library
    voice_soundboard.presets      - 50+ voice preset catalog with search

Experimental (may change between releases):
    voice_soundboard.codecs       - Neural audio codecs (Mimi, DualCodec)
    voice_soundboard.conversion   - Real-time voice conversion
    voice_soundboard.llm          - LLM conversation pipeline
    voice_soundboard.vocology     - Voice science / humanization
    voice_soundboard.studio       - Voice Studio editing sessions

Example:
    from voice_soundboard import VoiceEngine
    engine = VoiceEngine()
    result = engine.speak("Hello world!", preset="assistant")
"""

__version__ = "1.1.0"

# ---- Public API (stable) ----
from voice_soundboard.engine import VoiceEngine, SpeechResult, quick_speak
from voice_soundboard.config import Config, KOKORO_VOICES, VOICE_PRESETS
from voice_soundboard.audio import play_audio, stop_playback
from voice_soundboard.effects import get_effect, play_effect, list_effects
from voice_soundboard.emotions import get_emotion_params, list_emotions, EMOTIONS

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
    # Effects
    "get_effect",
    "play_effect",
    "list_effects",
    # Emotions
    "get_emotion_params",
    "list_emotions",
    "EMOTIONS",
]
