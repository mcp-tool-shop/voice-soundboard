"""
Exception hierarchy for Voice Soundboard.

All public exceptions inherit from VoiceSoundboardError so callers
can catch everything with a single except clause.

Exception tree:

    VoiceSoundboardError
    ├── ConfigurationError      # Bad config, missing env vars
    ├── ModelNotFoundError       # Models not downloaded
    ├── VoiceNotFoundError       # Invalid voice ID
    ├── EngineError              # TTS engine failure during synthesis
    ├── AudioError               # Playback or file I/O failure
    └── StreamingError           # Streaming pipeline failure

Usage:

    from voice_soundboard.exceptions import ModelNotFoundError

    try:
        engine.speak("Hello!")
    except ModelNotFoundError as e:
        print(e)           # Human-readable message
        print(e.hint)      # Suggested fix
"""


class VoiceSoundboardError(Exception):
    """Base exception for all Voice Soundboard errors.

    Attributes:
        hint: Optional actionable suggestion for fixing the error.
    """

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


class ConfigurationError(VoiceSoundboardError):
    """Raised when configuration is invalid or incomplete."""


class ModelNotFoundError(VoiceSoundboardError):
    """Raised when required model files are missing.

    Includes download instructions in the hint.
    """

    def __init__(self, path: str, model_name: str = "Kokoro"):
        super().__init__(
            f"{model_name} model not found: {path}",
            hint=(
                "Download models with:\n"
                "  mkdir models && cd models\n"
                "  curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/"
                "download/model-files-v1.0/kokoro-v1.0.onnx\n"
                "  curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/"
                "download/model-files-v1.0/voices-v1.0.bin"
            ),
        )
        self.path = path


class VoiceNotFoundError(VoiceSoundboardError):
    """Raised when a voice ID is not recognized."""

    def __init__(self, voice: str, available: list[str] | None = None):
        preview = ", ".join(sorted(available)[:10]) if available else "unknown"
        super().__init__(
            f"Unknown voice: {voice}",
            hint=f"Available voices (first 10): {preview}",
        )
        self.voice = voice


class EngineError(VoiceSoundboardError):
    """Raised when the TTS engine fails during synthesis."""


class AudioError(VoiceSoundboardError):
    """Raised when audio playback or file I/O fails."""


class StreamingError(VoiceSoundboardError):
    """Raised when the streaming pipeline encounters an error."""
