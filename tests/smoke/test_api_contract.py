"""
API contract regression tests.

These tests enforce the public API guarantees documented in
docs/API_STABILITY.md and docs/ARCHITECTURE.md. If any of these
tests fail, it means a breaking change was introduced.

Run with: pytest tests/smoke/test_api_contract.py -v
"""

import pytest
from dataclasses import fields
from unittest.mock import patch


# ========================================================================
# Public API surface - these symbols MUST be importable
# ========================================================================


class TestPublicAPIExports:
    """Every symbol in __all__ must be importable."""

    def test_api_version_exists(self):
        from voice_soundboard import API_VERSION
        assert isinstance(API_VERSION, int)
        assert API_VERSION >= 1

    def test_version_string(self):
        from voice_soundboard import __version__
        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_all_exports_importable(self):
        import voice_soundboard
        for name in voice_soundboard.__all__:
            assert hasattr(voice_soundboard, name), (
                f"{name} is in __all__ but not importable"
            )


# ========================================================================
# SpeechResult contract - field names and types are frozen
# ========================================================================


class TestSpeechResultContract:
    """SpeechResult fields cannot be removed or renamed."""

    REQUIRED_FIELDS = {
        "audio_path",
        "duration_seconds",
        "generation_time",
        "voice_used",
        "sample_rate",
        "realtime_factor",
    }

    def test_has_all_required_fields(self):
        from voice_soundboard import SpeechResult
        actual = {f.name for f in fields(SpeechResult)}
        missing = self.REQUIRED_FIELDS - actual
        assert not missing, f"SpeechResult is missing frozen fields: {missing}"

    def test_field_types(self):
        from voice_soundboard import SpeechResult
        from pathlib import Path

        result = SpeechResult(
            audio_path=Path("/tmp/test.wav"),
            duration_seconds=1.5,
            generation_time=0.3,
            voice_used="af_bella",
            sample_rate=24000,
            realtime_factor=5.0,
        )
        assert isinstance(result.audio_path, Path)
        assert isinstance(result.duration_seconds, float)
        assert isinstance(result.generation_time, float)
        assert isinstance(result.voice_used, str)
        assert isinstance(result.sample_rate, int)
        assert isinstance(result.realtime_factor, float)


# ========================================================================
# Config contract - defaults are frozen
# ========================================================================


class TestConfigContract:
    """Config defaults and fields are part of the public API."""

    @patch('pathlib.Path.mkdir')
    def test_default_voice(self, _):
        from voice_soundboard import Config
        config = Config()
        assert config.default_voice == "af_bella"

    @patch('pathlib.Path.mkdir')
    def test_default_speed(self, _):
        from voice_soundboard import Config
        config = Config()
        assert config.default_speed == 1.0

    @patch('pathlib.Path.mkdir')
    def test_default_sample_rate(self, _):
        from voice_soundboard import Config
        config = Config()
        assert config.sample_rate == 24000

    @patch('pathlib.Path.mkdir')
    def test_has_model_dir(self, _):
        from voice_soundboard import Config
        config = Config()
        assert hasattr(config, "model_dir")

    @patch('pathlib.Path.mkdir')
    def test_to_dict(self, _):
        from voice_soundboard import Config
        config = Config()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "output_dir" in d
        assert "model_dir" in d
        assert "device" in d
        assert "default_voice" in d


# ========================================================================
# VoiceEngine contract - method signatures are frozen
# ========================================================================


class TestVoiceEngineContract:
    """VoiceEngine method signatures are part of the public API."""

    def test_speak_accepts_all_params(self):
        """speak() must accept these keyword arguments."""
        from voice_soundboard import VoiceEngine
        import inspect
        sig = inspect.signature(VoiceEngine.speak)
        params = set(sig.parameters.keys()) - {"self"}
        required = {"text", "voice", "preset", "speed", "style", "emotion",
                     "save_as", "normalize"}
        missing = required - params
        assert not missing, f"speak() is missing frozen parameters: {missing}"

    def test_speak_raw_accepts_params(self):
        """speak_raw() must accept these keyword arguments."""
        from voice_soundboard import VoiceEngine
        import inspect
        sig = inspect.signature(VoiceEngine.speak_raw)
        params = set(sig.parameters.keys()) - {"self"}
        required = {"text", "voice", "speed", "normalize"}
        missing = required - params
        assert not missing, f"speak_raw() is missing frozen parameters: {missing}"

    def test_list_presets_returns_dict(self):
        from voice_soundboard import VoiceEngine
        with patch('pathlib.Path.mkdir'):
            engine = VoiceEngine()
        presets = engine.list_presets()
        assert isinstance(presets, dict)
        assert len(presets) >= 3

    def test_get_voice_info_returns_dict(self):
        from voice_soundboard import VoiceEngine
        with patch('pathlib.Path.mkdir'):
            engine = VoiceEngine()
        info = engine.get_voice_info("af_bella")
        assert isinstance(info, dict)
        assert "name" in info
        assert "gender" in info


# ========================================================================
# Preset contract - curated presets are frozen
# ========================================================================


class TestPresetContract:
    """The curated presets must always exist."""

    REQUIRED_PRESETS = {"assistant", "narrator", "announcer", "storyteller", "whisper"}

    def test_all_required_presets_exist(self):
        from voice_soundboard import VOICE_PRESETS
        missing = self.REQUIRED_PRESETS - set(VOICE_PRESETS.keys())
        assert not missing, f"Missing curated presets: {missing}"

    def test_presets_have_voice_and_description(self):
        from voice_soundboard import VOICE_PRESETS
        for name, preset in VOICE_PRESETS.items():
            assert "voice" in preset, f"Preset '{name}' missing 'voice'"
            assert "description" in preset, f"Preset '{name}' missing 'description'"


# ========================================================================
# Emotion contract - core emotions are frozen
# ========================================================================


class TestEmotionContract:
    """Core emotions must always exist."""

    REQUIRED_EMOTIONS = {"happy", "sad", "angry", "excited", "calm", "neutral"}

    def test_all_required_emotions_exist(self):
        from voice_soundboard import list_emotions
        available = set(list_emotions())
        missing = self.REQUIRED_EMOTIONS - available
        assert not missing, f"Missing core emotions: {missing}"


# ========================================================================
# Exception contract - hierarchy is frozen
# ========================================================================


class TestExceptionContract:
    """Exception hierarchy must follow the documented tree."""

    def test_base_exception(self):
        from voice_soundboard import VoiceSoundboardError
        assert issubclass(VoiceSoundboardError, Exception)

    def test_all_exceptions_inherit_from_base(self):
        from voice_soundboard import (
            VoiceSoundboardError,
            ConfigurationError,
            ModelNotFoundError,
            VoiceNotFoundError,
            EngineError,
            AudioError,
            StreamingError,
        )
        for exc_class in [ConfigurationError, ModelNotFoundError, VoiceNotFoundError,
                          EngineError, AudioError, StreamingError]:
            assert issubclass(exc_class, VoiceSoundboardError), (
                f"{exc_class.__name__} must inherit from VoiceSoundboardError"
            )

    def test_exceptions_have_hint(self):
        from voice_soundboard import ModelNotFoundError
        exc = ModelNotFoundError("/fake/path")
        assert exc.hint is not None
        assert "curl" in exc.hint  # Download instructions


# ========================================================================
# CLI contract - entrypoint and subcommands are frozen
# ========================================================================


class TestCLIContract:
    """CLI interface is part of the public API."""

    def test_speak_subcommand_exists(self):
        from voice_soundboard.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["speak", "hello"])
        assert args.command == "speak"

    def test_speak_accepts_all_options(self):
        from voice_soundboard.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "speak", "hello",
            "--voice", "af_bella",
            "--preset", "narrator",
            "--emotion", "calm",
            "--speed", "0.9",
            "-o", "test",
        ])
        assert args.voice == "af_bella"
        assert args.preset == "narrator"
        assert args.emotion == "calm"
        assert args.speed == 0.9
        assert args.output == "test"
