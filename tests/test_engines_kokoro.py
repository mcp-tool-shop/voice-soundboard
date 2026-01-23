"""
Tests for Kokoro TTS Engine (engines/kokoro.py).

Tests cover:
- Engine name and capabilities
- Voice listing and info
- Speak with various parameters
- Model loading/unloading
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from voice_soundboard.engines.base import EngineResult, EngineCapabilities
from voice_soundboard.engines.kokoro import KokoroEngine
from voice_soundboard.config import Config, KOKORO_VOICES, VOICE_PRESETS


class TestKokoroEngineName:
    """Tests for engine name property."""

    def test_engine_name(self):
        """TEST-EK01: KokoroEngine.name returns 'kokoro'."""
        engine = KokoroEngine.__new__(KokoroEngine)
        assert engine.name == "kokoro"


class TestKokoroEngineCapabilities:
    """Tests for engine capabilities."""

    def test_capabilities_returns_correct_structure(self):
        """TEST-EK02: KokoroEngine.capabilities reports correct features."""
        engine = KokoroEngine.__new__(KokoroEngine)
        caps = engine.capabilities

        assert isinstance(caps, EngineCapabilities)
        assert caps.supports_streaming is True
        assert caps.supports_ssml is True

    def test_capabilities_no_paralinguistic_tags(self):
        """TEST-EK03: KokoroEngine.capabilities.supports_paralinguistic_tags is False."""
        engine = KokoroEngine.__new__(KokoroEngine)
        caps = engine.capabilities

        assert caps.supports_paralinguistic_tags is False
        assert caps.paralinguistic_tags == []

    def test_capabilities_no_voice_cloning(self):
        """TEST-EK04: KokoroEngine.capabilities.supports_voice_cloning is False."""
        engine = KokoroEngine.__new__(KokoroEngine)
        caps = engine.capabilities

        assert caps.supports_voice_cloning is False

    def test_capabilities_emotion_control(self):
        """Test that Kokoro supports emotion control via presets."""
        engine = KokoroEngine.__new__(KokoroEngine)
        caps = engine.capabilities

        assert caps.supports_emotion_control is True
        assert caps.supports_emotion_exaggeration is False

    def test_capabilities_languages(self):
        """Test supported languages."""
        engine = KokoroEngine.__new__(KokoroEngine)
        caps = engine.capabilities

        assert "en" in caps.languages
        assert "ja" in caps.languages
        assert "zh" in caps.languages

    def test_capabilities_performance(self):
        """Test performance characteristics."""
        engine = KokoroEngine.__new__(KokoroEngine)
        caps = engine.capabilities

        assert caps.typical_rtf == 5.0
        assert caps.min_latency_ms == 150.0


class TestKokoroEngineInit:
    """Tests for engine initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        engine = KokoroEngine()

        assert engine.config is not None
        assert isinstance(engine.config, Config)
        assert engine._kokoro is None
        assert engine._model_loaded is False

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = Config()
        engine = KokoroEngine(config=config)

        assert engine.config is config

    def test_model_not_loaded_initially(self):
        """Test that model is not loaded on init."""
        engine = KokoroEngine()
        assert engine.is_loaded() is False


class TestKokoroEngineVoiceInfo:
    """Tests for voice information retrieval."""

    def test_get_voice_info_known_voice(self):
        """TEST-EK10: KokoroEngine.get_voice_info() returns metadata dict."""
        engine = KokoroEngine.__new__(KokoroEngine)

        # Test with a known voice from KOKORO_VOICES
        info = engine.get_voice_info("af_bella")

        assert info["id"] == "af_bella"
        assert "name" in info
        assert "gender" in info
        assert "accent" in info

    def test_get_voice_info_unknown_voice(self):
        """Test get_voice_info with unknown voice returns fallback."""
        engine = KokoroEngine.__new__(KokoroEngine)

        info = engine.get_voice_info("unknown_voice")

        assert info["id"] == "unknown_voice"
        assert info["name"] == "unknown_voice"
        assert info["gender"] == "unknown"
        assert info["accent"] == "unknown"


class TestKokoroEngineIsLoaded:
    """Tests for is_loaded method."""

    def test_is_loaded_false_initially(self):
        """TEST-EK11 partial: is_loaded returns False initially."""
        engine = KokoroEngine()
        assert engine.is_loaded() is False

    def test_is_loaded_after_model_load(self):
        """Test that is_loaded returns True after model is loaded."""
        engine = KokoroEngine()
        engine._model_loaded = True

        assert engine.is_loaded() is True


class TestKokoroEngineUnload:
    """Tests for unload method."""

    def test_unload_clears_model(self):
        """TEST-EK12: KokoroEngine.unload() clears model."""
        engine = KokoroEngine()
        engine._kokoro = Mock()
        engine._model_loaded = True

        engine.unload()

        assert engine._kokoro is None
        assert engine._model_loaded is False
        assert engine.is_loaded() is False


class TestKokoroEngineSpeakRawMocked:
    """Tests for speak_raw with mocked model."""

    @patch.object(KokoroEngine, '_ensure_model_loaded')
    def test_speak_raw_returns_tuple(self, mock_load):
        """TEST-EK08: KokoroEngine.speak_raw() returns (samples, sample_rate)."""
        engine = KokoroEngine()
        engine._kokoro = Mock()
        engine._kokoro.create = Mock(return_value=(np.zeros(24000), 24000))

        samples, sr = engine.speak_raw("Hello world")

        assert isinstance(samples, np.ndarray)
        assert sr == 24000

    @patch.object(KokoroEngine, '_ensure_model_loaded')
    def test_speak_raw_with_voice(self, mock_load):
        """Test speak_raw with specific voice."""
        engine = KokoroEngine()
        engine._kokoro = Mock()
        engine._kokoro.create = Mock(return_value=(np.zeros(24000), 24000))

        engine.speak_raw("Hello", voice="am_michael")

        engine._kokoro.create.assert_called_once()
        call_kwargs = engine._kokoro.create.call_args[1]
        assert call_kwargs["voice"] == "am_michael"

    @patch.object(KokoroEngine, '_ensure_model_loaded')
    def test_speak_raw_with_speed(self, mock_load):
        """Test speak_raw with speed parameter."""
        engine = KokoroEngine()
        engine._kokoro = Mock()
        engine._kokoro.create = Mock(return_value=(np.zeros(24000), 24000))

        engine.speak_raw("Hello", speed=1.5)

        call_kwargs = engine._kokoro.create.call_args[1]
        assert call_kwargs["speed"] == 1.5


class TestKokoroEngineSpeakMocked:
    """Tests for speak with mocked model."""

    @patch.object(KokoroEngine, '_ensure_model_loaded')
    def test_speak_returns_engine_result(self, mock_load, tmp_path):
        """TEST-EK05: KokoroEngine.speak() returns EngineResult."""
        engine = KokoroEngine()
        engine._kokoro = Mock()
        engine._kokoro.create = Mock(return_value=(np.zeros(24000, dtype=np.float32), 24000))
        engine._kokoro.get_voices = Mock(return_value=["af_bella"])

        result = engine.speak("Hello world", voice="af_bella", save_path=tmp_path / "test.wav")

        assert isinstance(result, EngineResult)
        assert result.engine_name == "kokoro"
        assert result.voice_used == "af_bella"

    @patch.object(KokoroEngine, '_ensure_model_loaded')
    def test_speak_with_preset(self, mock_load, tmp_path):
        """TEST-EK06: KokoroEngine.speak() with preset applies voice/speed."""
        engine = KokoroEngine()
        engine._kokoro = Mock()
        engine._kokoro.get_voices = Mock(return_value=list(KOKORO_VOICES.keys()))
        engine._kokoro.create = Mock(return_value=(np.zeros(24000, dtype=np.float32), 24000))

        result = engine.speak(
            "Hello world",
            preset="narrator",
            save_path=tmp_path / "test.wav"
        )

        # Narrator preset should use bm_george
        assert result.voice_used == VOICE_PRESETS["narrator"]["voice"]

    @patch.object(KokoroEngine, '_ensure_model_loaded')
    @patch('voice_soundboard.interpreter.apply_style_to_params')
    def test_speak_with_style(self, mock_style, mock_load, tmp_path):
        """TEST-EK07: KokoroEngine.speak() with style interprets natural language."""
        mock_style.return_value = ("af_bella", 0.9, None)

        engine = KokoroEngine()
        engine._kokoro = Mock()
        engine._kokoro.get_voices = Mock(return_value=["af_bella"])
        engine._kokoro.create = Mock(return_value=(np.zeros(24000, dtype=np.float32), 24000))

        result = engine.speak(
            "Hello world",
            style="warmly and softly",
            save_path=tmp_path / "test.wav"
        )

        mock_style.assert_called_once()

    @patch.object(KokoroEngine, '_ensure_model_loaded')
    def test_speak_invalid_voice(self, mock_load):
        """Test speak with invalid voice raises error."""
        engine = KokoroEngine()
        engine._kokoro = Mock()
        engine._kokoro.get_voices = Mock(return_value=["af_bella", "am_adam"])

        with pytest.raises(ValueError) as exc_info:
            engine.speak("Hello", voice="invalid_voice")

        assert "Unknown voice" in str(exc_info.value)


class TestKokoroEngineListVoicesMocked:
    """Tests for list_voices with mocked model."""

    @patch.object(KokoroEngine, '_ensure_model_loaded')
    def test_list_voices_returns_list(self, mock_load):
        """TEST-EK09: KokoroEngine.list_voices() returns voice list."""
        engine = KokoroEngine()
        engine._kokoro = Mock()
        engine._kokoro.get_voices = Mock(return_value=["af_bella", "am_adam", "bf_emma"])

        voices = engine.list_voices()

        assert isinstance(voices, list)
        assert len(voices) == 3
        assert "af_bella" in voices


class TestKokoroEngineEnsureModelLoaded:
    """Tests for _ensure_model_loaded."""

    def test_ensure_model_loaded_skips_if_loaded(self):
        """Test that _ensure_model_loaded skips if already loaded."""
        engine = KokoroEngine()
        engine._model_loaded = True
        engine._kokoro = Mock()

        # Should not raise or do anything
        engine._ensure_model_loaded()

        # Kokoro should still be the mock, not reloaded
        assert engine._kokoro is not None

    @pytest.mark.skipif(
        True,  # Skip in environments without onnxruntime
        reason="Requires onnxruntime which may not be available"
    )
    def test_ensure_model_loaded_raises_if_model_missing(self, tmp_path):
        """Test that _ensure_model_loaded raises if model file missing."""
        engine = KokoroEngine()
        engine._model_dir = tmp_path
        engine._model_path = tmp_path / "nonexistent.onnx"
        engine._voices_path = tmp_path / "nonexistent.bin"

        with pytest.raises(FileNotFoundError) as exc_info:
            engine._ensure_model_loaded()

        assert "Model not found" in str(exc_info.value)


class TestKokoroEngineSpeedValidation:
    """Tests for speed validation in Kokoro engine."""

    @patch.object(KokoroEngine, '_ensure_model_loaded')
    def test_speed_clamped_low(self, mock_load, tmp_path):
        """Test speed below 0.5 is clamped."""
        engine = KokoroEngine()
        engine._kokoro = Mock()
        engine._kokoro.get_voices = Mock(return_value=["af_bella"])
        engine._kokoro.create = Mock(return_value=(np.zeros(24000, dtype=np.float32), 24000))

        engine.speak("Hello", voice="af_bella", speed=0.1, save_path=tmp_path / "test.wav")

        call_kwargs = engine._kokoro.create.call_args[1]
        assert call_kwargs["speed"] == 0.5

    @patch.object(KokoroEngine, '_ensure_model_loaded')
    def test_speed_clamped_high(self, mock_load, tmp_path):
        """Test speed above 2.0 is clamped."""
        engine = KokoroEngine()
        engine._kokoro = Mock()
        engine._kokoro.get_voices = Mock(return_value=["af_bella"])
        engine._kokoro.create = Mock(return_value=(np.zeros(24000, dtype=np.float32), 24000))

        engine.speak("Hello", voice="af_bella", speed=5.0, save_path=tmp_path / "test.wav")

        call_kwargs = engine._kokoro.create.call_args[1]
        assert call_kwargs["speed"] == 2.0


class TestKokoroEngineOutputPath:
    """Tests for output path handling."""

    @patch.object(KokoroEngine, '_ensure_model_loaded')
    def test_auto_generate_filename(self, mock_load, tmp_path):
        """Test that filename is auto-generated if not provided."""
        engine = KokoroEngine()
        engine.config = Config()
        engine.config.output_dir = tmp_path
        engine._kokoro = Mock()
        engine._kokoro.get_voices = Mock(return_value=["af_bella"])
        engine._kokoro.create = Mock(return_value=(np.zeros(24000, dtype=np.float32), 24000))

        result = engine.speak("Hello world", voice="af_bella")

        assert result.audio_path is not None
        assert result.audio_path.suffix == ".wav"
        assert "af_bella" in result.audio_path.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
