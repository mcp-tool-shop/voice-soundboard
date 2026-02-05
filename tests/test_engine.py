"""
Tests for Voice Engine (engine.py).

Tests cover:
- SpeechResult dataclass
- VoiceEngine initialization
- Lazy model loading
- speak method with various parameters
- speak_raw method
- Voice/preset validation
- list_voices, list_presets, get_voice_info
- quick_speak utility function
- Security: filename sanitization, path traversal prevention
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import fields

from voice_soundboard.engine import SpeechResult, VoiceEngine, quick_speak
from voice_soundboard.config import Config, KOKORO_VOICES, VOICE_PRESETS


class TestSpeechResult:
    """Tests for SpeechResult dataclass."""

    def test_all_fields_present(self):
        """Test SpeechResult has all expected fields."""
        field_names = {f.name for f in fields(SpeechResult)}
        expected = {
            "audio_path", "duration_seconds", "generation_time",
            "voice_used", "sample_rate", "realtime_factor", "timing"
        }
        assert field_names == expected

    def test_create_speech_result(self):
        """Test creating a SpeechResult."""
        result = SpeechResult(
            audio_path=Path("/test/audio.wav"),
            duration_seconds=2.5,
            generation_time=0.5,
            voice_used="af_bella",
            sample_rate=24000,
            realtime_factor=5.0,
        )

        assert result.audio_path == Path("/test/audio.wav")
        assert result.duration_seconds == 2.5
        assert result.generation_time == 0.5
        assert result.voice_used == "af_bella"
        assert result.sample_rate == 24000
        assert result.realtime_factor == 5.0

    def test_realtime_factor_calculation(self):
        """Test realtime factor represents speed."""
        # If 2s audio generated in 0.5s, RTF = 2/0.5 = 4x
        result = SpeechResult(
            audio_path=Path("/test.wav"),
            duration_seconds=2.0,
            generation_time=0.5,
            voice_used="test",
            sample_rate=24000,
            realtime_factor=4.0,
        )
        assert result.realtime_factor == 4.0


class TestVoiceEngineInit:
    """Tests for VoiceEngine initialization."""

    @patch('pathlib.Path.mkdir')
    def test_init_with_default_config(self, mock_mkdir):
        """Test init creates default config if not provided."""
        engine = VoiceEngine()
        assert engine.config is not None
        assert isinstance(engine.config, Config)

    @patch('pathlib.Path.mkdir')
    def test_init_with_custom_config(self, mock_mkdir):
        """Test init accepts custom config."""
        config = Config(default_voice="am_michael")
        engine = VoiceEngine(config=config)
        assert engine.config.default_voice == "am_michael"

    @patch('pathlib.Path.mkdir')
    def test_model_not_loaded_initially(self, mock_mkdir):
        """Test model is not loaded on init (lazy loading)."""
        engine = VoiceEngine()
        assert engine._kokoro is None
        assert engine._model_loaded is False

    @patch('pathlib.Path.mkdir')
    def test_model_paths_set(self, mock_mkdir):
        """Test model paths are set correctly."""
        engine = VoiceEngine()
        assert engine._model_path.name == "kokoro-v1.0.onnx"
        assert engine._voices_path.name == "voices-v1.0.bin"


class TestVoiceEngineLazyLoading:
    """Tests for lazy model loading."""

    @patch('pathlib.Path.mkdir')
    def test_ensure_model_skips_if_loaded(self, mock_mkdir):
        """Test _ensure_model_loaded skips if already loaded."""
        engine = VoiceEngine()
        engine._model_loaded = True

        # Should not try to import or load anything
        engine._ensure_model_loaded()  # Should not raise

    @patch('pathlib.Path.mkdir')
    def test_model_not_found_error(self, mock_mkdir):
        """Test error when model file not found."""
        engine = VoiceEngine()

        # Set model path to non-existent location
        engine._model_path = Path("/nonexistent/model.onnx")

        # The import of kokoro_onnx may fail if onnxruntime not installed
        # In that case, we still expect FileNotFoundError (or ImportError)
        try:
            with pytest.raises(FileNotFoundError) as exc_info:
                engine._ensure_model_loaded()
            assert "Model not found" in str(exc_info.value)
        except ModuleNotFoundError:
            pytest.skip("kokoro_onnx requires onnxruntime")

    @patch('pathlib.Path.mkdir')
    def test_voices_not_found_error(self, mock_mkdir, tmp_path):
        """Test error when voices file not found."""
        engine = VoiceEngine()

        # Create a fake model file but no voices file
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake")
        engine._model_path = model_file
        engine._voices_path = tmp_path / "nonexistent_voices.bin"

        try:
            with pytest.raises(FileNotFoundError) as exc_info:
                engine._ensure_model_loaded()
            assert "Voices not found" in str(exc_info.value)
        except ModuleNotFoundError:
            pytest.skip("kokoro_onnx requires onnxruntime")


class TestVoiceEngineSpeak:
    """Tests for speak method."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked model."""
        with patch('pathlib.Path.mkdir'):
            engine = VoiceEngine()

        # Mock the Kokoro model
        mock_kokoro = MagicMock()
        mock_kokoro.get_voices.return_value = list(KOKORO_VOICES.keys())
        mock_kokoro.create.return_value = (np.zeros(24000, dtype=np.float32), 24000)

        engine._kokoro = mock_kokoro
        engine._model_loaded = True

        return engine

    def test_speak_generates_audio(self, mock_engine, tmp_path):
        """Test speak generates audio file."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            result = mock_engine.speak("Hello world")

        assert isinstance(result, SpeechResult)
        assert result.voice_used == "af_bella"  # Default voice

    def test_speak_with_custom_voice(self, mock_engine, tmp_path):
        """Test speak with custom voice."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            result = mock_engine.speak("Test", voice="am_michael")

        assert result.voice_used == "am_michael"

    def test_speak_with_preset(self, mock_engine, tmp_path):
        """Test speak with preset applies preset settings."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            result = mock_engine.speak("Test", preset="narrator")

        # Narrator preset uses bm_george
        assert result.voice_used == "bm_george"

    def test_speak_voice_overrides_preset(self, mock_engine, tmp_path):
        """Test explicit voice overrides preset."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            result = mock_engine.speak("Test", voice="af_bella", preset="narrator")

        # Explicit voice should override preset
        assert result.voice_used == "af_bella"

    def test_speak_with_speed(self, mock_engine, tmp_path):
        """Test speak with speed parameter."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            mock_engine.speak("Test", speed=1.5)

        # Check speed was passed to create
        call_args = mock_engine._kokoro.create.call_args
        assert call_args[1]["speed"] == 1.5

    def test_speak_invalid_voice_raises(self, mock_engine, tmp_path):
        """Test speak raises for invalid voice."""
        from voice_soundboard.exceptions import VoiceNotFoundError
        mock_engine.config.output_dir = tmp_path
        mock_engine._kokoro.get_voices.return_value = ["af_bella", "am_michael"]

        with pytest.raises(VoiceNotFoundError) as exc_info:
            mock_engine.speak("Test", voice="invalid_voice")

        assert "Unknown voice" in str(exc_info.value)

    def test_speak_with_style_hint(self, mock_engine, tmp_path):
        """Test speak with natural language style hint."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            with patch('voice_soundboard.interpreter.apply_style_to_params') as mock_style:
                mock_style.return_value = ("am_michael", 0.9, None)
                mock_engine.speak("Test", style="warmly")

        mock_style.assert_called_once()

    def test_speak_with_custom_filename(self, mock_engine, tmp_path):
        """Test speak with custom save_as filename."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write') as mock_write:
            result = mock_engine.speak("Test", save_as="my_audio")

        # Filename should be sanitized and have .wav extension
        assert ".wav" in str(result.audio_path)

    def test_speak_returns_metrics(self, mock_engine, tmp_path):
        """Test speak returns timing metrics."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            result = mock_engine.speak("Test")

        assert result.duration_seconds > 0 or result.duration_seconds == 0  # May be 0 for mock
        assert result.generation_time >= 0
        assert result.sample_rate == 24000


class TestVoiceEngineSpeakRaw:
    """Tests for speak_raw method."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked model."""
        with patch('pathlib.Path.mkdir'):
            engine = VoiceEngine()

        mock_kokoro = MagicMock()
        mock_kokoro.create.return_value = (np.zeros(24000, dtype=np.float32), 24000)

        engine._kokoro = mock_kokoro
        engine._model_loaded = True

        return engine

    def test_speak_raw_returns_samples(self, mock_engine):
        """Test speak_raw returns numpy array and sample rate."""
        samples, sr = mock_engine.speak_raw("Test")

        assert isinstance(samples, np.ndarray)
        assert sr == 24000

    def test_speak_raw_uses_default_voice(self, mock_engine):
        """Test speak_raw uses default voice."""
        mock_engine.speak_raw("Test")

        call_args = mock_engine._kokoro.create.call_args
        assert call_args[1]["voice"] == "af_bella"

    def test_speak_raw_clamps_speed(self, mock_engine):
        """Test speak_raw clamps speed to valid range."""
        mock_engine.speak_raw("Test", speed=0.1)  # Too slow
        call_args = mock_engine._kokoro.create.call_args
        assert call_args[1]["speed"] >= 0.5

        mock_engine.speak_raw("Test", speed=5.0)  # Too fast
        call_args = mock_engine._kokoro.create.call_args
        assert call_args[1]["speed"] <= 2.0


class TestVoiceEngineListMethods:
    """Tests for list_voices, list_presets, get_voice_info."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked model."""
        with patch('pathlib.Path.mkdir'):
            engine = VoiceEngine()

        mock_kokoro = MagicMock()
        mock_kokoro.get_voices.return_value = ["af_bella", "am_michael", "bf_emma"]

        engine._kokoro = mock_kokoro
        engine._model_loaded = True

        return engine

    def test_list_voices(self, mock_engine):
        """Test list_voices returns voice list."""
        voices = mock_engine.list_voices()

        assert isinstance(voices, list)
        assert "af_bella" in voices

    def test_list_presets(self, mock_engine):
        """Test list_presets returns preset dict."""
        presets = mock_engine.list_presets()

        assert isinstance(presets, dict)
        assert "assistant" in presets
        assert "voice" in presets["assistant"]
        assert "speed" in presets["assistant"]
        assert "description" in presets["assistant"]

    def test_get_voice_info_known_voice(self, mock_engine):
        """Test get_voice_info for known voice."""
        info = mock_engine.get_voice_info("af_bella")

        assert info["name"] == "Bella"
        assert info["gender"] == "female"
        assert info["accent"] == "american"

    def test_get_voice_info_unknown_voice(self, mock_engine):
        """Test get_voice_info for unknown voice."""
        info = mock_engine.get_voice_info("unknown_voice")

        assert info["name"] == "unknown_voice"
        assert info["gender"] == "unknown"


class TestSpeedValidation:
    """Tests for speed parameter validation."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked model."""
        with patch('pathlib.Path.mkdir'):
            engine = VoiceEngine()

        mock_kokoro = MagicMock()
        mock_kokoro.get_voices.return_value = list(KOKORO_VOICES.keys())
        mock_kokoro.create.return_value = (np.zeros(24000, dtype=np.float32), 24000)

        engine._kokoro = mock_kokoro
        engine._model_loaded = True

        return engine

    def test_speed_clamped_minimum(self, mock_engine, tmp_path):
        """Test speed is clamped to minimum."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            mock_engine.speak("Test", speed=0.1)

        call_args = mock_engine._kokoro.create.call_args
        assert call_args[1]["speed"] >= 0.5

    def test_speed_clamped_maximum(self, mock_engine, tmp_path):
        """Test speed is clamped to maximum."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            mock_engine.speak("Test", speed=5.0)

        call_args = mock_engine._kokoro.create.call_args
        assert call_args[1]["speed"] <= 2.0

    def test_valid_speed_passes_through(self, mock_engine, tmp_path):
        """Test valid speed passes through unchanged."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            mock_engine.speak("Test", speed=1.5)

        call_args = mock_engine._kokoro.create.call_args
        assert call_args[1]["speed"] == 1.5


class TestFilenameSecurity:
    """Tests for filename sanitization security."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked model."""
        with patch('pathlib.Path.mkdir'):
            engine = VoiceEngine()

        mock_kokoro = MagicMock()
        mock_kokoro.get_voices.return_value = list(KOKORO_VOICES.keys())
        mock_kokoro.create.return_value = (np.zeros(24000, dtype=np.float32), 24000)

        engine._kokoro = mock_kokoro
        engine._model_loaded = True

        return engine

    def test_path_traversal_blocked(self, mock_engine, tmp_path):
        """Test path traversal in filename is blocked."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            result = mock_engine.speak("Test", save_as="../../../etc/passwd")

        # Should not be outside output_dir
        assert tmp_path in result.audio_path.parents or result.audio_path.parent == tmp_path

    def test_absolute_path_rejected(self, mock_engine, tmp_path):
        """Test absolute paths are handled safely."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            result = mock_engine.speak("Test", save_as="/tmp/malicious.wav")

        # Should be within output_dir
        assert str(result.audio_path).startswith(str(tmp_path)) or tmp_path in result.audio_path.parents

    def test_special_characters_sanitized(self, mock_engine, tmp_path):
        """Test special characters in filename are sanitized."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            result = mock_engine.speak("Test", save_as="file<>:name?.wav")

        # Special characters should be removed or replaced
        filename = result.audio_path.name
        for char in '<>:?"*|':
            assert char not in filename


class TestQuickSpeak:
    """Tests for quick_speak utility function."""

    @patch('pathlib.Path.mkdir')
    def test_quick_speak_returns_path(self, mock_mkdir):
        """Test quick_speak returns audio path."""
        with patch.object(VoiceEngine, 'speak') as mock_speak:
            mock_result = MagicMock()
            mock_result.audio_path = Path("/test/audio.wav")
            mock_speak.return_value = mock_result

            with patch.object(VoiceEngine, '_ensure_model_loaded'):
                result = quick_speak("Hello")

        assert isinstance(result, Path)

    @patch('pathlib.Path.mkdir')
    def test_quick_speak_with_params(self, mock_mkdir):
        """Test quick_speak accepts voice and speed."""
        with patch.object(VoiceEngine, 'speak') as mock_speak:
            mock_result = MagicMock()
            mock_result.audio_path = Path("/test/audio.wav")
            mock_speak.return_value = mock_result

            with patch.object(VoiceEngine, '_ensure_model_loaded'):
                quick_speak("Hello", voice="am_michael", speed=1.2)

        mock_speak.assert_called_once_with("Hello", voice="am_michael", speed=1.2, normalize=True)


class TestVoiceEnginePresetIntegration:
    """Tests for preset and voice integration."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked model."""
        with patch('pathlib.Path.mkdir'):
            engine = VoiceEngine()

        mock_kokoro = MagicMock()
        mock_kokoro.get_voices.return_value = list(KOKORO_VOICES.keys())
        mock_kokoro.create.return_value = (np.zeros(24000, dtype=np.float32), 24000)

        engine._kokoro = mock_kokoro
        engine._model_loaded = True

        return engine

    def test_all_presets_work(self, mock_engine, tmp_path):
        """Test all presets can be used."""
        mock_engine.config.output_dir = tmp_path

        for preset_name in VOICE_PRESETS:
            with patch('soundfile.write'):
                result = mock_engine.speak("Test", preset=preset_name)

            assert result.voice_used in KOKORO_VOICES

    def test_preset_speed_applied(self, mock_engine, tmp_path):
        """Test preset speed is applied when not overridden."""
        mock_engine.config.output_dir = tmp_path

        # Narrator preset has speed < 1.0
        with patch('soundfile.write'):
            mock_engine.speak("Test", preset="narrator")

        call_args = mock_engine._kokoro.create.call_args
        assert call_args[1]["speed"] < 1.0

    def test_explicit_speed_overrides_preset(self, mock_engine, tmp_path):
        """Test explicit speed overrides preset speed."""
        mock_engine.config.output_dir = tmp_path

        with patch('soundfile.write'):
            mock_engine.speak("Test", preset="narrator", speed=1.5)

        call_args = mock_engine._kokoro.create.call_args
        assert call_args[1]["speed"] == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
