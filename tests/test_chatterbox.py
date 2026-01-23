"""
Tests for Chatterbox TTS Engine integration.

Tests cover:
- Paralinguistic tag parsing
- Emotion exaggeration control
- Voice cloning registration
- Engine capabilities
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from voice_soundboard.engines.base import TTSEngine, EngineResult, EngineCapabilities
from voice_soundboard.engines.chatterbox import (
    ChatterboxEngine,
    PARALINGUISTIC_TAGS,
    validate_paralinguistic_tags,
    has_paralinguistic_tags,
)


class TestParalinguisticTags:
    """Tests for paralinguistic tag parsing."""

    def test_all_tags_defined(self):
        """Verify all expected paralinguistic tags are defined."""
        expected_tags = [
            "laugh", "chuckle", "cough", "sigh", "gasp",
            "groan", "sniff", "shush", "clear throat"
        ]
        for tag in expected_tags:
            assert tag in PARALINGUISTIC_TAGS, f"Missing tag: {tag}"

    def test_validate_tags_single(self):
        """Test extracting a single tag."""
        text = "Hello [laugh] world"
        tags = validate_paralinguistic_tags(text)
        assert tags == ["laugh"]

    def test_validate_tags_multiple(self):
        """Test extracting multiple tags."""
        text = "That's hilarious! [laugh] Oh man, [sigh] I needed that."
        tags = validate_paralinguistic_tags(text)
        assert set(tags) == {"laugh", "sigh"}

    def test_validate_tags_case_insensitive(self):
        """Test that tag matching is case-insensitive."""
        text = "Hello [LAUGH] world [Sigh]"
        tags = validate_paralinguistic_tags(text)
        assert set(tags) == {"laugh", "sigh"}

    def test_validate_tags_none(self):
        """Test text with no tags."""
        text = "Hello world, no tags here!"
        tags = validate_paralinguistic_tags(text)
        assert tags == []

    def test_validate_tags_invalid_ignored(self):
        """Test that invalid tags are ignored."""
        text = "Hello [laugh] [invalid_tag] [sigh] world"
        tags = validate_paralinguistic_tags(text)
        assert set(tags) == {"laugh", "sigh"}
        assert "invalid_tag" not in tags

    def test_has_paralinguistic_tags_true(self):
        """Test detection of tags in text."""
        assert has_paralinguistic_tags("Hello [laugh] world")
        assert has_paralinguistic_tags("[cough] excuse me")
        assert has_paralinguistic_tags("Let me think... [sigh]")

    def test_has_paralinguistic_tags_false(self):
        """Test text without tags."""
        assert not has_paralinguistic_tags("Hello world")
        assert not has_paralinguistic_tags("No tags here [invalid]")
        assert not has_paralinguistic_tags("")

    def test_clear_throat_tag(self):
        """Test multi-word tag 'clear throat'."""
        text = "Ahem [clear throat] as I was saying..."
        tags = validate_paralinguistic_tags(text)
        assert "clear throat" in tags


class TestChatterboxEngineCapabilities:
    """Tests for engine capability reporting."""

    def test_engine_name(self):
        """Test engine name property."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine.model_variant = "turbo"
        assert engine.name == "chatterbox-turbo"

        engine.model_variant = "standard"
        assert engine.name == "chatterbox-standard"

    def test_capabilities_structure(self):
        """Test that capabilities return correct structure."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        caps = engine.capabilities

        assert isinstance(caps, EngineCapabilities)
        assert caps.supports_paralinguistic_tags is True
        assert caps.supports_emotion_exaggeration is True
        assert caps.supports_voice_cloning is True
        assert caps.supports_streaming is True

    def test_capabilities_tags_list(self):
        """Test that capabilities include paralinguistic tags list."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        caps = engine.capabilities

        assert len(caps.paralinguistic_tags) > 0
        assert "laugh" in caps.paralinguistic_tags
        assert "sigh" in caps.paralinguistic_tags


class TestChatterboxEngineVoiceCloning:
    """Tests for voice cloning functionality."""

    def test_clone_voice_registers_voice(self, tmp_path):
        """Test that clone_voice registers the voice."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._cloned_voices = {}

        # Create a dummy audio file
        audio_file = tmp_path / "reference.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)  # Minimal WAV header

        voice_id = engine.clone_voice(audio_file, "my_voice")

        assert voice_id == "my_voice"
        assert "my_voice" in engine._cloned_voices
        assert engine._cloned_voices["my_voice"] == audio_file

    def test_clone_voice_file_not_found(self):
        """Test clone_voice raises error for missing file."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._cloned_voices = {}

        with pytest.raises(FileNotFoundError):
            engine.clone_voice(Path("/nonexistent/audio.wav"), "test")

    def test_list_cloned_voices(self, tmp_path):
        """Test listing cloned voices."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._cloned_voices = {}

        # Register some voices
        audio1 = tmp_path / "voice1.wav"
        audio2 = tmp_path / "voice2.wav"
        audio1.write_bytes(b"RIFF" + b"\x00" * 100)
        audio2.write_bytes(b"RIFF" + b"\x00" * 100)

        engine.clone_voice(audio1, "alice")
        engine.clone_voice(audio2, "bob")

        voices = engine.list_cloned_voices()

        assert len(voices) == 2
        assert "alice" in voices
        assert "bob" in voices

    def test_remove_cloned_voice(self, tmp_path):
        """Test removing a cloned voice."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._cloned_voices = {}

        audio = tmp_path / "voice.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        engine.clone_voice(audio, "temp_voice")
        assert "temp_voice" in engine._cloned_voices

        result = engine.remove_cloned_voice("temp_voice")
        assert result is True
        assert "temp_voice" not in engine._cloned_voices

    def test_remove_nonexistent_voice(self):
        """Test removing a voice that doesn't exist."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._cloned_voices = {}

        result = engine.remove_cloned_voice("nonexistent")
        assert result is False


class TestChatterboxEngineFormatWithTags:
    """Tests for the format_with_tags utility."""

    def test_format_with_single_tag(self):
        """Test inserting a single tag."""
        result = ChatterboxEngine.format_with_tags(
            "Hello how are you",
            {"laugh": [1]}
        )
        assert "[laugh]" in result
        assert result == "Hello how [laugh] are you"

    def test_format_with_multiple_tags(self):
        """Test inserting multiple tags."""
        result = ChatterboxEngine.format_with_tags(
            "Hello world this is great",
            {"laugh": [1], "sigh": [3]}
        )
        assert "[laugh]" in result
        assert "[sigh]" in result

    def test_format_with_no_tags(self):
        """Test with empty tags dict."""
        result = ChatterboxEngine.format_with_tags("Hello world", {})
        assert result == "Hello world"


class TestChatterboxEngineInit:
    """Tests for engine initialization."""

    def test_init_defaults(self):
        """Test default initialization values."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine.__init__()

        assert engine.model_variant == "turbo"
        assert engine.device == "cuda"
        assert engine.default_exaggeration == 0.5
        assert engine.default_cfg_weight == 0.5
        assert engine._model is None
        assert engine._model_loaded is False

    def test_init_custom_variant(self):
        """Test initialization with custom model variant."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine.__init__(model_variant="standard", device="cpu")

        assert engine.model_variant == "standard"
        assert engine.device == "cpu"

    def test_is_loaded_false_initially(self):
        """Test that model is not loaded initially."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._model_loaded = False

        assert engine.is_loaded() is False

    def test_unload_clears_model(self):
        """Test that unload clears the model."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine._model = Mock()
        engine._model_loaded = True

        engine.unload()

        assert engine._model is None
        assert engine._model_loaded is False


class TestChatterboxEngineSpeakRaw:
    """Tests for speak_raw with mocked model."""

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_speak_raw_returns_tuple(self, mock_load):
        """Test that speak_raw returns (samples, sample_rate) tuple."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine.config = Mock()
        engine.config.default_voice = None
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._cloned_voices = {}
        engine.default_exaggeration = 0.5
        engine.default_cfg_weight = 0.5

        samples, sr = engine.speak_raw("Hello world")

        assert isinstance(samples, np.ndarray)
        assert sr == 24000

    @patch('voice_soundboard.engines.chatterbox.ChatterboxEngine._ensure_model_loaded')
    def test_speak_raw_with_exaggeration(self, mock_load):
        """Test speak_raw passes exaggeration parameter."""
        engine = ChatterboxEngine.__new__(ChatterboxEngine)
        engine.config = Mock()
        engine.config.default_voice = None
        engine._model = Mock()
        engine._model.sr = 24000
        engine._model.generate = Mock(return_value=np.zeros(24000))
        engine._cloned_voices = {}
        engine.default_exaggeration = 0.5
        engine.default_cfg_weight = 0.5

        engine.speak_raw("Hello", emotion_exaggeration=0.9, cfg_weight=0.3)

        # Verify generate was called with correct params
        engine._model.generate.assert_called_once()
        call_kwargs = engine._model.generate.call_args[1]
        assert call_kwargs["exaggeration"] == 0.9
        assert call_kwargs["cfg_weight"] == 0.3


class TestEngineResultMetadata:
    """Tests for EngineResult metadata handling."""

    def test_result_includes_paralinguistic_tags(self):
        """Test that result metadata includes detected tags."""
        result = EngineResult(
            audio_path=Path("/tmp/test.wav"),
            duration_seconds=1.0,
            engine_name="chatterbox-turbo",
            metadata={
                "paralinguistic_tags": ["laugh", "sigh"],
                "emotion_exaggeration": 0.7,
            }
        )

        assert "paralinguistic_tags" in result.metadata
        assert result.metadata["paralinguistic_tags"] == ["laugh", "sigh"]
        assert result.metadata["emotion_exaggeration"] == 0.7


class TestListParalinguisticTags:
    """Tests for the static list_paralinguistic_tags method."""

    def test_returns_copy(self):
        """Test that list returns a copy, not the original."""
        tags1 = ChatterboxEngine.list_paralinguistic_tags()
        tags2 = ChatterboxEngine.list_paralinguistic_tags()

        # Should be equal but not the same object
        assert tags1 == tags2
        assert tags1 is not tags2

        # Modifying one shouldn't affect the other
        tags1.append("test")
        assert "test" not in tags2

    def test_contains_expected_tags(self):
        """Test that all expected tags are included."""
        tags = ChatterboxEngine.list_paralinguistic_tags()

        expected = ["laugh", "chuckle", "cough", "sigh", "gasp"]
        for tag in expected:
            assert tag in tags


# Integration test (requires chatterbox to be installed)
@pytest.mark.skip(reason="Integration test - requires chatterbox-tts installed")
class TestChatterboxIntegration:
    """Integration tests that require actual chatterbox installation."""

    def test_full_generation(self, tmp_path):
        """Test full speech generation with Chatterbox."""
        engine = ChatterboxEngine(device="cuda")

        result = engine.speak(
            "Hello! [laugh] That's funny.",
            emotion_exaggeration=0.7,
            save_path=tmp_path / "output.wav"
        )

        assert result.audio_path.exists()
        assert result.duration_seconds > 0
        assert "laugh" in result.metadata.get("paralinguistic_tags", [])
