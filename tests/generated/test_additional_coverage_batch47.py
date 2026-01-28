"""
Test Additional Coverage Batch 47: Engine Extensions Tests

Tests for:
- EngineResult dataclass
- EngineCapabilities dataclass
- TTSEngine abstract base class
- KokoroEngine implementation
- ChatterboxEngine implementation
- F5TTSEngine implementation
- Paralinguistic tag validation
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


# ============== EngineResult Tests ==============

class TestEngineResult:
    """Tests for EngineResult dataclass."""

    def test_engine_result_default_values(self):
        """Test EngineResult has correct defaults."""
        from voice_soundboard.engines.base import EngineResult
        result = EngineResult()
        assert result.audio_path is None
        assert result.samples is None
        assert result.sample_rate == 24000
        assert result.duration_seconds == 0.0
        assert result.voice_used == ""

    def test_engine_result_with_audio_path(self):
        """Test EngineResult with audio path."""
        from voice_soundboard.engines.base import EngineResult
        result = EngineResult(audio_path=Path("/tmp/audio.wav"))
        assert result.audio_path == Path("/tmp/audio.wav")

    def test_engine_result_with_samples(self):
        """Test EngineResult with samples array."""
        from voice_soundboard.engines.base import EngineResult
        samples = np.zeros(1000, dtype=np.float32)
        result = EngineResult(samples=samples, sample_rate=22050)
        assert len(result.samples) == 1000
        assert result.sample_rate == 22050

    def test_engine_result_with_metadata(self):
        """Test EngineResult with custom metadata."""
        from voice_soundboard.engines.base import EngineResult
        result = EngineResult(
            engine_name="kokoro",
            metadata={"preset": "narrator", "speed": 1.2}
        )
        assert result.engine_name == "kokoro"
        assert result.metadata["preset"] == "narrator"

    def test_engine_result_realtime_factor(self):
        """Test EngineResult realtime factor."""
        from voice_soundboard.engines.base import EngineResult
        result = EngineResult(
            duration_seconds=2.0,
            generation_time=0.4,
            realtime_factor=5.0
        )
        assert result.realtime_factor == 5.0


# ============== EngineCapabilities Tests ==============

class TestEngineCapabilities:
    """Tests for EngineCapabilities dataclass."""

    def test_capabilities_default_values(self):
        """Test EngineCapabilities default values."""
        from voice_soundboard.engines.base import EngineCapabilities
        caps = EngineCapabilities()
        assert caps.supports_streaming is False
        assert caps.supports_ssml is False
        assert caps.supports_voice_cloning is False
        assert caps.languages == ["en"]

    def test_capabilities_with_streaming(self):
        """Test EngineCapabilities with streaming support."""
        from voice_soundboard.engines.base import EngineCapabilities
        caps = EngineCapabilities(supports_streaming=True)
        assert caps.supports_streaming is True

    def test_capabilities_with_voice_cloning(self):
        """Test EngineCapabilities with voice cloning support."""
        from voice_soundboard.engines.base import EngineCapabilities
        caps = EngineCapabilities(supports_voice_cloning=True)
        assert caps.supports_voice_cloning is True

    def test_capabilities_paralinguistic_tags(self):
        """Test EngineCapabilities paralinguistic tags."""
        from voice_soundboard.engines.base import EngineCapabilities
        tags = ["laugh", "cough", "sigh"]
        caps = EngineCapabilities(
            supports_paralinguistic_tags=True,
            paralinguistic_tags=tags
        )
        assert caps.supports_paralinguistic_tags is True
        assert "laugh" in caps.paralinguistic_tags

    def test_capabilities_multiple_languages(self):
        """Test EngineCapabilities with multiple languages."""
        from voice_soundboard.engines.base import EngineCapabilities
        caps = EngineCapabilities(languages=["en", "ja", "zh", "fr"])
        assert len(caps.languages) == 4
        assert "ja" in caps.languages


# ============== TTSEngine Abstract Tests ==============

class TestTTSEngineAbstract:
    """Tests for TTSEngine abstract base class."""

    def test_tts_engine_is_abstract(self):
        """Test TTSEngine cannot be instantiated directly."""
        from voice_soundboard.engines.base import TTSEngine
        with pytest.raises(TypeError):
            TTSEngine()

    def test_tts_engine_clone_voice_not_supported(self):
        """Test clone_voice raises error when not supported."""
        from voice_soundboard.engines.base import TTSEngine, EngineCapabilities

        class TestEngine(TTSEngine):
            @property
            def name(self):
                return "test"

            @property
            def capabilities(self):
                return EngineCapabilities(supports_voice_cloning=False)

            def speak(self, text, voice=None, speed=1.0, **kwargs):
                pass

            def speak_raw(self, text, voice=None, speed=1.0, **kwargs):
                return np.zeros(100), 24000

            def list_voices(self):
                return []

        engine = TestEngine()
        with pytest.raises(NotImplementedError):
            engine.clone_voice(Path("/tmp/sample.wav"))

    def test_tts_engine_get_voice_info_default(self):
        """Test get_voice_info default implementation."""
        from voice_soundboard.engines.base import TTSEngine, EngineCapabilities

        class TestEngine(TTSEngine):
            @property
            def name(self):
                return "test"

            @property
            def capabilities(self):
                return EngineCapabilities()

            def speak(self, text, voice=None, speed=1.0, **kwargs):
                pass

            def speak_raw(self, text, voice=None, speed=1.0, **kwargs):
                return np.zeros(100), 24000

            def list_voices(self):
                return ["voice_a"]

        engine = TestEngine()
        info = engine.get_voice_info("voice_a")
        assert info["id"] == "voice_a"

    def test_tts_engine_is_loaded_default(self):
        """Test is_loaded returns False by default."""
        from voice_soundboard.engines.base import TTSEngine, EngineCapabilities

        class TestEngine(TTSEngine):
            @property
            def name(self):
                return "test"

            @property
            def capabilities(self):
                return EngineCapabilities()

            def speak(self, text, voice=None, speed=1.0, **kwargs):
                pass

            def speak_raw(self, text, voice=None, speed=1.0, **kwargs):
                return np.zeros(100), 24000

            def list_voices(self):
                return []

        engine = TestEngine()
        assert engine.is_loaded() is False


# ============== Paralinguistic Tag Tests ==============

class TestParalinguisticTags:
    """Tests for paralinguistic tag validation."""

    def test_validate_paralinguistic_tags_found(self):
        """Test validate_paralinguistic_tags finds tags."""
        from voice_soundboard.engines.chatterbox import validate_paralinguistic_tags
        text = "That's hilarious! [laugh] Oh man..."
        tags = validate_paralinguistic_tags(text)
        assert "laugh" in tags

    def test_validate_paralinguistic_tags_multiple(self):
        """Test validate_paralinguistic_tags finds multiple tags."""
        from voice_soundboard.engines.chatterbox import validate_paralinguistic_tags
        text = "[sigh] I don't know... [cough] Excuse me."
        tags = validate_paralinguistic_tags(text)
        assert "sigh" in tags
        assert "cough" in tags

    def test_validate_paralinguistic_tags_case_insensitive(self):
        """Test paralinguistic tags are case insensitive."""
        from voice_soundboard.engines.chatterbox import validate_paralinguistic_tags
        text = "[LAUGH] and [Sigh]"
        tags = validate_paralinguistic_tags(text)
        assert "laugh" in tags
        assert "sigh" in tags

    def test_has_paralinguistic_tags_true(self):
        """Test has_paralinguistic_tags returns True when present."""
        from voice_soundboard.engines.chatterbox import has_paralinguistic_tags
        text = "That's funny [laugh]"
        assert has_paralinguistic_tags(text) is True

    def test_has_paralinguistic_tags_false(self):
        """Test has_paralinguistic_tags returns False when absent."""
        from voice_soundboard.engines.chatterbox import has_paralinguistic_tags
        text = "That's funny"
        assert has_paralinguistic_tags(text) is False


# ============== ChatterboxEngine Tests ==============

class TestChatterboxEngine:
    """Tests for ChatterboxEngine."""

    def test_chatterbox_engine_name(self):
        """Test ChatterboxEngine name property."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine
        engine = ChatterboxEngine(model_variant="turbo")
        assert engine.name == "chatterbox-turbo"

    def test_chatterbox_engine_capabilities(self):
        """Test ChatterboxEngine capabilities."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine
        engine = ChatterboxEngine()
        caps = engine.capabilities
        assert caps.supports_voice_cloning is True
        assert caps.supports_paralinguistic_tags is True
        assert "laugh" in caps.paralinguistic_tags

    def test_chatterbox_multilingual_languages(self):
        """Test ChatterboxEngine multilingual has 23 languages."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine
        engine = ChatterboxEngine(model_variant="multilingual")
        caps = engine.capabilities
        assert len(caps.languages) == 23

    def test_chatterbox_turbo_english_only(self):
        """Test ChatterboxEngine turbo is English only."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine
        engine = ChatterboxEngine(model_variant="turbo")
        caps = engine.capabilities
        assert caps.languages == ["en"]

    def test_chatterbox_languages_constant(self):
        """Test CHATTERBOX_LANGUAGES constant."""
        from voice_soundboard.engines.chatterbox import CHATTERBOX_LANGUAGES
        assert "en" in CHATTERBOX_LANGUAGES
        assert "fr" in CHATTERBOX_LANGUAGES
        assert "ja" in CHATTERBOX_LANGUAGES
        assert len(CHATTERBOX_LANGUAGES) == 23


# ============== F5TTSEngine Tests ==============

class TestF5TTSEngine:
    """Tests for F5TTSEngine."""

    def test_f5tts_engine_name(self):
        """Test F5TTSEngine name property."""
        from voice_soundboard.engines.f5tts import F5TTSEngine
        engine = F5TTSEngine()
        assert "f5-tts" in engine.name

    def test_f5tts_engine_capabilities(self):
        """Test F5TTSEngine capabilities."""
        from voice_soundboard.engines.f5tts import F5TTSEngine
        engine = F5TTSEngine()
        caps = engine.capabilities
        assert caps.supports_voice_cloning is True
        assert caps.supports_streaming is False  # Diffusion models

    def test_f5tts_default_parameters(self):
        """Test F5TTSEngine default parameters."""
        from voice_soundboard.engines.f5tts import F5TTSEngine
        engine = F5TTSEngine()
        assert engine.default_cfg_strength == 2.0
        assert engine.default_nfe_step == 32


# ============== KokoroEngine Tests ==============

class TestKokoroEngine:
    """Tests for KokoroEngine."""

    def test_kokoro_engine_name(self):
        """Test KokoroEngine name property."""
        from voice_soundboard.engines.kokoro import KokoroEngine
        engine = KokoroEngine()
        assert engine.name == "kokoro"

    def test_kokoro_engine_capabilities(self):
        """Test KokoroEngine capabilities."""
        from voice_soundboard.engines.kokoro import KokoroEngine
        engine = KokoroEngine()
        caps = engine.capabilities
        assert caps.supports_streaming is True
        assert caps.supports_ssml is True
        assert caps.supports_voice_cloning is False

    def test_kokoro_supported_languages(self):
        """Test KokoroEngine supported languages."""
        from voice_soundboard.engines.kokoro import KokoroEngine
        engine = KokoroEngine()
        caps = engine.capabilities
        assert "en" in caps.languages
        assert "ja" in caps.languages
