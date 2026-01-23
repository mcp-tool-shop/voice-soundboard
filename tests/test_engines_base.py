"""
Tests for TTS Engine Base Interface (engines/base.py).

Tests cover:
- EngineResult dataclass structure
- EngineCapabilities dataclass defaults
- TTSEngine abstract methods
- Default implementations
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from voice_soundboard.engines.base import TTSEngine, EngineResult, EngineCapabilities


class TestEngineResult:
    """Tests for EngineResult dataclass."""

    def test_result_default_values(self):
        """TEST-EB02: EngineResult dataclass has all required fields with defaults."""
        result = EngineResult()

        assert result.audio_path is None
        assert result.samples is None
        assert result.sample_rate == 24000
        assert result.duration_seconds == 0.0
        assert result.generation_time == 0.0
        assert result.voice_used == ""
        assert result.realtime_factor == 0.0
        assert result.engine_name == ""
        assert isinstance(result.metadata, dict)

    def test_result_with_all_fields(self):
        """Test EngineResult with all fields populated."""
        samples = np.zeros(24000, dtype=np.float32)
        result = EngineResult(
            audio_path=Path("/tmp/test.wav"),
            samples=samples,
            sample_rate=44100,
            duration_seconds=1.5,
            generation_time=0.3,
            voice_used="af_bella",
            realtime_factor=5.0,
            engine_name="kokoro",
            metadata={"preset": "narrator", "speed": 1.0}
        )

        assert result.audio_path == Path("/tmp/test.wav")
        assert np.array_equal(result.samples, samples)
        assert result.sample_rate == 44100
        assert result.duration_seconds == 1.5
        assert result.generation_time == 0.3
        assert result.voice_used == "af_bella"
        assert result.realtime_factor == 5.0
        assert result.engine_name == "kokoro"
        assert result.metadata["preset"] == "narrator"


class TestEngineCapabilities:
    """Tests for EngineCapabilities dataclass."""

    def test_capabilities_defaults(self):
        """TEST-EB03: EngineCapabilities dataclass defaults are correct."""
        caps = EngineCapabilities()

        # Basic features default to False
        assert caps.supports_streaming is False
        assert caps.supports_ssml is False

        # Advanced features default to False
        assert caps.supports_voice_cloning is False
        assert caps.supports_emotion_control is False
        assert caps.supports_paralinguistic_tags is False
        assert caps.supports_emotion_exaggeration is False

        # Tags list defaults to empty
        assert caps.paralinguistic_tags == []

        # Languages defaults to English
        assert caps.languages == ["en"]

        # Performance defaults
        assert caps.typical_rtf == 1.0
        assert caps.min_latency_ms == 200.0

    def test_capabilities_custom_values(self):
        """Test EngineCapabilities with custom values."""
        caps = EngineCapabilities(
            supports_streaming=True,
            supports_voice_cloning=True,
            supports_paralinguistic_tags=True,
            paralinguistic_tags=["laugh", "sigh", "cough"],
            languages=["en", "ja", "zh"],
            typical_rtf=5.0,
            min_latency_ms=150.0,
        )

        assert caps.supports_streaming is True
        assert caps.supports_voice_cloning is True
        assert caps.supports_paralinguistic_tags is True
        assert len(caps.paralinguistic_tags) == 3
        assert "laugh" in caps.paralinguistic_tags
        assert len(caps.languages) == 3
        assert caps.typical_rtf == 5.0
        assert caps.min_latency_ms == 150.0


class TestTTSEngineAbstract:
    """Tests for TTSEngine abstract base class."""

    def test_engine_is_abstract(self):
        """TEST-EB01: TTSEngine is abstract base class."""
        # Cannot instantiate directly
        with pytest.raises(TypeError):
            TTSEngine()

    def test_name_property_is_abstract(self):
        """TEST-EB04: TTSEngine.name property is abstract."""
        # Create a minimal concrete class without implementing name
        class IncompleteEngine(TTSEngine):
            @property
            def capabilities(self):
                return EngineCapabilities()

            def speak(self, text, voice=None, speed=1.0, **kwargs):
                return EngineResult()

            def speak_raw(self, text, voice=None, speed=1.0, **kwargs):
                return np.zeros(24000), 24000

            def list_voices(self):
                return []

        # Should fail because name is not implemented
        with pytest.raises(TypeError):
            IncompleteEngine()

    def test_capabilities_property_is_abstract(self):
        """TEST-EB05: TTSEngine.capabilities property is abstract."""
        class IncompleteEngine(TTSEngine):
            @property
            def name(self):
                return "test"

            def speak(self, text, voice=None, speed=1.0, **kwargs):
                return EngineResult()

            def speak_raw(self, text, voice=None, speed=1.0, **kwargs):
                return np.zeros(24000), 24000

            def list_voices(self):
                return []

        # Should fail because capabilities is not implemented
        with pytest.raises(TypeError):
            IncompleteEngine()


class TestTTSEngineConcreteImplementation:
    """Tests using a minimal concrete implementation."""

    @pytest.fixture
    def concrete_engine(self):
        """Create a minimal concrete TTSEngine implementation."""
        class TestEngine(TTSEngine):
            @property
            def name(self):
                return "test-engine"

            @property
            def capabilities(self):
                return EngineCapabilities(
                    supports_streaming=True,
                    supports_voice_cloning=False,
                )

            def speak(self, text, voice=None, speed=1.0, **kwargs):
                return EngineResult(
                    duration_seconds=1.0,
                    engine_name=self.name,
                )

            def speak_raw(self, text, voice=None, speed=1.0, **kwargs):
                samples = np.zeros(24000, dtype=np.float32)
                return samples, 24000

            def list_voices(self):
                return ["voice1", "voice2"]

        return TestEngine()

    def test_speak_is_abstract(self, concrete_engine):
        """TEST-EB06: TTSEngine.speak() returns EngineResult."""
        result = concrete_engine.speak("Hello world")
        assert isinstance(result, EngineResult)
        assert result.engine_name == "test-engine"

    def test_speak_raw_is_abstract(self, concrete_engine):
        """TEST-EB07: TTSEngine.speak_raw() returns tuple."""
        samples, sr = concrete_engine.speak_raw("Hello world")
        assert isinstance(samples, np.ndarray)
        assert sr == 24000

    def test_list_voices_is_abstract(self, concrete_engine):
        """TEST-EB08: TTSEngine.list_voices() returns list."""
        voices = concrete_engine.list_voices()
        assert isinstance(voices, list)
        assert len(voices) == 2

    def test_get_voice_info_default(self, concrete_engine):
        """TEST-EB09: TTSEngine.get_voice_info() has default implementation."""
        info = concrete_engine.get_voice_info("voice1")
        assert info["id"] == "voice1"
        assert info["name"] == "voice1"

    def test_is_loaded_default(self, concrete_engine):
        """TEST-EB12: TTSEngine.is_loaded() default returns False."""
        assert concrete_engine.is_loaded() is False

    def test_unload_default(self, concrete_engine):
        """TEST-EB13: TTSEngine.unload() default does nothing."""
        # Should not raise
        concrete_engine.unload()

    def test_clone_voice_raises_not_implemented(self, concrete_engine):
        """TEST-EB11: TTSEngine.clone_voice() raises NotImplementedError when unsupported."""
        with pytest.raises(NotImplementedError) as exc_info:
            concrete_engine.clone_voice(Path("/tmp/audio.wav"), "cloned")

        assert "does not support voice cloning" in str(exc_info.value)


class TestTTSEngineStream:
    """Tests for the default stream implementation."""

    @pytest.fixture
    def streaming_engine(self):
        """Create an engine with default stream behavior."""
        class StreamTestEngine(TTSEngine):
            @property
            def name(self):
                return "stream-test"

            @property
            def capabilities(self):
                return EngineCapabilities()

            def speak(self, text, voice=None, speed=1.0, **kwargs):
                return EngineResult()

            def speak_raw(self, text, voice=None, speed=1.0, **kwargs):
                # Return 1 second of audio
                samples = np.zeros(24000, dtype=np.float32)
                return samples, 24000

            def list_voices(self):
                return ["default"]

        return StreamTestEngine()

    @pytest.mark.asyncio
    async def test_stream_default_single_chunk(self, streaming_engine):
        """TEST-EB10: TTSEngine.stream() default yields single chunk."""
        chunks = []
        async for samples, sr in streaming_engine.stream("Hello"):
            chunks.append((samples, sr))

        # Default implementation yields all at once
        assert len(chunks) == 1
        samples, sr = chunks[0]
        assert isinstance(samples, np.ndarray)
        assert sr == 24000


class TestEngineResultMetadata:
    """Tests for EngineResult metadata handling."""

    def test_metadata_isolation(self):
        """Test that metadata dict is isolated between instances."""
        result1 = EngineResult()
        result2 = EngineResult()

        result1.metadata["key"] = "value1"

        # result2 should have its own empty metadata
        assert "key" not in result2.metadata

    def test_metadata_with_complex_data(self):
        """Test metadata with nested structures."""
        result = EngineResult(
            metadata={
                "paralinguistic_tags": ["laugh", "sigh"],
                "emotion_exaggeration": 0.7,
                "timings": {"first_chunk": 0.15, "total": 1.2},
            }
        )

        assert len(result.metadata["paralinguistic_tags"]) == 2
        assert result.metadata["emotion_exaggeration"] == 0.7
        assert result.metadata["timings"]["first_chunk"] == 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
