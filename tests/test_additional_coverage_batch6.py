"""
Additional tests - Batch 6.

Covers remaining unchecked items from TEST_PLAN.md:
- llm/streaming.py (StreamConfig, StreamBuffer, SentenceBoundaryDetector, StreamingLLMSpeaker)
- llm/providers.py (LLMConfig, LLMProvider, MockLLMProvider, create_provider)
- conversion/base.py (ConversionConfig, ConversionResult, VoiceConverter, MockVoiceConverter)
"""

import pytest
import numpy as np
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import time


# =============================================================================
# Module: llm/streaming.py - StreamConfig Tests
# =============================================================================

class TestStreamConfig:
    """Tests for StreamConfig dataclass."""

    def test_stream_config_defaults(self):
        """Test StreamConfig default values."""
        from voice_soundboard.llm.streaming import StreamConfig

        config = StreamConfig()

        assert config.sentence_end_chars == ".!?"
        assert config.min_sentence_length == 10
        assert config.max_buffer_length == 500
        assert config.flush_timeout_ms == 2000.0
        assert config.inter_sentence_pause_ms == 200.0
        assert config.speed == 1.0
        assert config.allow_partial_sentences is True
        assert config.smart_punctuation is True

    def test_stream_config_custom_values(self):
        """Test StreamConfig with custom values."""
        from voice_soundboard.llm.streaming import StreamConfig

        config = StreamConfig(
            sentence_end_chars=".!?;",
            min_sentence_length=20,
            speed=1.5,
            voice="custom_voice",
        )

        assert config.sentence_end_chars == ".!?;"
        assert config.min_sentence_length == 20
        assert config.speed == 1.5
        assert config.voice == "custom_voice"


class TestStreamBuffer:
    """Tests for StreamBuffer dataclass."""

    def test_stream_buffer_initialization(self):
        """Test StreamBuffer initialization."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()

        assert buffer.content == ""
        assert buffer.sentences_spoken == 0
        assert buffer.tokens_received == 0

    def test_stream_buffer_append(self):
        """Test StreamBuffer.append()."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("Hello")
        buffer.append(" world")

        assert buffer.content == "Hello world"
        assert buffer.tokens_received == 2

    def test_stream_buffer_clear(self):
        """Test StreamBuffer.clear()."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("Hello world")

        content = buffer.clear()

        assert content == "Hello world"
        assert buffer.content == ""

    def test_stream_buffer_peek(self):
        """Test StreamBuffer.peek()."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("Hello")

        assert buffer.peek() == "Hello"
        assert buffer.content == "Hello"  # Not cleared

    def test_stream_buffer_length(self):
        """Test StreamBuffer.length property."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        assert buffer.length == 0

        buffer.append("Hello")
        assert buffer.length == 5

    def test_stream_buffer_age_ms(self):
        """Test StreamBuffer.age_ms property."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("Hello")

        # Age should be very small (just created)
        assert buffer.age_ms < 100  # Less than 100ms

        # Wait a bit
        time.sleep(0.05)  # 50ms
        assert buffer.age_ms >= 45  # At least 45ms (allowing for timing variance)


class TestStreamState:
    """Tests for StreamState enum."""

    def test_stream_state_values(self):
        """Test StreamState enum values."""
        from voice_soundboard.llm.streaming import StreamState

        assert StreamState.IDLE.value == "idle"
        assert StreamState.BUFFERING.value == "buffering"
        assert StreamState.SPEAKING.value == "speaking"
        assert StreamState.FINISHING.value == "finishing"
        assert StreamState.ERROR.value == "error"


# =============================================================================
# Module: llm/streaming.py - SentenceBoundaryDetector Tests
# =============================================================================

class TestSentenceBoundaryDetector:
    """Tests for SentenceBoundaryDetector class."""

    def test_detector_initialization(self):
        """Test SentenceBoundaryDetector initialization."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        assert detector.config is not None

    def test_find_boundary_simple_sentence(self):
        """Test finding boundary in simple sentence."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        text = "Hello world. How are you?"
        boundary = detector.find_boundary(text)

        assert boundary is not None
        assert boundary == 13  # After "Hello world. "

    def test_find_boundary_question(self):
        """Test finding boundary with question mark."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        text = "How are you? I am fine."
        boundary = detector.find_boundary(text)

        assert boundary is not None
        # Should find the first sentence ending

    def test_find_boundary_exclamation(self):
        """Test finding boundary with exclamation mark."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        text = "Hello world! This is great."
        boundary = detector.find_boundary(text)

        assert boundary is not None

    def test_find_boundary_too_short(self):
        """Test no boundary found for short text."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector, StreamConfig

        config = StreamConfig(min_sentence_length=10)
        detector = SentenceBoundaryDetector(config)

        text = "Hi."  # Too short
        boundary = detector.find_boundary(text)

        assert boundary is None

    def test_find_boundary_abbreviation(self):
        """Test abbreviations are not treated as sentence endings."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        text = "Dr. Smith is here. He is a doctor."
        boundary = detector.find_boundary(text)

        # Should find boundary after "here." not after "Dr."
        assert boundary is not None
        assert "Dr. Smith is here" in text[:boundary]

    def test_find_boundary_decimal_number(self):
        """Test decimal numbers are not treated as sentence endings."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        text = "The value is 3.14 and that is pi."
        boundary = detector.find_boundary(text)

        # Should find boundary after "pi." not after "3.14"
        assert boundary is not None

    def test_split_sentences(self):
        """Test splitting text into sentences."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        text = "Hello world. How are you? I am fine!"
        sentences = detector.split_sentences(text)

        assert len(sentences) >= 2

    def test_extract_complete(self):
        """Test extracting complete sentences with remainder."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        text = "Hello world. This is incomplete"
        sentences, remaining = detector.extract_complete(text)

        assert len(sentences) == 1
        assert "Hello world" in sentences[0]
        assert "incomplete" in remaining


# =============================================================================
# Module: llm/streaming.py - StreamingLLMSpeaker Tests
# =============================================================================

class TestStreamingLLMSpeaker:
    """Tests for StreamingLLMSpeaker class."""

    def test_speaker_initialization(self):
        """Test StreamingLLMSpeaker initialization."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState

        speaker = StreamingLLMSpeaker()

        assert speaker.state == StreamState.IDLE
        assert speaker.sentences_spoken == 0
        assert speaker.total_tokens == 0

    def test_speaker_with_config(self):
        """Test StreamingLLMSpeaker with config."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamConfig

        config = StreamConfig(speed=1.5, voice="test_voice")
        speaker = StreamingLLMSpeaker(config=config)

        assert speaker.config.speed == 1.5
        assert speaker.config.voice == "test_voice"

    def test_speaker_reset(self):
        """Test StreamingLLMSpeaker.reset()."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState

        speaker = StreamingLLMSpeaker()
        speaker.sentences_spoken = 5
        speaker.total_tokens = 100

        speaker.reset()

        assert speaker.state == StreamState.IDLE
        assert speaker.sentences_spoken == 0
        assert speaker.total_tokens == 0

    def test_speaker_stats(self):
        """Test StreamingLLMSpeaker.stats property."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker

        speaker = StreamingLLMSpeaker()

        stats = speaker.stats

        assert "state" in stats
        assert "sentences_spoken" in stats
        assert "total_tokens" in stats
        assert "buffer_length" in stats

    @pytest.mark.asyncio
    async def test_speaker_feed_accumulates_tokens(self):
        """Test that feed() accumulates tokens."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState

        speaker = StreamingLLMSpeaker()

        # Mock the engines to avoid actual TTS
        speaker._engine = Mock()
        speaker._streaming_engine = Mock()

        await speaker.feed("Hello ")
        await speaker.feed("world")

        assert speaker.total_tokens == 2
        assert speaker.state == StreamState.BUFFERING

    @pytest.mark.asyncio
    async def test_speaker_finish(self):
        """Test StreamingLLMSpeaker.finish()."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState, StreamConfig

        config = StreamConfig(allow_partial_sentences=False)
        speaker = StreamingLLMSpeaker(config=config)

        await speaker.finish()

        assert speaker.state == StreamState.IDLE


# =============================================================================
# Module: llm/providers.py - LLMConfig Tests
# =============================================================================

class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_llm_config_defaults(self):
        """Test LLMConfig default values."""
        from voice_soundboard.llm.providers import LLMConfig

        config = LLMConfig()

        assert config.model == "llama3.2"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.stream is True

    def test_llm_config_custom(self):
        """Test LLMConfig with custom values."""
        from voice_soundboard.llm.providers import LLMConfig

        config = LLMConfig(
            model="gpt-4",
            temperature=0.5,
            max_tokens=2048,
            api_key="test_key",
        )

        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.api_key == "test_key"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_creation(self):
        """Test LLMResponse creation."""
        from voice_soundboard.llm.providers import LLMResponse

        response = LLMResponse(
            content="Hello world",
            model="test-model",
            finish_reason="stop",
            latency_ms=100.0,
        )

        assert response.content == "Hello world"
        assert response.model == "test-model"
        assert response.finish_reason == "stop"
        assert response.latency_ms == 100.0


class TestProviderType:
    """Tests for ProviderType enum."""

    def test_provider_type_values(self):
        """Test ProviderType enum values."""
        from voice_soundboard.llm.providers import ProviderType

        assert ProviderType.OLLAMA.value == "ollama"
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.VLLM.value == "vllm"
        assert ProviderType.MOCK.value == "mock"


# =============================================================================
# Module: llm/providers.py - MockLLMProvider Tests
# =============================================================================

class TestMockLLMProvider:
    """Tests for MockLLMProvider class."""

    def test_mock_provider_initialization(self):
        """Test MockLLMProvider initialization."""
        from voice_soundboard.llm.providers import MockLLMProvider, ProviderType

        provider = MockLLMProvider()

        assert provider.provider_type == ProviderType.MOCK

    def test_mock_provider_custom_response(self):
        """Test MockLLMProvider with custom responses."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            responses={"hello": "Hi there!"},
            default_response="I don't understand.",
        )

        assert provider.default_response == "I don't understand."
        assert "hello" in provider.responses

    @pytest.mark.asyncio
    async def test_mock_provider_generate(self):
        """Test MockLLMProvider.generate()."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(default_response="Test response")

        response = await provider.generate("Hello")

        assert response.content == "Test response"
        assert response.provider == "mock"
        assert response.model == "mock"

    @pytest.mark.asyncio
    async def test_mock_provider_generate_with_match(self):
        """Test MockLLMProvider.generate() with pattern match."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            responses={"weather": "It's sunny today!"},
            default_response="Default",
        )

        response = await provider.generate("What's the weather?")

        assert response.content == "It's sunny today!"

    @pytest.mark.asyncio
    async def test_mock_provider_stream(self):
        """Test MockLLMProvider.stream()."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            default_response="Hello world",
            token_delay_ms=10,
        )

        tokens = []
        async for token in provider.stream("Test"):
            tokens.append(token)

        # Should receive tokens for "Hello world"
        assert len(tokens) == 2
        assert "Hello" in tokens[0]
        assert "world" in tokens[1]


class TestOllamaProvider:
    """Tests for OllamaProvider class."""

    def test_ollama_provider_initialization(self):
        """Test OllamaProvider initialization."""
        from voice_soundboard.llm.providers import OllamaProvider, ProviderType

        provider = OllamaProvider()

        assert provider.provider_type == ProviderType.OLLAMA
        assert provider.base_url == "http://localhost:11434"

    def test_ollama_provider_custom_url(self):
        """Test OllamaProvider with custom URL."""
        from voice_soundboard.llm.providers import OllamaProvider, LLMConfig

        config = LLMConfig(base_url="http://remote:11434")
        provider = OllamaProvider(config)

        assert provider.base_url == "http://remote:11434"


class TestOpenAIProvider:
    """Tests for OpenAIProvider class."""

    def test_openai_provider_initialization(self):
        """Test OpenAIProvider initialization."""
        from voice_soundboard.llm.providers import OpenAIProvider, ProviderType

        provider = OpenAIProvider()

        assert provider.provider_type == ProviderType.OPENAI
        assert provider.base_url == "https://api.openai.com/v1"


class TestVLLMProvider:
    """Tests for VLLMProvider class."""

    def test_vllm_provider_initialization(self):
        """Test VLLMProvider initialization."""
        from voice_soundboard.llm.providers import VLLMProvider, ProviderType

        provider = VLLMProvider()

        assert provider.provider_type == ProviderType.VLLM
        assert provider.base_url == "http://localhost:8000/v1"


class TestCreateProvider:
    """Tests for create_provider factory function."""

    def test_create_provider_mock(self):
        """Test create_provider with mock type."""
        from voice_soundboard.llm.providers import create_provider, MockLLMProvider

        provider = create_provider("mock")

        assert isinstance(provider, MockLLMProvider)

    def test_create_provider_ollama(self):
        """Test create_provider with ollama type."""
        from voice_soundboard.llm.providers import create_provider, OllamaProvider

        provider = create_provider("ollama")

        assert isinstance(provider, OllamaProvider)

    def test_create_provider_openai(self):
        """Test create_provider with openai type."""
        from voice_soundboard.llm.providers import create_provider, OpenAIProvider

        provider = create_provider("openai")

        assert isinstance(provider, OpenAIProvider)

    def test_create_provider_vllm(self):
        """Test create_provider with vllm type."""
        from voice_soundboard.llm.providers import create_provider, VLLMProvider

        provider = create_provider("vllm")

        assert isinstance(provider, VLLMProvider)

    def test_create_provider_with_enum(self):
        """Test create_provider with ProviderType enum."""
        from voice_soundboard.llm.providers import create_provider, ProviderType, MockLLMProvider

        provider = create_provider(ProviderType.MOCK)

        assert isinstance(provider, MockLLMProvider)

    def test_create_provider_unknown_raises(self):
        """Test create_provider raises for unknown type."""
        from voice_soundboard.llm.providers import create_provider

        with pytest.raises(ValueError):
            create_provider("unknown_provider")


# =============================================================================
# Module: conversion/base.py - LatencyMode Tests
# =============================================================================

class TestLatencyMode:
    """Tests for LatencyMode enum."""

    def test_latency_mode_values(self):
        """Test LatencyMode enum has expected values."""
        from voice_soundboard.conversion.base import LatencyMode

        assert hasattr(LatencyMode, "ULTRA_LOW")
        assert hasattr(LatencyMode, "LOW")
        assert hasattr(LatencyMode, "BALANCED")
        assert hasattr(LatencyMode, "HIGH_QUALITY")


class TestConversionState:
    """Tests for ConversionState enum."""

    def test_conversion_state_values(self):
        """Test ConversionState enum has expected values."""
        from voice_soundboard.conversion.base import ConversionState

        assert hasattr(ConversionState, "IDLE")
        assert hasattr(ConversionState, "STARTING")
        assert hasattr(ConversionState, "RUNNING")
        assert hasattr(ConversionState, "STOPPING")
        assert hasattr(ConversionState, "ERROR")


# =============================================================================
# Module: conversion/base.py - ConversionConfig Tests
# =============================================================================

class TestConversionConfig:
    """Tests for ConversionConfig dataclass."""

    def test_conversion_config_defaults(self):
        """Test ConversionConfig default values."""
        from voice_soundboard.conversion.base import ConversionConfig, LatencyMode

        config = ConversionConfig()

        assert config.latency_mode == LatencyMode.BALANCED
        assert config.sample_rate == 24000
        assert config.channels == 1
        assert config.chunk_size_ms == 20.0
        assert config.preserve_pitch is True
        assert config.use_gpu is True

    def test_conversion_config_get_latency_ms(self):
        """Test ConversionConfig.get_latency_ms()."""
        from voice_soundboard.conversion.base import ConversionConfig, LatencyMode

        # Test each latency mode
        config_ultra = ConversionConfig(latency_mode=LatencyMode.ULTRA_LOW)
        assert config_ultra.get_latency_ms() == 60.0

        config_low = ConversionConfig(latency_mode=LatencyMode.LOW)
        assert config_low.get_latency_ms() == 100.0

        config_balanced = ConversionConfig(latency_mode=LatencyMode.BALANCED)
        assert config_balanced.get_latency_ms() == 150.0

        config_hq = ConversionConfig(latency_mode=LatencyMode.HIGH_QUALITY)
        assert config_hq.get_latency_ms() == 300.0

    def test_conversion_config_target_latency_override(self):
        """Test ConversionConfig target_latency_ms overrides mode."""
        from voice_soundboard.conversion.base import ConversionConfig, LatencyMode

        config = ConversionConfig(
            latency_mode=LatencyMode.HIGH_QUALITY,
            target_latency_ms=50.0,
        )

        assert config.get_latency_ms() == 50.0

    def test_conversion_config_get_chunk_samples(self):
        """Test ConversionConfig.get_chunk_samples()."""
        from voice_soundboard.conversion.base import ConversionConfig

        config = ConversionConfig(sample_rate=24000, chunk_size_ms=20.0)

        # 24000 * 20 / 1000 = 480
        assert config.get_chunk_samples() == 480

    def test_conversion_config_get_buffer_samples(self):
        """Test ConversionConfig.get_buffer_samples()."""
        from voice_soundboard.conversion.base import ConversionConfig

        config = ConversionConfig(sample_rate=24000)

        # 24000 * 100 / 1000 = 2400
        assert config.get_buffer_samples(100.0) == 2400


# =============================================================================
# Module: conversion/base.py - ConversionResult Tests
# =============================================================================

class TestConversionResult:
    """Tests for ConversionResult dataclass."""

    def test_conversion_result_creation(self):
        """Test ConversionResult creation."""
        from voice_soundboard.conversion.base import ConversionResult

        audio = np.zeros(24000, dtype=np.float32)
        result = ConversionResult(
            audio=audio,
            sample_rate=24000,
            input_duration_ms=1000.0,
            output_duration_ms=1000.0,
            processing_time_ms=50.0,
            latency_ms=50.0,
        )

        assert len(result.audio) == 24000
        assert result.sample_rate == 24000
        assert result.latency_ms == 50.0

    def test_conversion_result_realtime_factor(self):
        """Test ConversionResult.realtime_factor property."""
        from voice_soundboard.conversion.base import ConversionResult

        result = ConversionResult(
            audio=np.zeros(1000),
            sample_rate=24000,
            input_duration_ms=1000.0,
            output_duration_ms=1000.0,
            processing_time_ms=100.0,  # 10% of input duration
            latency_ms=100.0,
        )

        assert result.realtime_factor == pytest.approx(0.1, abs=0.01)

    def test_conversion_result_is_realtime(self):
        """Test ConversionResult.is_realtime property."""
        from voice_soundboard.conversion.base import ConversionResult

        # Fast processing (realtime)
        fast_result = ConversionResult(
            audio=np.zeros(1000),
            sample_rate=24000,
            input_duration_ms=1000.0,
            output_duration_ms=1000.0,
            processing_time_ms=100.0,
            latency_ms=100.0,
        )
        assert fast_result.is_realtime is True

        # Slow processing (not realtime)
        slow_result = ConversionResult(
            audio=np.zeros(1000),
            sample_rate=24000,
            input_duration_ms=1000.0,
            output_duration_ms=1000.0,
            processing_time_ms=2000.0,
            latency_ms=2000.0,
        )
        assert slow_result.is_realtime is False


# =============================================================================
# Module: conversion/base.py - ConversionStats Tests
# =============================================================================

class TestConversionStats:
    """Tests for ConversionStats dataclass."""

    def test_conversion_stats_defaults(self):
        """Test ConversionStats default values."""
        from voice_soundboard.conversion.base import ConversionStats

        stats = ConversionStats()

        assert stats.chunks_processed == 0
        assert stats.chunks_dropped == 0
        assert stats.buffer_underruns == 0

    def test_conversion_stats_update_latency(self):
        """Test ConversionStats.update_latency()."""
        from voice_soundboard.conversion.base import ConversionStats

        stats = ConversionStats()

        stats.update_latency(100.0)
        stats.update_latency(200.0)
        stats.update_latency(150.0)

        assert stats.min_latency_ms == 100.0
        assert stats.max_latency_ms == 200.0
        assert stats.avg_latency_ms == 150.0

    def test_conversion_stats_to_dict(self):
        """Test ConversionStats.to_dict()."""
        from voice_soundboard.conversion.base import ConversionStats

        stats = ConversionStats()
        stats.chunks_processed = 10
        stats.update_latency(100.0)

        data = stats.to_dict()

        assert data["chunks_processed"] == 10
        assert "min_latency_ms" in data
        assert "avg_latency_ms" in data


# =============================================================================
# Module: conversion/base.py - VoiceConverter Tests
# =============================================================================

class TestVoiceConverterAbstract:
    """Tests for VoiceConverter abstract class."""

    def test_voice_converter_is_abstract(self):
        """Test VoiceConverter is abstract."""
        from voice_soundboard.conversion.base import VoiceConverter

        with pytest.raises(TypeError):
            VoiceConverter()


class TestMockVoiceConverter:
    """Tests for MockVoiceConverter class."""

    def test_mock_converter_initialization(self):
        """Test MockVoiceConverter initialization."""
        from voice_soundboard.conversion.base import MockVoiceConverter, ConversionState

        converter = MockVoiceConverter()

        assert converter.name == "mock"
        assert converter.state == ConversionState.IDLE

    def test_mock_converter_load(self):
        """Test MockVoiceConverter.load()."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter()

        converter.load()

        assert converter._loaded is True

    def test_mock_converter_convert(self):
        """Test MockVoiceConverter.convert()."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)

        audio = np.random.randn(24000).astype(np.float32)
        result = converter.convert(audio, 24000)

        assert result.audio is not None
        assert result.sample_rate == 24000
        assert result.similarity_score == 0.85
        assert result.naturalness_score == 0.90

    def test_mock_converter_convert_chunk(self):
        """Test MockVoiceConverter.convert_chunk()."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)

        chunk = np.random.randn(480).astype(np.float32)
        result = converter.convert_chunk(chunk)

        assert result is not None
        assert result.dtype == np.float32

    def test_mock_converter_with_target_voice(self):
        """Test MockVoiceConverter with target voice."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)

        audio = np.random.randn(24000).astype(np.float32)
        result = converter.convert(audio, 24000, target_voice="test_voice")

        assert result.target_voice == "test_voice"

    def test_mock_converter_set_target_voice_string(self):
        """Test MockVoiceConverter.set_target_voice() with string."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter()

        converter.set_target_voice("my_voice")

        assert converter._target_voice_id == "my_voice"
        assert converter._target_embedding is not None

    def test_mock_converter_set_target_voice_array(self):
        """Test MockVoiceConverter.set_target_voice() with array."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter()

        embedding = np.random.randn(256).astype(np.float32)
        converter.set_target_voice(embedding)

        assert converter._target_voice_id == "custom"
        assert np.array_equal(converter._target_embedding, embedding)

    def test_mock_converter_is_running(self):
        """Test MockVoiceConverter.is_running property."""
        from voice_soundboard.conversion.base import MockVoiceConverter, ConversionState

        converter = MockVoiceConverter()

        assert converter.is_running is False

        converter._state = ConversionState.RUNNING
        assert converter.is_running is True

    def test_mock_converter_reset_stats(self):
        """Test MockVoiceConverter.reset_stats()."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)

        # Do some conversion to generate stats
        audio = np.random.randn(24000).astype(np.float32)
        converter.convert(audio, 24000)

        assert converter.stats.chunks_processed > 0

        converter.reset_stats()

        assert converter.stats.chunks_processed == 0

    def test_mock_converter_pitch_shift(self):
        """Test MockVoiceConverter with pitch shift."""
        from voice_soundboard.conversion.base import MockVoiceConverter, ConversionConfig

        config = ConversionConfig(pitch_shift_semitones=2.0)
        converter = MockVoiceConverter(config=config, simulate_latency=False)

        audio = np.random.randn(24000).astype(np.float32)
        result = converter.convert(audio, 24000)

        # Audio should be shorter (pitch up = fewer samples)
        assert len(result.audio) < len(audio)

    def test_mock_converter_formant_shift(self):
        """Test MockVoiceConverter with formant shift."""
        from voice_soundboard.conversion.base import MockVoiceConverter, ConversionConfig

        config = ConversionConfig(formant_shift_ratio=1.2)
        converter = MockVoiceConverter(config=config, simulate_latency=False)

        audio = np.random.randn(24000).astype(np.float32)
        result = converter.convert(audio, 24000)

        assert result.audio is not None


# =============================================================================
# Additional LLM Provider Tests
# =============================================================================

class TestLLMProviderChat:
    """Tests for LLMProvider.chat() method."""

    @pytest.mark.asyncio
    async def test_mock_provider_chat(self):
        """Test MockLLMProvider.chat() with messages."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(default_response="Chat response")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        response = await provider.chat(messages)

        assert response.content == "Chat response"

    @pytest.mark.asyncio
    async def test_mock_provider_chat_stream(self):
        """Test MockLLMProvider.chat_stream()."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            default_response="Hello world",
            token_delay_ms=5,
        )

        messages = [
            {"role": "user", "content": "Hi!"},
        ]

        tokens = []
        async for token in provider.chat_stream(messages):
            tokens.append(token)

        assert len(tokens) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
