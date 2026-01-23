"""
Additional coverage tests - Batch 30: Streaming and Final Coverage Gaps.

Tests for voice_soundboard/streaming.py and remaining coverage gaps.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile


# =============================================================================
# StreamChunk Tests
# =============================================================================

class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_stream_chunk_creation(self):
        """Test creating StreamChunk."""
        from voice_soundboard.streaming import StreamChunk

        samples = np.zeros(1000, dtype=np.float32)
        chunk = StreamChunk(
            samples=samples,
            sample_rate=24000,
            chunk_index=0,
            is_final=False,
            text_segment="Hello",
        )

        assert len(chunk.samples) == 1000
        assert chunk.sample_rate == 24000
        assert chunk.chunk_index == 0
        assert chunk.is_final is False
        assert chunk.text_segment == "Hello"

    def test_stream_chunk_final(self):
        """Test final StreamChunk."""
        from voice_soundboard.streaming import StreamChunk

        chunk = StreamChunk(
            samples=np.array([], dtype=np.float32),
            sample_rate=24000,
            chunk_index=5,
            is_final=True,
            text_segment="",
        )

        assert chunk.is_final is True
        assert len(chunk.samples) == 0


class TestStreamResult:
    """Tests for StreamResult dataclass."""

    def test_stream_result_creation(self):
        """Test creating StreamResult."""
        from voice_soundboard.streaming import StreamResult

        result = StreamResult(
            audio_path=Path("/tmp/test.wav"),
            total_duration=5.5,
            total_chunks=10,
            generation_time=1.2,
            voice_used="af_bella",
        )

        assert result.audio_path == Path("/tmp/test.wav")
        assert result.total_duration == 5.5
        assert result.total_chunks == 10
        assert result.generation_time == 1.2
        assert result.voice_used == "af_bella"

    def test_stream_result_no_audio(self):
        """Test StreamResult with no audio path."""
        from voice_soundboard.streaming import StreamResult

        result = StreamResult(
            audio_path=None,
            total_duration=0.0,
            total_chunks=0,
            generation_time=0.1,
            voice_used="af_bella",
        )

        assert result.audio_path is None


# =============================================================================
# StreamingEngine Tests
# =============================================================================

class TestStreamingEngine:
    """Tests for StreamingEngine class."""

    def test_init_default(self):
        """Test default initialization."""
        from voice_soundboard.streaming import StreamingEngine

        engine = StreamingEngine()

        assert engine._model_loaded is False
        assert engine._kokoro is None

    def test_init_with_config(self):
        """Test initialization with config."""
        from voice_soundboard.streaming import StreamingEngine
        from voice_soundboard.config import Config

        config = Config()
        engine = StreamingEngine(config=config)

        assert engine.config == config

    def test_ensure_model_loaded_caches(self):
        """Test model loading is cached."""
        from voice_soundboard.streaming import StreamingEngine

        engine = StreamingEngine()
        engine._model_loaded = True

        # Should not try to load again
        engine._ensure_model_loaded()
        assert engine._model_loaded is True

    def test_ensure_model_loaded_file_not_found(self):
        """Test model loading with missing file."""
        from voice_soundboard.streaming import StreamingEngine

        engine = StreamingEngine()
        engine._model_path = Path("/nonexistent/model.onnx")

        # May raise FileNotFoundError or ModuleNotFoundError depending on setup
        with pytest.raises((FileNotFoundError, ModuleNotFoundError)):
            engine._ensure_model_loaded()


# =============================================================================
# RealtimeStreamResult Tests
# =============================================================================

class TestRealtimeStreamResult:
    """Tests for RealtimeStreamResult dataclass."""

    def test_result_creation(self):
        """Test creating RealtimeStreamResult."""
        from voice_soundboard.streaming import RealtimeStreamResult

        result = RealtimeStreamResult(
            total_duration=10.5,
            total_chunks=15,
            generation_time=2.3,
            playback_started_at_chunk=0,
            voice_used="af_bella",
        )

        assert result.total_duration == 10.5
        assert result.total_chunks == 15
        assert result.generation_time == 2.3
        assert result.playback_started_at_chunk == 0
        assert result.voice_used == "af_bella"


# =============================================================================
# RealtimePlayer Tests
# =============================================================================

class TestRealtimePlayer:
    """Tests for RealtimePlayer class."""

    def test_init(self):
        """Test RealtimePlayer initialization."""
        from voice_soundboard.streaming import RealtimePlayer

        player = RealtimePlayer(sample_rate=24000, buffer_chunks=2)

        assert player.sample_rate == 24000
        assert player.buffer_chunks == 2
        assert player._is_playing is False

    @pytest.mark.asyncio
    async def test_add_chunk(self):
        """Test adding chunk to queue."""
        from voice_soundboard.streaming import RealtimePlayer

        player = RealtimePlayer()
        samples = np.zeros(1000, dtype=np.float32)

        await player.add_chunk(samples)

        # Queue should have one item
        assert player._queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stop method."""
        from voice_soundboard.streaming import RealtimePlayer

        player = RealtimePlayer()

        # No task set - just test sentinel is added
        await player.stop()

        # Should put sentinel (None)
        assert player._queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_stop_immediate(self):
        """Test stop_immediate method."""
        from voice_soundboard.streaming import RealtimePlayer

        player = RealtimePlayer()

        # Create a mock task that can be cancelled
        async def dummy_task():
            await asyncio.sleep(10)

        player._playback_task = asyncio.create_task(dummy_task())

        await player.stop_immediate()

        assert player._stop_event.is_set()


# =============================================================================
# LLM Providers Additional Tests
# =============================================================================

class TestLLMProvidersAdditional:
    """Additional tests for LLM providers."""

    @pytest.mark.asyncio
    async def test_mock_provider_stream_with_delay(self):
        """Test MockLLMProvider stream with token delay."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            default_response="Test",
            token_delay_ms=1,  # Very small delay for test
        )

        tokens = []
        async for token in provider.stream("Hello"):
            tokens.append(token)

        assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_openai_provider_chat_stream(self):
        """Test OpenAIProvider chat_stream method."""
        from voice_soundboard.llm.providers import OpenAIProvider, LLMConfig

        config = LLMConfig(api_key="test-key")
        provider = OpenAIProvider(config)

        # Mock the stream method
        async def mock_stream(*args, **kwargs):
            yield "Hello "
            yield "world"

        provider.stream = mock_stream

        messages = [{"role": "user", "content": "Hi"}]
        tokens = []
        async for token in provider.chat_stream(messages):
            tokens.append(token)

        assert tokens == ["Hello ", "world"]


# =============================================================================
# Server MCP Handlers Additional Tests
# =============================================================================

class TestServerMCPHandlersAdditional:
    """Additional tests for server MCP handlers."""

    @pytest.mark.asyncio
    async def test_handle_list_effects_detail(self):
        """Test list_effects handler with effect details."""
        from voice_soundboard.server import handle_list_effects

        result = await handle_list_effects({})

        assert len(result) == 1
        assert "effects" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_list_emotions_detail(self):
        """Test list_emotions handler."""
        from voice_soundboard.server import handle_list_emotions

        result = await handle_list_emotions({})

        assert len(result) == 1
        # Should contain emotion names

    @pytest.mark.asyncio
    async def test_handle_speak_with_save_path(self):
        """Test speak handler with custom save path."""
        from voice_soundboard.server import handle_speak

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "custom.wav"

            with patch("voice_soundboard.server.get_engine") as mock_get:
                mock_engine = Mock()
                mock_result = Mock()
                mock_result.audio_path = save_path
                mock_result.duration_seconds = 1.0
                mock_result.voice_used = "af_bella"
                mock_engine.speak.return_value = mock_result
                mock_get.return_value = mock_engine

                result = await handle_speak({
                    "text": "Hello",
                    "save_path": str(save_path),
                })

                assert len(result) == 1


# =============================================================================
# Engines Additional Tests
# =============================================================================

class TestEnginesAdditional:
    """Additional tests for engine modules."""

    def test_engines_init_imports(self):
        """Test engines module imports."""
        from voice_soundboard import engines

        assert engines is not None


# =============================================================================
# Config Additional Tests
# =============================================================================

class TestConfigAdditional:
    """Additional tests for config module."""

    def test_config_kokoro_voices(self):
        """Test KOKORO_VOICES configuration."""
        from voice_soundboard.config import KOKORO_VOICES

        assert len(KOKORO_VOICES) > 0
        # Check voice has required fields
        for voice_id, info in KOKORO_VOICES.items():
            assert "name" in info
            assert "gender" in info

    def test_config_voice_presets(self):
        """Test VOICE_PRESETS configuration."""
        from voice_soundboard.config import VOICE_PRESETS

        assert len(VOICE_PRESETS) > 0
        # Check preset has voice
        for name, config in VOICE_PRESETS.items():
            assert "voice" in config


# =============================================================================
# Emotions Additional Tests
# =============================================================================

class TestEmotionsAdditional:
    """Additional tests for emotions module."""

    def test_emotions_dict(self):
        """Test EMOTIONS dictionary."""
        from voice_soundboard.emotions import EMOTIONS

        assert len(EMOTIONS) > 0
        assert "happy" in EMOTIONS
        assert "sad" in EMOTIONS

    def test_emotion_params(self):
        """Test EmotionParams structure."""
        from voice_soundboard.emotions import EMOTIONS

        happy = EMOTIONS["happy"]
        assert hasattr(happy, "speed")
        assert hasattr(happy, "voice_preference")


# =============================================================================
# Interpreter Additional Tests
# =============================================================================

class TestInterpreterAdditional:
    """Additional tests for interpreter module."""

    def test_interpreter_imports(self):
        """Test interpreter module imports."""
        from voice_soundboard import interpreter

        assert interpreter is not None


# =============================================================================
# LLM Context Additional Tests
# =============================================================================

class TestLLMContextAdditional:
    """Additional tests for LLM context module."""

    def test_context_config_defaults(self):
        """Test ContextConfig defaults."""
        from voice_soundboard.llm.context import ContextConfig

        config = ContextConfig()
        assert config.enable_auto_emotion is True


# =============================================================================
# Cloning Extractor Additional Tests
# =============================================================================

class TestCloningExtractorAdditional:
    """Additional tests for cloning extractor."""

    def test_cloning_extractor_exists(self):
        """Test cloning extractor module exists."""
        from voice_soundboard.cloning import extractor

        assert extractor is not None


# =============================================================================
# Conversion Devices Additional Tests
# =============================================================================

class TestConversionDevicesAdditional:
    """Additional tests for conversion devices."""

    def test_audio_device(self):
        """Test AudioDevice exists."""
        from voice_soundboard.conversion.devices import AudioDevice

        assert AudioDevice is not None


# =============================================================================
# SSML Additional Tests
# =============================================================================

class TestSSMLAdditional:
    """Additional tests for SSML module."""

    def test_parse_ssml_basic(self):
        """Test basic SSML parsing."""
        from voice_soundboard.ssml import parse_ssml

        ssml = '<speak><prosody rate="slow">Hello</prosody></speak>'
        text, params = parse_ssml(ssml)

        assert "Hello" in text


# =============================================================================
# Effects Additional Tests
# =============================================================================

class TestEffectsAdditional:
    """Additional tests for effects module."""

    def test_list_effects(self):
        """Test list_effects function."""
        from voice_soundboard.effects import list_effects

        effects = list_effects()

        assert isinstance(effects, list)
        assert len(effects) > 0

    def test_get_effect(self):
        """Test get_effect function."""
        from voice_soundboard.effects import get_effect

        effect = get_effect("chime")

        assert effect is not None
        assert hasattr(effect, "play")
        assert hasattr(effect, "duration")


# =============================================================================
# Audio Module Additional Tests
# =============================================================================

class TestAudioAdditional:
    """Additional tests for audio module."""

    def test_stop_playback_safe(self):
        """Test stop_playback doesn't raise."""
        from voice_soundboard.audio import stop_playback

        # Should not raise even if nothing is playing
        stop_playback()


# =============================================================================
# Web Server Additional Tests
# =============================================================================

class TestWebServerAdditional:
    """Additional tests for web server."""

    def test_web_server_exists(self):
        """Test web server module exists."""
        from voice_soundboard import web_server

        assert web_server is not None


# =============================================================================
# Security Additional Tests
# =============================================================================

class TestSecurityAdditional:
    """Additional tests for security module."""

    def test_validate_speed_bounds(self):
        """Test validate_speed with bounds."""
        from voice_soundboard.security import validate_speed

        assert validate_speed(0.3) == 0.5  # Clamped to min
        assert validate_speed(3.0) == 2.0  # Clamped to max
        assert validate_speed(1.0) == 1.0  # Within bounds

    def test_safe_error_message(self):
        """Test safe_error_message sanitization."""
        from voice_soundboard.security import safe_error_message

        error = Exception("File not found: /secret/path/file.txt")
        safe_msg = safe_error_message(error)

        # Should not expose full path
        assert isinstance(safe_msg, str)


# =============================================================================
# Dialogue Engine Additional Tests
# =============================================================================

class TestDialogueEngineAdditional:
    """Additional tests for dialogue engine."""

    def test_dialogue_engine_exists(self):
        """Test DialogueEngine exists."""
        from voice_soundboard.dialogue.engine import DialogueEngine

        assert DialogueEngine is not None


# =============================================================================
# Normalizer Additional Tests
# =============================================================================

class TestNormalizerAdditional:
    """Additional tests for normalizer module."""

    def test_normalizer_exists(self):
        """Test normalizer module exists."""
        from voice_soundboard import normalizer

        assert normalizer is not None


# =============================================================================
# Emotion Module Additional Tests
# =============================================================================

class TestEmotionModuleAdditional:
    """Additional tests for emotion module."""

    def test_emotion_vad_values(self):
        """Test VAD values for emotions."""
        from voice_soundboard.emotion import vad

        # Module exists
        assert vad is not None

    def test_emotion_blending(self):
        """Test emotion blending."""
        from voice_soundboard.emotion.blending import blend_emotions

        result = blend_emotions([("happy", 0.7), ("sad", 0.3)])

        assert result is not None


# =============================================================================
# Codec Mock Tests
# =============================================================================

class TestCodecMock:
    """Tests for MockCodec."""

    def test_mock_codec_encode_decode(self):
        """Test MockCodec encode and decode."""
        from voice_soundboard.codecs.mock import MockCodec

        codec = MockCodec()
        codec.load()

        audio = np.random.randn(24000).astype(np.float32)
        encoded = codec.encode(audio, sample_rate=24000)

        assert encoded.tokens is not None

        decoded, sr = codec.decode(encoded)
        assert sr == 24000
        assert len(decoded) > 0


# =============================================================================
# LLM Streaming Additional Tests
# =============================================================================

class TestLLMStreamingAdditional:
    """Additional tests for LLM streaming module."""

    def test_stream_config_defaults(self):
        """Test StreamConfig defaults."""
        from voice_soundboard.llm.streaming import StreamConfig

        config = StreamConfig()

        assert config.voice is None
        assert config.preset == "assistant"
        assert config.speed == 1.0


# =============================================================================
# LLM Interruption Additional Tests
# =============================================================================

class TestLLMInterruptionAdditional:
    """Additional tests for LLM interruption module."""

    def test_interruption_strategy_enum(self):
        """Test InterruptionStrategy enum."""
        from voice_soundboard.llm.interruption import InterruptionStrategy

        assert InterruptionStrategy.STOP_IMMEDIATE is not None
        # Just check one known value exists

    def test_barge_in_config(self):
        """Test BargeInConfig creation."""
        from voice_soundboard.llm.interruption import BargeInConfig

        config = BargeInConfig(enabled=True, vad_threshold_db=-30.0)

        assert config.enabled is True
        assert config.vad_threshold_db == -30.0
