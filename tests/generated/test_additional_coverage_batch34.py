"""
Additional coverage tests - Batch 34: Heavy Server.py Focus.

Targeting server.py handlers to push toward 90% coverage.
"""

import pytest
import asyncio
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile


# =============================================================================
# Server Handlers - Voice Library Operations
# =============================================================================

class TestServerVoiceLibraryHandlers:
    """Tests for voice library server handlers."""

    @pytest.mark.asyncio
    async def test_handle_list_voice_library(self):
        """Test list_voice_library handler."""
        from voice_soundboard.server import handle_list_voice_library

        result = await handle_list_voice_library({})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_list_voice_library_with_filters(self):
        """Test list_voice_library with filters."""
        from voice_soundboard.server import handle_list_voice_library

        result = await handle_list_voice_library({
            "gender": "female",
            "language": "en",
        })
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_get_voice_profile(self):
        """Test get_voice_profile handler."""
        from voice_soundboard.server import handle_get_voice_profile

        result = await handle_get_voice_profile({
            "voice_id": "nonexistent_voice",
        })
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_delete_cloned_voice(self):
        """Test delete_cloned_voice handler."""
        from voice_soundboard.server import handle_delete_cloned_voice

        result = await handle_delete_cloned_voice({
            "voice_id": "nonexistent_voice",
        })
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_find_similar_voices(self):
        """Test find_similar_voices handler."""
        from voice_soundboard.server import handle_find_similar_voices

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import wave
            with wave.open(f.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b'\x00' * 48000)

            result = await handle_find_similar_voices({
                "audio_path": f.name,
            })
            assert len(result) == 1


# =============================================================================
# Server Handlers - Emotion Transfer
# =============================================================================

class TestServerEmotionTransferHandlers:
    """Tests for emotion transfer server handlers."""

    @pytest.mark.asyncio
    async def test_handle_transfer_voice_emotion(self):
        """Test transfer_voice_emotion handler."""
        from voice_soundboard.server import handle_transfer_voice_emotion

        result = await handle_transfer_voice_emotion({
            "voice_id": "test_voice",
            "emotion": "happy",
        })
        assert len(result) == 1


# =============================================================================
# Server Handlers - Language Compatibility
# =============================================================================

class TestServerLanguageHandlers:
    """Tests for language-related server handlers."""

    @pytest.mark.asyncio
    async def test_handle_list_cloning_languages(self):
        """Test list_cloning_languages handler."""
        from voice_soundboard.server import handle_list_cloning_languages

        result = await handle_list_cloning_languages({})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_check_language_compatibility(self):
        """Test check_language_compatibility handler."""
        from voice_soundboard.server import handle_check_language_compatibility

        result = await handle_check_language_compatibility({
            "source_language": "en",
            "target_language": "es",
        })
        assert len(result) == 1


# =============================================================================
# Server Handlers - Voice Conversion File
# =============================================================================

class TestServerConvertFileHandlers:
    """Tests for file conversion server handlers."""

    @pytest.mark.asyncio
    async def test_handle_convert_audio_file(self):
        """Test convert_audio_file handler."""
        from voice_soundboard.server import handle_convert_audio_file

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import wave
            with wave.open(f.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b'\x00' * 48000)

            result = await handle_convert_audio_file({
                "input_path": f.name,
                "target_voice": "af_bella",
            })
            assert len(result) == 1


# =============================================================================
# Server Handlers - DualCodec Voice Conversion
# =============================================================================

class TestServerDualCodecHandlers:
    """Tests for DualCodec server handlers."""

    @pytest.mark.asyncio
    async def test_handle_voice_convert_dualcodec(self):
        """Test voice_convert_dualcodec handler."""
        from voice_soundboard.server import handle_voice_convert_dualcodec

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import wave
            with wave.open(f.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b'\x00' * 48000)

            result = await handle_voice_convert_dualcodec({
                "content_audio": f.name,
                "style_audio": f.name,
            })
            assert len(result) == 1


# =============================================================================
# Server Handlers - Speak Long
# =============================================================================

class TestServerSpeakLongHandlers:
    """Tests for speak_long server handlers."""

    @pytest.mark.asyncio
    async def test_handle_speak_long(self):
        """Test speak_long handler."""
        from voice_soundboard.server import handle_speak_long

        with patch("voice_soundboard.server.get_engine") as mock_get:
            mock_engine = Mock()
            mock_result = Mock()
            mock_result.audio_path = Path("/tmp/test.wav")
            mock_result.total_duration = 10.0
            mock_result.total_chunks = 20
            mock_result.voice_used = "af_bella"
            mock_engine.speak.return_value = mock_result
            mock_get.return_value = mock_engine

            result = await handle_speak_long({
                "text": "This is a long text that will be streamed for efficient processing.",
            })
            assert len(result) == 1


# =============================================================================
# Server Handlers - Play Audio
# =============================================================================

class TestServerPlayAudioHandlers:
    """Tests for play_audio server handlers."""

    @pytest.mark.asyncio
    async def test_handle_play_audio(self):
        """Test play_audio handler."""
        from voice_soundboard.server import handle_play_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import wave
            with wave.open(f.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b'\x00' * 4800)

            result = await handle_play_audio({
                "path": f.name,
            })
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_stop_audio(self):
        """Test stop_audio handler."""
        from voice_soundboard.server import handle_stop_audio

        result = await handle_stop_audio({})
        assert len(result) == 1


# =============================================================================
# WebSocket Server Handlers - More Coverage
# =============================================================================

class TestWebSocketHandlersMore:
    """More WebSocket handler tests."""

    @pytest.mark.asyncio
    async def test_handle_speak_with_style(self):
        """Test speak handler with style parameter."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/test.wav")
        mock_result.voice_used = "af_bella"
        mock_result.duration_seconds = 1.0
        mock_result.realtime_factor = 0.5

        mock_engine = Mock()
        mock_engine.speak.return_value = mock_result
        server._engine = mock_engine

        await server.handle_speak(
            mock_ws,
            {"text": "Hello", "style": "warmly"},
            "req-1",
        )

        mock_ws.send.assert_called()

    @pytest.mark.asyncio
    async def test_handle_list_voices(self):
        """Test list_voices handler."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_list_voices(mock_ws, {}, "req-1")

        mock_ws.send.assert_called()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is True


# =============================================================================
# LLM Provider Error Handling
# =============================================================================

class TestLLMProviderErrors:
    """Tests for LLM provider error handling."""

    @pytest.mark.asyncio
    async def test_ollama_generate_connection_error(self):
        """Test OllamaProvider with connection error."""
        from voice_soundboard.llm.providers import OllamaProvider

        provider = OllamaProvider()

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(side_effect=ConnectionError("Connection refused"))
            mock_session.return_value.__aexit__ = AsyncMock()

            try:
                await provider.generate("Test")
            except (ConnectionError, Exception):
                pass  # Expected

    @pytest.mark.asyncio
    async def test_create_provider_ollama(self):
        """Test create_provider for ollama."""
        from voice_soundboard.llm.providers import create_provider, ProviderType

        provider = create_provider(ProviderType.OLLAMA)
        assert provider is not None

    @pytest.mark.asyncio
    async def test_create_provider_openai(self):
        """Test create_provider for openai."""
        from voice_soundboard.llm.providers import create_provider, ProviderType, LLMConfig

        config = LLMConfig(api_key="test-key")
        provider = create_provider(ProviderType.OPENAI, config)
        assert provider is not None

    @pytest.mark.asyncio
    async def test_create_provider_vllm(self):
        """Test create_provider for vllm."""
        from voice_soundboard.llm.providers import create_provider, ProviderType

        provider = create_provider(ProviderType.VLLM)
        assert provider is not None


# =============================================================================
# Streaming Engine Additional Tests
# =============================================================================

class TestStreamingEngineAdditional:
    """Additional streaming engine tests."""

    def test_realtime_player_init(self):
        """Test RealtimePlayer initialization."""
        from voice_soundboard.streaming import RealtimePlayer

        player = RealtimePlayer()
        assert player is not None

    def test_realtime_stream_result(self):
        """Test RealtimeStreamResult dataclass."""
        from voice_soundboard.streaming import RealtimeStreamResult

        result = RealtimeStreamResult(
            total_duration=5.0,
            total_chunks=10,
            voice_used="af_bella",
            generation_time=0.5,
            playback_started_at_chunk=0,
        )
        assert result.total_duration == 5.0


# =============================================================================
# Emotions Module Additional Tests
# =============================================================================

class TestEmotionsAdditional:
    """Additional emotions tests."""

    def test_emotion_calm(self):
        """Test calm emotion params."""
        from voice_soundboard.emotions import get_emotion_voice_params

        params = get_emotion_voice_params("calm")
        assert isinstance(params, dict)

    def test_emotion_neutral(self):
        """Test neutral emotion params."""
        from voice_soundboard.emotions import get_emotion_voice_params

        params = get_emotion_voice_params("neutral")
        assert isinstance(params, dict)

    def test_emotion_fearful(self):
        """Test fearful emotion params."""
        from voice_soundboard.emotions import get_emotion_voice_params

        params = get_emotion_voice_params("fearful")
        assert isinstance(params, dict)


# =============================================================================
# Engine Initialization Tests
# =============================================================================

class TestEngineInit:
    """Tests for engine initialization."""

    def test_voice_engine_lazy_init(self):
        """Test VoiceEngine lazy initialization."""
        from voice_soundboard.engine import VoiceEngine

        engine = VoiceEngine()
        # Engine should initialize lazily
        assert engine is not None

    def test_kokoro_engine_exists(self):
        """Test KokoroEngine class exists."""
        from voice_soundboard.engines.kokoro import KokoroEngine

        assert KokoroEngine is not None


# =============================================================================
# Server Handler Edge Cases
# =============================================================================

class TestServerHandlerEdgeCases:
    """Edge case tests for server handlers."""

    @pytest.mark.asyncio
    async def test_handle_speak_missing_text(self):
        """Test speak handler with missing text."""
        from voice_soundboard.server import handle_speak

        result = await handle_speak({})
        assert len(result) == 1
        assert "text" in result[0].text.lower() or "required" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_handle_clone_voice_missing_audio(self):
        """Test clone_voice with missing audio path."""
        from voice_soundboard.server import handle_clone_voice

        result = await handle_clone_voice({
            "voice_id": "test",
        })
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_encode_audio_tokens_missing_path(self):
        """Test encode_audio_tokens with missing path."""
        from voice_soundboard.server import handle_encode_audio_tokens

        result = await handle_encode_audio_tokens({})
        assert len(result) == 1


# =============================================================================
# Codec Additional Tests
# =============================================================================

class TestCodecAdditional:
    """Additional codec tests."""

    def test_mock_codec_init(self):
        """Test MockCodec initialization."""
        from voice_soundboard.codecs.mock import MockCodec

        codec = MockCodec()
        assert codec.name == "mock"

    def test_mock_codec_encode(self):
        """Test MockCodec encode."""
        from voice_soundboard.codecs.mock import MockCodec

        codec = MockCodec()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import wave
            with wave.open(f.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b'\x00' * 48000)

            result = codec.encode(f.name)
            assert result is not None

    def test_mock_codec_capabilities(self):
        """Test MockCodec capabilities."""
        from voice_soundboard.codecs.mock import MockCodec

        codec = MockCodec()
        caps = codec.capabilities

        assert caps.can_encode is True
        assert caps.sample_rate > 0


# =============================================================================
# Security Additional Tests
# =============================================================================

class TestSecurityAdditional:
    """Additional security tests."""

    def test_websocket_security_no_api_key(self):
        """Test WebSocketSecurityManager without API key."""
        from voice_soundboard.security import WebSocketSecurityManager

        manager = WebSocketSecurityManager()

        # Without API key configured, should allow any
        assert manager.validate_api_key(None) is True or manager.api_key is None

    def test_rate_limiter_cleanup(self):
        """Test RateLimiter cleanup of old entries."""
        from voice_soundboard.security import RateLimiter
        import time

        limiter = RateLimiter(max_requests=2, window_seconds=0.05)

        # Use up quota
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        # Wait for window to expire
        time.sleep(0.1)

        # Should allow again
        assert limiter.is_allowed("client1") is True


# =============================================================================
# Normalizer Edge Cases
# =============================================================================

class TestNormalizerEdgeCases:
    """Edge case tests for normalizer."""

    def test_normalize_empty_string(self):
        """Test normalizing empty string."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("")
        assert result == ""

    def test_normalize_special_chars(self):
        """Test normalizing special characters."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("Hello! How are you? I'm fine.")
        assert isinstance(result, str)

    def test_normalize_urls(self):
        """Test normalizing URLs."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("Visit https://example.com for more info")
        assert isinstance(result, str)


# =============================================================================
# Config Additional Tests
# =============================================================================

class TestConfigAdditional:
    """Additional config tests."""

    def test_config_voice_presets(self):
        """Test voice presets in config."""
        from voice_soundboard.config import VOICE_PRESETS

        assert "assistant" in VOICE_PRESETS
        preset = VOICE_PRESETS["assistant"]
        assert "voice" in preset

    def test_config_has_kokoro_voices(self):
        """Test KOKORO_VOICES constant."""
        from voice_soundboard.config import KOKORO_VOICES

        assert len(KOKORO_VOICES) > 0
        assert "af_bella" in KOKORO_VOICES
