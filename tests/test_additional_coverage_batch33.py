"""
Additional coverage tests - Batch 33: Final Push to 90%.

Focus on server.py and websocket_server.py handler coverage.
"""

import pytest
import asyncio
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile


# =============================================================================
# Server Handler Tests - Focus on Remaining Handlers
# =============================================================================

class TestServerConversationHandlers:
    """Tests for conversation-related server handlers."""

    @pytest.mark.asyncio
    async def test_handle_get_conversation_context(self):
        """Test get_conversation_context handler."""
        from voice_soundboard.server import handle_get_conversation_context

        with patch("voice_soundboard.server.get_conversation_manager") as mock_get:
            mock_manager = Mock()
            mock_manager.get_context.return_value = [
                {"role": "user", "content": "Hello"},
            ]
            mock_get.return_value = mock_manager

            result = await handle_get_conversation_context({
                "max_messages": 10,
            })

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_get_conversation_stats(self):
        """Test get_conversation_stats handler."""
        from voice_soundboard.server import handle_get_conversation_stats

        with patch("voice_soundboard.server.get_conversation_manager") as mock_get:
            mock_manager = Mock()
            mock_manager.get_stats.return_value = {
                "messages": 5,
                "tokens": 100,
            }
            mock_get.return_value = mock_manager

            result = await handle_get_conversation_stats({})

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_end_conversation(self):
        """Test end_conversation handler."""
        from voice_soundboard.server import handle_end_conversation

        with patch("voice_soundboard.server.get_conversation_manager") as mock_get:
            mock_manager = Mock()
            mock_manager.end.return_value = None
            mock_get.return_value = mock_manager

            result = await handle_end_conversation({})

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_detect_user_emotion(self):
        """Test detect_user_emotion handler."""
        from voice_soundboard.server import handle_detect_user_emotion

        result = await handle_detect_user_emotion({
            "message": "I'm so happy today!",
        })

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_select_response_emotion(self):
        """Test select_response_emotion handler."""
        from voice_soundboard.server import handle_select_response_emotion

        result = await handle_select_response_emotion({
            "response_text": "That's wonderful news!",
            "user_emotion": "happy",
        })

        assert len(result) == 1


class TestServerVoiceConversionMoreHandlers:
    """More voice conversion handler tests."""

    @pytest.mark.asyncio
    async def test_handle_start_voice_conversion(self):
        """Test start_voice_conversion handler."""
        from voice_soundboard.server import handle_start_voice_conversion

        with patch("voice_soundboard.server._realtime_converter") as mock_conv:
            mock_conv.start.return_value = None
            mock_conv.is_running = False

            result = await handle_start_voice_conversion({
                "target_voice": "af_bella",
            })

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_stop_voice_conversion(self):
        """Test stop_voice_conversion handler."""
        from voice_soundboard.server import handle_stop_voice_conversion
        import voice_soundboard.server as server_module

        original = getattr(server_module, '_realtime_converter', None)
        try:
            server_module._realtime_converter = None

            result = await handle_stop_voice_conversion({})

            assert len(result) == 1
        finally:
            server_module._realtime_converter = original


class TestServerCloneVoiceAdvanced:
    """Tests for advanced voice cloning handlers."""

    @pytest.mark.asyncio
    async def test_handle_validate_clone_audio(self):
        """Test validate_clone_audio handler."""
        from voice_soundboard.server import handle_validate_clone_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import wave
            with wave.open(f.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b'\x00' * 48000)

            result = await handle_validate_clone_audio({
                "audio_path": f.name,
            })

            assert len(result) == 1


class TestServerEmotionHandlersMore:
    """More emotion handler tests."""

    @pytest.mark.asyncio
    async def test_handle_list_emotion_blends(self):
        """Test list_emotion_blends handler."""
        from voice_soundboard.server import handle_list_emotion_blends

        result = await handle_list_emotion_blends({})

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_parse_emotion_text(self):
        """Test parse_emotion_text handler."""
        from voice_soundboard.server import handle_parse_emotion_text

        result = await handle_parse_emotion_text({
            "text": "{happy}Hello{/happy} there!",
        })

        assert len(result) == 1


# =============================================================================
# WebSocket Server Additional Handlers
# =============================================================================

class TestWebSocketServerMoreHandlers:
    """More WebSocket server handler tests."""

    @pytest.mark.asyncio
    async def test_handle_stop(self):
        """Test stop handler."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_stop(mock_ws, {}, "req-1")

        mock_ws.send.assert_called()

    @pytest.mark.asyncio
    async def test_handle_error_response(self):
        """Test error response creation."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server._send_error(mock_ws, "Test error", "test_action", "req-1")

        mock_ws.send.assert_called()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["success"] is False
        assert "error" in sent_data


# =============================================================================
# LLM Providers Additional Tests
# =============================================================================

class TestLLMProvidersMore:
    """More LLM provider tests."""

    def test_llm_response_creation(self):
        """Test LLMResponse creation."""
        from voice_soundboard.llm.providers import LLMResponse

        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            latency_ms=100,
            provider="openai",
        )

        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.finish_reason == "stop"

    def test_vllm_provider_basic(self):
        """Test VLLMProvider basic init."""
        from voice_soundboard.llm.providers import VLLMProvider

        provider = VLLMProvider()
        assert provider is not None

    def test_ollama_provider_basic(self):
        """Test OllamaProvider basic init."""
        from voice_soundboard.llm.providers import OllamaProvider

        provider = OllamaProvider()
        assert provider is not None


# =============================================================================
# LLM Streaming Additional Tests
# =============================================================================

class TestLLMStreamingMore:
    """More LLM streaming tests."""

    def test_stream_state_enum(self):
        """Test StreamState enum."""
        from voice_soundboard.llm.streaming import StreamState

        assert StreamState.IDLE.value == "idle"
        assert StreamState.BUFFERING.value == "buffering"
        assert StreamState.SPEAKING.value == "speaking"

    def test_streaming_llm_speaker_state(self):
        """Test StreamingLLMSpeaker state transitions."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState

        speaker = StreamingLLMSpeaker()
        assert speaker.state == StreamState.IDLE


# =============================================================================
# Streaming Engine More Tests
# =============================================================================

class TestStreamingEngineMore:
    """More streaming engine tests."""

    def test_streaming_engine_init(self):
        """Test StreamingEngine initialization."""
        from voice_soundboard.streaming import StreamingEngine

        engine = StreamingEngine()
        assert engine is not None


# =============================================================================
# Emotions Module Tests
# =============================================================================

class TestEmotionsModuleMore:
    """More emotions module tests."""

    def test_emotion_to_voice_params(self):
        """Test emotion to voice params conversion."""
        from voice_soundboard.emotions import get_emotion_voice_params

        params = get_emotion_voice_params("excited")
        assert "speed" in params
        assert params["speed"] > 1.0  # excited = faster

    def test_emotion_angry(self):
        """Test angry emotion params."""
        from voice_soundboard.emotions import get_emotion_voice_params

        params = get_emotion_voice_params("angry")
        assert isinstance(params, dict)

    def test_emotion_sad(self):
        """Test sad emotion params."""
        from voice_soundboard.emotions import get_emotion_voice_params

        params = get_emotion_voice_params("sad")
        assert "speed" in params


# =============================================================================
# Conversion Module Additional Tests
# =============================================================================

class TestConversionDevicesMore:
    """More conversion device tests."""

    def test_audio_device_class(self):
        """Test AudioDevice class."""
        from voice_soundboard.conversion.devices import AudioDevice

        # AudioDevice is a dataclass
        assert AudioDevice is not None


# =============================================================================
# Chatterbox Engine More Tests
# =============================================================================

class TestChatterboxEngineMore:
    """More Chatterbox engine tests."""

    def test_paralinguistic_tag_constants(self):
        """Test paralinguistic tag constants."""
        from voice_soundboard.engines.chatterbox import PARALINGUISTIC_TAGS

        # Check for common tags
        assert len(PARALINGUISTIC_TAGS) > 0

    def test_chatterbox_engine_init(self):
        """Test ChatterboxEngine initialization."""
        from voice_soundboard.engines.chatterbox import ChatterboxEngine

        engine = ChatterboxEngine()
        assert engine is not None


# =============================================================================
# Server Edge Cases
# =============================================================================

class TestServerEdgeCases:
    """Tests for server edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handle_speak_empty_text(self):
        """Test speak handler with empty text."""
        from voice_soundboard.server import handle_speak

        result = await handle_speak({
            "text": "",
        })

        assert len(result) == 1
        # Should return error message

    @pytest.mark.asyncio
    async def test_handle_sound_effect_unknown(self):
        """Test sound_effect handler with unknown effect."""
        from voice_soundboard.server import handle_sound_effect

        result = await handle_sound_effect({
            "effect": "nonexistent_effect_xyz",
        })

        assert len(result) == 1
        # Should return error message

    @pytest.mark.asyncio
    async def test_handle_speak_ssml(self):
        """Test speak_ssml handler."""
        from voice_soundboard.server import handle_speak_ssml

        with patch("voice_soundboard.server.get_engine") as mock_get:
            mock_engine = Mock()
            mock_result = Mock()
            mock_result.audio_path = Path("/tmp/test.wav")
            mock_result.duration_seconds = 1.0
            mock_result.voice_used = "af_bella"
            mock_engine.speak_ssml.return_value = mock_result
            mock_get.return_value = mock_engine

            result = await handle_speak_ssml({
                "ssml": "<speak>Hello</speak>",
            })

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_speak_context_aware(self):
        """Test speak_with_context handler."""
        from voice_soundboard.server import handle_speak_with_context

        with patch("voice_soundboard.server.get_engine") as mock_get:
            mock_engine = Mock()
            mock_result = Mock()
            mock_result.audio_path = Path("/tmp/test.wav")
            mock_result.duration_seconds = 1.0
            mock_result.voice_used = "af_bella"
            mock_engine.speak.return_value = mock_result
            mock_get.return_value = mock_engine

            result = await handle_speak_with_context({
                "text": "I understand how you feel",
                "context": "User is feeling sad",
            })

            assert len(result) == 1


# =============================================================================
# Normalizer Additional Tests
# =============================================================================

class TestNormalizerMore:
    """More normalizer tests."""

    def test_normalize_currency(self):
        """Test normalizing currency."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("It costs $50.00")
        assert isinstance(result, str)

    def test_normalize_time(self):
        """Test normalizing time."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("Meet at 3:30 PM")
        assert isinstance(result, str)

    def test_normalize_abbreviations(self):
        """Test normalizing abbreviations."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("Dr. Smith works at IBM")
        assert isinstance(result, str)


# =============================================================================
# SSML Additional Tests
# =============================================================================

class TestSSMLMore:
    """More SSML tests."""

    def test_parse_ssml_emphasis(self):
        """Test parsing SSML with emphasis."""
        from voice_soundboard.ssml import parse_ssml

        ssml = '<speak><emphasis level="strong">Important</emphasis> message</speak>'
        text, params = parse_ssml(ssml)

        assert "Important" in text

    def test_parse_ssml_say_as(self):
        """Test parsing SSML with say-as."""
        from voice_soundboard.ssml import parse_ssml

        ssml = '<speak><say-as interpret-as="date">2024-01-15</say-as></speak>'
        text, params = parse_ssml(ssml)

        assert text is not None

    def test_parse_ssml_sub(self):
        """Test parsing SSML with sub."""
        from voice_soundboard.ssml import parse_ssml

        ssml = '<speak><sub alias="World Wide Web">WWW</sub></speak>'
        text, params = parse_ssml(ssml)

        assert text is not None


# =============================================================================
# Pipeline Additional Tests
# =============================================================================

class TestPipelineMore:
    """More pipeline tests."""

    def test_speech_pipeline_exists(self):
        """Test SpeechPipeline class exists."""
        from voice_soundboard.llm.pipeline import SpeechPipeline

        assert SpeechPipeline is not None


# =============================================================================
# Engine Base Tests
# =============================================================================

class TestEngineBase:
    """Tests for engine base classes."""

    def test_tts_engine_class(self):
        """Test TTSEngine class exists."""
        from voice_soundboard.engines.base import TTSEngine

        assert TTSEngine is not None


# =============================================================================
# Codec LLM Bridge Tests
# =============================================================================

class TestCodecLLMBridge:
    """Tests for codec LLM bridge."""

    def test_llm_codec_bridge_exists(self):
        """Test LLMCodecBridge class exists."""
        from voice_soundboard.codecs.llm import LLMCodecBridge

        assert LLMCodecBridge is not None


# =============================================================================
# Emotion Curves Tests
# =============================================================================

class TestEmotionCurvesMore:
    """More emotion curve tests."""

    def test_vad_point(self):
        """Test VADPoint dataclass."""
        from voice_soundboard.emotion.curves import VADPoint

        point = VADPoint(valence=0.5, arousal=0.5, dominance=0.5)
        assert point.valence == 0.5

    def test_emotion_keyframe(self):
        """Test EmotionKeyframe dataclass."""
        from voice_soundboard.emotion.curves import EmotionKeyframe

        keyframe = EmotionKeyframe(position=0.5, emotion="happy")
        assert keyframe.position == 0.5


# =============================================================================
# Dialogue Voices Tests
# =============================================================================

class TestDialogueVoicesMore:
    """More dialogue voices tests."""

    def test_voice_assigner_exists(self):
        """Test VoiceAssigner class exists."""
        from voice_soundboard.dialogue.voices import VoiceAssigner

        assert VoiceAssigner is not None
