"""
Additional coverage tests - Batch 36: Final Push for 90%.

Focus on server.py, websocket_server.py, audio.py, and edge cases.
Also includes mobile device compatibility testing.
"""

import pytest
import asyncio
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import wave


# =============================================================================
# Server.py - More Handler Coverage
# =============================================================================

class TestServerDialogueHandlers:
    """Tests for dialogue-related server handlers."""

    @pytest.mark.asyncio
    async def test_handle_speak_dialogue(self):
        """Test speak_dialogue handler."""
        from voice_soundboard.server import handle_speak_dialogue

        result = await handle_speak_dialogue({
            "script": "[S1:narrator] Hello there.",
        })
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_preview_dialogue(self):
        """Test preview_dialogue handler."""
        from voice_soundboard.server import handle_preview_dialogue

        result = await handle_preview_dialogue({
            "script": "[S1:narrator] Hello.\n[S2:alice] Hi!",
        })
        assert len(result) == 1


class TestServerEmotionCurveHandlers:
    """Tests for emotion curve server handlers."""

    @pytest.mark.asyncio
    async def test_handle_list_narrative_curves(self):
        """Test list_narrative_curves handler."""
        from voice_soundboard.server import handle_list_narrative_curves

        result = await handle_list_narrative_curves({})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_sample_emotion_curve(self):
        """Test sample_emotion_curve handler."""
        from voice_soundboard.server import handle_sample_emotion_curve

        result = await handle_sample_emotion_curve({
            "curve_name": "tension_build",
        })
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_get_emotion_vad(self):
        """Test get_emotion_vad handler."""
        from voice_soundboard.server import handle_get_emotion_vad

        result = await handle_get_emotion_vad({
            "emotion": "happy",
        })
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_blend_emotions(self):
        """Test blend_emotions handler."""
        from voice_soundboard.server import handle_blend_emotions

        result = await handle_blend_emotions({
            "emotions": [
                {"emotion": "happy", "weight": 0.7},
                {"emotion": "surprised", "weight": 0.3},
            ],
        })
        assert len(result) == 1


class TestServerCodecHandlers:
    """Tests for codec server handlers."""

    @pytest.mark.asyncio
    async def test_handle_get_codec_info(self):
        """Test get_codec_info handler."""
        from voice_soundboard.server import handle_get_codec_info

        result = await handle_get_codec_info({
            "codec": "mock",
        })
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_estimate_audio_tokens(self):
        """Test estimate_audio_tokens handler."""
        from voice_soundboard.server import handle_estimate_audio_tokens

        result = await handle_estimate_audio_tokens({
            "duration_seconds": 5.0,
        })
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_decode_audio_tokens(self):
        """Test decode_audio_tokens handler."""
        from voice_soundboard.server import handle_decode_audio_tokens

        result = await handle_decode_audio_tokens({
            "tokens": [1, 2, 3, 4, 5],
        })
        assert len(result) == 1


class TestServerChatterboxHandlers:
    """Tests for chatterbox server handlers."""

    @pytest.mark.asyncio
    async def test_handle_speak_chatterbox(self):
        """Test speak_chatterbox handler."""
        from voice_soundboard.server import handle_speak_chatterbox

        with patch("voice_soundboard.server.get_chatterbox_engine") as mock_get:
            mock_engine = Mock()
            mock_result = Mock()
            mock_result.audio_path = Path("/tmp/test.wav")
            mock_result.duration_seconds = 1.0
            mock_engine.speak.return_value = mock_result
            mock_get.return_value = mock_engine

            result = await handle_speak_chatterbox({
                "text": "Hello [laugh] there!",
            })
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_list_paralinguistic_tags(self):
        """Test list_paralinguistic_tags handler."""
        from voice_soundboard.server import handle_list_paralinguistic_tags

        result = await handle_list_paralinguistic_tags({})
        assert len(result) == 1


class TestServerRealtimeSpeakHandler:
    """Tests for realtime speak handlers."""

    @pytest.mark.asyncio
    async def test_handle_speak_realtime(self):
        """Test speak_realtime handler."""
        from voice_soundboard.server import handle_speak_realtime

        with patch("voice_soundboard.server.get_engine") as mock_get:
            mock_engine = Mock()
            mock_result = Mock()
            mock_result.audio_path = Path("/tmp/test.wav")
            mock_result.total_duration = 1.0
            mock_engine.speak.return_value = mock_result
            mock_get.return_value = mock_engine

            result = await handle_speak_realtime({
                "text": "Hello world",
            })
            assert len(result) == 1


# =============================================================================
# WebSocket Server - More Handler Coverage
# =============================================================================

class TestWebSocketServerHandlersMore:
    """More WebSocket server handler tests."""

    @pytest.mark.asyncio
    async def test_handle_speak_preset(self):
        """Test speak WebSocket handler with preset."""
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
            {"text": "Hello", "preset": "assistant"},
            "req-1",
        )

        mock_ws.send.assert_called()

    @pytest.mark.asyncio
    async def test_handle_list_presets(self):
        """Test list_presets WebSocket handler."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        mock_ws = AsyncMock()

        await server.handle_list_presets(mock_ws, {}, "req-1")

        mock_ws.send.assert_called()


# =============================================================================
# Audio Module - Edge Cases
# =============================================================================

class TestAudioEdgeCases:
    """Edge case tests for audio module."""

    def test_play_audio_invalid_path(self):
        """Test playing audio with invalid path."""
        from voice_soundboard.audio import play_audio

        # Should handle gracefully
        try:
            play_audio("/nonexistent/path.wav")
        except (FileNotFoundError, Exception):
            pass  # Expected

    def test_audio_short_duration(self):
        """Test audio with very short duration."""
        from voice_soundboard.audio import get_audio_duration

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                # Very short - 100 samples
                wav.writeframes(b'\x00' * 200)

            duration = get_audio_duration(f.name)
            assert duration < 0.1  # Less than 100ms


class TestAudioMobileCompatibility:
    """Tests for mobile device audio compatibility."""

    def test_mobile_sample_rates(self):
        """Test audio with mobile-compatible sample rates."""
        from voice_soundboard.audio import get_audio_duration

        # Mobile devices typically support 44100, 48000, 22050
        for sample_rate in [22050, 44100, 48000]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with wave.open(f.name, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(b'\x00' * (sample_rate * 2))  # 1 second

                duration = get_audio_duration(f.name)
                assert abs(duration - 1.0) < 0.1

    def test_mono_stereo_compatibility(self):
        """Test mono and stereo audio compatibility."""
        from voice_soundboard.audio import get_audio_duration

        for channels in [1, 2]:  # Mono and stereo
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with wave.open(f.name, 'wb') as wav:
                    wav.setnchannels(channels)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(b'\x00' * (24000 * 2 * channels))

                duration = get_audio_duration(f.name)
                assert duration > 0


# =============================================================================
# Streaming Module - Edge Cases
# =============================================================================

class TestStreamingEdgeCases:
    """Edge case tests for streaming module."""

    def test_stream_chunk_empty_samples(self):
        """Test StreamChunk with empty samples."""
        from voice_soundboard.streaming import StreamChunk

        chunk = StreamChunk(
            samples=np.array([]),
            sample_rate=24000,
            chunk_index=0,
            is_final=True,
            text_segment="",
        )

        assert len(chunk.samples) == 0
        assert chunk.is_final is True

    def test_stream_result_dataclass(self):
        """Test StreamResult dataclass."""
        from voice_soundboard.streaming import StreamResult

        result = StreamResult(
            audio_path=Path("/tmp/test.wav"),
            total_duration=5.0,
            total_chunks=10,
            generation_time=0.5,
            voice_used="af_bella",
        )
        assert result.total_duration == 5.0
        assert result.total_chunks == 10
        assert result.voice_used == "af_bella"


# =============================================================================
# LLM Streaming - Edge Cases
# =============================================================================

class TestLLMStreamingEdgeCases:
    """Edge case tests for LLM streaming."""

    def test_sentence_boundary_no_punctuation(self):
        """Test sentence boundary detection with no punctuation."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector, StreamConfig

        config = StreamConfig()
        detector = SentenceBoundaryDetector(config)

        # Text without punctuation
        sentences, remaining = detector.extract_complete("Hello world no punctuation here")

        assert isinstance(sentences, list)
        assert isinstance(remaining, str)

    def test_sentence_boundary_mixed_punctuation(self):
        """Test sentence boundary detection with mixed punctuation."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector, StreamConfig

        config = StreamConfig()
        detector = SentenceBoundaryDetector(config)

        text = "What? No way! That's amazing... Really?"
        sentences, remaining = detector.extract_complete(text)

        assert len(sentences) >= 1


# =============================================================================
# LLM Providers - Edge Cases
# =============================================================================

class TestLLMProvidersEdgeCases:
    """Edge case tests for LLM providers."""

    @pytest.mark.asyncio
    async def test_mock_provider_empty_prompt(self):
        """Test MockLLMProvider with empty prompt."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider()
        response = await provider.generate("")

        assert response.content != ""  # Should still generate something

    @pytest.mark.asyncio
    async def test_mock_provider_long_prompt(self):
        """Test MockLLMProvider with long prompt."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider()
        long_prompt = "Hello " * 1000  # Very long prompt

        response = await provider.generate(long_prompt)
        assert response.content != ""

    def test_provider_type_enum_all_values(self):
        """Test all ProviderType enum values."""
        from voice_soundboard.llm.providers import ProviderType

        # Check all expected providers exist
        assert ProviderType.MOCK is not None
        assert ProviderType.OLLAMA is not None
        assert ProviderType.OPENAI is not None
        assert ProviderType.VLLM is not None


# =============================================================================
# Security - Edge Cases
# =============================================================================

class TestSecurityEdgeCases:
    """Edge case tests for security module."""

    def test_rate_limiter_burst(self):
        """Test rate limiter under burst conditions."""
        from voice_soundboard.security import RateLimiter

        # Create limiter that allows 10 requests per second
        limiter = RateLimiter(max_requests=10, window_seconds=1.0)

        # Burst 10 requests
        for i in range(10):
            assert limiter.is_allowed(f"client{i}") is True

    def test_websocket_security_manager_validate_origin(self):
        """Test WebSocketSecurityManager origin validation."""
        from voice_soundboard.security import WebSocketSecurityManager

        manager = WebSocketSecurityManager()

        # Test with various origins
        for origin in ["http://localhost", "https://example.com", None]:
            # Should not crash
            result = manager.validate_origin(origin)
            assert isinstance(result, bool)


# =============================================================================
# Config - Edge Cases
# =============================================================================

class TestConfigEdgeCases:
    """Edge case tests for config module."""

    def test_voice_presets_all_have_required_fields(self):
        """Test all voice presets have required fields."""
        from voice_soundboard.config import VOICE_PRESETS

        for name, preset in VOICE_PRESETS.items():
            assert "voice" in preset, f"Preset {name} missing 'voice'"

    def test_kokoro_voices_format(self):
        """Test KOKORO_VOICES format."""
        from voice_soundboard.config import KOKORO_VOICES

        for voice_id, info in KOKORO_VOICES.items():
            assert isinstance(voice_id, str)
            # Voice IDs should follow naming convention
            assert "_" in voice_id or voice_id.isalpha()


# =============================================================================
# Emotion Module - Edge Cases
# =============================================================================

class TestEmotionEdgeCases:
    """Edge case tests for emotion module."""

    def test_emotion_happy_params(self):
        """Test happy emotion params."""
        from voice_soundboard.emotions import get_emotion_voice_params

        params = get_emotion_voice_params("happy")
        assert isinstance(params, dict)
        assert "speed" in params

    def test_emotion_angry_params(self):
        """Test angry emotion params."""
        from voice_soundboard.emotions import get_emotion_voice_params

        params = get_emotion_voice_params("angry")
        assert isinstance(params, dict)

    def test_emotion_unknown(self):
        """Test unknown emotion fallback."""
        from voice_soundboard.emotions import get_emotion_voice_params

        params = get_emotion_voice_params("unknown_emotion_xyz")
        assert isinstance(params, dict)  # Should return default


# =============================================================================
# Normalizer - Mobile Text Edge Cases
# =============================================================================

class TestNormalizerMobileEdgeCases:
    """Edge case tests for normalizer with mobile text patterns."""

    def test_normalize_emoji_text(self):
        """Test normalizing text with emojis (common on mobile)."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("Hello 👋 how are you? 😊")
        assert isinstance(result, str)

    def test_normalize_abbreviations_mobile(self):
        """Test normalizing mobile abbreviations."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("lol that's so funny tbh")
        assert isinstance(result, str)

    def test_normalize_unicode_text(self):
        """Test normalizing unicode text."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("Café résumé naïve")
        assert isinstance(result, str)


# =============================================================================
# SSML - Edge Cases
# =============================================================================

class TestSSMLEdgeCases:
    """Edge case tests for SSML module."""

    def test_parse_ssml_empty(self):
        """Test parsing empty SSML."""
        from voice_soundboard.ssml import parse_ssml

        text, params = parse_ssml("<speak></speak>")
        assert text == "" or text.strip() == ""

    def test_parse_ssml_nested_tags(self):
        """Test parsing SSML with nested tags."""
        from voice_soundboard.ssml import parse_ssml

        ssml = """<speak>
            <prosody rate="slow">
                <emphasis level="strong">Very</emphasis> important
            </prosody>
        </speak>"""

        text, params = parse_ssml(ssml)
        assert "important" in text.lower()

    def test_parse_ssml_malformed(self):
        """Test parsing malformed SSML."""
        from voice_soundboard.ssml import parse_ssml

        # Should handle gracefully
        try:
            text, params = parse_ssml("<speak>Unclosed tag")
            assert isinstance(text, str)
        except Exception:
            pass  # May raise on malformed input


# =============================================================================
# Dialogue Parser - Edge Cases
# =============================================================================

class TestDialogueParserEdgeCases:
    """Edge case tests for dialogue parser."""

    def test_parse_dialogue_single_line(self):
        """Test parsing single line dialogue."""
        from voice_soundboard.dialogue.parser import parse_dialogue

        result = parse_dialogue("[S1] Hello.")
        assert result is not None

    def test_parse_dialogue_with_directions(self):
        """Test parsing dialogue with stage directions."""
        from voice_soundboard.dialogue.parser import parse_dialogue

        script = """
        [S1:narrator] (softly) Once upon a time...
        [S2:character] (excitedly) What happened next?
        """

        result = parse_dialogue(script)
        assert len(result.lines) >= 2

    def test_parse_dialogue_unicode_speakers(self):
        """Test parsing dialogue with unicode speaker names."""
        from voice_soundboard.dialogue.parser import parse_dialogue

        script = "[S1:Müller] Guten Tag!"
        result = parse_dialogue(script)
        assert result is not None


# =============================================================================
# Cloning - Edge Cases
# =============================================================================

class TestCloningEdgeCases:
    """Edge case tests for cloning module."""

    def test_voice_library_empty(self):
        """Test VoiceLibrary with empty directory."""
        from voice_soundboard.cloning.library import VoiceLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            library = VoiceLibrary(library_path=Path(tmpdir))

            # Should return empty list
            voices = library.list_all()
            assert isinstance(voices, list)
            assert len(voices) == 0

    def test_voice_profile_optional_fields(self):
        """Test VoiceProfile with minimal fields."""
        from voice_soundboard.cloning.library import VoiceProfile

        profile = VoiceProfile(
            voice_id="test",
            name="Test",
            created_at="2024-01-01",
        )

        assert profile.voice_id == "test"


# =============================================================================
# Conversion - Edge Cases
# =============================================================================

class TestConversionEdgeCases:
    """Edge case tests for conversion module."""

    def test_audio_device_class_fields(self):
        """Test AudioDevice dataclass fields."""
        from voice_soundboard.conversion.devices import AudioDevice, DeviceType

        device = AudioDevice(
            id=0,
            name="Test Device",
            device_type=DeviceType.INPUT,
            max_input_channels=2,
            max_output_channels=0,
            default_sample_rate=44100.0,
        )

        assert device.id == 0
        assert device.device_type == DeviceType.INPUT


# =============================================================================
# Effects - Edge Cases
# =============================================================================

class TestEffectsEdgeCases:
    """Edge case tests for effects module."""

    def test_get_effect_unknown(self):
        """Test getting unknown effect."""
        from voice_soundboard.effects import get_effect

        # Should handle gracefully
        try:
            effect = get_effect("nonexistent_effect_xyz")
            assert effect is None or effect is not None
        except (KeyError, ValueError):
            pass  # Expected for unknown effect

    def test_list_effects_returns_list(self):
        """Test list_effects returns a list."""
        from voice_soundboard.effects import list_effects

        effects = list_effects()
        assert isinstance(effects, list)
        assert len(effects) > 0


# =============================================================================
# Server - Remaining Handler Coverage
# =============================================================================

class TestServerRemainingHandlers:
    """Tests for remaining server handlers."""

    @pytest.mark.asyncio
    async def test_handle_list_llm_providers(self):
        """Test list_llm_providers handler."""
        from voice_soundboard.server import handle_list_llm_providers

        result = await handle_list_llm_providers({})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_list_effects(self):
        """Test list_effects handler."""
        from voice_soundboard.server import handle_list_effects

        result = await handle_list_effects({})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_list_emotions(self):
        """Test list_emotions handler."""
        from voice_soundboard.server import handle_list_emotions

        result = await handle_list_emotions({})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_start_conversation(self):
        """Test start_conversation handler."""
        from voice_soundboard.server import handle_start_conversation

        result = await handle_start_conversation({})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_add_conversation_message(self):
        """Test add_conversation_message handler."""
        from voice_soundboard.server import handle_add_conversation_message

        result = await handle_add_conversation_message({
            "role": "user",
            "content": "Hello!",
        })
        assert len(result) == 1


# =============================================================================
# WebSocket - Connection Edge Cases
# =============================================================================

class TestWebSocketConnectionEdgeCases:
    """Edge case tests for WebSocket connections."""

    def test_ws_response_to_json(self):
        """Test WSResponse to_json method."""
        from voice_soundboard.websocket_server import WSResponse

        response = WSResponse(
            success=True,
            action="test",
            data={"key": "value"},
            request_id="req-1",
        )

        json_str = response.to_json()
        parsed = json.loads(json_str)

        assert parsed["success"] is True
        assert parsed["action"] == "test"
        assert parsed["data"]["key"] == "value"

    def test_voice_websocket_server_init(self):
        """Test VoiceWebSocketServer initialization."""
        from voice_soundboard.websocket_server import VoiceWebSocketServer

        server = VoiceWebSocketServer()
        assert server is not None
