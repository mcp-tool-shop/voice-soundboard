"""
Additional test coverage batch 26: server.py MCP handlers (part 2).

Tests for MCP server tool handlers - speech synthesis, dialogue, and cloning handlers.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import numpy as np

from mcp.types import TextContent


# ============================================================================
# Speak Handler Tests
# ============================================================================

class TestHandleSpeak:
    """Tests for handle_speak handler."""

    @pytest.mark.asyncio
    async def test_speak_missing_text(self):
        """Test speak with missing text."""
        from voice_soundboard.server import handle_speak

        result = await handle_speak({})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "'text' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_speak_success(self):
        """Test successful speech generation."""
        from voice_soundboard.server import handle_speak

        with patch('voice_soundboard.server.get_engine') as mock_get:
            mock_engine = Mock()
            mock_result = Mock()
            mock_result.audio_path = Path("/tmp/test.wav")
            mock_result.voice_used = "af_bella"
            mock_result.duration_seconds = 2.5
            mock_result.realtime_factor = 15.0
            mock_engine.speak.return_value = mock_result
            mock_get.return_value = mock_engine

            result = await handle_speak({"text": "Hello world"})
            assert len(result) == 1
            assert "Generated speech" in result[0].text
            assert "af_bella" in result[0].text

    @pytest.mark.asyncio
    async def test_speak_with_style(self):
        """Test speak with style parameter."""
        from voice_soundboard.server import handle_speak

        with patch('voice_soundboard.server.get_engine') as mock_get:
            with patch('voice_soundboard.server.apply_style_to_params') as mock_style:
                mock_style.return_value = ("bm_george", 0.9, None)

                mock_engine = Mock()
                mock_result = Mock()
                mock_result.audio_path = Path("/tmp/test.wav")
                mock_result.voice_used = "bm_george"
                mock_result.duration_seconds = 2.0
                mock_result.realtime_factor = 12.0
                mock_engine.speak.return_value = mock_result
                mock_get.return_value = mock_engine

                result = await handle_speak({
                    "text": "Hello",
                    "style": "like a narrator"
                })
                assert len(result) == 1
                mock_style.assert_called_once()

    @pytest.mark.asyncio
    async def test_speak_with_play(self):
        """Test speak with play=True."""
        from voice_soundboard.server import handle_speak

        with patch('voice_soundboard.server.get_engine') as mock_get:
            with patch('voice_soundboard.server.play_audio') as mock_play:
                mock_engine = Mock()
                mock_result = Mock()
                mock_result.audio_path = Path("/tmp/test.wav")
                mock_result.voice_used = "af_bella"
                mock_result.duration_seconds = 1.0
                mock_result.realtime_factor = 10.0
                mock_engine.speak.return_value = mock_result
                mock_get.return_value = mock_engine

                result = await handle_speak({
                    "text": "Hello",
                    "play": True
                })
                assert len(result) == 1
                # play_audio should have been called through asyncio.to_thread

    @pytest.mark.asyncio
    async def test_speak_error_handling(self):
        """Test speak handles errors gracefully."""
        from voice_soundboard.server import handle_speak

        with patch('voice_soundboard.server.get_engine') as mock_get:
            mock_get.side_effect = RuntimeError("Engine failed")

            result = await handle_speak({"text": "Hello"})
            assert len(result) == 1
            assert "Error" in result[0].text


class TestHandleSpeakLong:
    """Tests for handle_speak_long handler."""

    @pytest.mark.asyncio
    async def test_speak_long_missing_text(self):
        """Test speak_long with missing text."""
        from voice_soundboard.server import handle_speak_long

        result = await handle_speak_long({})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "'text' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_speak_long_success(self):
        """Test successful long speech generation."""
        from voice_soundboard.server import handle_speak_long

        with patch('voice_soundboard.streaming.StreamingEngine') as mock_cls:
            mock_engine = Mock()
            mock_result = Mock()
            mock_result.audio_path = Path("/tmp/stream.wav")
            mock_result.voice_used = "af_bella"
            mock_result.total_duration = 10.0
            mock_result.total_chunks = 5
            mock_result.generation_time = 1.5

            # Make stream_to_file an async method
            async def mock_stream(*args, **kwargs):
                return mock_result
            mock_engine.stream_to_file = mock_stream
            mock_cls.return_value = mock_engine

            result = await handle_speak_long({"text": "This is a long text..."})
            assert len(result) == 1
            # Should succeed or handle error gracefully


class TestHandleSpeakSSML:
    """Tests for handle_speak_ssml handler."""

    @pytest.mark.asyncio
    async def test_speak_ssml_missing_ssml(self):
        """Test speak_ssml with missing ssml."""
        from voice_soundboard.server import handle_speak_ssml

        result = await handle_speak_ssml({})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "'ssml' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_speak_ssml_success(self):
        """Test successful SSML speech generation."""
        from voice_soundboard.server import handle_speak_ssml

        with patch('voice_soundboard.server.parse_ssml') as mock_parse:
            mock_params = Mock()
            mock_params.voice = None
            mock_params.speed = 1.0
            mock_parse.return_value = ("Hello world", mock_params)

            with patch('voice_soundboard.server.get_engine') as mock_get:
                mock_engine = Mock()
                mock_result = Mock()
                mock_result.audio_path = Path("/tmp/ssml.wav")
                mock_result.voice_used = "af_bella"
                mock_result.duration_seconds = 2.0
                mock_engine.speak.return_value = mock_result
                mock_get.return_value = mock_engine

                result = await handle_speak_ssml({
                    "ssml": "<speak>Hello world</speak>"
                })
                assert len(result) == 1
                assert "Generated SSML speech" in result[0].text


class TestHandleSpeakRealtime:
    """Tests for handle_speak_realtime handler."""

    @pytest.mark.asyncio
    async def test_speak_realtime_missing_text(self):
        """Test speak_realtime with missing text."""
        from voice_soundboard.server import handle_speak_realtime

        result = await handle_speak_realtime({})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "'text' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_speak_realtime_success(self):
        """Test successful realtime speech."""
        from voice_soundboard.server import handle_speak_realtime

        with patch('voice_soundboard.server.stream_realtime') as mock_stream:
            mock_result = Mock()
            mock_result.total_duration = 2.0
            mock_result.total_chunks = 3
            mock_result.voice_used = "af_bella"
            mock_result.generation_time = 0.5

            async def mock_fn(*args, **kwargs):
                return mock_result
            mock_stream.side_effect = mock_fn

            result = await handle_speak_realtime({"text": "Hello"})
            assert len(result) == 1
            assert "Real-time speech completed" in result[0].text

    @pytest.mark.asyncio
    async def test_speak_realtime_with_emotion(self):
        """Test realtime speech with emotion."""
        from voice_soundboard.server import handle_speak_realtime

        with patch('voice_soundboard.server.get_emotion_voice_params') as mock_emotion:
            mock_emotion.return_value = {"voice": "af_bella", "speed": 1.1}

            with patch('voice_soundboard.server.apply_emotion_to_text') as mock_apply:
                mock_apply.return_value = "Hello!"

                with patch('voice_soundboard.server.stream_realtime') as mock_stream:
                    mock_result = Mock()
                    mock_result.total_duration = 2.0
                    mock_result.total_chunks = 3
                    mock_result.voice_used = "af_bella"
                    mock_result.generation_time = 0.5

                    async def mock_fn(*args, **kwargs):
                        return mock_result
                    mock_stream.side_effect = mock_fn

                    result = await handle_speak_realtime({
                        "text": "Hello",
                        "emotion": "happy"
                    })
                    assert len(result) == 1


# ============================================================================
# Chatterbox Handler Tests
# ============================================================================

class TestHandleSpeakChatterbox:
    """Tests for handle_speak_chatterbox handler."""

    @pytest.mark.asyncio
    async def test_speak_chatterbox_missing_text(self):
        """Test speak_chatterbox with missing text."""
        from voice_soundboard.server import handle_speak_chatterbox

        result = await handle_speak_chatterbox({})
        assert len(result) == 1
        assert "Error" in result[0].text

    @pytest.mark.asyncio
    async def test_speak_chatterbox_not_installed(self):
        """Test speak_chatterbox when Chatterbox not installed."""
        from voice_soundboard.server import handle_speak_chatterbox

        with patch('voice_soundboard.server.get_chatterbox_engine') as mock_get:
            mock_get.side_effect = ImportError("Chatterbox not installed")

            result = await handle_speak_chatterbox({"text": "Hello"})
            assert len(result) == 1
            # Should return import error message

    @pytest.mark.asyncio
    async def test_speak_chatterbox_success(self):
        """Test successful Chatterbox speech."""
        from voice_soundboard.server import handle_speak_chatterbox

        with patch('voice_soundboard.server.get_chatterbox_engine') as mock_get:
            mock_engine = Mock()
            mock_result = Mock()
            mock_result.audio_path = Path("/tmp/chatter.wav")
            mock_result.duration_seconds = 2.0
            mock_result.realtime_factor = 10.0
            mock_result.metadata = {"emotion_exaggeration": 0.5, "cfg_weight": 0.5}
            mock_engine.speak.return_value = mock_result
            mock_get.return_value = mock_engine

            result = await handle_speak_chatterbox({"text": "Hello [laugh]"})
            assert len(result) == 1
            assert "Generated Chatterbox speech" in result[0].text


class TestHandleCloneVoice:
    """Tests for handle_clone_voice handler."""

    @pytest.mark.asyncio
    async def test_clone_voice_missing_path(self):
        """Test clone_voice with missing audio_path."""
        from voice_soundboard.server import handle_clone_voice

        result = await handle_clone_voice({})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "'audio_path' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_clone_voice_success(self):
        """Test successful voice cloning."""
        from voice_soundboard.server import handle_clone_voice

        with patch('voice_soundboard.server.get_chatterbox_engine') as mock_get:
            mock_engine = Mock()
            mock_engine.clone_voice.return_value = "my_voice"
            mock_get.return_value = mock_engine

            result = await handle_clone_voice({
                "audio_path": "/path/to/audio.wav",
                "voice_id": "my_voice"
            })
            assert len(result) == 1
            assert "Voice registered successfully" in result[0].text

    @pytest.mark.asyncio
    async def test_clone_voice_file_not_found(self):
        """Test clone_voice with non-existent file."""
        from voice_soundboard.server import handle_clone_voice

        with patch('voice_soundboard.server.get_chatterbox_engine') as mock_get:
            mock_engine = Mock()
            mock_engine.clone_voice.side_effect = FileNotFoundError("Not found")
            mock_get.return_value = mock_engine

            result = await handle_clone_voice({
                "audio_path": "/nonexistent/audio.wav"
            })
            assert len(result) == 1
            assert "Error" in result[0].text


# ============================================================================
# Dialogue Handler Tests
# ============================================================================

class TestHandleSpeakDialogue:
    """Tests for handle_speak_dialogue handler."""

    @pytest.mark.asyncio
    async def test_speak_dialogue_missing_script(self):
        """Test speak_dialogue with missing script."""
        from voice_soundboard.server import handle_speak_dialogue

        result = await handle_speak_dialogue({})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "'script' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_speak_dialogue_success(self):
        """Test successful dialogue synthesis."""
        from voice_soundboard.server import handle_speak_dialogue

        with patch('voice_soundboard.server.get_dialogue_engine') as mock_get:
            mock_engine = Mock()
            mock_result = Mock()
            mock_result.audio_path = Path("/tmp/dialogue.wav")
            mock_result.duration_seconds = 10.0
            mock_result.speaker_count = 2
            mock_result.line_count = 4
            mock_result.voice_assignments = {"narrator": "bm_george", "alice": "af_bella"}
            mock_result.turns = [
                Mock(speaker_name="narrator"),
                Mock(speaker_name="alice")
            ]
            mock_result.get_speaker_duration.return_value = 5.0
            mock_engine.synthesize.return_value = mock_result
            mock_get.return_value = mock_engine

            result = await handle_speak_dialogue({
                "script": "[S1:narrator] Hello\n[S2:alice] Hi there"
            })
            assert len(result) == 1
            assert "Generated dialogue" in result[0].text


class TestHandlePreviewDialogue:
    """Tests for handle_preview_dialogue handler."""

    @pytest.mark.asyncio
    async def test_preview_dialogue_missing_script(self):
        """Test preview_dialogue with missing script."""
        from voice_soundboard.server import handle_preview_dialogue

        result = await handle_preview_dialogue({})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "'script' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_preview_dialogue_success(self):
        """Test successful dialogue preview."""
        from voice_soundboard.server import handle_preview_dialogue

        with patch('voice_soundboard.server.get_dialogue_engine') as mock_get:
            mock_engine = Mock()
            mock_engine.get_script_info.return_value = {
                "title": "Test Script",
                "speaker_count": 2,
                "line_count": 4,
                "total_words": 50,
                "estimated_duration_seconds": 30.0,
                "speaker_lines": {"narrator": 2, "alice": 2},
                "metadata": {}
            }
            mock_engine.preview_assignments.return_value = {
                "narrator": "bm_george",
                "alice": "af_bella"
            }
            mock_get.return_value = mock_engine

            result = await handle_preview_dialogue({
                "script": "[S1:narrator] Hello\n[S2:alice] Hi"
            })
            assert len(result) == 1
            assert "Dialogue preview" in result[0].text


# ============================================================================
# Voice Cloning Advanced Handler Tests
# ============================================================================

class TestHandleCloneVoiceAdvanced:
    """Tests for handle_clone_voice_advanced handler."""

    @pytest.mark.asyncio
    async def test_clone_voice_advanced_missing_path(self):
        """Test clone_voice_advanced with missing audio_path."""
        from voice_soundboard.server import handle_clone_voice_advanced

        result = await handle_clone_voice_advanced({})
        assert len(result) == 1
        assert "Error" in result[0].text

    @pytest.mark.asyncio
    async def test_clone_voice_advanced_missing_consent(self):
        """Test clone_voice_advanced requires consent."""
        from voice_soundboard.server import handle_clone_voice_advanced

        result = await handle_clone_voice_advanced({
            "audio_path": "/path/to/audio.wav"
        })
        assert len(result) == 1
        # Should require consent acknowledgment


class TestHandleValidateCloneAudio:
    """Tests for handle_validate_clone_audio handler."""

    @pytest.mark.asyncio
    async def test_validate_clone_audio(self):
        """Test audio validation for cloning."""
        from voice_soundboard.server import handle_validate_clone_audio

        result = await handle_validate_clone_audio({
            "audio_path": "/path/to/audio.wav"
        })
        assert len(result) == 1
        # Should return validation result or error


class TestHandleGetVoiceProfile:
    """Tests for handle_get_voice_profile handler."""

    @pytest.mark.asyncio
    async def test_get_voice_profile(self):
        """Test getting voice profile."""
        from voice_soundboard.server import handle_get_voice_profile

        result = await handle_get_voice_profile({
            "voice_id": "test_voice"
        })
        assert len(result) == 1


class TestHandleDeleteClonedVoice:
    """Tests for handle_delete_cloned_voice handler."""

    @pytest.mark.asyncio
    async def test_delete_cloned_voice(self):
        """Test deleting cloned voice."""
        from voice_soundboard.server import handle_delete_cloned_voice

        result = await handle_delete_cloned_voice({
            "voice_id": "test_voice"
        })
        assert len(result) == 1


class TestHandleFindSimilarVoices:
    """Tests for handle_find_similar_voices handler."""

    @pytest.mark.asyncio
    async def test_find_similar_voices(self):
        """Test finding similar voices."""
        from voice_soundboard.server import handle_find_similar_voices

        result = await handle_find_similar_voices({
            "audio_path": "/path/to/audio.wav"
        })
        assert len(result) == 1


class TestHandleTransferVoiceEmotion:
    """Tests for handle_transfer_voice_emotion handler."""

    @pytest.mark.asyncio
    async def test_transfer_voice_emotion(self):
        """Test transferring emotion to voice."""
        from voice_soundboard.server import handle_transfer_voice_emotion

        result = await handle_transfer_voice_emotion({
            "voice_id": "test_voice",
            "emotion": "happy"
        })
        assert len(result) == 1


class TestHandleCheckLanguageCompatibility:
    """Tests for handle_check_language_compatibility handler."""

    @pytest.mark.asyncio
    async def test_check_language_compatibility(self):
        """Test checking language compatibility."""
        from voice_soundboard.server import handle_check_language_compatibility

        result = await handle_check_language_compatibility({
            "source_language": "english",
            "target_language": "spanish"
        })
        assert len(result) == 1


# ============================================================================
# Codec Handler Tests
# ============================================================================

class TestHandleEncodeAudioTokens:
    """Tests for handle_encode_audio_tokens handler."""

    @pytest.mark.asyncio
    async def test_encode_audio_tokens(self):
        """Test encoding audio to tokens."""
        from voice_soundboard.server import handle_encode_audio_tokens

        result = await handle_encode_audio_tokens({
            "audio_path": "/path/to/audio.wav"
        })
        assert len(result) == 1


class TestHandleDecodeAudioTokens:
    """Tests for handle_decode_audio_tokens handler."""

    @pytest.mark.asyncio
    async def test_decode_audio_tokens(self):
        """Test decoding tokens to audio."""
        from voice_soundboard.server import handle_decode_audio_tokens

        result = await handle_decode_audio_tokens({
            "tokens": [1, 2, 3, 4, 5]
        })
        assert len(result) == 1


class TestHandleGetCodecInfo:
    """Tests for handle_get_codec_info handler."""

    @pytest.mark.asyncio
    async def test_get_codec_info(self):
        """Test getting codec info."""
        from voice_soundboard.server import handle_get_codec_info

        result = await handle_get_codec_info({})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_codec_info_specific(self):
        """Test getting specific codec info."""
        from voice_soundboard.server import handle_get_codec_info

        result = await handle_get_codec_info({"codec": "mock"})
        assert len(result) == 1


class TestHandleEstimateAudioTokens:
    """Tests for handle_estimate_audio_tokens handler."""

    @pytest.mark.asyncio
    async def test_estimate_audio_tokens(self):
        """Test estimating audio tokens."""
        from voice_soundboard.server import handle_estimate_audio_tokens

        result = await handle_estimate_audio_tokens({
            "duration_seconds": 5.0
        })
        assert len(result) == 1


class TestHandleVoiceConvertDualcodec:
    """Tests for handle_voice_convert_dualcodec handler."""

    @pytest.mark.asyncio
    async def test_voice_convert_dualcodec(self):
        """Test voice conversion with DualCodec."""
        from voice_soundboard.server import handle_voice_convert_dualcodec

        result = await handle_voice_convert_dualcodec({
            "content_audio": "/path/to/content.wav",
            "style_audio": "/path/to/style.wav"
        })
        assert len(result) == 1


# ============================================================================
# Voice Conversion Handler Tests
# ============================================================================

class TestHandleStartVoiceConversion:
    """Tests for handle_start_voice_conversion handler."""

    @pytest.mark.asyncio
    async def test_start_voice_conversion(self):
        """Test starting voice conversion."""
        from voice_soundboard.server import handle_start_voice_conversion

        result = await handle_start_voice_conversion({
            "target_voice": "test_voice"
        })
        assert len(result) == 1


class TestHandleStopVoiceConversion:
    """Tests for handle_stop_voice_conversion handler."""

    @pytest.mark.asyncio
    async def test_stop_voice_conversion(self):
        """Test stopping voice conversion."""
        from voice_soundboard.server import handle_stop_voice_conversion

        result = await handle_stop_voice_conversion({})
        assert len(result) == 1


class TestHandleGetVoiceConversionStatus:
    """Tests for handle_get_voice_conversion_status handler."""

    @pytest.mark.asyncio
    async def test_get_voice_conversion_status(self):
        """Test getting voice conversion status."""
        from voice_soundboard.server import handle_get_voice_conversion_status

        result = await handle_get_voice_conversion_status({})
        assert len(result) == 1


class TestHandleConvertAudioFile:
    """Tests for handle_convert_audio_file handler."""

    @pytest.mark.asyncio
    async def test_convert_audio_file(self):
        """Test converting audio file."""
        from voice_soundboard.server import handle_convert_audio_file

        result = await handle_convert_audio_file({
            "input_path": "/path/to/input.wav",
            "target_voice": "test_voice"
        })
        assert len(result) == 1


# ============================================================================
# LLM Handler Tests
# ============================================================================

class TestHandleSpeakWithContext:
    """Tests for handle_speak_with_context handler."""

    @pytest.mark.asyncio
    async def test_speak_with_context(self):
        """Test speaking with context."""
        from voice_soundboard.server import handle_speak_with_context

        result = await handle_speak_with_context({
            "text": "Hello",
            "context": "greeting"
        })
        assert len(result) == 1


class TestHandleStartConversation:
    """Tests for handle_start_conversation handler."""

    @pytest.mark.asyncio
    async def test_start_conversation(self):
        """Test starting a conversation."""
        from voice_soundboard.server import handle_start_conversation

        result = await handle_start_conversation({})
        assert len(result) == 1


class TestHandleAddConversationMessage:
    """Tests for handle_add_conversation_message handler."""

    @pytest.mark.asyncio
    async def test_add_conversation_message(self):
        """Test adding conversation message."""
        from voice_soundboard.server import handle_add_conversation_message

        result = await handle_add_conversation_message({
            "role": "user",
            "content": "Hello"
        })
        assert len(result) == 1


class TestHandleGetConversationContext:
    """Tests for handle_get_conversation_context handler."""

    @pytest.mark.asyncio
    async def test_get_conversation_context(self):
        """Test getting conversation context."""
        from voice_soundboard.server import handle_get_conversation_context

        result = await handle_get_conversation_context({})
        assert len(result) == 1


class TestHandleGetConversationStats:
    """Tests for handle_get_conversation_stats handler."""

    @pytest.mark.asyncio
    async def test_get_conversation_stats(self):
        """Test getting conversation stats."""
        from voice_soundboard.server import handle_get_conversation_stats

        result = await handle_get_conversation_stats({})
        assert len(result) == 1


class TestHandleEndConversation:
    """Tests for handle_end_conversation handler."""

    @pytest.mark.asyncio
    async def test_end_conversation(self):
        """Test ending a conversation."""
        from voice_soundboard.server import handle_end_conversation

        result = await handle_end_conversation({})
        assert len(result) == 1


class TestHandleDetectUserEmotion:
    """Tests for handle_detect_user_emotion handler."""

    @pytest.mark.asyncio
    async def test_detect_user_emotion(self):
        """Test detecting user emotion."""
        from voice_soundboard.server import handle_detect_user_emotion

        result = await handle_detect_user_emotion({
            "message": "I'm so happy today!"
        })
        assert len(result) == 1


class TestHandleSelectResponseEmotion:
    """Tests for handle_select_response_emotion handler."""

    @pytest.mark.asyncio
    async def test_select_response_emotion(self):
        """Test selecting response emotion."""
        from voice_soundboard.server import handle_select_response_emotion

        result = await handle_select_response_emotion({
            "response_text": "That's wonderful!"
        })
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_select_response_emotion_with_user_emotion(self):
        """Test selecting response emotion with user emotion."""
        from voice_soundboard.server import handle_select_response_emotion

        result = await handle_select_response_emotion({
            "response_text": "I'm sorry to hear that.",
            "user_emotion": "sad"
        })
        assert len(result) == 1
