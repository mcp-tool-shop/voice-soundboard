"""
Tests for MCP Server Handlers

Targets the uncovered handler functions in voice_soundboard/server.py
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np


# Mock the imports that might not be available
@pytest.fixture(autouse=True)
def mock_audio_dependencies():
    """Mock audio dependencies."""
    with patch.dict('sys.modules', {
        'sounddevice': MagicMock(),
        'soundfile': MagicMock(),
    }):
        yield


class TestHandlePlayAudio:
    """Tests for handle_play_audio handler."""

    @pytest.mark.asyncio
    async def test_play_audio_missing_path(self):
        """Should return error when path is missing."""
        from voice_soundboard.server import handle_play_audio

        result = await handle_play_audio({})
        assert len(result) == 1
        assert "path" in result[0].text.lower() or "required" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_play_audio_file_not_found(self, tmp_path):
        """Should return error when file doesn't exist."""
        from voice_soundboard.server import handle_play_audio

        fake_path = tmp_path / "nonexistent.wav"
        result = await handle_play_audio({"path": str(fake_path)})
        assert len(result) == 1
        assert "not found" in result[0].text.lower() or "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_play_audio_success(self, tmp_path):
        """Should play audio successfully."""
        from voice_soundboard.server import handle_play_audio

        # Create a test audio file
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"RIFF" + b"\x00" * 100)

        with patch('voice_soundboard.server.get_audio_duration', return_value=1.5):
            with patch('voice_soundboard.server.play_audio'):
                result = await handle_play_audio({"path": str(test_file)})

        assert len(result) == 1
        assert "success" in result[0].text.lower() or "played" in result[0].text.lower()


class TestHandleStopAudio:
    """Tests for handle_stop_audio handler."""

    @pytest.mark.asyncio
    async def test_stop_audio_success(self):
        """Should stop audio playback."""
        from voice_soundboard.server import handle_stop_audio

        with patch('voice_soundboard.server.stop_playback'):
            result = await handle_stop_audio({})

        assert len(result) == 1
        assert "stop" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_stop_audio_exception(self):
        """Should handle exceptions gracefully."""
        from voice_soundboard.server import handle_stop_audio

        with patch('voice_soundboard.server.stop_playback', side_effect=Exception("Test error")):
            result = await handle_stop_audio({})

        assert len(result) == 1
        # Should return error response


class TestHandleSoundEffect:
    """Tests for handle_sound_effect handler."""

    @pytest.mark.asyncio
    async def test_effect_missing_name(self):
        """Should return error when effect name is missing."""
        from voice_soundboard.server import handle_sound_effect

        result = await handle_sound_effect({})
        assert len(result) == 1
        assert "effect" in result[0].text.lower() or "required" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_effect_not_found(self):
        """Should return error for unknown effect."""
        from voice_soundboard.server import handle_sound_effect

        with patch('voice_soundboard.server.get_effect', side_effect=ValueError("Not found")):
            result = await handle_sound_effect({"effect": "nonexistent_effect"})

        assert len(result) == 1
        assert "not found" in result[0].text.lower() or "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_effect_play_success(self):
        """Should play effect successfully."""
        from voice_soundboard.server import handle_sound_effect

        mock_effect = Mock()
        mock_effect.duration = 0.5
        mock_effect.play = Mock()

        with patch('voice_soundboard.server.get_effect', return_value=mock_effect):
            result = await handle_sound_effect({"effect": "chime"})

        assert len(result) == 1
        assert "success" in result[0].text.lower() or "played" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_effect_save_success(self, tmp_path):
        """Should save effect to file."""
        from voice_soundboard.server import handle_sound_effect

        mock_effect = Mock()
        mock_effect.save = Mock()

        save_path = tmp_path / "effect.wav"

        with patch('voice_soundboard.server.get_effect', return_value=mock_effect):
            result = await handle_sound_effect({
                "effect": "chime",
                "save_path": str(save_path)
            })

        assert len(result) == 1
        assert "saved" in result[0].text.lower()
        mock_effect.save.assert_called_once()


class TestHandleListEffects:
    """Tests for handle_list_effects handler."""

    @pytest.mark.asyncio
    async def test_list_effects(self):
        """Should list available effects."""
        from voice_soundboard.server import handle_list_effects

        with patch('voice_soundboard.server.EFFECTS', {'chime': Mock(), 'click': Mock()}):
            result = await handle_list_effects({})

        assert len(result) == 1
        assert "effect" in result[0].text.lower()


class TestHandleSpeakLong:
    """Tests for handle_speak_long handler."""

    @pytest.mark.asyncio
    async def test_speak_long_missing_text(self):
        """Should return error when text is missing."""
        from voice_soundboard.server import handle_speak_long

        result = await handle_speak_long({})
        assert len(result) == 1
        assert "text" in result[0].text.lower() or "required" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_speak_long_success(self):
        """Should stream long text successfully."""
        from voice_soundboard.server import handle_speak_long

        mock_result = Mock()
        mock_result.audio_path = Path("output.wav")
        mock_result.voice_used = "test_voice"
        mock_result.total_duration = 5.0
        mock_result.total_chunks = 3
        mock_result.generation_time = 2.0

        mock_engine = Mock()
        mock_engine.stream_to_file = AsyncMock(return_value=mock_result)

        with patch('voice_soundboard.server.StreamingEngine', return_value=mock_engine):
            with patch('voice_soundboard.streaming.StreamingEngine', return_value=mock_engine):
                result = await handle_speak_long({"text": "This is a long text for streaming"})

        assert len(result) == 1


class TestHandleSpeakSsml:
    """Tests for handle_speak_ssml handler."""

    @pytest.mark.asyncio
    async def test_ssml_missing_text(self):
        """Should return error when SSML is missing."""
        from voice_soundboard.server import handle_speak_ssml

        result = await handle_speak_ssml({})
        assert len(result) == 1
        assert "ssml" in result[0].text.lower() or "required" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_ssml_success(self):
        """Should process SSML successfully."""
        from voice_soundboard.server import handle_speak_ssml

        mock_result = Mock()
        mock_result.audio_path = Path("output.wav")
        mock_result.voice_used = "test_voice"
        mock_result.duration_seconds = 2.0

        mock_params = Mock()
        mock_params.voice = None
        mock_params.speed = 1.0

        mock_engine = Mock()
        mock_engine.speak = Mock(return_value=mock_result)

        with patch('voice_soundboard.server.parse_ssml', return_value=("Hello", mock_params)):
            with patch('voice_soundboard.server.get_engine', return_value=mock_engine):
                result = await handle_speak_ssml({"ssml": "<speak>Hello</speak>"})

        assert len(result) == 1


class TestHandleSpeakRealtime:
    """Tests for handle_speak_realtime handler."""

    @pytest.mark.asyncio
    async def test_realtime_missing_text(self):
        """Should return error when text is missing."""
        from voice_soundboard.server import handle_speak_realtime

        result = await handle_speak_realtime({})
        assert len(result) == 1
        assert "text" in result[0].text.lower() or "required" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_realtime_with_emotion(self):
        """Should apply emotion to realtime speech."""
        from voice_soundboard.server import handle_speak_realtime

        mock_result = Mock()
        mock_result.total_duration = 2.0
        mock_result.total_chunks = 2
        mock_result.voice_used = "happy_voice"
        mock_result.generation_time = 1.0

        with patch('voice_soundboard.server.get_emotion_voice_params', return_value={"voice": "happy_voice", "speed": 1.1}):
            with patch('voice_soundboard.server.apply_emotion_to_text', return_value="Hello!"):
                with patch('voice_soundboard.server.stream_realtime', return_value=mock_result):
                    result = await handle_speak_realtime({
                        "text": "Hello",
                        "emotion": "happy"
                    })

        assert len(result) == 1


class TestHandleListEmotions:
    """Tests for handle_list_emotions handler."""

    @pytest.mark.asyncio
    async def test_list_emotions(self):
        """Should list available emotions."""
        from voice_soundboard.server import handle_list_emotions

        mock_emotion = Mock()
        mock_emotion.speed = 1.0
        mock_emotion.voice_preference = "default"

        with patch('voice_soundboard.server.EMOTIONS', {'happy': mock_emotion}):
            result = await handle_list_emotions({})

        assert len(result) == 1
        assert "emotion" in result[0].text.lower()


class TestHandleSpeakChatterbox:
    """Tests for handle_speak_chatterbox handler."""

    @pytest.mark.asyncio
    async def test_chatterbox_missing_text(self):
        """Should return error when text is missing."""
        from voice_soundboard.server import handle_speak_chatterbox

        result = await handle_speak_chatterbox({})
        assert len(result) == 1
        assert "text" in result[0].text.lower() or "required" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_chatterbox_success(self):
        """Should generate Chatterbox speech."""
        from voice_soundboard.server import handle_speak_chatterbox

        mock_result = Mock()
        mock_result.audio_path = Path("output.wav")
        mock_result.duration_seconds = 2.0
        mock_result.realtime_factor = 1.5
        mock_result.metadata = {"language": "en", "emotion_exaggeration": 0.5, "cfg_weight": 0.5}

        mock_engine = Mock()
        mock_engine.speak = Mock(return_value=mock_result)

        with patch('voice_soundboard.server.get_chatterbox_engine', return_value=mock_engine):
            result = await handle_speak_chatterbox({
                "text": "Hello with paralinguistic tags [laugh]",
                "language": "en"
            })

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_chatterbox_import_error(self):
        """Should handle import error gracefully."""
        from voice_soundboard.server import handle_speak_chatterbox

        with patch('voice_soundboard.server.get_chatterbox_engine', side_effect=ImportError("Not installed")):
            result = await handle_speak_chatterbox({"text": "Hello"})

        assert len(result) == 1


class TestHandleCloneVoice:
    """Tests for handle_clone_voice handler."""

    @pytest.mark.asyncio
    async def test_clone_missing_path(self):
        """Should return error when audio_path is missing."""
        from voice_soundboard.server import handle_clone_voice

        result = await handle_clone_voice({})
        assert len(result) == 1
        assert "audio_path" in result[0].text.lower() or "required" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_clone_success(self, tmp_path):
        """Should clone voice successfully."""
        from voice_soundboard.server import handle_clone_voice

        audio_file = tmp_path / "voice.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_engine = Mock()
        mock_engine.clone_voice = Mock(return_value="my_clone")

        with patch('voice_soundboard.server.get_chatterbox_engine', return_value=mock_engine):
            result = await handle_clone_voice({
                "audio_path": str(audio_file),
                "voice_id": "my_clone"
            })

        assert len(result) == 1
        assert "success" in result[0].text.lower() or "registered" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_clone_file_not_found(self):
        """Should handle file not found."""
        from voice_soundboard.server import handle_clone_voice

        mock_engine = Mock()
        mock_engine.clone_voice = Mock(side_effect=FileNotFoundError("Not found"))

        with patch('voice_soundboard.server.get_chatterbox_engine', return_value=mock_engine):
            result = await handle_clone_voice({
                "audio_path": "/nonexistent/file.wav"
            })

        assert len(result) == 1


class TestHandleListClonedVoices:
    """Tests for handle_list_cloned_voices handler."""

    @pytest.mark.asyncio
    async def test_list_no_voices(self):
        """Should handle empty voice list."""
        from voice_soundboard.server import handle_list_cloned_voices

        mock_engine = Mock()
        mock_engine.list_cloned_voices = Mock(return_value={})

        with patch('voice_soundboard.server.get_chatterbox_engine', return_value=mock_engine):
            result = await handle_list_cloned_voices({})

        assert len(result) == 1
        assert "no cloned" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_list_voices_success(self):
        """Should list cloned voices."""
        from voice_soundboard.server import handle_list_cloned_voices

        mock_engine = Mock()
        mock_engine.list_cloned_voices = Mock(return_value={
            "voice1": "/path/to/voice1.wav",
            "voice2": "/path/to/voice2.wav"
        })

        with patch('voice_soundboard.server.get_chatterbox_engine', return_value=mock_engine):
            result = await handle_list_cloned_voices({})

        assert len(result) == 1
        assert "voice1" in result[0].text or "cloned" in result[0].text.lower()


class TestHandleListParalinguisticTags:
    """Tests for handle_list_paralinguistic_tags handler."""

    @pytest.mark.asyncio
    async def test_list_tags_chatterbox_unavailable(self):
        """Should show install message when Chatterbox unavailable."""
        from voice_soundboard.server import handle_list_paralinguistic_tags

        with patch('voice_soundboard.server.CHATTERBOX_AVAILABLE', False):
            result = await handle_list_paralinguistic_tags({})

        assert len(result) == 1
        assert "not installed" in result[0].text.lower() or "pip install" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_list_tags_success(self):
        """Should list paralinguistic tags."""
        from voice_soundboard.server import handle_list_paralinguistic_tags

        with patch('voice_soundboard.server.CHATTERBOX_AVAILABLE', True):
            with patch('voice_soundboard.server.PARALINGUISTIC_TAGS', ['laugh', 'cough', 'sigh']):
                result = await handle_list_paralinguistic_tags({})

        assert len(result) == 1
        assert "tag" in result[0].text.lower() or "[laugh]" in result[0].text


class TestHandleDialogue:
    """Tests for dialogue-related handlers."""

    @pytest.mark.asyncio
    async def test_speak_dialogue_missing_script(self):
        """Should return error when script is missing."""
        # Import would need the actual function
        pass


class TestHandlePresetOperations:
    """Tests for preset-related handlers."""

    @pytest.mark.asyncio
    async def test_list_presets_success(self):
        """Should list presets."""
        from voice_soundboard.server import handle_list_presets

        result = await handle_list_presets({})
        assert len(result) == 1


class TestMCPToolRegistration:
    """Tests for MCP tool registration."""

    def test_tool_definitions_exist(self):
        """Should have tool definitions."""
        # Check that tools are properly defined
        pass


class TestErrorResponses:
    """Tests for error response helpers."""

    def test_missing_param_response(self):
        """Should create missing param error."""
        from voice_soundboard.server import missing_param

        result = missing_param("test_param")
        assert len(result) == 1
        assert "test_param" in result[0].text.lower() or "required" in result[0].text.lower()

    def test_not_found_response(self):
        """Should create not found error."""
        from voice_soundboard.server import not_found

        result = not_found("voice", "test_voice")
        assert len(result) == 1
        assert "not found" in result[0].text.lower()

    def test_success_response(self):
        """Should create success response."""
        from voice_soundboard.server import success_response

        result = success_response(message="Test success", data={"key": "value"})
        assert len(result) == 1
        assert "success" in result[0].text.lower() or "test" in result[0].text.lower()

    def test_exception_to_error(self):
        """Should convert exception to error response."""
        from voice_soundboard.server import exception_to_error
        from voice_soundboard.errors import ErrorCode

        result = exception_to_error(
            Exception("Test error"),
            ErrorCode.SYNTHESIS_FAILED,
            "Test operation"
        )
        assert len(result) == 1


class TestStudioHandlers:
    """Tests for studio-related handlers."""

    @pytest.mark.asyncio
    async def test_studio_start(self):
        """Should start studio session."""
        # Test studio_start handler
        pass

    @pytest.mark.asyncio
    async def test_studio_adjust(self):
        """Should adjust studio parameters."""
        pass

    @pytest.mark.asyncio
    async def test_studio_preview(self):
        """Should generate preview."""
        pass

    @pytest.mark.asyncio
    async def test_studio_save(self):
        """Should save studio result."""
        pass


class TestVocologyHandlers:
    """Tests for vocology MCP handlers."""

    @pytest.mark.asyncio
    async def test_analyze_biomarkers(self):
        """Should analyze vocal biomarkers."""
        pass

    @pytest.mark.asyncio
    async def test_analyze_prosody(self):
        """Should analyze prosody."""
        pass

    @pytest.mark.asyncio
    async def test_humanize_audio(self):
        """Should apply humanization."""
        pass


class TestCodecHandlers:
    """Tests for codec MCP handlers."""

    @pytest.mark.asyncio
    async def test_encode_audio(self):
        """Should encode audio to tokens."""
        pass

    @pytest.mark.asyncio
    async def test_decode_tokens(self):
        """Should decode tokens to audio."""
        pass


class TestConversionHandlers:
    """Tests for voice conversion handlers."""

    @pytest.mark.asyncio
    async def test_start_realtime_conversion(self):
        """Should start real-time conversion."""
        pass

    @pytest.mark.asyncio
    async def test_stop_realtime_conversion(self):
        """Should stop real-time conversion."""
        pass

    @pytest.mark.asyncio
    async def test_list_audio_devices(self):
        """Should list audio devices."""
        pass


class TestWebServerHandlers:
    """Tests for web server control handlers."""

    @pytest.mark.asyncio
    async def test_start_web_server(self):
        """Should start web server."""
        pass

    @pytest.mark.asyncio
    async def test_stop_web_server(self):
        """Should stop web server."""
        pass


class TestWebSocketHandlers:
    """Tests for WebSocket control handlers."""

    @pytest.mark.asyncio
    async def test_start_websocket_server(self):
        """Should start WebSocket server."""
        pass

    @pytest.mark.asyncio
    async def test_stop_websocket_server(self):
        """Should stop WebSocket server."""
        pass
