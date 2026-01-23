"""
Tests for Phase 8: server.py MCP tools for F5-TTS and Chatterbox multilingual.

Tests cover:
- speak_f5tts MCP tool (TEST-SV-F5-01 to TEST-SV-F5-10)
- clone_voice_f5tts MCP tool (TEST-SV-F5-11 to TEST-SV-F5-17)
- list_chatterbox_languages MCP tool (TEST-SV-F5-18 to TEST-SV-F5-22)
- speak_chatterbox with language parameter (TEST-SV-F5-23 to TEST-SV-F5-26)
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import numpy as np


# =============================================================================
# TEST-SV-F5-01 to TEST-SV-F5-10: speak_f5tts MCP Tool Tests
# =============================================================================

class TestSpeakF5TTSMCP:
    """Tests for speak_f5tts MCP tool (TEST-SV-F5-01 to TEST-SV-F5-10)."""

    @pytest.mark.asyncio
    async def test_sv_f5_01_speak_f5tts_tool_exists(self):
        """TEST-SV-F5-01: speak_f5tts tool is registered in MCP server."""
        from voice_soundboard.server import list_tools

        # Get tools via the list_tools handler
        tools = await list_tools()
        tool_names = [t.name for t in tools]

        assert "speak_f5tts" in tool_names

    @pytest.mark.asyncio
    async def test_sv_f5_02_speak_f5tts_requires_text(self):
        """TEST-SV-F5-02: speak_f5tts returns error when text is empty."""
        from voice_soundboard.server import handle_speak_f5tts

        result = await handle_speak_f5tts({"text": ""})

        assert len(result) == 1
        assert "Error" in result[0].text
        assert "text" in result[0].text.lower()

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_f5tts_engine')
    async def test_sv_f5_03_speak_f5tts_calls_engine(self, mock_get_engine):
        """TEST-SV-F5-03: speak_f5tts calls F5TTSEngine.speak()."""
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/output.wav")
        mock_result.duration_seconds = 2.0
        mock_result.realtime_factor = 1.5
        mock_result.sample_rate = 24000
        mock_result.metadata = {'cfg_strength': 2.0, 'nfe_step': 32, 'has_reference': False}
        mock_engine.speak.return_value = mock_result
        mock_get_engine.return_value = mock_engine

        from voice_soundboard.server import handle_speak_f5tts

        result = await handle_speak_f5tts({"text": "Hello world"})

        mock_engine.speak.assert_called_once()
        assert "output.wav" in result[0].text

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_f5tts_engine')
    async def test_sv_f5_04_speak_f5tts_passes_voice(self, mock_get_engine):
        """TEST-SV-F5-04: speak_f5tts passes voice parameter."""
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/output.wav")
        mock_result.duration_seconds = 2.0
        mock_result.realtime_factor = 1.5
        mock_result.sample_rate = 24000
        mock_result.metadata = {'cfg_strength': 2.0, 'nfe_step': 32, 'has_reference': True}
        mock_engine.speak.return_value = mock_result
        mock_get_engine.return_value = mock_engine

        from voice_soundboard.server import handle_speak_f5tts

        await handle_speak_f5tts({"text": "Hello", "voice": "my_voice"})

        call_kwargs = mock_engine.speak.call_args[1]
        assert call_kwargs["voice"] == "my_voice"

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_f5tts_engine')
    async def test_sv_f5_05_speak_f5tts_passes_ref_text(self, mock_get_engine):
        """TEST-SV-F5-05: speak_f5tts passes ref_text parameter."""
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/output.wav")
        mock_result.duration_seconds = 2.0
        mock_result.realtime_factor = 1.5
        mock_result.sample_rate = 24000
        mock_result.metadata = {'cfg_strength': 2.0, 'nfe_step': 32, 'has_reference': True}
        mock_engine.speak.return_value = mock_result
        mock_get_engine.return_value = mock_engine

        from voice_soundboard.server import handle_speak_f5tts

        await handle_speak_f5tts({
            "text": "Hello",
            "voice": "ref.wav",
            "ref_text": "Reference transcription"
        })

        call_kwargs = mock_engine.speak.call_args[1]
        assert call_kwargs["ref_text"] == "Reference transcription"

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_f5tts_engine')
    async def test_sv_f5_06_speak_f5tts_passes_cfg_strength(self, mock_get_engine):
        """TEST-SV-F5-06: speak_f5tts passes cfg_strength parameter."""
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/output.wav")
        mock_result.duration_seconds = 2.0
        mock_result.realtime_factor = 1.5
        mock_result.sample_rate = 24000
        mock_result.metadata = {'cfg_strength': 3.0, 'nfe_step': 32, 'has_reference': False}
        mock_engine.speak.return_value = mock_result
        mock_get_engine.return_value = mock_engine

        from voice_soundboard.server import handle_speak_f5tts

        await handle_speak_f5tts({"text": "Hello", "cfg_strength": 3.0})

        call_kwargs = mock_engine.speak.call_args[1]
        assert call_kwargs["cfg_strength"] == 3.0

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_f5tts_engine')
    async def test_sv_f5_07_speak_f5tts_passes_speed(self, mock_get_engine):
        """TEST-SV-F5-07: speak_f5tts passes speed parameter."""
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/output.wav")
        mock_result.duration_seconds = 2.0
        mock_result.realtime_factor = 1.5
        mock_result.sample_rate = 24000
        mock_result.metadata = {'cfg_strength': 2.0, 'nfe_step': 32, 'has_reference': False}
        mock_engine.speak.return_value = mock_result
        mock_get_engine.return_value = mock_engine

        from voice_soundboard.server import handle_speak_f5tts

        await handle_speak_f5tts({"text": "Hello", "speed": 1.5})

        call_kwargs = mock_engine.speak.call_args[1]
        assert call_kwargs["speed"] == 1.5

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_f5tts_engine')
    async def test_sv_f5_08_speak_f5tts_passes_nfe_step(self, mock_get_engine):
        """TEST-SV-F5-08: speak_f5tts passes nfe_step parameter."""
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/output.wav")
        mock_result.duration_seconds = 2.0
        mock_result.realtime_factor = 1.5
        mock_result.sample_rate = 24000
        mock_result.metadata = {'cfg_strength': 2.0, 'nfe_step': 64, 'has_reference': False}
        mock_engine.speak.return_value = mock_result
        mock_get_engine.return_value = mock_engine

        from voice_soundboard.server import handle_speak_f5tts

        await handle_speak_f5tts({"text": "Hello", "nfe_step": 64})

        call_kwargs = mock_engine.speak.call_args[1]
        assert call_kwargs["nfe_step"] == 64

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_f5tts_engine')
    async def test_sv_f5_09_speak_f5tts_returns_audio_path(self, mock_get_engine):
        """TEST-SV-F5-09: speak_f5tts returns audio_path in response."""
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/f5tts_output.wav")
        mock_result.duration_seconds = 2.5
        mock_result.realtime_factor = 1.5
        mock_result.sample_rate = 24000
        mock_result.metadata = {'cfg_strength': 2.0, 'nfe_step': 32, 'has_reference': False}
        mock_engine.speak.return_value = mock_result
        mock_get_engine.return_value = mock_engine

        from voice_soundboard.server import handle_speak_f5tts

        result = await handle_speak_f5tts({"text": "Hello"})

        assert "f5tts_output.wav" in result[0].text

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_f5tts_engine')
    async def test_sv_f5_10_speak_f5tts_handles_import_error(self, mock_get_engine):
        """TEST-SV-F5-10: speak_f5tts handles ImportError gracefully."""
        mock_get_engine.side_effect = ImportError("F5-TTS is not installed")

        from voice_soundboard.server import handle_speak_f5tts

        result = await handle_speak_f5tts({"text": "Hello"})

        # Should return install instructions
        text = result[0].text
        assert "F5-TTS" in text or "not installed" in text or "pip install" in text


# =============================================================================
# TEST-SV-F5-11 to TEST-SV-F5-17: clone_voice_f5tts MCP Tool Tests
# =============================================================================

class TestCloneVoiceF5TTSMCP:
    """Tests for clone_voice_f5tts MCP tool (TEST-SV-F5-11 to TEST-SV-F5-17)."""

    @pytest.mark.asyncio
    async def test_sv_f5_11_clone_voice_f5tts_tool_exists(self):
        """TEST-SV-F5-11: clone_voice_f5tts tool is registered in MCP server."""
        from voice_soundboard.server import list_tools

        tools = await list_tools()
        tool_names = [t.name for t in tools]

        assert "clone_voice_f5tts" in tool_names

    @pytest.mark.asyncio
    async def test_sv_f5_12_clone_voice_f5tts_requires_audio_path(self):
        """TEST-SV-F5-12: clone_voice_f5tts returns error when audio_path is missing."""
        from voice_soundboard.server import handle_clone_voice_f5tts

        result = await handle_clone_voice_f5tts({})

        assert len(result) == 1
        assert "Error" in result[0].text
        assert "audio_path" in result[0].text.lower()

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_f5tts_engine')
    async def test_sv_f5_13_clone_voice_f5tts_calls_clone_voice(self, mock_get_engine, tmp_path):
        """TEST-SV-F5-13: clone_voice_f5tts calls engine.clone_voice()."""
        mock_engine = Mock()
        mock_engine.clone_voice.return_value = "test_voice_id"
        mock_get_engine.return_value = mock_engine

        audio_file = tmp_path / "ref.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        from voice_soundboard.server import handle_clone_voice_f5tts

        result = await handle_clone_voice_f5tts({
            "audio_path": str(audio_file),
            "voice_id": "test_voice"
        })

        mock_engine.clone_voice.assert_called_once()
        assert "Voice registered" in result[0].text

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_f5tts_engine')
    async def test_sv_f5_14_clone_voice_f5tts_passes_transcription(self, mock_get_engine, tmp_path):
        """TEST-SV-F5-14: clone_voice_f5tts passes transcription parameter."""
        mock_engine = Mock()
        mock_engine.clone_voice.return_value = "test_voice_id"
        mock_get_engine.return_value = mock_engine

        audio_file = tmp_path / "ref.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        from voice_soundboard.server import handle_clone_voice_f5tts

        await handle_clone_voice_f5tts({
            "audio_path": str(audio_file),
            "voice_id": "test_voice",
            "transcription": "This is the reference text"
        })

        call_kwargs = mock_engine.clone_voice.call_args[1]
        assert call_kwargs["transcription"] == "This is the reference text"

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_f5tts_engine')
    async def test_sv_f5_15_clone_voice_f5tts_generates_voice_id(self, mock_get_engine, tmp_path):
        """TEST-SV-F5-15: clone_voice_f5tts generates voice_id when not provided."""
        mock_engine = Mock()
        mock_engine.clone_voice.return_value = "cloned"  # Default voice_id
        mock_get_engine.return_value = mock_engine

        audio_file = tmp_path / "my_reference.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        from voice_soundboard.server import handle_clone_voice_f5tts

        result = await handle_clone_voice_f5tts({
            "audio_path": str(audio_file)
            # voice_id not provided
        })

        # Should have been called with a generated voice ID
        mock_engine.clone_voice.assert_called_once()
        assert "Voice registered" in result[0].text

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_f5tts_engine')
    async def test_sv_f5_16_clone_voice_f5tts_notes_no_transcription(self, mock_get_engine, tmp_path):
        """TEST-SV-F5-16: clone_voice_f5tts includes note when transcription is missing."""
        mock_engine = Mock()
        mock_engine.clone_voice.return_value = "test_voice_id"
        mock_get_engine.return_value = mock_engine

        audio_file = tmp_path / "ref.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        from voice_soundboard.server import handle_clone_voice_f5tts

        result = await handle_clone_voice_f5tts({
            "audio_path": str(audio_file),
            "voice_id": "test_voice"
            # No transcription
        })

        # Should include a note about providing transcription for better results
        text = result[0].text
        assert "not provided" in text.lower() or "no transcription" in text.lower()

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_f5tts_engine')
    async def test_sv_f5_17_clone_voice_f5tts_handles_import_error(self, mock_get_engine):
        """TEST-SV-F5-17: clone_voice_f5tts handles ImportError gracefully."""
        mock_get_engine.side_effect = ImportError("F5-TTS is not installed")

        from voice_soundboard.server import handle_clone_voice_f5tts

        result = await handle_clone_voice_f5tts({
            "audio_path": "/some/path.wav"
        })

        # Should return install instructions
        text = result[0].text
        assert "F5-TTS" in text or "not installed" in text or "pip install" in text


# =============================================================================
# TEST-SV-F5-18 to TEST-SV-F5-22: list_chatterbox_languages MCP Tool Tests
# =============================================================================

class TestListChatterboxLanguagesMCP:
    """Tests for list_chatterbox_languages MCP tool (TEST-SV-F5-18 to TEST-SV-F5-22)."""

    @pytest.mark.asyncio
    async def test_sv_f5_18_list_chatterbox_languages_tool_exists(self):
        """TEST-SV-F5-18: list_chatterbox_languages tool is registered."""
        from voice_soundboard.server import list_tools

        tools = await list_tools()
        tool_names = [t.name for t in tools]

        assert "list_chatterbox_languages" in tool_names

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.CHATTERBOX_AVAILABLE', True)
    async def test_sv_f5_19_list_chatterbox_languages_returns_23(self):
        """TEST-SV-F5-19: list_chatterbox_languages returns 23 languages."""
        from voice_soundboard.server import handle_list_chatterbox_languages

        result = await handle_list_chatterbox_languages({})

        text = result[0].text
        # Should list all 23 languages
        assert "23" in text

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.CHATTERBOX_AVAILABLE', True)
    async def test_sv_f5_20_list_chatterbox_languages_includes_codes(self):
        """TEST-SV-F5-20: list_chatterbox_languages includes language codes."""
        from voice_soundboard.server import handle_list_chatterbox_languages

        result = await handle_list_chatterbox_languages({})

        text = result[0].text
        # Should include some language codes
        assert "en" in text
        assert "fr" in text or "French" in text

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.CHATTERBOX_AVAILABLE', False)
    async def test_sv_f5_21_list_chatterbox_languages_not_installed(self):
        """TEST-SV-F5-21: list_chatterbox_languages returns error when not installed."""
        from voice_soundboard.server import handle_list_chatterbox_languages

        result = await handle_list_chatterbox_languages({})

        text = result[0].text
        assert "not installed" in text.lower() or "Chatterbox" in text

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.CHATTERBOX_AVAILABLE', True)
    async def test_sv_f5_22_list_chatterbox_languages_includes_usage(self):
        """TEST-SV-F5-22: list_chatterbox_languages includes usage example."""
        from voice_soundboard.server import handle_list_chatterbox_languages

        result = await handle_list_chatterbox_languages({})

        text = result[0].text
        # Should include how to use language parameter
        assert "speak_chatterbox" in text or "language" in text.lower()


# =============================================================================
# TEST-SV-F5-23 to TEST-SV-F5-26: speak_chatterbox with Language Parameter Tests
# =============================================================================

class TestSpeakChatterboxLanguageMCP:
    """Tests for speak_chatterbox with language parameter (TEST-SV-F5-23 to TEST-SV-F5-26)."""

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_chatterbox_engine')
    async def test_sv_f5_23_speak_chatterbox_accepts_language(self, mock_get_engine, tmp_path):
        """TEST-SV-F5-23: speak_chatterbox accepts language parameter."""
        mock_engine = Mock()
        mock_engine._is_multilingual = True
        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/output.wav")
        mock_result.duration_seconds = 2.0
        mock_result.realtime_factor = 1.5
        mock_result.sample_rate = 24000
        mock_result.metadata = {"language": "fr", "emotion_exaggeration": 0.5, "cfg_weight": 0.5}
        mock_engine.speak.return_value = mock_result
        mock_get_engine.return_value = mock_engine

        from voice_soundboard.server import handle_speak_chatterbox

        result = await handle_speak_chatterbox({
            "text": "Bonjour",
            "language": "fr"
        })

        mock_engine.speak.assert_called_once()
        call_kwargs = mock_engine.speak.call_args[1]
        assert call_kwargs.get("language") == "fr"

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_chatterbox_engine')
    async def test_sv_f5_24_speak_chatterbox_language_in_response(self, mock_get_engine):
        """TEST-SV-F5-24: speak_chatterbox includes language in response."""
        mock_engine = Mock()
        mock_engine._is_multilingual = True
        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/output.wav")
        mock_result.duration_seconds = 2.0
        mock_result.realtime_factor = 1.5
        mock_result.sample_rate = 24000
        mock_result.metadata = {"language": "ja", "emotion_exaggeration": 0.5, "cfg_weight": 0.5}
        mock_engine.speak.return_value = mock_result
        mock_get_engine.return_value = mock_engine

        from voice_soundboard.server import handle_speak_chatterbox

        result = await handle_speak_chatterbox({
            "text": "こんにちは",
            "language": "ja"
        })

        # Should mention language in response
        text = result[0].text
        assert "ja" in text or "Japanese" in text or "Language" in text

    @pytest.mark.asyncio
    @patch('voice_soundboard.server.get_chatterbox_engine')
    async def test_sv_f5_25_speak_chatterbox_default_language_en(self, mock_get_engine):
        """TEST-SV-F5-25: speak_chatterbox defaults to English when language not specified."""
        mock_engine = Mock()
        mock_engine._is_multilingual = True
        mock_engine.default_language = "en"
        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/output.wav")
        mock_result.duration_seconds = 2.0
        mock_result.realtime_factor = 1.5
        mock_result.sample_rate = 24000
        mock_result.metadata = {"language": "en", "emotion_exaggeration": 0.5, "cfg_weight": 0.5}
        mock_engine.speak.return_value = mock_result
        mock_get_engine.return_value = mock_engine

        from voice_soundboard.server import handle_speak_chatterbox

        await handle_speak_chatterbox({
            "text": "Hello"
            # No language specified
        })

        mock_engine.speak.assert_called_once()
        # Should default to 'en'
        call_kwargs = mock_engine.speak.call_args[1]
        assert call_kwargs.get("language") == "en"

    @pytest.mark.asyncio
    async def test_sv_f5_26_speak_chatterbox_tool_schema_has_language(self):
        """TEST-SV-F5-26: speak_chatterbox tool schema includes language property."""
        from voice_soundboard.server import list_tools

        tools = await list_tools()
        speak_tool = next(t for t in tools if t.name == "speak_chatterbox")

        properties = speak_tool.inputSchema.get("properties", {})

        assert "language" in properties
        assert properties["language"].get("type") == "string"
