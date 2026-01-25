"""
Tests for MCP Server tools (server.py).

Tests cover:
- Tool handler functions
- Tool listing and schemas
- Error handling
- Response formatting (structured JSON responses)
"""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from mcp.types import TextContent

from voice_soundboard.config import KOKORO_VOICES, VOICE_PRESETS


def parse_response(result):
    """Parse MCP TextContent result to JSON dict."""
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    return json.loads(result[0].text)


class TestServerEngineSingletons:
    """Tests for engine singleton getters."""

    def test_get_engine_creates_singleton(self):
        """Test that get_engine returns a singleton."""
        from voice_soundboard import server

        # Reset global
        server._engine = None

        with patch('voice_soundboard.server.VoiceEngine') as mock_engine:
            mock_engine.return_value = Mock()

            engine1 = server.get_engine()
            engine2 = server.get_engine()

            assert engine1 is engine2
            mock_engine.assert_called_once()

    def test_get_chatterbox_raises_when_unavailable(self):
        """Test that get_chatterbox_engine raises ImportError when unavailable."""
        from voice_soundboard import server

        server._chatterbox_engine = None

        with patch.object(server, 'CHATTERBOX_AVAILABLE', False):
            with pytest.raises(ImportError) as exc_info:
                server.get_chatterbox_engine()

            assert "Chatterbox is not installed" in str(exc_info.value)


class TestHandleListVoices:
    """Tests for handle_list_voices handler."""

    @pytest.mark.asyncio
    async def test_list_voices_returns_all(self):
        """TEST-T17: list_voices returns voice list."""
        from voice_soundboard.server import handle_list_voices

        result = await handle_list_voices({})
        data = parse_response(result)

        assert data["ok"] is True
        assert "voices" in data["data"]
        assert data["data"]["count"] == 50  # 50 voices

    @pytest.mark.asyncio
    async def test_list_voices_filter_gender(self):
        """TEST-T18: list_voices with filter_gender filters correctly."""
        from voice_soundboard.server import handle_list_voices

        result = await handle_list_voices({"filter_gender": "female"})
        data = parse_response(result)

        assert data["ok"] is True
        # All results should be female
        for voice in data["data"]["voices"]:
            assert voice["gender"] == "female"

    @pytest.mark.asyncio
    async def test_list_voices_no_matches(self):
        """Test list_voices with filters that match nothing."""
        from voice_soundboard.server import handle_list_voices

        result = await handle_list_voices({
            "filter_gender": "female",
            "filter_accent": "nonexistent"
        })
        data = parse_response(result)

        assert data["ok"] is True
        assert data["data"]["count"] == 0
        assert data["data"]["voices"] == []


class TestHandleListPresets:
    """Tests for handle_list_presets handler."""

    @pytest.mark.asyncio
    async def test_list_presets_returns_all(self):
        """TEST-T19: list_presets returns preset list."""
        from voice_soundboard.server import handle_list_presets

        result = await handle_list_presets({})
        data = parse_response(result)

        assert data["ok"] is True
        assert "presets" in data["data"]
        # Check for expected presets
        preset_ids = [p["id"] for p in data["data"]["presets"]]
        assert "narrator" in preset_ids
        assert "assistant" in preset_ids


class TestHandleListEffects:
    """Tests for handle_list_effects handler."""

    @pytest.mark.asyncio
    async def test_list_effects_returns_all(self):
        """TEST-T20: list_effects returns effect list."""
        from voice_soundboard.server import handle_list_effects

        result = await handle_list_effects({})
        data = parse_response(result)

        assert data["ok"] is True
        assert "effects" in data["data"]
        # Check for expected effect
        effect_ids = [e["id"] for e in data["data"]["effects"]]
        assert "chime" in effect_ids


class TestHandleListEmotions:
    """Tests for handle_list_emotions handler."""

    @pytest.mark.asyncio
    async def test_list_emotions_returns_all(self):
        """TEST-T21: list_emotions returns emotion list."""
        from voice_soundboard.server import handle_list_emotions

        result = await handle_list_emotions({})

        assert len(result) == 1
        assert "emotions" in result[0].text.lower()
        assert "happy" in result[0].text


class TestHandleSpeak:
    """Tests for handle_speak handler."""

    @pytest.mark.asyncio
    async def test_speak_missing_text(self):
        """TEST-T26: Missing required parameter returns error."""
        from voice_soundboard.server import handle_speak

        result = await handle_speak({})
        data = parse_response(result)

        assert data["ok"] is False
        assert data["error"]["code"] == "missing_required"
        assert "'text'" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_speak_success(self):
        """TEST-T01: speak tool generates audio file."""
        from voice_soundboard import server

        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/test.wav")
        mock_result.voice_used = "af_bella"
        mock_result.duration_seconds = 1.5
        mock_result.realtime_factor = 5.0

        mock_engine = Mock()
        mock_engine.speak = Mock(return_value=mock_result)

        with patch.object(server, 'get_engine', return_value=mock_engine):
            result = await server.handle_speak({"text": "Hello world"})
            data = parse_response(result)

        assert data["ok"] is True
        assert data["data"]["voice"] == "af_bella"
        assert data["data"]["duration_seconds"] == 1.5

    @pytest.mark.asyncio
    async def test_speak_with_style(self):
        """TEST-T03: speak tool with style parameter."""
        from voice_soundboard import server

        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/test.wav")
        mock_result.voice_used = "af_bella"
        mock_result.duration_seconds = 1.5
        mock_result.realtime_factor = 5.0

        mock_engine = Mock()
        mock_engine.speak = Mock(return_value=mock_result)

        with patch.object(server, 'get_engine', return_value=mock_engine), \
             patch.object(server, 'apply_style_to_params', return_value=("af_bella", 0.9, None)):
            result = await server.handle_speak({
                "text": "Hello world",
                "style": "warmly"
            })
            data = parse_response(result)

        assert data["ok"] is True
        assert data["data"]["voice"] == "af_bella"


class TestHandleSoundEffect:
    """Tests for handle_sound_effect handler."""

    @pytest.mark.asyncio
    async def test_sound_effect_missing_effect(self):
        """Test sound_effect with missing effect parameter."""
        from voice_soundboard.server import handle_sound_effect

        result = await handle_sound_effect({})
        data = parse_response(result)

        assert data["ok"] is False
        assert data["error"]["code"] == "missing_required"
        assert "'effect'" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_sound_effect_invalid_name(self):
        """TEST-T16: sound_effect with invalid effect name."""
        from voice_soundboard.server import handle_sound_effect

        result = await handle_sound_effect({"effect": "nonexistent_effect"})
        data = parse_response(result)

        # Should return error about unknown effect
        assert data["ok"] is False
        assert data["error"]["code"] == "effect_not_found"


class TestHandlePlayAudio:
    """Tests for handle_play_audio handler."""

    @pytest.mark.asyncio
    async def test_play_audio_missing_path(self):
        """Test play_audio with missing path parameter."""
        from voice_soundboard.server import handle_play_audio

        result = await handle_play_audio({})
        data = parse_response(result)

        assert data["ok"] is False
        assert data["error"]["code"] == "missing_required"
        assert "'path'" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_play_audio_file_not_found(self):
        """TEST-T23: play_audio with nonexistent file."""
        from voice_soundboard.server import handle_play_audio

        result = await handle_play_audio({"path": "/nonexistent/audio.wav"})
        data = parse_response(result)

        assert data["ok"] is False
        assert data["error"]["code"] == "file_not_found"


class TestHandleStopAudio:
    """Tests for handle_stop_audio handler."""

    @pytest.mark.asyncio
    async def test_stop_audio_success(self):
        """TEST-T24: stop_audio tool stops playback."""
        from voice_soundboard import server

        with patch.object(server, 'stop_playback'):
            result = await server.handle_stop_audio({})
            data = parse_response(result)

        assert data["ok"] is True
        assert "stopped" in data["message"].lower()


class TestCallTool:
    """Tests for call_tool dispatcher."""

    @pytest.mark.asyncio
    async def test_call_tool_unknown(self):
        """TEST-T25: Unknown tool name returns error."""
        from voice_soundboard.server import call_tool

        result = await call_tool("nonexistent_tool", {})
        data = parse_response(result)

        assert data["ok"] is False
        assert data["error"]["code"] == "unknown_tool"

    @pytest.mark.asyncio
    async def test_call_tool_routes_correctly(self):
        """Test that call_tool routes to correct handler."""
        from voice_soundboard import server

        # Test routing to list_presets
        result = await server.call_tool("list_presets", {})
        data = parse_response(result)

        assert data["ok"] is True
        assert "presets" in data["data"]


class TestHandleSpeakSSML:
    """Tests for handle_speak_ssml handler."""

    @pytest.mark.asyncio
    async def test_speak_ssml_missing_ssml(self):
        """Test speak_ssml with missing ssml parameter."""
        from voice_soundboard.server import handle_speak_ssml

        result = await handle_speak_ssml({})
        data = parse_response(result)

        assert data["ok"] is False
        assert data["error"]["code"] == "missing_required"
        assert "'ssml'" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_speak_ssml_success(self):
        """TEST-T07: speak_ssml tool processes SSML correctly."""
        from voice_soundboard import server

        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/test.wav")
        mock_result.voice_used = "af_bella"
        mock_result.duration_seconds = 1.5
        mock_result.realtime_factor = 5.0

        mock_engine = Mock()
        mock_engine.speak = Mock(return_value=mock_result)

        with patch.object(server, 'get_engine', return_value=mock_engine):
            result = await server.handle_speak_ssml({
                "ssml": "<speak>Hello world</speak>"
            })

        assert "SSML speech" in result[0].text


class TestHandleSpeakRealtime:
    """Tests for handle_speak_realtime handler."""

    @pytest.mark.asyncio
    async def test_speak_realtime_missing_text(self):
        """Test speak_realtime with missing text parameter."""
        from voice_soundboard.server import handle_speak_realtime

        result = await handle_speak_realtime({})

        assert "Error" in result[0].text
        assert "'text' is required" in result[0].text


class TestHandleSpeakLong:
    """Tests for handle_speak_long handler."""

    @pytest.mark.asyncio
    async def test_speak_long_missing_text(self):
        """Test speak_long with missing text parameter."""
        from voice_soundboard.server import handle_speak_long

        result = await handle_speak_long({})
        data = parse_response(result)

        assert data["ok"] is False
        assert data["error"]["code"] == "missing_required"
        assert "'text'" in data["error"]["message"]


class TestListTools:
    """Tests for list_tools function."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_all(self):
        """Test that list_tools returns all tools."""
        from voice_soundboard.server import list_tools

        tools = await list_tools()

        # Check expected tools are present
        tool_names = {t.name for t in tools}
        expected = {
            "speak", "list_voices", "list_presets", "play_audio",
            "stop_audio", "sound_effect", "list_effects", "speak_long",
            "speak_ssml", "speak_realtime", "list_emotions",
        }
        assert expected.issubset(tool_names)

    @pytest.mark.asyncio
    async def test_list_tools_schemas_valid(self):
        """Test that tool schemas are valid."""
        from voice_soundboard.server import list_tools

        tools = await list_tools()

        for tool in tools:
            assert tool.name is not None
            assert tool.description is not None
            assert tool.inputSchema is not None
            assert "type" in tool.inputSchema


class TestChatterboxHandlers:
    """Tests for Chatterbox tool handlers."""

    @pytest.mark.asyncio
    async def test_list_paralinguistic_tags_when_unavailable(self):
        """Test list_paralinguistic_tags when Chatterbox not installed."""
        from voice_soundboard import server

        with patch.object(server, 'CHATTERBOX_AVAILABLE', False):
            result = await server.handle_list_paralinguistic_tags({})

        assert "not installed" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_speak_chatterbox_missing_text(self):
        """Test speak_chatterbox with missing text."""
        from voice_soundboard.server import handle_speak_chatterbox

        result = await handle_speak_chatterbox({})

        assert "Error" in result[0].text
        assert "'text' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_clone_voice_missing_path(self):
        """Test clone_voice with missing audio_path."""
        from voice_soundboard.server import handle_clone_voice

        result = await handle_clone_voice({})

        assert "Error" in result[0].text
        assert "'audio_path' is required" in result[0].text


class TestDialogueHandlers:
    """Tests for Dialogue tool handlers."""

    @pytest.mark.asyncio
    async def test_speak_dialogue_missing_script(self):
        """Test speak_dialogue with missing script."""
        from voice_soundboard.server import handle_speak_dialogue

        result = await handle_speak_dialogue({})

        assert "Error" in result[0].text
        assert "'script' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_preview_dialogue_missing_script(self):
        """Test preview_dialogue with missing script."""
        from voice_soundboard.server import handle_preview_dialogue

        result = await handle_preview_dialogue({})

        assert "Error" in result[0].text
        assert "'script' is required" in result[0].text


class TestErrorHandling:
    """Tests for error handling in handlers."""

    @pytest.mark.asyncio
    async def test_speak_handles_engine_error(self):
        """Test that speak handles engine errors gracefully."""
        from voice_soundboard import server

        mock_engine = Mock()
        mock_engine.speak = Mock(side_effect=Exception("Model not found"))

        with patch.object(server, 'get_engine', return_value=mock_engine):
            result = await server.handle_speak({"text": "Hello"})
            data = parse_response(result)

        assert data["ok"] is False
        assert data["error"]["code"] in ["synthesis_failed", "internal_error"]

    @pytest.mark.asyncio
    async def test_sound_effect_handles_play_error(self):
        """Test that sound_effect handles play errors."""
        from voice_soundboard import server

        mock_effect = Mock()
        mock_effect.play = Mock(side_effect=Exception("Audio device error"))
        mock_effect.duration = 0.5

        with patch.object(server, 'get_effect', return_value=mock_effect):
            result = await server.handle_sound_effect({"effect": "chime"})
            data = parse_response(result)

        assert data["ok"] is False
        assert "error" in data


class TestEmotionToolHandlers:
    """Tests for advanced emotion control tool handlers."""

    @pytest.mark.asyncio
    async def test_blend_emotions_missing_emotions(self):
        """Test blend_emotions with missing emotions parameter."""
        from voice_soundboard.server import handle_blend_emotions

        result = await handle_blend_emotions({})

        assert "Error" in result[0].text
        assert "'emotions'" in result[0].text

    @pytest.mark.asyncio
    async def test_blend_emotions_success(self):
        """Test blend_emotions with valid input."""
        from voice_soundboard.server import handle_blend_emotions

        result = await handle_blend_emotions({
            "emotions": [
                {"emotion": "happy", "weight": 0.7},
                {"emotion": "surprised", "weight": 0.3}
            ]
        })

        assert "blend result" in result[0].text.lower()
        assert "Dominant" in result[0].text
        assert "VAD" in result[0].text

    @pytest.mark.asyncio
    async def test_parse_emotion_text_missing_text(self):
        """Test parse_emotion_text with missing text parameter."""
        from voice_soundboard.server import handle_parse_emotion_text

        result = await handle_parse_emotion_text({})

        assert "Error" in result[0].text
        assert "'text' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_parse_emotion_text_no_tags(self):
        """Test parse_emotion_text with plain text (no tags)."""
        from voice_soundboard.server import handle_parse_emotion_text

        result = await handle_parse_emotion_text({"text": "Hello world"})

        assert "No emotion tags found" in result[0].text

    @pytest.mark.asyncio
    async def test_parse_emotion_text_with_tags(self):
        """Test parse_emotion_text with emotion tags."""
        from voice_soundboard.server import handle_parse_emotion_text

        result = await handle_parse_emotion_text({
            "text": "I'm {happy}so glad{/happy} to see you!"
        })

        assert "Parsed emotion text" in result[0].text
        assert "happy" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_get_emotion_vad_missing_emotion(self):
        """Test get_emotion_vad with missing emotion parameter."""
        from voice_soundboard.server import handle_get_emotion_vad

        result = await handle_get_emotion_vad({})

        assert "Error" in result[0].text
        assert "'emotion' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_get_emotion_vad_success(self):
        """Test get_emotion_vad with valid emotion."""
        from voice_soundboard.server import handle_get_emotion_vad

        result = await handle_get_emotion_vad({"emotion": "happy"})

        assert "VAD values" in result[0].text
        assert "happy" in result[0].text.lower()
        assert "Valence" in result[0].text
        assert "Arousal" in result[0].text
        assert "Dominance" in result[0].text

    @pytest.mark.asyncio
    async def test_get_emotion_vad_unknown_emotion(self):
        """Test get_emotion_vad with unknown emotion."""
        from voice_soundboard.server import handle_get_emotion_vad

        result = await handle_get_emotion_vad({"emotion": "nonexistent_xyz"})

        # Should return error message
        assert "Unknown emotion" in result[0].text or "Error" in result[0].text

    @pytest.mark.asyncio
    async def test_list_emotion_blends(self):
        """Test list_emotion_blends returns blends list."""
        from voice_soundboard.server import handle_list_emotion_blends

        result = await handle_list_emotion_blends({})

        assert "emotion blends" in result[0].text.lower()
        assert "bittersweet" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_list_narrative_curves(self):
        """Test list_narrative_curves returns curves list."""
        from voice_soundboard.server import handle_list_narrative_curves

        result = await handle_list_narrative_curves({})

        assert "narrative" in result[0].text.lower() or "curve" in result[0].text.lower()
        assert "tension_build" in result[0].text

    @pytest.mark.asyncio
    async def test_sample_emotion_curve_with_name(self):
        """Test sample_emotion_curve with named curve."""
        from voice_soundboard.server import handle_sample_emotion_curve

        result = await handle_sample_emotion_curve({
            "curve_name": "tension_build",
            "num_samples": 5
        })

        assert "samples" in result[0].text.lower() or "keyframes" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_sample_emotion_curve_unknown_name(self):
        """Test sample_emotion_curve with unknown curve name."""
        from voice_soundboard.server import handle_sample_emotion_curve

        result = await handle_sample_emotion_curve({
            "curve_name": "nonexistent_curve"
        })

        assert "Unknown curve" in result[0].text or "unknown" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_sample_emotion_curve_with_keyframes(self):
        """Test sample_emotion_curve with custom keyframes."""
        from voice_soundboard.server import handle_sample_emotion_curve

        result = await handle_sample_emotion_curve({
            "keyframes": [
                {"position": 0.0, "emotion": "calm"},
                {"position": 0.5, "emotion": "excited"},
                {"position": 1.0, "emotion": "happy"}
            ],
            "num_samples": 3
        })

        assert "samples" in result[0].text.lower() or "keyframes" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_sample_emotion_curve_no_params(self):
        """Test sample_emotion_curve with no curve_name or keyframes."""
        from voice_soundboard.server import handle_sample_emotion_curve

        result = await handle_sample_emotion_curve({})

        assert "Error" in result[0].text or "Provide either" in result[0].text


class TestCallToolEmotionRouting:
    """Tests for call_tool routing to emotion handlers."""

    @pytest.mark.asyncio
    async def test_call_tool_blend_emotions(self):
        """Test that call_tool routes to blend_emotions."""
        from voice_soundboard.server import call_tool

        result = await call_tool("blend_emotions", {
            "emotions": [{"emotion": "happy", "weight": 1.0}]
        })

        assert "blend" in result[0].text.lower() or "Dominant" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_parse_emotion_text(self):
        """Test that call_tool routes to parse_emotion_text."""
        from voice_soundboard.server import call_tool

        result = await call_tool("parse_emotion_text", {"text": "Hello"})

        assert "No emotion tags" in result[0].text or "Parsed" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_get_emotion_vad(self):
        """Test that call_tool routes to get_emotion_vad."""
        from voice_soundboard.server import call_tool

        result = await call_tool("get_emotion_vad", {"emotion": "happy"})

        assert "VAD" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_list_emotion_blends(self):
        """Test that call_tool routes to list_emotion_blends."""
        from voice_soundboard.server import call_tool

        result = await call_tool("list_emotion_blends", {})

        assert "blend" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_call_tool_list_narrative_curves(self):
        """Test that call_tool routes to list_narrative_curves."""
        from voice_soundboard.server import call_tool

        result = await call_tool("list_narrative_curves", {})

        assert "curve" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_call_tool_sample_emotion_curve(self):
        """Test that call_tool routes to sample_emotion_curve."""
        from voice_soundboard.server import call_tool

        result = await call_tool("sample_emotion_curve", {
            "curve_name": "joy_arc"
        })

        assert "sample" in result[0].text.lower() or "keyframe" in result[0].text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
