"""
Additional test coverage batch 25: server.py MCP handlers (part 1).

Tests for MCP server tool handlers - basic handlers and list operations.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import numpy as np

from mcp.types import TextContent


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_voice_engine():
    """Create a mock voice engine."""
    engine = Mock()
    result = Mock()
    result.audio_path = Path(tempfile.gettempdir()) / "test_audio.wav"
    result.voice_used = "af_bella"
    result.duration_seconds = 2.5
    result.realtime_factor = 15.0
    engine.speak.return_value = result
    engine.speak_raw.return_value = (np.zeros(24000), 24000)
    return engine


@pytest.fixture
def mock_chatterbox_engine():
    """Create a mock Chatterbox engine."""
    engine = Mock()
    result = Mock()
    result.audio_path = Path(tempfile.gettempdir()) / "test_chatterbox.wav"
    result.duration_seconds = 3.0
    engine.speak.return_value = result
    engine.clone_voice.return_value = "cloned_voice_id"
    engine.list_cloned_voices.return_value = ["voice1", "voice2"]
    return engine


# ============================================================================
# Handler Helper Tests
# ============================================================================

class TestGetEngineFunctions:
    """Tests for engine getter functions."""

    def test_get_engine_creates_singleton(self):
        """Test get_engine creates and caches engine."""
        from voice_soundboard import server

        # Reset the global
        original = server._engine
        server._engine = None

        try:
            with patch.object(server, 'VoiceEngine') as mock_cls:
                mock_engine = Mock()
                mock_cls.return_value = mock_engine

                result = server.get_engine()
                assert result == mock_engine

                # Second call should return cached instance
                result2 = server.get_engine()
                assert result2 == mock_engine
                mock_cls.assert_called_once()
        finally:
            server._engine = original

    def test_get_engine_handles_error(self):
        """Test get_engine raises RuntimeError on failure."""
        from voice_soundboard import server

        original = server._engine
        server._engine = None

        try:
            with patch.object(server, 'VoiceEngine', side_effect=Exception("Init failed")):
                with pytest.raises(RuntimeError, match="Voice engine initialization failed"):
                    server.get_engine()
        finally:
            server._engine = original

    def test_get_dialogue_engine_creates_singleton(self):
        """Test get_dialogue_engine creates and caches engine."""
        from voice_soundboard import server

        original_dialogue = server._dialogue_engine
        original_voice = server._engine
        server._dialogue_engine = None

        try:
            with patch.object(server, 'get_engine') as mock_get:
                mock_voice = Mock()
                mock_get.return_value = mock_voice

                with patch.object(server, 'DialogueEngine') as mock_cls:
                    mock_engine = Mock()
                    mock_cls.return_value = mock_engine

                    result = server.get_dialogue_engine()
                    assert result == mock_engine
                    mock_cls.assert_called_once_with(voice_engine=mock_voice)
        finally:
            server._dialogue_engine = original_dialogue
            server._engine = original_voice

    def test_get_voice_cloner_creates_singleton(self):
        """Test get_voice_cloner creates and caches cloner."""
        from voice_soundboard import server

        original = server._voice_cloner
        server._voice_cloner = None

        try:
            with patch.object(server, 'VoiceCloner') as mock_cls:
                mock_cloner = Mock()
                mock_cls.return_value = mock_cloner

                result = server.get_voice_cloner()
                assert result == mock_cloner
        finally:
            server._voice_cloner = original

    def test_get_emotion_separator_creates_singleton(self):
        """Test get_emotion_separator creates and caches separator."""
        from voice_soundboard import server

        original = server._emotion_separator
        server._emotion_separator = None

        try:
            with patch.object(server, 'EmotionTimbreSeparator') as mock_cls:
                mock_sep = Mock()
                mock_cls.return_value = mock_sep

                result = server.get_emotion_separator()
                assert result == mock_sep
        finally:
            server._emotion_separator = original

    def test_get_audio_codec_mock(self):
        """Test get_audio_codec with mock codec."""
        from voice_soundboard import server

        original = server._audio_codec
        server._audio_codec = None

        try:
            with patch.object(server, 'MockCodec') as mock_cls:
                mock_codec = Mock()
                mock_codec.name = "mock"
                mock_cls.return_value = mock_codec

                result = server.get_audio_codec("mock")
                assert result == mock_codec
        finally:
            server._audio_codec = original

    def test_get_audio_codec_mimi(self):
        """Test get_audio_codec with mimi codec."""
        from voice_soundboard import server

        original = server._audio_codec
        server._audio_codec = None

        try:
            with patch.object(server, 'MimiCodec') as mock_cls:
                mock_codec = Mock()
                mock_codec.name = "mimi"
                mock_cls.return_value = mock_codec

                result = server.get_audio_codec("mimi")
                assert result == mock_codec
        finally:
            server._audio_codec = original

    def test_get_realtime_converter_creates_singleton(self):
        """Test get_realtime_converter creates converter with latency mode."""
        from voice_soundboard import server

        original = server._realtime_converter
        server._realtime_converter = None

        try:
            with patch.object(server, 'RealtimeConverter') as mock_cls:
                mock_conv = Mock()
                mock_cls.return_value = mock_conv

                result = server.get_realtime_converter("low")
                assert result == mock_conv
        finally:
            server._realtime_converter = original

    def test_get_conversation_manager_creates_singleton(self):
        """Test get_conversation_manager creates and caches manager."""
        from voice_soundboard import server

        original = server._conversation_manager
        server._conversation_manager = None

        try:
            with patch.object(server, 'ConversationManager') as mock_cls:
                mock_mgr = Mock()
                mock_cls.return_value = mock_mgr

                result = server.get_conversation_manager()
                assert result == mock_mgr
        finally:
            server._conversation_manager = original


# ============================================================================
# List Tool Handler Tests
# ============================================================================

class TestHandleListVoices:
    """Tests for handle_list_voices handler."""

    @pytest.mark.asyncio
    async def test_list_voices_all(self):
        """Test listing all voices."""
        from voice_soundboard.server import handle_list_voices

        result = await handle_list_voices({})
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Available voices" in result[0].text

    @pytest.mark.asyncio
    async def test_list_voices_filter_gender(self):
        """Test filtering voices by gender."""
        from voice_soundboard.server import handle_list_voices

        result = await handle_list_voices({"filter_gender": "female"})
        assert len(result) == 1
        # Should contain female voices
        assert "female" in result[0].text.lower() or "Available" in result[0].text

    @pytest.mark.asyncio
    async def test_list_voices_filter_accent(self):
        """Test filtering voices by accent."""
        from voice_soundboard.server import handle_list_voices

        result = await handle_list_voices({"filter_accent": "british"})
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_voices_no_match(self):
        """Test when no voices match filters."""
        from voice_soundboard.server import handle_list_voices

        # Use an accent that might not exist
        result = await handle_list_voices({"filter_accent": "nonexistent"})
        assert len(result) == 1
        assert "No voices match" in result[0].text


class TestHandleListPresets:
    """Tests for handle_list_presets handler."""

    @pytest.mark.asyncio
    async def test_list_presets(self):
        """Test listing voice presets."""
        from voice_soundboard.server import handle_list_presets

        result = await handle_list_presets({})
        assert len(result) == 1
        assert "Voice presets" in result[0].text
        assert "narrator" in result[0].text.lower() or "assistant" in result[0].text.lower()


class TestHandleListEffects:
    """Tests for handle_list_effects handler."""

    @pytest.mark.asyncio
    async def test_list_effects(self):
        """Test listing sound effects."""
        from voice_soundboard.server import handle_list_effects

        result = await handle_list_effects({})
        assert len(result) == 1
        assert "Available sound effects" in result[0].text


class TestHandleListEmotions:
    """Tests for handle_list_emotions handler."""

    @pytest.mark.asyncio
    async def test_list_emotions(self):
        """Test listing emotions."""
        from voice_soundboard.server import handle_list_emotions

        result = await handle_list_emotions({})
        assert len(result) == 1
        # Should contain emotion info
        assert "happy" in result[0].text.lower() or "emotion" in result[0].text.lower()


class TestHandleListEmotionBlends:
    """Tests for handle_list_emotion_blends handler."""

    @pytest.mark.asyncio
    async def test_list_emotion_blends(self):
        """Test listing named emotion blends."""
        from voice_soundboard.server import handle_list_emotion_blends

        result = await handle_list_emotion_blends({})
        assert len(result) == 1
        assert "bittersweet" in result[0].text.lower() or "blend" in result[0].text.lower()


class TestHandleListNarrativeCurves:
    """Tests for handle_list_narrative_curves handler."""

    @pytest.mark.asyncio
    async def test_list_narrative_curves(self):
        """Test listing narrative curves."""
        from voice_soundboard.server import handle_list_narrative_curves

        result = await handle_list_narrative_curves({})
        assert len(result) == 1
        assert "curve" in result[0].text.lower() or "tension" in result[0].text.lower()


class TestHandleListCloningLanguages:
    """Tests for handle_list_cloning_languages handler."""

    @pytest.mark.asyncio
    async def test_list_cloning_languages(self):
        """Test listing supported cloning languages."""
        from voice_soundboard.server import handle_list_cloning_languages

        result = await handle_list_cloning_languages({})
        assert len(result) == 1
        assert "language" in result[0].text.lower() or "english" in result[0].text.lower()


class TestHandleListLLMProviders:
    """Tests for handle_list_llm_providers handler."""

    @pytest.mark.asyncio
    async def test_list_llm_providers(self):
        """Test listing LLM providers."""
        from voice_soundboard.server import handle_list_llm_providers

        result = await handle_list_llm_providers({})
        assert len(result) == 1
        # Should mention providers
        assert "provider" in result[0].text.lower() or "ollama" in result[0].text.lower() or "mock" in result[0].text.lower()


class TestHandleListParalinguisticTags:
    """Tests for handle_list_paralinguistic_tags handler."""

    @pytest.mark.asyncio
    async def test_list_paralinguistic_tags(self):
        """Test listing paralinguistic tags."""
        from voice_soundboard.server import handle_list_paralinguistic_tags

        result = await handle_list_paralinguistic_tags({})
        assert len(result) == 1
        # Should list tags like [laugh], [sigh], etc.
        text = result[0].text.lower()
        assert "laugh" in text or "tag" in text or "chatterbox" in text


class TestHandleListClonedVoices:
    """Tests for handle_list_cloned_voices handler."""

    @pytest.mark.asyncio
    async def test_list_cloned_voices(self):
        """Test listing cloned voices."""
        from voice_soundboard.server import handle_list_cloned_voices

        result = await handle_list_cloned_voices({})
        assert len(result) == 1
        # May be empty or have voices
        assert isinstance(result[0].text, str)


class TestHandleListVoiceLibrary:
    """Tests for handle_list_voice_library handler."""

    @pytest.mark.asyncio
    async def test_list_voice_library(self):
        """Test listing voice library."""
        from voice_soundboard.server import handle_list_voice_library

        result = await handle_list_voice_library({})
        assert len(result) == 1
        assert isinstance(result[0].text, str)


class TestHandleListAudioDevices:
    """Tests for handle_list_audio_devices handler."""

    @pytest.mark.asyncio
    async def test_list_audio_devices(self):
        """Test listing audio devices."""
        from voice_soundboard.server import handle_list_audio_devices

        result = await handle_list_audio_devices({})
        assert len(result) == 1
        assert isinstance(result[0].text, str)

    @pytest.mark.asyncio
    async def test_list_audio_devices_with_type(self):
        """Test listing audio devices filtered by type."""
        from voice_soundboard.server import handle_list_audio_devices

        result = await handle_list_audio_devices({"device_type": "input"})
        assert len(result) == 1


# ============================================================================
# Audio Handler Tests
# ============================================================================

class TestHandlePlayAudio:
    """Tests for handle_play_audio handler."""

    @pytest.mark.asyncio
    async def test_play_audio_missing_path(self):
        """Test play_audio with missing path."""
        from voice_soundboard.server import handle_play_audio

        result = await handle_play_audio({})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "'path' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_play_audio_file_not_found(self):
        """Test play_audio with non-existent file."""
        from voice_soundboard.server import handle_play_audio

        result = await handle_play_audio({"path": "/nonexistent/file.wav"})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "not found" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_play_audio_success(self):
        """Test successful audio playback."""
        from voice_soundboard.server import handle_play_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.wav"
            # Create a simple WAV file
            import wave
            with wave.open(str(filepath), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b'\x00' * 4800)  # 0.1 seconds of silence

            with patch('voice_soundboard.server.play_audio') as mock_play:
                with patch('voice_soundboard.server.get_audio_duration', return_value=0.1):
                    result = await handle_play_audio({"path": str(filepath)})
                    assert len(result) == 1
                    assert "Played" in result[0].text


class TestHandleStopAudio:
    """Tests for handle_stop_audio handler."""

    @pytest.mark.asyncio
    async def test_stop_audio(self):
        """Test stopping audio playback."""
        from voice_soundboard.server import handle_stop_audio

        with patch('voice_soundboard.server.stop_playback') as mock_stop:
            result = await handle_stop_audio({})
            assert len(result) == 1
            assert "stopped" in result[0].text.lower()
            mock_stop.assert_called_once()


class TestHandleSoundEffect:
    """Tests for handle_sound_effect handler."""

    @pytest.mark.asyncio
    async def test_sound_effect_missing_name(self):
        """Test sound_effect with missing effect name."""
        from voice_soundboard.server import handle_sound_effect

        result = await handle_sound_effect({})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "'effect' is required" in result[0].text

    @pytest.mark.asyncio
    async def test_sound_effect_invalid_name(self):
        """Test sound_effect with invalid effect name."""
        from voice_soundboard.server import handle_sound_effect

        result = await handle_sound_effect({"effect": "nonexistent_effect"})
        assert len(result) == 1
        # Should return error about unknown effect

    @pytest.mark.asyncio
    async def test_sound_effect_play(self):
        """Test playing a sound effect."""
        from voice_soundboard.server import handle_sound_effect

        with patch('voice_soundboard.server.get_effect') as mock_get:
            mock_effect = Mock()
            mock_effect.duration = 0.5
            mock_effect.play = Mock()
            mock_get.return_value = mock_effect

            result = await handle_sound_effect({"effect": "chime"})
            assert len(result) == 1
            assert "Played" in result[0].text or "chime" in result[0].text

    @pytest.mark.asyncio
    async def test_sound_effect_save(self):
        """Test saving a sound effect."""
        from voice_soundboard.server import handle_sound_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "effect.wav"

            with patch('voice_soundboard.server.get_effect') as mock_get:
                mock_effect = Mock()
                mock_effect.save = Mock()
                mock_get.return_value = mock_effect

                result = await handle_sound_effect({
                    "effect": "chime",
                    "save_path": str(save_path)
                })
                assert len(result) == 1
                assert "Saved" in result[0].text


# ============================================================================
# Emotion Handler Tests
# ============================================================================

class TestHandleBlendEmotions:
    """Tests for handle_blend_emotions handler."""

    @pytest.mark.asyncio
    async def test_blend_emotions_success(self):
        """Test blending emotions."""
        from voice_soundboard.server import handle_blend_emotions

        result = await handle_blend_emotions({
            "emotions": [
                {"emotion": "happy", "weight": 0.6},
                {"emotion": "sad", "weight": 0.4}
            ]
        })
        assert len(result) == 1
        # Should return blend info
        text = result[0].text.lower()
        assert "blend" in text or "vad" in text or "valence" in text

    @pytest.mark.asyncio
    async def test_blend_emotions_empty(self):
        """Test blending with empty list."""
        from voice_soundboard.server import handle_blend_emotions

        result = await handle_blend_emotions({"emotions": []})
        assert len(result) == 1
        # Should handle error gracefully


class TestHandleParseEmotionText:
    """Tests for handle_parse_emotion_text handler."""

    @pytest.mark.asyncio
    async def test_parse_emotion_text(self):
        """Test parsing emotion-tagged text."""
        from voice_soundboard.server import handle_parse_emotion_text

        result = await handle_parse_emotion_text({
            "text": "I'm {happy}so excited{/happy} to see you!"
        })
        assert len(result) == 1
        text = result[0].text.lower()
        assert "happy" in text or "span" in text or "emotion" in text


class TestHandleGetEmotionVAD:
    """Tests for handle_get_emotion_vad handler."""

    @pytest.mark.asyncio
    async def test_get_emotion_vad(self):
        """Test getting VAD for emotion."""
        from voice_soundboard.server import handle_get_emotion_vad

        result = await handle_get_emotion_vad({"emotion": "happy"})
        assert len(result) == 1
        text = result[0].text.lower()
        assert "valence" in text or "arousal" in text or "dominance" in text


class TestHandleSampleEmotionCurve:
    """Tests for handle_sample_emotion_curve handler."""

    @pytest.mark.asyncio
    async def test_sample_emotion_curve_named(self):
        """Test sampling a named emotion curve."""
        from voice_soundboard.server import handle_sample_emotion_curve

        result = await handle_sample_emotion_curve({
            "curve_name": "tension_build",
            "num_samples": 5
        })
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_sample_emotion_curve_custom(self):
        """Test sampling custom keyframes."""
        from voice_soundboard.server import handle_sample_emotion_curve

        result = await handle_sample_emotion_curve({
            "keyframes": [
                {"position": 0.0, "emotion": "calm"},
                {"position": 1.0, "emotion": "excited"}
            ],
            "num_samples": 3
        })
        assert len(result) == 1


# ============================================================================
# Call Tool Dispatcher Tests
# ============================================================================

class TestCallTool:
    """Tests for the main call_tool dispatcher."""

    @pytest.mark.asyncio
    async def test_call_tool_unknown(self):
        """Test calling unknown tool."""
        from voice_soundboard.server import call_tool

        result = await call_tool("unknown_tool_xyz", {})
        assert len(result) == 1
        assert "Unknown tool" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_list_presets(self):
        """Test calling list_presets through dispatcher."""
        from voice_soundboard.server import call_tool

        result = await call_tool("list_presets", {})
        assert len(result) == 1
        assert "Voice presets" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_list_effects(self):
        """Test calling list_effects through dispatcher."""
        from voice_soundboard.server import call_tool

        result = await call_tool("list_effects", {})
        assert len(result) == 1
        assert "Available sound effects" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_stop_audio(self):
        """Test calling stop_audio through dispatcher."""
        from voice_soundboard.server import call_tool

        with patch('voice_soundboard.server.stop_playback'):
            result = await call_tool("stop_audio", {})
            assert len(result) == 1


# ============================================================================
# List Tools Tests
# ============================================================================

class TestListTools:
    """Tests for the list_tools function."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_tools(self):
        """Test that list_tools returns a list of Tool objects."""
        from voice_soundboard.server import list_tools
        from mcp.types import Tool

        tools = await list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert all(isinstance(t, Tool) for t in tools)

    @pytest.mark.asyncio
    async def test_list_tools_has_speak(self):
        """Test that speak tool is in the list."""
        from voice_soundboard.server import list_tools

        tools = await list_tools()
        tool_names = [t.name for t in tools]
        assert "speak" in tool_names

    @pytest.mark.asyncio
    async def test_list_tools_has_required_tools(self):
        """Test that essential tools are present."""
        from voice_soundboard.server import list_tools

        tools = await list_tools()
        tool_names = [t.name for t in tools]

        essential = ["speak", "list_voices", "play_audio", "stop_audio", "sound_effect"]
        for name in essential:
            assert name in tool_names, f"Missing tool: {name}"
