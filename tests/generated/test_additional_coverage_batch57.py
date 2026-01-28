"""
Test Additional Coverage Batch 57: LLM Pipeline Tests

Tests for:
- TurnType enum
- PipelineState enum
- ConversationTurn dataclass
- PipelineConfig dataclass
- SpeechPipeline class
- quick_converse convenience function
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import time


# ============== TurnType Enum Tests ==============

class TestTurnTypeEnum:
    """Tests for TurnType enum."""

    def test_turn_type_user(self):
        """Test TurnType.USER value."""
        from voice_soundboard.llm.pipeline import TurnType
        assert TurnType.USER.value == "user"

    def test_turn_type_assistant(self):
        """Test TurnType.ASSISTANT value."""
        from voice_soundboard.llm.pipeline import TurnType
        assert TurnType.ASSISTANT.value == "assistant"

    def test_turn_type_system(self):
        """Test TurnType.SYSTEM value."""
        from voice_soundboard.llm.pipeline import TurnType
        assert TurnType.SYSTEM.value == "system"


# ============== PipelineState Enum Tests ==============

class TestPipelineStateEnum:
    """Tests for PipelineState enum."""

    def test_pipeline_state_idle(self):
        """Test PipelineState.IDLE value."""
        from voice_soundboard.llm.pipeline import PipelineState
        assert PipelineState.IDLE.value == "idle"

    def test_pipeline_state_listening(self):
        """Test PipelineState.LISTENING value."""
        from voice_soundboard.llm.pipeline import PipelineState
        assert PipelineState.LISTENING.value == "listening"

    def test_pipeline_state_transcribing(self):
        """Test PipelineState.TRANSCRIBING value."""
        from voice_soundboard.llm.pipeline import PipelineState
        assert PipelineState.TRANSCRIBING.value == "transcribing"

    def test_pipeline_state_thinking(self):
        """Test PipelineState.THINKING value."""
        from voice_soundboard.llm.pipeline import PipelineState
        assert PipelineState.THINKING.value == "thinking"

    def test_pipeline_state_speaking(self):
        """Test PipelineState.SPEAKING value."""
        from voice_soundboard.llm.pipeline import PipelineState
        assert PipelineState.SPEAKING.value == "speaking"

    def test_pipeline_state_interrupted(self):
        """Test PipelineState.INTERRUPTED value."""
        from voice_soundboard.llm.pipeline import PipelineState
        assert PipelineState.INTERRUPTED.value == "interrupted"

    def test_pipeline_state_error(self):
        """Test PipelineState.ERROR value."""
        from voice_soundboard.llm.pipeline import PipelineState
        assert PipelineState.ERROR.value == "error"


# ============== ConversationTurn Tests ==============

class TestConversationTurn:
    """Tests for ConversationTurn dataclass."""

    def test_conversation_turn_creation(self):
        """Test ConversationTurn basic creation."""
        from voice_soundboard.llm.pipeline import ConversationTurn, TurnType
        turn = ConversationTurn(
            type=TurnType.USER,
            content="Hello, how are you?"
        )
        assert turn.type == TurnType.USER
        assert turn.content == "Hello, how are you?"
        assert turn.timestamp > 0

    def test_conversation_turn_with_audio(self):
        """Test ConversationTurn with audio data."""
        from voice_soundboard.llm.pipeline import ConversationTurn, TurnType
        audio = np.random.randn(16000).astype(np.float32)
        turn = ConversationTurn(
            type=TurnType.USER,
            content="Test",
            audio=audio,
            duration_ms=1000.0
        )
        assert turn.audio is not None
        assert len(turn.audio) == 16000
        assert turn.duration_ms == 1000.0

    def test_conversation_turn_with_metadata(self):
        """Test ConversationTurn with metadata."""
        from voice_soundboard.llm.pipeline import ConversationTurn, TurnType
        turn = ConversationTurn(
            type=TurnType.ASSISTANT,
            content="I'm doing well!",
            emotion="happy",
            voice="narrator",
            confidence=0.95
        )
        assert turn.emotion == "happy"
        assert turn.voice == "narrator"
        assert turn.confidence == 0.95

    def test_conversation_turn_with_timing(self):
        """Test ConversationTurn with timing information."""
        from voice_soundboard.llm.pipeline import ConversationTurn, TurnType
        turn = ConversationTurn(
            type=TurnType.ASSISTANT,
            content="Response",
            transcription_ms=150.0,
            llm_ms=500.0,
            tts_ms=200.0
        )
        assert turn.transcription_ms == 150.0
        assert turn.llm_ms == 500.0
        assert turn.tts_ms == 200.0


# ============== PipelineConfig Tests ==============

class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        from voice_soundboard.llm.pipeline import PipelineConfig
        config = PipelineConfig()
        assert config.stt_backend == "whisper"
        assert config.stt_model == "base"
        assert config.llm_backend == "ollama"
        assert config.tts_backend == "kokoro"

    def test_pipeline_config_llm_settings(self):
        """Test PipelineConfig LLM settings."""
        from voice_soundboard.llm.pipeline import PipelineConfig
        config = PipelineConfig()
        assert config.llm_model == "llama3.2"
        assert config.llm_temperature == 0.7
        assert config.llm_max_tokens == 1024

    def test_pipeline_config_tts_settings(self):
        """Test PipelineConfig TTS settings."""
        from voice_soundboard.llm.pipeline import PipelineConfig
        config = PipelineConfig()
        assert config.tts_preset == "assistant"
        assert config.tts_speed == 1.0
        assert config.tts_voice is None

    def test_pipeline_config_custom_values(self):
        """Test PipelineConfig with custom values."""
        from voice_soundboard.llm.pipeline import PipelineConfig
        config = PipelineConfig(
            stt_backend="faster_whisper",
            llm_backend="openai",
            llm_model="gpt-4",
            tts_backend="elevenlabs"
        )
        assert config.stt_backend == "faster_whisper"
        assert config.llm_backend == "openai"
        assert config.llm_model == "gpt-4"

    def test_pipeline_config_interruption_settings(self):
        """Test PipelineConfig interruption settings."""
        from voice_soundboard.llm.pipeline import PipelineConfig
        config = PipelineConfig()
        assert config.allow_interruption is True
        assert config.interruption_threshold_db == -35.0

    def test_pipeline_config_turn_taking(self):
        """Test PipelineConfig turn-taking settings."""
        from voice_soundboard.llm.pipeline import PipelineConfig
        config = PipelineConfig()
        assert config.silence_threshold_ms == 1500.0
        assert config.max_turn_duration_ms == 30000.0


# ============== SpeechPipeline Tests ==============

class TestSpeechPipeline:
    """Tests for SpeechPipeline class."""

    def test_speech_pipeline_init_default(self):
        """Test SpeechPipeline default initialization."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState
        pipeline = SpeechPipeline()
        assert pipeline.state == PipelineState.IDLE
        assert pipeline.conversation_history == []

    def test_speech_pipeline_init_with_backends(self):
        """Test SpeechPipeline with custom backends."""
        from voice_soundboard.llm.pipeline import SpeechPipeline
        pipeline = SpeechPipeline(stt="faster_whisper", llm="openai", tts="elevenlabs")
        assert pipeline.config.stt_backend == "faster_whisper"
        assert pipeline.config.llm_backend == "openai"
        assert pipeline.config.tts_backend == "elevenlabs"

    def test_speech_pipeline_init_with_config(self):
        """Test SpeechPipeline with config object."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineConfig
        config = PipelineConfig(llm_model="gpt-4", tts_speed=1.2)
        pipeline = SpeechPipeline(config=config)
        assert pipeline.config.llm_model == "gpt-4"
        assert pipeline.config.tts_speed == 1.2

    def test_speech_pipeline_state_change_callback(self):
        """Test SpeechPipeline state change callback."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState
        pipeline = SpeechPipeline()
        states = []

        def on_state(state):
            states.append(state)

        pipeline.on_state_change = on_state
        pipeline._set_state(PipelineState.LISTENING)
        pipeline._set_state(PipelineState.THINKING)

        assert len(states) == 2
        assert states[0] == PipelineState.LISTENING
        assert states[1] == PipelineState.THINKING

    def test_speech_pipeline_transcription_callback(self):
        """Test SpeechPipeline transcription callback property."""
        from voice_soundboard.llm.pipeline import SpeechPipeline
        pipeline = SpeechPipeline()

        callback = Mock()
        pipeline.on_transcription = callback
        assert pipeline.on_transcription == callback

    def test_speech_pipeline_response_callback(self):
        """Test SpeechPipeline response callback property."""
        from voice_soundboard.llm.pipeline import SpeechPipeline
        pipeline = SpeechPipeline()

        callback = Mock()
        pipeline.on_response = callback
        assert pipeline.on_response == callback

    def test_speech_pipeline_interrupt_not_speaking(self):
        """Test SpeechPipeline.interrupt when not speaking."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState
        pipeline = SpeechPipeline()
        pipeline.state = PipelineState.IDLE
        pipeline.interrupt()
        # Should not change state if not speaking
        assert pipeline.state == PipelineState.IDLE

    def test_speech_pipeline_interrupt_when_speaking(self):
        """Test SpeechPipeline.interrupt when speaking."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState
        pipeline = SpeechPipeline()
        pipeline.state = PipelineState.SPEAKING

        with patch('voice_soundboard.llm.pipeline.stop_playback', side_effect=ImportError()):
            pipeline.interrupt()

        assert pipeline.state == PipelineState.INTERRUPTED

    def test_speech_pipeline_reset(self):
        """Test SpeechPipeline.reset method."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState, ConversationTurn, TurnType
        pipeline = SpeechPipeline()
        pipeline.conversation_history.append(
            ConversationTurn(type=TurnType.USER, content="Test")
        )
        pipeline.state = PipelineState.THINKING

        pipeline.reset()

        assert pipeline.state == PipelineState.IDLE
        assert len(pipeline.conversation_history) == 0

    def test_speech_pipeline_stats(self):
        """Test SpeechPipeline.stats property."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, ConversationTurn, TurnType
        pipeline = SpeechPipeline()
        pipeline.conversation_history = [
            ConversationTurn(type=TurnType.USER, content="Hi", transcription_ms=100.0),
            ConversationTurn(type=TurnType.ASSISTANT, content="Hello!", llm_ms=200.0, tts_ms=150.0)
        ]

        stats = pipeline.stats
        assert stats["turn_count"] == 2
        assert stats["total_transcription_ms"] == 100.0
        assert stats["total_llm_ms"] == 200.0
        assert stats["total_tts_ms"] == 150.0

    @pytest.mark.asyncio
    async def test_speech_pipeline_transcribe_mock(self):
        """Test SpeechPipeline.transcribe with mock whisper."""
        from voice_soundboard.llm.pipeline import SpeechPipeline
        pipeline = SpeechPipeline()

        audio = np.random.randn(16000).astype(np.float32)

        # Should return mock transcription when whisper not available
        result = await pipeline.transcribe(audio, sample_rate=16000)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_speech_pipeline_generate_response(self):
        """Test SpeechPipeline.generate_response method."""
        from voice_soundboard.llm.pipeline import SpeechPipeline

        pipeline = SpeechPipeline()

        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = "Hello! I'm doing great."
        mock_llm.chat.return_value = mock_response

        with patch.object(pipeline, '_get_llm', return_value=mock_llm):
            response = await pipeline.generate_response("Hello, how are you?")

        assert response == "Hello! I'm doing great."
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_speech_pipeline_generate_response_error(self):
        """Test SpeechPipeline.generate_response handles errors."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState

        pipeline = SpeechPipeline()

        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = Exception("LLM error")

        with patch.object(pipeline, '_get_llm', return_value=mock_llm):
            with pytest.raises(RuntimeError, match="Failed to generate LLM response"):
                await pipeline.generate_response("Test")

        assert pipeline.state == PipelineState.ERROR


# ============== quick_converse Function Tests ==============

class TestQuickConverse:
    """Tests for quick_converse convenience function."""

    @pytest.mark.asyncio
    async def test_quick_converse(self):
        """Test quick_converse function."""
        from voice_soundboard.llm.pipeline import quick_converse, ConversationTurn, TurnType

        mock_turn = ConversationTurn(
            type=TurnType.ASSISTANT,
            content="Hello! I'm a helpful assistant."
        )

        with patch('voice_soundboard.llm.pipeline.SpeechPipeline') as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.converse.return_value = mock_turn
            mock_pipeline_class.return_value = mock_pipeline

            audio = np.random.randn(16000).astype(np.float32)
            result = await quick_converse(audio, llm="ollama")

        assert result == "Hello! I'm a helpful assistant."

    @pytest.mark.asyncio
    async def test_quick_converse_with_system_prompt(self):
        """Test quick_converse with custom system prompt."""
        from voice_soundboard.llm.pipeline import quick_converse, PipelineConfig

        with patch('voice_soundboard.llm.pipeline.SpeechPipeline') as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_turn = Mock()
            mock_turn.content = "Response"
            mock_pipeline.converse.return_value = mock_turn
            mock_pipeline_class.return_value = mock_pipeline

            audio = np.random.randn(16000).astype(np.float32)
            await quick_converse(audio, system_prompt="You are a pirate.")

            # Verify config was passed
            call_kwargs = mock_pipeline_class.call_args[1]
            assert call_kwargs['config'].system_prompt == "You are a pirate."
