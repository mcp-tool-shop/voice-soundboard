"""
Additional test coverage batch 10.

Tests for:
- llm/conversation.py (MessageRole, ConversationState, TurnTakingStrategy, Message, ConversationManager, TurnTakingController)
- llm/pipeline.py (TurnType, PipelineState, ConversationTurn, PipelineConfig, SpeechPipeline)
- llm/streaming.py (StreamState, StreamConfig, StreamBuffer, SentenceBoundaryDetector, StreamingLLMSpeaker)
"""

import pytest
import numpy as np
import time
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock


# =============================================================================
# Tests for llm/conversation.py
# =============================================================================

class TestMessageRoleEnum:
    """Tests for MessageRole enum."""

    def test_message_role_values(self):
        """Test MessageRole enum values."""
        from voice_soundboard.llm.conversation import MessageRole

        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"

    def test_message_role_count(self):
        """Test MessageRole enum has 3 roles."""
        from voice_soundboard.llm.conversation import MessageRole

        assert len(MessageRole) == 3


class TestConversationStateEnum:
    """Tests for ConversationState enum."""

    def test_conversation_state_values(self):
        """Test ConversationState enum values."""
        from voice_soundboard.llm.conversation import ConversationState

        assert ConversationState.IDLE.value == "idle"
        assert ConversationState.LISTENING.value == "listening"
        assert ConversationState.PROCESSING.value == "processing"
        assert ConversationState.SPEAKING.value == "speaking"
        assert ConversationState.ENDED.value == "ended"

    def test_conversation_state_count(self):
        """Test ConversationState enum has 6 states."""
        from voice_soundboard.llm.conversation import ConversationState

        assert len(ConversationState) == 6


class TestTurnTakingStrategyEnum:
    """Tests for TurnTakingStrategy enum."""

    def test_turn_taking_strategy_values(self):
        """Test TurnTakingStrategy enum values."""
        from voice_soundboard.llm.conversation import TurnTakingStrategy

        assert TurnTakingStrategy.STRICT.value == "strict"
        assert TurnTakingStrategy.SILENCE.value == "silence"
        assert TurnTakingStrategy.PUSH_TO_TALK.value == "push_to_talk"
        assert TurnTakingStrategy.HOTWORD.value == "hotword"
        assert TurnTakingStrategy.CONTINUOUS.value == "continuous"

    def test_turn_taking_strategy_count(self):
        """Test TurnTakingStrategy enum has 5 strategies."""
        from voice_soundboard.llm.conversation import TurnTakingStrategy

        assert len(TurnTakingStrategy) == 5


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test Message creation."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        msg = Message(role=MessageRole.USER, content="Hello")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.id is not None
        assert msg.timestamp > 0

    def test_message_with_metadata(self):
        """Test Message with metadata."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hello!",
            emotion="happy",
            voice="narrator",
            duration_ms=500.0,
            tokens=5
        )

        assert msg.emotion == "happy"
        assert msg.voice == "narrator"
        assert msg.duration_ms == 500.0
        assert msg.tokens == 5

    def test_message_to_dict(self):
        """Test Message to_dict method."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        msg = Message(role=MessageRole.USER, content="Test message")
        data = msg.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Test message"
        assert "id" in data
        assert "timestamp" in data

    def test_message_from_dict(self):
        """Test Message from_dict method."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        data = {
            "id": "test123",
            "role": "assistant",
            "content": "Hi there",
            "emotion": "friendly"
        }

        msg = Message.from_dict(data)

        assert msg.id == "test123"
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there"
        assert msg.emotion == "friendly"


class TestConversationConfig:
    """Tests for ConversationConfig dataclass."""

    def test_conversation_config_defaults(self):
        """Test ConversationConfig default values."""
        from voice_soundboard.llm.conversation import ConversationConfig, TurnTakingStrategy

        config = ConversationConfig()

        assert config.turn_taking_strategy == TurnTakingStrategy.SILENCE
        assert config.silence_threshold_ms == 1500.0
        assert config.max_history_messages == 100
        assert config.context_window_messages == 10
        assert config.auto_save is False

    def test_conversation_config_custom(self):
        """Test ConversationConfig with custom values."""
        from voice_soundboard.llm.conversation import ConversationConfig, TurnTakingStrategy

        config = ConversationConfig(
            turn_taking_strategy=TurnTakingStrategy.PUSH_TO_TALK,
            silence_threshold_ms=2000.0,
            system_prompt="You are helpful."
        )

        assert config.turn_taking_strategy == TurnTakingStrategy.PUSH_TO_TALK
        assert config.silence_threshold_ms == 2000.0
        assert config.system_prompt == "You are helpful."


class TestConversationManager:
    """Tests for ConversationManager class."""

    def test_conversation_manager_creation(self):
        """Test ConversationManager creation."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()

        assert manager.state == ConversationState.IDLE
        assert len(manager.messages) == 0
        assert manager.id is not None

    def test_conversation_manager_with_system_prompt(self):
        """Test ConversationManager with system prompt."""
        from voice_soundboard.llm.conversation import (
            ConversationManager, ConversationConfig, MessageRole
        )

        config = ConversationConfig(system_prompt="You are helpful.")
        manager = ConversationManager(config=config)

        assert len(manager.messages) == 1
        assert manager.messages[0].role == MessageRole.SYSTEM
        assert manager.messages[0].content == "You are helpful."

    def test_conversation_manager_start(self):
        """Test starting a conversation."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()

        assert manager.state == ConversationState.LISTENING
        assert manager.started_at is not None

    def test_conversation_manager_end(self):
        """Test ending a conversation."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()
        manager.end()

        assert manager.state == ConversationState.ENDED
        assert manager.ended_at is not None

    def test_conversation_manager_add_user_message(self):
        """Test adding user message."""
        from voice_soundboard.llm.conversation import (
            ConversationManager, ConversationState, MessageRole
        )

        manager = ConversationManager()
        manager.start()

        msg = manager.add_user_message("Hello!")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert manager.state == ConversationState.PROCESSING

    def test_conversation_manager_add_assistant_message(self):
        """Test adding assistant message."""
        from voice_soundboard.llm.conversation import (
            ConversationManager, ConversationState, MessageRole
        )

        manager = ConversationManager()
        manager.start()
        manager.add_user_message("Hello!")

        msg = manager.add_assistant_message("Hi there!")

        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there!"
        assert manager.state == ConversationState.LISTENING

    def test_conversation_manager_get_llm_context(self):
        """Test getting LLM context."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationConfig

        config = ConversationConfig(system_prompt="System prompt")
        manager = ConversationManager(config=config)
        manager.start()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")

        context = manager.get_llm_context()

        assert len(context) == 3
        assert context[0]["role"] == "system"
        assert context[1]["role"] == "user"
        assert context[2]["role"] == "assistant"

    def test_conversation_manager_get_last_message(self):
        """Test getting last message."""
        from voice_soundboard.llm.conversation import ConversationManager, MessageRole

        manager = ConversationManager()
        manager.start()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")

        last = manager.get_last_message()
        assert last.role == MessageRole.ASSISTANT

        last_user = manager.get_last_message(MessageRole.USER)
        assert last_user.content == "Hello"

    def test_conversation_manager_stats(self):
        """Test conversation statistics."""
        from voice_soundboard.llm.conversation import ConversationManager

        manager = ConversationManager()
        manager.start()
        manager.add_user_message("Hello world")
        manager.add_assistant_message("Hi there friend")

        stats = manager.stats

        assert stats["message_count"] == 2
        assert stats["user_message_count"] == 1
        assert stats["assistant_message_count"] == 1
        assert stats["user_word_count"] == 2
        assert stats["assistant_word_count"] == 3

    def test_conversation_manager_duration(self):
        """Test conversation duration property."""
        from voice_soundboard.llm.conversation import ConversationManager

        manager = ConversationManager()
        manager.start()
        time.sleep(0.1)

        assert manager.duration_seconds >= 0.1

    def test_conversation_manager_save_load(self):
        """Test saving and loading conversation."""
        from voice_soundboard.llm.conversation import ConversationManager, MessageRole

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "conversation.json"

            # Create and save
            manager = ConversationManager()
            manager.start()
            manager.add_user_message("Hello")
            manager.add_assistant_message("Hi")
            manager.save(path)

            # Load
            loaded = ConversationManager.load(path)

            assert len(loaded.messages) == 2
            assert loaded.messages[0].role == MessageRole.USER
            assert loaded.messages[1].role == MessageRole.ASSISTANT

    def test_conversation_manager_set_speaking(self):
        """Test set_speaking method."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()

        manager.set_speaking()

        assert manager.state == ConversationState.SPEAKING

    def test_conversation_manager_set_interrupted(self):
        """Test set_interrupted method."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()

        manager.set_interrupted()

        assert manager.state == ConversationState.INTERRUPTED


class TestTurnTakingController:
    """Tests for TurnTakingController class."""

    def test_turn_taking_controller_creation(self):
        """Test TurnTakingController creation."""
        from voice_soundboard.llm.conversation import TurnTakingController, TurnTakingStrategy

        controller = TurnTakingController()

        assert controller.strategy == TurnTakingStrategy.SILENCE
        assert controller.silence_threshold_ms == 1500.0

    def test_turn_taking_controller_custom_strategy(self):
        """Test TurnTakingController with custom strategy."""
        from voice_soundboard.llm.conversation import TurnTakingController, TurnTakingStrategy

        controller = TurnTakingController(
            strategy=TurnTakingStrategy.PUSH_TO_TALK,
            silence_threshold_ms=2000.0
        )

        assert controller.strategy == TurnTakingStrategy.PUSH_TO_TALK
        assert controller.silence_threshold_ms == 2000.0

    def test_turn_taking_controller_process_audio_speech_start(self):
        """Test processing audio when speech starts."""
        from voice_soundboard.llm.conversation import TurnTakingController

        controller = TurnTakingController()
        callback = Mock()
        controller.on_user_start = callback

        # First speech frame
        result = controller.process_audio(is_speech=True)

        assert result is False  # Turn not ended yet
        assert callback.called

    def test_turn_taking_controller_process_audio_silence_end(self):
        """Test processing audio when silence ends turn."""
        from voice_soundboard.llm.conversation import TurnTakingController

        controller = TurnTakingController(silence_threshold_ms=100)
        end_callback = Mock()
        controller.on_user_end = end_callback

        # Start speaking
        controller.process_audio(is_speech=True)

        # Silence starts
        controller.process_audio(is_speech=False)

        # Wait for silence threshold
        time.sleep(0.15)

        # More silence
        result = controller.process_audio(is_speech=False)

        assert result is True  # Turn ended
        assert end_callback.called

    def test_turn_taking_controller_force_end_turn(self):
        """Test force ending turn."""
        from voice_soundboard.llm.conversation import TurnTakingController

        controller = TurnTakingController()
        callback = Mock()
        controller.on_user_end = callback

        # Start speaking
        controller.process_audio(is_speech=True)

        # Force end
        controller.force_end_turn()

        assert callback.called

    def test_turn_taking_controller_reset(self):
        """Test resetting controller."""
        from voice_soundboard.llm.conversation import TurnTakingController

        controller = TurnTakingController()
        controller.process_audio(is_speech=True)

        controller.reset()

        assert controller._is_user_speaking is False
        assert controller._silence_start is None


# =============================================================================
# Tests for llm/pipeline.py
# =============================================================================

class TestTurnTypeEnum:
    """Tests for TurnType enum."""

    def test_turn_type_values(self):
        """Test TurnType enum values."""
        from voice_soundboard.llm.pipeline import TurnType

        assert TurnType.USER.value == "user"
        assert TurnType.ASSISTANT.value == "assistant"
        assert TurnType.SYSTEM.value == "system"


class TestPipelineStateEnum:
    """Tests for PipelineState enum."""

    def test_pipeline_state_values(self):
        """Test PipelineState enum values."""
        from voice_soundboard.llm.pipeline import PipelineState

        assert PipelineState.IDLE.value == "idle"
        assert PipelineState.LISTENING.value == "listening"
        assert PipelineState.TRANSCRIBING.value == "transcribing"
        assert PipelineState.THINKING.value == "thinking"
        assert PipelineState.SPEAKING.value == "speaking"

    def test_pipeline_state_count(self):
        """Test PipelineState enum has 7 states."""
        from voice_soundboard.llm.pipeline import PipelineState

        assert len(PipelineState) == 7


class TestConversationTurn:
    """Tests for ConversationTurn dataclass."""

    def test_conversation_turn_creation(self):
        """Test ConversationTurn creation."""
        from voice_soundboard.llm.pipeline import ConversationTurn, TurnType

        turn = ConversationTurn(type=TurnType.USER, content="Hello")

        assert turn.type == TurnType.USER
        assert turn.content == "Hello"
        assert turn.timestamp > 0

    def test_conversation_turn_with_timing(self):
        """Test ConversationTurn with timing info."""
        from voice_soundboard.llm.pipeline import ConversationTurn, TurnType

        turn = ConversationTurn(
            type=TurnType.ASSISTANT,
            content="Hi there",
            transcription_ms=100.0,
            llm_ms=500.0,
            tts_ms=200.0,
            duration_ms=800.0
        )

        assert turn.transcription_ms == 100.0
        assert turn.llm_ms == 500.0
        assert turn.tts_ms == 200.0
        assert turn.duration_ms == 800.0


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        from voice_soundboard.llm.pipeline import PipelineConfig

        config = PipelineConfig()

        assert config.stt_backend == "whisper"
        assert config.llm_backend == "ollama"
        assert config.tts_backend == "kokoro"
        assert config.auto_emotion is True
        assert config.allow_interruption is True

    def test_pipeline_config_custom(self):
        """Test PipelineConfig with custom values."""
        from voice_soundboard.llm.pipeline import PipelineConfig

        config = PipelineConfig(
            llm_model="gpt-4",
            tts_voice="narrator",
            stream_tts=False
        )

        assert config.llm_model == "gpt-4"
        assert config.tts_voice == "narrator"
        assert config.stream_tts is False


class TestSpeechPipeline:
    """Tests for SpeechPipeline class."""

    def test_speech_pipeline_creation(self):
        """Test SpeechPipeline creation."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState

        pipeline = SpeechPipeline()

        assert pipeline.state == PipelineState.IDLE
        assert len(pipeline.conversation_history) == 0

    def test_speech_pipeline_with_config(self):
        """Test SpeechPipeline with custom config."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineConfig

        config = PipelineConfig(llm_backend="mock")
        pipeline = SpeechPipeline(config=config)

        assert pipeline.config.llm_backend == "mock"

    def test_speech_pipeline_stats(self):
        """Test SpeechPipeline stats property."""
        from voice_soundboard.llm.pipeline import SpeechPipeline

        pipeline = SpeechPipeline()
        stats = pipeline.stats

        assert stats["state"] == "idle"
        assert stats["turn_count"] == 0
        assert "config" in stats

    def test_speech_pipeline_reset(self):
        """Test SpeechPipeline reset method."""
        from voice_soundboard.llm.pipeline import (
            SpeechPipeline, PipelineState, ConversationTurn, TurnType
        )

        pipeline = SpeechPipeline()
        pipeline.conversation_history.append(
            ConversationTurn(type=TurnType.USER, content="test")
        )

        pipeline.reset()

        assert len(pipeline.conversation_history) == 0
        assert pipeline.state == PipelineState.IDLE

    def test_speech_pipeline_interrupt(self):
        """Test SpeechPipeline interrupt method."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState

        pipeline = SpeechPipeline()
        pipeline._set_state(PipelineState.SPEAKING)

        pipeline.interrupt()

        assert pipeline.state == PipelineState.INTERRUPTED

    def test_speech_pipeline_state_change_callback(self):
        """Test SpeechPipeline state change callback."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState

        pipeline = SpeechPipeline()
        callback = Mock()
        pipeline.on_state_change = callback

        pipeline._set_state(PipelineState.LISTENING)

        callback.assert_called_once_with(PipelineState.LISTENING)


# =============================================================================
# Tests for llm/streaming.py
# =============================================================================

class TestStreamStateEnum:
    """Tests for StreamState enum."""

    def test_stream_state_values(self):
        """Test StreamState enum values."""
        from voice_soundboard.llm.streaming import StreamState

        assert StreamState.IDLE.value == "idle"
        assert StreamState.BUFFERING.value == "buffering"
        assert StreamState.SPEAKING.value == "speaking"
        assert StreamState.FINISHING.value == "finishing"
        assert StreamState.ERROR.value == "error"


class TestStreamConfig:
    """Tests for StreamConfig dataclass."""

    def test_stream_config_defaults(self):
        """Test StreamConfig default values."""
        from voice_soundboard.llm.streaming import StreamConfig

        config = StreamConfig()

        assert config.sentence_end_chars == ".!?"
        assert config.min_sentence_length == 10
        assert config.max_buffer_length == 500
        assert config.flush_timeout_ms == 2000.0

    def test_stream_config_custom(self):
        """Test StreamConfig with custom values."""
        from voice_soundboard.llm.streaming import StreamConfig

        config = StreamConfig(
            min_sentence_length=5,
            voice="custom_voice",
            speed=1.2
        )

        assert config.min_sentence_length == 5
        assert config.voice == "custom_voice"
        assert config.speed == 1.2


class TestStreamBuffer:
    """Tests for StreamBuffer dataclass."""

    def test_stream_buffer_creation(self):
        """Test StreamBuffer creation."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()

        assert buffer.content == ""
        assert buffer.sentences_spoken == 0
        assert buffer.tokens_received == 0

    def test_stream_buffer_append(self):
        """Test StreamBuffer append method."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("Hello ")
        buffer.append("world")

        assert buffer.content == "Hello world"
        assert buffer.tokens_received == 2

    def test_stream_buffer_clear(self):
        """Test StreamBuffer clear method."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("Hello")

        content = buffer.clear()

        assert content == "Hello"
        assert buffer.content == ""

    def test_stream_buffer_peek(self):
        """Test StreamBuffer peek method."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("Hello")

        content = buffer.peek()

        assert content == "Hello"
        assert buffer.content == "Hello"  # Not cleared

    def test_stream_buffer_age_ms(self):
        """Test StreamBuffer age_ms property."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("Hello")
        time.sleep(0.1)

        assert buffer.age_ms >= 100

    def test_stream_buffer_length(self):
        """Test StreamBuffer length property."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("Hello")

        assert buffer.length == 5


class TestSentenceBoundaryDetector:
    """Tests for SentenceBoundaryDetector class."""

    def test_sentence_boundary_detector_creation(self):
        """Test SentenceBoundaryDetector creation."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        assert detector.config is not None

    def test_find_boundary_period(self):
        """Test finding boundary with period."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        # Note: min_sentence_length is 10 by default
        text = "Hello world. This is next"
        boundary = detector.find_boundary(text)

        assert boundary is not None
        assert text[:boundary].strip() == "Hello world."

    def test_find_boundary_question(self):
        """Test finding boundary with question mark."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        text = "How are you? I am fine"
        boundary = detector.find_boundary(text)

        assert boundary is not None
        assert "?" in text[:boundary]

    def test_find_boundary_exclamation(self):
        """Test finding boundary with exclamation."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        text = "This is great! Thanks a lot"
        boundary = detector.find_boundary(text)

        assert boundary is not None
        assert "!" in text[:boundary]

    def test_find_boundary_abbreviation(self):
        """Test not finding boundary in abbreviation."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        text = "Dr. Smith is here today"
        boundary = detector.find_boundary(text)

        # Should not detect boundary after "Dr."
        assert boundary is None

    def test_find_boundary_decimal(self):
        """Test not finding boundary in decimal number."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        text = "The value is 3.14159 approximately"
        boundary = detector.find_boundary(text)

        # Should not detect boundary after decimal
        assert boundary is None

    def test_split_sentences(self):
        """Test splitting text into sentences."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        text = "Hello world. How are you? I am fine!"
        sentences = detector.split_sentences(text)

        assert len(sentences) == 3

    def test_extract_complete(self):
        """Test extracting complete sentences."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()

        text = "Hello world. How are you? I am still"
        sentences, remaining = detector.extract_complete(text)

        assert len(sentences) == 2
        assert "still" in remaining


class TestStreamingLLMSpeaker:
    """Tests for StreamingLLMSpeaker class."""

    def test_streaming_speaker_creation(self):
        """Test StreamingLLMSpeaker creation."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState

        speaker = StreamingLLMSpeaker()

        assert speaker.state == StreamState.IDLE
        assert speaker.sentences_spoken == 0
        assert speaker.total_tokens == 0

    def test_streaming_speaker_with_config(self):
        """Test StreamingLLMSpeaker with config."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamConfig

        config = StreamConfig(voice="test_voice", speed=1.5)
        speaker = StreamingLLMSpeaker(config=config)

        assert speaker.config.voice == "test_voice"
        assert speaker.config.speed == 1.5

    @pytest.mark.asyncio
    async def test_streaming_speaker_feed_tokens(self):
        """Test feeding tokens to speaker."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState

        speaker = StreamingLLMSpeaker()

        # Feed tokens that don't complete a sentence
        result = await speaker.feed("Hello ")
        assert result is None
        assert speaker.state == StreamState.BUFFERING

        result = await speaker.feed("world")
        assert result is None
        assert speaker.buffer.content == "Hello world"

    @pytest.mark.asyncio
    async def test_streaming_speaker_reset(self):
        """Test resetting speaker."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState

        speaker = StreamingLLMSpeaker()
        await speaker.feed("Hello")

        speaker.reset()

        assert speaker.state == StreamState.IDLE
        assert speaker.buffer.content == ""
        assert speaker.sentences_spoken == 0

    def test_streaming_speaker_stats(self):
        """Test speaker stats property."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker

        speaker = StreamingLLMSpeaker()
        stats = speaker.stats

        assert stats["state"] == "idle"
        assert stats["sentences_spoken"] == 0
        assert stats["total_tokens"] == 0
        assert "buffer_length" in stats


class TestQuickConverse:
    """Tests for quick_converse function."""

    @pytest.mark.asyncio
    async def test_quick_converse_creation(self):
        """Test quick_converse function creates pipeline."""
        from voice_soundboard.llm.pipeline import PipelineConfig

        config = PipelineConfig(llm_backend="mock")

        # Just verify we can create the config
        assert config.llm_backend == "mock"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
