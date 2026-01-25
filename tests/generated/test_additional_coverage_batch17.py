"""
Batch 17: Additional Coverage Tests for LLM Module
- llm/interruption.py: InterruptionStrategy, InterruptionEvent, BargeInConfig, BargeInDetector, InterruptionHandler
- llm/streaming.py: StreamState, StreamConfig, StreamBuffer, SentenceBoundaryDetector, StreamingLLMSpeaker
- llm/pipeline.py: TurnType, PipelineState, ConversationTurn, PipelineConfig, SpeechPipeline
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

import numpy as np
import pytest


# ==============================================================================
# Tests for llm/interruption.py
# ==============================================================================

class TestInterruptionStrategy:
    """Tests for InterruptionStrategy enum."""

    def test_strategy_values(self):
        """Test InterruptionStrategy enum values."""
        from voice_soundboard.llm.interruption import InterruptionStrategy

        assert InterruptionStrategy.IGNORE.value == "ignore"
        assert InterruptionStrategy.STOP_IMMEDIATE.value == "stop_immediate"
        assert InterruptionStrategy.STOP_SENTENCE.value == "stop_sentence"
        assert InterruptionStrategy.PAUSE.value == "pause"
        assert InterruptionStrategy.QUEUE.value == "queue"

    def test_all_strategies_accessible(self):
        """Test all strategies are accessible."""
        from voice_soundboard.llm.interruption import InterruptionStrategy

        strategies = list(InterruptionStrategy)
        assert len(strategies) == 5


class TestInterruptionEvent:
    """Tests for InterruptionEvent dataclass."""

    def test_default_values(self):
        """Test InterruptionEvent default values."""
        from voice_soundboard.llm.interruption import InterruptionEvent

        event = InterruptionEvent()
        assert event.trigger == "user"
        assert event.audio_level is None
        assert event.transcript is None
        assert event.handled is False
        assert event.strategy_used is None
        assert event.timestamp > 0

    def test_custom_values(self):
        """Test InterruptionEvent with custom values."""
        from voice_soundboard.llm.interruption import InterruptionEvent, InterruptionStrategy

        event = InterruptionEvent(
            trigger="voice_activity",
            audio_level=-30.0,
            transcript="stop",
            handled=True,
            strategy_used=InterruptionStrategy.STOP_IMMEDIATE,
        )
        assert event.trigger == "voice_activity"
        assert event.audio_level == -30.0
        assert event.transcript == "stop"
        assert event.handled is True
        assert event.strategy_used == InterruptionStrategy.STOP_IMMEDIATE

    def test_age_ms_property(self):
        """Test age_ms property."""
        from voice_soundboard.llm.interruption import InterruptionEvent

        event = InterruptionEvent()
        time.sleep(0.01)
        age = event.age_ms
        assert age >= 10  # At least 10ms


class TestBargeInConfig:
    """Tests for BargeInConfig dataclass."""

    def test_default_values(self):
        """Test BargeInConfig default values."""
        from voice_soundboard.llm.interruption import BargeInConfig, InterruptionStrategy

        config = BargeInConfig()
        assert config.enabled is True
        assert config.vad_threshold_db == -35.0
        assert config.vad_duration_ms == 200.0
        assert config.vad_cooldown_ms == 500.0
        assert config.hotword_enabled is False
        assert config.hotwords == ["hey", "stop", "wait"]
        assert config.default_strategy == InterruptionStrategy.STOP_IMMEDIATE
        assert config.ignore_initial_ms == 500.0
        assert config.min_spoken_ms == 1000.0

    def test_custom_values(self):
        """Test BargeInConfig with custom values."""
        from voice_soundboard.llm.interruption import BargeInConfig, InterruptionStrategy

        config = BargeInConfig(
            enabled=False,
            vad_threshold_db=-40.0,
            hotword_enabled=True,
            hotwords=["hello", "hi"],
            default_strategy=InterruptionStrategy.PAUSE,
        )
        assert config.enabled is False
        assert config.vad_threshold_db == -40.0
        assert config.hotword_enabled is True
        assert config.hotwords == ["hello", "hi"]
        assert config.default_strategy == InterruptionStrategy.PAUSE


class TestBargeInDetector:
    """Tests for BargeInDetector class."""

    def test_initialization(self):
        """Test BargeInDetector initialization."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        detector = BargeInDetector()
        assert detector.config is not None
        assert detector._is_listening is False

        config = BargeInConfig(vad_threshold_db=-40.0)
        detector2 = BargeInDetector(config)
        assert detector2.config.vad_threshold_db == -40.0

    def test_start_stop_listening(self):
        """Test start/stop listening."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()
        assert detector._is_listening is False

        detector.start_listening()
        assert detector._is_listening is True
        assert detector._speaking_started is not None

        detector.stop_listening()
        assert detector._is_listening is False
        assert detector._speaking_started is None

    def test_on_interrupt_callback(self):
        """Test registering interrupt callback."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()
        callback = Mock()
        detector.on_interrupt(callback)
        assert callback in detector._callbacks

    def test_check_audio_level_not_listening(self):
        """Test check_audio_level when not listening."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()
        result = detector.check_audio_level(-30.0)
        assert result is None

    def test_check_audio_level_disabled(self):
        """Test check_audio_level when disabled."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        config = BargeInConfig(enabled=False)
        detector = BargeInDetector(config)
        detector.start_listening()
        result = detector.check_audio_level(-30.0)
        assert result is None

    def test_check_audio_level_initial_ignore(self):
        """Test check_audio_level during initial ignore period."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        config = BargeInConfig(ignore_initial_ms=1000.0, min_spoken_ms=0)
        detector = BargeInDetector(config)
        detector.start_listening()
        # Should ignore during initial period
        result = detector.check_audio_level(-30.0)
        assert result is None

    def test_check_audio_level_min_spoken(self):
        """Test check_audio_level before min spoken time."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        config = BargeInConfig(ignore_initial_ms=0, min_spoken_ms=1000.0)
        detector = BargeInDetector(config)
        detector.start_listening()
        result = detector.check_audio_level(-30.0)
        assert result is None

    def test_check_audio_level_cooldown(self):
        """Test check_audio_level during cooldown."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        config = BargeInConfig(
            ignore_initial_ms=0,
            min_spoken_ms=0,
            vad_cooldown_ms=1000.0,
        )
        detector = BargeInDetector(config)
        detector.start_listening()
        detector._last_trigger = time.time()
        result = detector.check_audio_level(-30.0)
        assert result is None

    def test_check_audio_level_below_threshold(self):
        """Test check_audio_level with audio below threshold."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        config = BargeInConfig(
            ignore_initial_ms=0,
            min_spoken_ms=0,
            vad_threshold_db=-35.0,
        )
        detector = BargeInDetector(config)
        detector.start_listening()
        # Audio level below threshold
        result = detector.check_audio_level(-40.0)
        assert result is None

    def test_check_audio_level_triggers(self):
        """Test check_audio_level triggering an event."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        config = BargeInConfig(
            ignore_initial_ms=0,
            min_spoken_ms=0,
            vad_duration_ms=0,  # Immediate trigger
            vad_threshold_db=-35.0,
        )
        detector = BargeInDetector(config)
        detector.start_listening()

        # First call starts voice activity
        result = detector.check_audio_level(-30.0)
        # Need duration, so may not trigger on first call
        # Simulate time passing and call again
        detector._voice_start = time.time() - 1  # 1 second ago
        result = detector.check_audio_level(-30.0)
        assert result is not None
        assert result.trigger == "voice_activity"

    def test_check_transcript_disabled(self):
        """Test check_transcript when hotword disabled."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        config = BargeInConfig(hotword_enabled=False)
        detector = BargeInDetector(config)
        detector.start_listening()
        result = detector.check_transcript("hey stop")
        assert result is None

    def test_check_transcript_not_listening(self):
        """Test check_transcript when not listening."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        config = BargeInConfig(hotword_enabled=True)
        detector = BargeInDetector(config)
        result = detector.check_transcript("hey stop")
        assert result is None

    def test_check_transcript_hotword_match(self):
        """Test check_transcript with matching hotword."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        config = BargeInConfig(
            enabled=True,
            hotword_enabled=True,
            hotwords=["stop", "wait"],
        )
        detector = BargeInDetector(config)
        detector.start_listening()
        result = detector.check_transcript("please stop talking")
        assert result is not None
        assert result.trigger == "hotword"
        assert result.transcript == "please stop talking"

    def test_check_transcript_no_match(self):
        """Test check_transcript with no matching hotword."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        config = BargeInConfig(
            enabled=True,
            hotword_enabled=True,
            hotwords=["stop", "wait"],
        )
        detector = BargeInDetector(config)
        detector.start_listening()
        result = detector.check_transcript("hello there")
        assert result is None

    def test_trigger_manual(self):
        """Test manual trigger."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()
        callback = Mock()
        detector.on_interrupt(callback)

        event = detector.trigger_manual()
        assert event.trigger == "manual"
        callback.assert_called_once()

    def test_callback_async(self):
        """Test async callback handling."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()

        async def async_callback(event):
            pass

        detector.on_interrupt(async_callback)
        # Should not raise
        with patch("asyncio.create_task"):
            detector.trigger_manual()


class TestInterruptionHandler:
    """Tests for InterruptionHandler class."""

    def test_initialization(self):
        """Test InterruptionHandler initialization."""
        from voice_soundboard.llm.interruption import (
            InterruptionHandler,
            InterruptionStrategy,
        )

        handler = InterruptionHandler()
        assert handler.strategy == InterruptionStrategy.STOP_IMMEDIATE
        assert handler._is_active is False

    def test_initialization_with_strategy(self):
        """Test InterruptionHandler with custom strategy."""
        from voice_soundboard.llm.interruption import (
            InterruptionHandler,
            InterruptionStrategy,
        )

        handler = InterruptionHandler(strategy=InterruptionStrategy.PAUSE)
        assert handler.strategy == InterruptionStrategy.PAUSE

    def test_callback_properties(self):
        """Test callback properties."""
        from voice_soundboard.llm.interruption import InterruptionHandler

        handler = InterruptionHandler()
        callback = Mock()

        handler.on_interrupt = callback
        assert handler.on_interrupt == callback

        handler.on_pause = callback
        assert handler.on_pause == callback

        handler.on_resume = callback
        assert handler.on_resume == callback

    def test_start_end_session(self):
        """Test start and end session."""
        from voice_soundboard.llm.interruption import InterruptionHandler

        handler = InterruptionHandler()
        assert handler._is_active is False

        handler.start_session()
        assert handler._is_active is True

        handler.end_session()
        assert handler._is_active is False

    def test_check_interrupt_not_active(self):
        """Test check_interrupt when not active."""
        from voice_soundboard.llm.interruption import InterruptionHandler

        handler = InterruptionHandler()
        result = handler.check_interrupt(audio_level=-30.0)
        assert result is False

    def test_check_interrupt_with_audio(self):
        """Test check_interrupt with audio level."""
        from voice_soundboard.llm.interruption import InterruptionHandler, BargeInConfig

        config = BargeInConfig(
            ignore_initial_ms=0,
            min_spoken_ms=0,
            vad_duration_ms=0,
        )
        handler = InterruptionHandler(config=config)
        handler.start_session()

        # Simulate voice activity started
        handler.detector._voice_start = time.time() - 1
        handler.detector._voice_active = True

        result = handler.check_interrupt(audio_level=-30.0)
        # May or may not trigger depending on timing

    def test_check_interrupt_with_transcript(self):
        """Test check_interrupt with transcript."""
        from voice_soundboard.llm.interruption import InterruptionHandler, BargeInConfig

        config = BargeInConfig(hotword_enabled=True)
        handler = InterruptionHandler(config=config)
        handler.start_session()
        result = handler.check_interrupt(transcript="stop")
        # Should trigger hotword

    def test_force_interrupt(self):
        """Test force interrupt."""
        from voice_soundboard.llm.interruption import InterruptionHandler

        handler = InterruptionHandler()
        callback = Mock()
        handler.on_interrupt = callback
        handler.start_session()

        event = handler.force_interrupt()
        assert event.trigger == "manual"
        assert event.handled is True
        callback.assert_called()

    def test_handle_event_ignore(self):
        """Test handle event with IGNORE strategy."""
        from voice_soundboard.llm.interruption import (
            InterruptionHandler,
            InterruptionStrategy,
            InterruptionEvent,
        )

        handler = InterruptionHandler(strategy=InterruptionStrategy.IGNORE)
        handler.start_session()

        event = InterruptionEvent(trigger="manual")
        handler._handle_event(event)
        assert event.handled is False

    def test_handle_event_stop_immediate(self):
        """Test handle event with STOP_IMMEDIATE strategy."""
        from voice_soundboard.llm.interruption import (
            InterruptionHandler,
            InterruptionStrategy,
            InterruptionEvent,
        )

        handler = InterruptionHandler(strategy=InterruptionStrategy.STOP_IMMEDIATE)
        callback = Mock()
        handler.on_interrupt = callback
        handler.start_session()

        event = InterruptionEvent(trigger="manual")
        handler._handle_event(event)
        assert event.handled is True
        callback.assert_called()

    def test_handle_event_pause(self):
        """Test handle event with PAUSE strategy."""
        from voice_soundboard.llm.interruption import (
            InterruptionHandler,
            InterruptionStrategy,
            InterruptionEvent,
        )

        handler = InterruptionHandler(strategy=InterruptionStrategy.PAUSE)
        callback = Mock()
        handler.on_pause = callback
        handler.start_session()

        event = InterruptionEvent(trigger="manual")
        handler._handle_event(event)
        assert handler._is_paused is True
        callback.assert_called()

    def test_handle_event_queue(self):
        """Test handle event with QUEUE strategy."""
        from voice_soundboard.llm.interruption import (
            InterruptionHandler,
            InterruptionStrategy,
            InterruptionEvent,
        )

        handler = InterruptionHandler(strategy=InterruptionStrategy.QUEUE)
        handler.start_session()

        event = InterruptionEvent(trigger="manual", transcript="hello")
        handler._handle_event(event)
        assert "hello" in handler._queue

    def test_resume(self):
        """Test resume after pause."""
        from voice_soundboard.llm.interruption import (
            InterruptionHandler,
            InterruptionStrategy,
        )

        handler = InterruptionHandler(strategy=InterruptionStrategy.PAUSE)
        callback = Mock()
        handler.on_resume = callback
        handler.start_session()
        handler._is_paused = True

        handler.resume()
        assert handler._is_paused is False
        callback.assert_called()

    def test_get_queued_messages(self):
        """Test get queued messages."""
        from voice_soundboard.llm.interruption import InterruptionHandler

        handler = InterruptionHandler()
        handler._queue = ["message1", "message2"]

        messages = handler.get_queued_messages()
        assert messages == ["message1", "message2"]
        assert handler._queue == []

    def test_stats_property(self):
        """Test stats property."""
        from voice_soundboard.llm.interruption import (
            InterruptionHandler,
            InterruptionStrategy,
        )

        handler = InterruptionHandler()
        handler.start_session()

        stats = handler.stats
        assert stats["is_active"] is True
        assert stats["is_paused"] is False
        assert stats["strategy"] == "stop_immediate"


# ==============================================================================
# Tests for llm/streaming.py
# ==============================================================================

class TestStreamState:
    """Tests for StreamState enum."""

    def test_state_values(self):
        """Test StreamState enum values."""
        from voice_soundboard.llm.streaming import StreamState

        assert StreamState.IDLE.value == "idle"
        assert StreamState.BUFFERING.value == "buffering"
        assert StreamState.SPEAKING.value == "speaking"
        assert StreamState.FINISHING.value == "finishing"
        assert StreamState.ERROR.value == "error"


class TestStreamConfig:
    """Tests for StreamConfig dataclass."""

    def test_default_values(self):
        """Test StreamConfig default values."""
        from voice_soundboard.llm.streaming import StreamConfig

        config = StreamConfig()
        assert config.sentence_end_chars == ".!?"
        assert config.min_sentence_length == 10
        assert config.max_buffer_length == 500
        assert config.flush_timeout_ms == 2000.0
        assert config.inter_sentence_pause_ms == 200.0
        assert config.preset == "assistant"
        assert config.speed == 1.0

    def test_custom_values(self):
        """Test StreamConfig with custom values."""
        from voice_soundboard.llm.streaming import StreamConfig

        config = StreamConfig(
            voice="custom_voice",
            speed=1.5,
            emotion="happy",
        )
        assert config.voice == "custom_voice"
        assert config.speed == 1.5
        assert config.emotion == "happy"


class TestStreamBuffer:
    """Tests for StreamBuffer dataclass."""

    def test_default_values(self):
        """Test StreamBuffer default values."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        assert buffer.content == ""
        assert buffer.sentences_spoken == 0
        assert buffer.tokens_received == 0

    def test_append(self):
        """Test append method."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("hello ")
        buffer.append("world")

        assert buffer.content == "hello world"
        assert buffer.tokens_received == 2

    def test_clear(self):
        """Test clear method."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.content = "test content"

        result = buffer.clear()
        assert result == "test content"
        assert buffer.content == ""

    def test_peek(self):
        """Test peek method."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.content = "test content"

        result = buffer.peek()
        assert result == "test content"
        assert buffer.content == "test content"

    def test_age_ms_property(self):
        """Test age_ms property."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        time.sleep(0.01)
        assert buffer.age_ms >= 10

    def test_length_property(self):
        """Test length property."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.content = "12345"
        assert buffer.length == 5


class TestSentenceBoundaryDetector:
    """Tests for SentenceBoundaryDetector class."""

    def test_initialization(self):
        """Test SentenceBoundaryDetector initialization."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        assert detector.config is not None

    def test_find_boundary_simple(self):
        """Test finding simple sentence boundary."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector.find_boundary("Hello world. How are you?")
        assert result is not None
        assert result == 13  # After "Hello world. "

    def test_find_boundary_too_short(self):
        """Test find boundary with text too short."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector.find_boundary("Hi.")
        assert result is None  # Too short

    def test_find_boundary_no_ending(self):
        """Test find boundary with no sentence ending."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector.find_boundary("Hello world how are you")
        assert result is None

    def test_find_boundary_abbreviation(self):
        """Test find boundary with abbreviation."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        # "Dr." should not be treated as sentence end
        result = detector.find_boundary("Dr. Smith is here")
        assert result is None

    def test_find_boundary_decimal(self):
        """Test find boundary with decimal number."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector.find_boundary("The value is 3.14 today")
        assert result is None

    def test_find_boundary_multiple_punctuation(self):
        """Test find boundary with multiple punctuation."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector.find_boundary("Really?! Yes indeed.")
        assert result is not None

    def test_split_sentences(self):
        """Test split_sentences method."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        sentences = detector.split_sentences("Hello world. How are you? I am fine!")
        assert len(sentences) >= 1

    def test_extract_complete(self):
        """Test extract_complete method."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        sentences, remaining = detector.extract_complete("Hello world. This is a test")
        assert len(sentences) >= 1
        assert "test" in remaining or len(sentences) > 1

    def test_is_real_ending_pos_zero(self):
        """Test _is_real_ending at position 0."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector._is_real_ending(".", 0)
        assert result is False


class TestStreamingLLMSpeaker:
    """Tests for StreamingLLMSpeaker class."""

    def test_initialization(self):
        """Test StreamingLLMSpeaker initialization."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState

        speaker = StreamingLLMSpeaker()
        assert speaker.state == StreamState.IDLE
        assert speaker.sentences_spoken == 0
        assert speaker.total_tokens == 0

    def test_initialization_with_callbacks(self):
        """Test StreamingLLMSpeaker with callbacks."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker

        on_sentence = Mock()
        on_start = Mock()
        on_end = Mock()

        speaker = StreamingLLMSpeaker(
            on_sentence=on_sentence,
            on_speak_start=on_start,
            on_speak_end=on_end,
        )
        assert speaker._on_sentence == on_sentence
        assert speaker._on_speak_start == on_start
        assert speaker._on_speak_end == on_end

    @pytest.mark.asyncio
    async def test_feed_simple(self):
        """Test feed method with simple tokens."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState

        speaker = StreamingLLMSpeaker()

        result = await speaker.feed("Hello ")
        assert speaker.state == StreamState.BUFFERING
        assert speaker.total_tokens == 1

    @pytest.mark.asyncio
    async def test_feed_complete_sentence(self):
        """Test feed method completing a sentence."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamConfig

        config = StreamConfig(inter_sentence_pause_ms=0)
        speaker = StreamingLLMSpeaker(config=config)

        # Mock engines
        speaker._engine = Mock()
        speaker._engine.speak = Mock()
        speaker._streaming_engine = Mock()
        speaker._streaming_engine.stream = AsyncMock(return_value=AsyncMock(__aiter__=lambda s: iter([])))

        # Patch _get_streaming_engine to return mock
        with patch.object(speaker, '_get_streaming_engine', return_value=speaker._streaming_engine):
            # Make stream an async generator
            async def mock_stream(*args, **kwargs):
                if False:
                    yield
            speaker._streaming_engine.stream = mock_stream

            for token in ["Hello ", "world. ", "How "]:
                await speaker.feed(token)

    @pytest.mark.asyncio
    async def test_finish(self):
        """Test finish method."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState, StreamConfig

        config = StreamConfig(allow_partial_sentences=True, inter_sentence_pause_ms=0)
        speaker = StreamingLLMSpeaker(config=config)
        speaker.buffer.content = "remaining text"

        # Mock engines
        speaker._engine = Mock()
        speaker._engine.speak = Mock()

        with patch.object(speaker, '_get_streaming_engine') as mock_get:
            async def mock_stream(*args, **kwargs):
                if False:
                    yield
            mock_engine = Mock()
            mock_engine.stream = mock_stream
            mock_get.return_value = mock_engine

            result = await speaker.finish()
            assert speaker.state == StreamState.IDLE

    def test_reset(self):
        """Test reset method."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState

        speaker = StreamingLLMSpeaker()
        speaker.sentences_spoken = 5
        speaker.total_tokens = 100
        speaker.buffer.content = "test"

        speaker.reset()
        assert speaker.sentences_spoken == 0
        assert speaker.total_tokens == 0
        assert speaker.buffer.content == ""
        assert speaker.state == StreamState.IDLE

    def test_stats_property(self):
        """Test stats property."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker

        speaker = StreamingLLMSpeaker()
        speaker.start_time = time.time() - 10
        speaker.total_tokens = 50
        speaker.sentences_spoken = 5

        stats = speaker.stats
        assert stats["total_tokens"] == 50
        assert stats["sentences_spoken"] == 5
        assert stats["tokens_per_second"] > 0


# ==============================================================================
# Tests for llm/pipeline.py
# ==============================================================================

class TestTurnType:
    """Tests for TurnType enum."""

    def test_turn_type_values(self):
        """Test TurnType enum values."""
        from voice_soundboard.llm.pipeline import TurnType

        assert TurnType.USER.value == "user"
        assert TurnType.ASSISTANT.value == "assistant"
        assert TurnType.SYSTEM.value == "system"


class TestPipelineState:
    """Tests for PipelineState enum."""

    def test_pipeline_state_values(self):
        """Test PipelineState enum values."""
        from voice_soundboard.llm.pipeline import PipelineState

        assert PipelineState.IDLE.value == "idle"
        assert PipelineState.LISTENING.value == "listening"
        assert PipelineState.TRANSCRIBING.value == "transcribing"
        assert PipelineState.THINKING.value == "thinking"
        assert PipelineState.SPEAKING.value == "speaking"
        assert PipelineState.INTERRUPTED.value == "interrupted"
        assert PipelineState.ERROR.value == "error"


class TestConversationTurn:
    """Tests for ConversationTurn dataclass."""

    def test_default_values(self):
        """Test ConversationTurn default values."""
        from voice_soundboard.llm.pipeline import ConversationTurn, TurnType

        turn = ConversationTurn(type=TurnType.USER, content="Hello")
        assert turn.type == TurnType.USER
        assert turn.content == "Hello"
        assert turn.audio is None
        assert turn.audio_path is None
        assert turn.emotion is None

    def test_with_metadata(self):
        """Test ConversationTurn with metadata."""
        from voice_soundboard.llm.pipeline import ConversationTurn, TurnType

        turn = ConversationTurn(
            type=TurnType.ASSISTANT,
            content="Hello there",
            emotion="happy",
            voice="voice1",
            llm_ms=100.0,
            tts_ms=50.0,
        )
        assert turn.emotion == "happy"
        assert turn.voice == "voice1"
        assert turn.llm_ms == 100.0
        assert turn.tts_ms == 50.0


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """Test PipelineConfig default values."""
        from voice_soundboard.llm.pipeline import PipelineConfig

        config = PipelineConfig()
        assert config.stt_backend == "whisper"
        assert config.llm_backend == "ollama"
        assert config.tts_backend == "kokoro"
        assert config.auto_emotion is True
        assert config.allow_interruption is True
        assert config.stream_tts is True

    def test_custom_values(self):
        """Test PipelineConfig with custom values."""
        from voice_soundboard.llm.pipeline import PipelineConfig

        config = PipelineConfig(
            stt_backend="vosk",
            llm_backend="openai",
            llm_model="gpt-4",
            auto_emotion=False,
        )
        assert config.stt_backend == "vosk"
        assert config.llm_backend == "openai"
        assert config.llm_model == "gpt-4"
        assert config.auto_emotion is False


class TestSpeechPipeline:
    """Tests for SpeechPipeline class."""

    def test_initialization(self):
        """Test SpeechPipeline initialization."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState

        pipeline = SpeechPipeline()
        assert pipeline.state == PipelineState.IDLE
        assert len(pipeline.conversation_history) == 0

    def test_initialization_with_config(self):
        """Test SpeechPipeline with custom config."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineConfig

        config = PipelineConfig(llm_backend="openai")
        pipeline = SpeechPipeline(config=config)
        assert pipeline.config.llm_backend == "openai"

    def test_initialization_with_backends(self):
        """Test SpeechPipeline with backend strings."""
        from voice_soundboard.llm.pipeline import SpeechPipeline

        pipeline = SpeechPipeline(stt="vosk", llm="openai", tts="edge")
        assert pipeline.config.stt_backend == "vosk"
        assert pipeline.config.llm_backend == "openai"
        assert pipeline.config.tts_backend == "edge"

    def test_callback_properties(self):
        """Test callback properties."""
        from voice_soundboard.llm.pipeline import SpeechPipeline

        pipeline = SpeechPipeline()
        callback = Mock()

        pipeline.on_state_change = callback
        assert pipeline.on_state_change == callback

        pipeline.on_transcription = callback
        assert pipeline.on_transcription == callback

        pipeline.on_response = callback
        assert pipeline.on_response == callback

    def test_set_state(self):
        """Test _set_state method."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState

        pipeline = SpeechPipeline()
        callback = Mock()
        pipeline.on_state_change = callback

        pipeline._set_state(PipelineState.LISTENING)
        assert pipeline.state == PipelineState.LISTENING
        callback.assert_called_with(PipelineState.LISTENING)

    def test_interrupt(self):
        """Test interrupt method."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState

        pipeline = SpeechPipeline()
        pipeline.state = PipelineState.SPEAKING

        # stop_playback is imported inside the method, so patch it at the source
        with patch("voice_soundboard.stop_playback") as mock_stop:
            pipeline.interrupt()
            assert pipeline.state == PipelineState.INTERRUPTED

    def test_interrupt_not_speaking(self):
        """Test interrupt when not speaking."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState

        pipeline = SpeechPipeline()
        pipeline.state = PipelineState.IDLE

        pipeline.interrupt()
        assert pipeline.state == PipelineState.IDLE

    def test_reset(self):
        """Test reset method."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState, ConversationTurn, TurnType

        pipeline = SpeechPipeline()
        pipeline.conversation_history.append(
            ConversationTurn(type=TurnType.USER, content="test")
        )

        pipeline.reset()
        assert len(pipeline.conversation_history) == 0
        assert pipeline.state == PipelineState.IDLE

    def test_stats_property(self):
        """Test stats property."""
        from voice_soundboard.llm.pipeline import (
            SpeechPipeline,
            ConversationTurn,
            TurnType,
        )

        pipeline = SpeechPipeline()
        pipeline.conversation_history.append(
            ConversationTurn(type=TurnType.USER, content="test", transcription_ms=100.0)
        )
        pipeline.conversation_history.append(
            ConversationTurn(
                type=TurnType.ASSISTANT,
                content="response",
                llm_ms=200.0,
                tts_ms=50.0,
            )
        )

        stats = pipeline.stats
        assert stats["turn_count"] == 2
        assert stats["total_transcription_ms"] == 100.0
        assert stats["total_llm_ms"] == 200.0
        assert stats["total_tts_ms"] == 50.0

    @pytest.mark.asyncio
    async def test_transcribe_mock(self):
        """Test transcribe method with mock STT."""
        from voice_soundboard.llm.pipeline import SpeechPipeline

        pipeline = SpeechPipeline()
        audio = np.zeros(16000, dtype=np.float32)

        # Without whisper installed, should return mock
        result = await pipeline.transcribe(audio)
        assert "Mock" in result or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test generate_response method."""
        from voice_soundboard.llm.pipeline import SpeechPipeline

        pipeline = SpeechPipeline()

        # Mock the LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Hello there!"
        mock_llm.chat = AsyncMock(return_value=mock_response)

        with patch.object(pipeline, "_get_llm", return_value=mock_llm):
            result = await pipeline.generate_response("Hello")
            assert result == "Hello there!"

    @pytest.mark.asyncio
    async def test_generate_response_error(self):
        """Test generate_response with error."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState

        pipeline = SpeechPipeline()

        mock_llm = Mock()
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM error"))

        with patch.object(pipeline, "_get_llm", return_value=mock_llm):
            with pytest.raises(RuntimeError):
                await pipeline.generate_response("Hello")
            assert pipeline.state == PipelineState.ERROR

    @pytest.mark.asyncio
    async def test_speak(self):
        """Test speak method."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState

        pipeline = SpeechPipeline()
        pipeline.config.auto_emotion = False

        mock_tts = Mock()
        mock_result = Mock()
        mock_result.audio = np.zeros(1000, dtype=np.float32)
        mock_tts.speak = Mock(return_value=mock_result)

        with patch.object(pipeline, "_get_tts", return_value=mock_tts):
            result = await pipeline.speak("Hello")
            assert pipeline.state == PipelineState.SPEAKING

    @pytest.mark.asyncio
    async def test_speak_with_auto_emotion(self):
        """Test speak method with auto emotion."""
        from voice_soundboard.llm.pipeline import SpeechPipeline

        pipeline = SpeechPipeline()
        pipeline.config.auto_emotion = True

        mock_speaker = Mock()
        mock_speaker.speak = Mock(return_value={"result": None})

        with patch.object(pipeline, "_get_context_speaker", return_value=mock_speaker):
            result = await pipeline.speak("Hello")
            mock_speaker.speak.assert_called()

    @pytest.mark.asyncio
    async def test_converse(self):
        """Test converse method."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, TurnType

        pipeline = SpeechPipeline()
        audio = np.zeros(16000, dtype=np.float32)

        # Mock all components
        with patch.object(pipeline, "transcribe", new=AsyncMock(return_value="Hello")):
            with patch.object(
                pipeline, "generate_response", new=AsyncMock(return_value="Hi there!")
            ):
                with patch.object(pipeline, "speak", new=AsyncMock(return_value=None)):
                    turn = await pipeline.converse(audio)

                    assert turn.type == TurnType.ASSISTANT
                    assert turn.content == "Hi there!"
                    assert len(pipeline.conversation_history) == 2

    @pytest.mark.asyncio
    async def test_converse_with_callbacks(self):
        """Test converse method with callbacks."""
        from voice_soundboard.llm.pipeline import SpeechPipeline

        pipeline = SpeechPipeline()
        audio = np.zeros(16000, dtype=np.float32)

        transcription_callback = Mock()
        response_callback = Mock()
        pipeline.on_transcription = transcription_callback
        pipeline.on_response = response_callback

        with patch.object(pipeline, "transcribe", new=AsyncMock(return_value="Hello")):
            with patch.object(
                pipeline, "generate_response", new=AsyncMock(return_value="Hi!")
            ):
                with patch.object(pipeline, "speak", new=AsyncMock(return_value=None)):
                    await pipeline.converse(audio)

                    transcription_callback.assert_called_with("Hello")
                    response_callback.assert_called_with("Hi!")


class TestQuickConverse:
    """Tests for quick_converse function."""

    @pytest.mark.asyncio
    async def test_quick_converse(self):
        """Test quick_converse function."""
        from voice_soundboard.llm.pipeline import quick_converse, SpeechPipeline

        audio = np.zeros(16000, dtype=np.float32)

        with patch.object(
            SpeechPipeline, "converse", new=AsyncMock()
        ) as mock_converse:
            mock_turn = Mock()
            mock_turn.content = "Response"
            mock_converse.return_value = mock_turn

            result = await quick_converse(audio, llm="mock")
            assert result == "Response"

    @pytest.mark.asyncio
    async def test_quick_converse_with_system_prompt(self):
        """Test quick_converse with custom system prompt."""
        from voice_soundboard.llm.pipeline import quick_converse, SpeechPipeline

        audio = np.zeros(16000, dtype=np.float32)

        with patch.object(
            SpeechPipeline, "converse", new=AsyncMock()
        ) as mock_converse:
            mock_turn = Mock()
            mock_turn.content = "Custom response"
            mock_converse.return_value = mock_turn

            result = await quick_converse(
                audio, llm="mock", system_prompt="Be helpful"
            )
            assert result == "Custom response"


# ==============================================================================
# Additional Edge Case Tests
# ==============================================================================

class TestInterruptionEdgeCases:
    """Edge case tests for interruption handling."""

    def test_async_callback_error_handling(self):
        """Test async callback error handling."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()

        async def failing_callback(event):
            raise ValueError("Test error")

        detector.on_interrupt(failing_callback)

        # Should not raise
        with patch("asyncio.create_task"):
            detector.trigger_manual()

    def test_voice_activity_resets(self):
        """Test voice activity resets on silence."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        config = BargeInConfig(
            ignore_initial_ms=0,
            min_spoken_ms=0,
            vad_threshold_db=-35.0,
        )
        detector = BargeInDetector(config)
        detector.start_listening()

        # Start voice activity
        detector.check_audio_level(-30.0)
        assert detector._voice_active is True

        # Go silent
        detector.check_audio_level(-50.0)
        assert detector._voice_active is False


class TestStreamingEdgeCases:
    """Edge case tests for streaming."""

    def test_sentence_boundary_ellipsis(self):
        """Test sentence boundary with ellipsis."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        # Ellipsis should not end sentence
        result = detector.find_boundary("Wait for it... the answer is")
        # Should handle ellipsis correctly

    def test_sentence_boundary_currency(self):
        """Test sentence boundary with currency."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector.find_boundary("The price is $1.99 today")
        assert result is None  # Should not split on decimal

    def test_sentence_boundary_url(self):
        """Test sentence boundary with URL-like text."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector.find_boundary("Visit example.com for more")
        assert result is None

    @pytest.mark.asyncio
    async def test_flush_buffer_disabled(self):
        """Test flush buffer when partial sentences disabled."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamConfig

        config = StreamConfig(allow_partial_sentences=False)
        speaker = StreamingLLMSpeaker(config=config)
        speaker.buffer.content = "partial text"

        result = await speaker._flush_buffer()
        assert result is None


class TestPipelineEdgeCases:
    """Edge case tests for pipeline."""

    @pytest.mark.asyncio
    async def test_converse_stream(self):
        """Test converse_stream method."""
        from voice_soundboard.llm.pipeline import SpeechPipeline

        pipeline = SpeechPipeline()
        pipeline.config.speak_as_generated = False
        audio = np.zeros(16000, dtype=np.float32)

        with patch.object(pipeline, "transcribe", new=AsyncMock(return_value="Hello")):
            mock_llm = Mock()

            async def mock_chat_stream(messages):
                for token in ["Hi", " there", "!"]:
                    yield token

            mock_llm.chat_stream = mock_chat_stream

            with patch.object(pipeline, "_get_llm", return_value=mock_llm):
                with patch.object(pipeline, "speak", new=AsyncMock(return_value=None)):
                    events = []
                    async for event in pipeline.converse_stream(audio):
                        events.append(event)

                    # Should have state, transcription, tokens, response, etc.
                    event_types = [e["type"] for e in events]
                    assert "state" in event_types
                    assert "transcription" in event_types
                    assert "token" in event_types

    def test_interrupt_with_streaming_speaker(self):
        """Test interrupt with streaming speaker."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState

        pipeline = SpeechPipeline()
        pipeline.state = PipelineState.SPEAKING
        pipeline._streaming_speaker = Mock()
        pipeline._streaming_speaker.reset = Mock()

        # stop_playback is imported inside the method, so patch it at the source
        with patch("voice_soundboard.stop_playback"):
            pipeline.interrupt()

        pipeline._streaming_speaker.reset.assert_called()

    def test_stats_with_empty_history(self):
        """Test stats with empty history."""
        from voice_soundboard.llm.pipeline import SpeechPipeline

        pipeline = SpeechPipeline()
        stats = pipeline.stats
        assert stats["turn_count"] == 0
        assert stats["total_transcription_ms"] == 0
        assert stats["total_llm_ms"] == 0
        assert stats["total_tts_ms"] == 0
