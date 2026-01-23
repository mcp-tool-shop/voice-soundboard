"""
Tests for LLM Integration module.

Run with: pytest tests/test_llm.py -v
"""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, patch, AsyncMock


# =============================================================================
# Module: streaming.py
# =============================================================================

class TestSentenceBoundaryDetector:
    """Tests for sentence boundary detection."""

    def test_find_simple_boundary(self):
        """Test finding simple sentence boundary."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector.find_boundary("Hello world. This is a test.")

        assert result is not None
        assert result == 13  # After "Hello world. "

    def test_find_question_boundary(self):
        """Test finding question boundary."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector.find_boundary("How are you? I'm fine.")

        assert result is not None
        assert result == 13  # After "How are you? "

    def test_find_exclamation_boundary(self):
        """Test finding exclamation boundary."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector.find_boundary("Wow! That's amazing!")

        assert result is not None
        assert result == 5  # After "Wow! "

    def test_no_boundary_short_text(self):
        """Test no boundary for short text."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector.find_boundary("Hi.")

        # Too short to be a sentence (min_sentence_length default is 10)
        assert result is None

    def test_abbreviation_not_boundary(self):
        """Test abbreviations don't trigger boundary."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector.find_boundary("Dr. Smith went to the store.")

        # Should find boundary at end, not after "Dr."
        assert result is not None
        assert result > 10  # After the full sentence

    def test_decimal_not_boundary(self):
        """Test decimal numbers don't trigger boundary."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        result = detector.find_boundary("The price is 3.14 dollars.")

        # Should find boundary at end, not after "3."
        assert result is not None
        assert "dollars" in "The price is 3.14 dollars."[:result]

    def test_extract_complete_sentences(self):
        """Test extracting complete sentences."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        sentences, remaining = detector.extract_complete(
            "Hello world. This is a test. More text"
        )

        assert len(sentences) == 2
        assert "Hello world." in sentences[0]
        assert "This is a test." in sentences[1]
        assert "More text" in remaining

    def test_split_sentences(self):
        """Test splitting text into sentences."""
        from voice_soundboard.llm.streaming import SentenceBoundaryDetector

        detector = SentenceBoundaryDetector()
        # Use longer sentences that meet the min_sentence_length requirement
        sentences = detector.split_sentences(
            "This is the first sentence here. And this is the second one. And here is the third."
        )

        # Should find at least 2 complete sentences
        assert len(sentences) >= 2


class TestStreamBuffer:
    """Tests for stream buffer."""

    def test_buffer_append(self):
        """Test appending to buffer."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("Hello ")
        buffer.append("world")

        assert buffer.content == "Hello world"
        assert buffer.tokens_received == 2

    def test_buffer_clear(self):
        """Test clearing buffer."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("test")

        content = buffer.clear()

        assert content == "test"
        assert buffer.content == ""

    def test_buffer_peek(self):
        """Test peeking at buffer."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("test")

        content = buffer.peek()

        assert content == "test"
        assert buffer.content == "test"  # Still there

    def test_buffer_length(self):
        """Test buffer length property."""
        from voice_soundboard.llm.streaming import StreamBuffer

        buffer = StreamBuffer()
        buffer.append("12345")

        assert buffer.length == 5


class TestStreamConfig:
    """Tests for stream configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from voice_soundboard.llm.streaming import StreamConfig

        config = StreamConfig()

        assert config.sentence_end_chars == ".!?"
        assert config.min_sentence_length == 10
        assert config.max_buffer_length == 500
        assert config.speed == 1.0
        assert config.preset == "assistant"

    def test_custom_config(self):
        """Test custom configuration."""
        from voice_soundboard.llm.streaming import StreamConfig

        config = StreamConfig(
            min_sentence_length=5,
            speed=1.5,
            voice="af_bella",
        )

        assert config.min_sentence_length == 5
        assert config.speed == 1.5
        assert config.voice == "af_bella"


class TestStreamingLLMSpeaker:
    """Tests for streaming LLM speaker."""

    def test_speaker_init(self):
        """Test speaker initialization."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState

        speaker = StreamingLLMSpeaker()

        assert speaker.state == StreamState.IDLE
        assert speaker.sentences_spoken == 0
        assert speaker.total_tokens == 0

    def test_speaker_reset(self):
        """Test speaker reset."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamState

        speaker = StreamingLLMSpeaker()
        speaker.total_tokens = 100
        speaker.sentences_spoken = 5

        speaker.reset()

        assert speaker.state == StreamState.IDLE
        assert speaker.total_tokens == 0
        assert speaker.sentences_spoken == 0

    def test_speaker_stats(self):
        """Test speaker statistics."""
        from voice_soundboard.llm.streaming import StreamingLLMSpeaker

        speaker = StreamingLLMSpeaker()
        speaker.start_time = time.time()
        speaker.total_tokens = 50
        speaker.sentences_spoken = 3

        stats = speaker.stats

        assert stats["total_tokens"] == 50
        assert stats["sentences_spoken"] == 3
        assert "state" in stats
        assert "elapsed_seconds" in stats


# =============================================================================
# Module: context.py
# =============================================================================

class TestEmotionSelector:
    """Tests for emotion selection."""

    def test_detect_frustration(self):
        """Test detecting user frustration."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        sentiment, emotion = selector.detect_user_sentiment(
            "This is so frustrating! It doesn't work!"
        )

        assert sentiment == "negative"
        assert emotion == "frustrated"

    def test_detect_happiness(self):
        """Test detecting user happiness."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        sentiment, emotion = selector.detect_user_sentiment(
            "Thanks so much! This is great!"
        )

        assert sentiment == "positive"
        assert emotion == "happy"

    def test_detect_confusion(self):
        """Test detecting user confusion."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        sentiment, emotion = selector.detect_user_sentiment(
            "I don't understand what you mean. I'm confused."
        )

        assert sentiment == "negative"
        assert emotion == "confused"

    def test_detect_neutral(self):
        """Test detecting neutral sentiment."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        sentiment, emotion = selector.detect_user_sentiment(
            "Please tell me about the weather."
        )

        assert sentiment == "neutral"
        assert emotion == "neutral"

    def test_select_emotion_from_text(self):
        """Test selecting emotion from response text."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        emotion, confidence = selector.select_emotion(
            "I'm so sorry to hear that. That must be really difficult."
        )

        assert emotion == "sympathetic"
        assert confidence > 0

    def test_select_emotion_excited(self):
        """Test selecting excited emotion."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        emotion, confidence = selector.select_emotion(
            "Wow! That's incredible! This is so exciting!"
        )

        assert emotion == "excited"
        assert confidence > 0


class TestConversationContext:
    """Tests for conversation context."""

    def test_add_message(self):
        """Test adding messages to context."""
        from voice_soundboard.llm.context import ConversationContext

        context = ConversationContext()
        context.add_message("user", "Hello")
        context.add_message("assistant", "Hi there!")

        assert len(context.messages) == 2
        assert context.turn_count == 2

    def test_get_recent_messages(self):
        """Test getting recent messages."""
        from voice_soundboard.llm.context import ConversationContext

        context = ConversationContext()
        for i in range(10):
            context.add_message("user", f"Message {i}")

        recent = context.get_recent_messages(5)

        assert len(recent) == 5
        assert "Message 9" in recent[-1]["content"]

    def test_get_last_user_message(self):
        """Test getting last user message."""
        from voice_soundboard.llm.context import ConversationContext

        context = ConversationContext()
        context.add_message("user", "First")
        context.add_message("assistant", "Response")
        context.add_message("user", "Second")

        last = context.get_last_user_message()

        assert last == "Second"


class TestContextConfig:
    """Tests for context configuration."""

    def test_default_config(self):
        """Test default context configuration."""
        from voice_soundboard.llm.context import ContextConfig

        config = ContextConfig()

        assert config.enable_auto_emotion is True
        assert config.default_emotion == "neutral"
        assert config.default_preset == "assistant"

    def test_custom_config(self):
        """Test custom context configuration."""
        from voice_soundboard.llm.context import ContextConfig

        config = ContextConfig(
            enable_auto_emotion=False,
            default_emotion="friendly",
        )

        assert config.enable_auto_emotion is False
        assert config.default_emotion == "friendly"


class TestContextAwareSpeaker:
    """Tests for context-aware speaker."""

    def test_speaker_init(self):
        """Test speaker initialization."""
        from voice_soundboard.llm.context import ContextAwareSpeaker

        speaker = ContextAwareSpeaker()

        assert speaker.context is not None
        assert speaker.emotion_selector is not None

    def test_update_context(self):
        """Test updating context with user message."""
        from voice_soundboard.llm.context import ContextAwareSpeaker

        speaker = ContextAwareSpeaker()
        speaker.update_context("I'm really frustrated with this problem!")

        assert speaker.context.user_emotion == "frustrated"
        assert speaker.context.user_sentiment == "negative"

    def test_reset_context(self):
        """Test resetting context."""
        from voice_soundboard.llm.context import ContextAwareSpeaker

        speaker = ContextAwareSpeaker()
        speaker.update_context("Test message")

        speaker.reset_context()

        assert len(speaker.context.messages) == 0


# =============================================================================
# Module: providers.py
# =============================================================================

class TestLLMConfig:
    """Tests for LLM configuration."""

    def test_default_config(self):
        """Test default LLM configuration."""
        from voice_soundboard.llm.providers import LLMConfig

        config = LLMConfig()

        assert config.model == "llama3.2"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.stream is True

    def test_custom_config(self):
        """Test custom LLM configuration."""
        from voice_soundboard.llm.providers import LLMConfig

        config = LLMConfig(
            model="gpt-4",
            temperature=0.5,
            max_tokens=2048,
        )

        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048


class TestMockLLMProvider:
    """Tests for mock LLM provider."""

    @pytest.mark.asyncio
    async def test_mock_generate(self):
        """Test mock provider generation."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(default_response="Hello from mock!")
        response = await provider.generate("Test prompt")

        assert response.content == "Hello from mock!"
        assert response.provider == "mock"

    @pytest.mark.asyncio
    async def test_mock_generate_with_pattern(self):
        """Test mock provider with pattern matching."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            responses={"weather": "It's sunny today!"},
            default_response="I don't know.",
        )

        response = await provider.generate("What's the weather?")

        assert response.content == "It's sunny today!"

    @pytest.mark.asyncio
    async def test_mock_stream(self):
        """Test mock provider streaming."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            default_response="Hello world",
            token_delay_ms=10,
        )

        tokens = []
        async for token in provider.stream("Test"):
            tokens.append(token)

        assert len(tokens) == 2
        assert "".join(tokens).strip() == "Hello world"


class TestProviderFactory:
    """Tests for provider factory."""

    def test_create_mock_provider(self):
        """Test creating mock provider."""
        from voice_soundboard.llm.providers import create_provider, ProviderType

        provider = create_provider("mock")

        assert provider.provider_type == ProviderType.MOCK

    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        from voice_soundboard.llm.providers import create_provider, ProviderType

        provider = create_provider(ProviderType.OLLAMA)

        assert provider.provider_type == ProviderType.OLLAMA

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        from voice_soundboard.llm.providers import create_provider, ProviderType

        provider = create_provider("openai")

        assert provider.provider_type == ProviderType.OPENAI


# =============================================================================
# Module: interruption.py
# =============================================================================

class TestBargeInConfig:
    """Tests for barge-in configuration."""

    def test_default_config(self):
        """Test default barge-in configuration."""
        from voice_soundboard.llm.interruption import BargeInConfig

        config = BargeInConfig()

        assert config.enabled is True
        assert config.vad_threshold_db == -35.0
        assert config.vad_duration_ms == 200.0

    def test_custom_config(self):
        """Test custom barge-in configuration."""
        from voice_soundboard.llm.interruption import BargeInConfig

        config = BargeInConfig(
            enabled=False,
            vad_threshold_db=-40.0,
        )

        assert config.enabled is False
        assert config.vad_threshold_db == -40.0


class TestInterruptionEvent:
    """Tests for interruption events."""

    def test_event_creation(self):
        """Test creating interruption event."""
        from voice_soundboard.llm.interruption import InterruptionEvent

        event = InterruptionEvent(trigger="voice_activity", audio_level=-30.0)

        assert event.trigger == "voice_activity"
        assert event.audio_level == -30.0
        assert event.handled is False

    def test_event_age(self):
        """Test event age calculation."""
        from voice_soundboard.llm.interruption import InterruptionEvent
        import time

        event = InterruptionEvent()
        time.sleep(0.1)

        assert event.age_ms >= 100


class TestBargeInDetector:
    """Tests for barge-in detector."""

    def test_detector_init(self):
        """Test detector initialization."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()

        assert detector._is_listening is False

    def test_detector_start_stop(self):
        """Test starting and stopping detector."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()
        detector.start_listening()

        assert detector._is_listening is True

        detector.stop_listening()

        assert detector._is_listening is False

    def test_manual_trigger(self):
        """Test manual interruption trigger."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()
        detector.start_listening()

        event = detector.trigger_manual()

        assert event.trigger == "manual"


class TestInterruptionHandler:
    """Tests for interruption handler."""

    def test_handler_init(self):
        """Test handler initialization."""
        from voice_soundboard.llm.interruption import (
            InterruptionHandler, InterruptionStrategy
        )

        handler = InterruptionHandler(strategy=InterruptionStrategy.STOP_IMMEDIATE)

        assert handler.strategy == InterruptionStrategy.STOP_IMMEDIATE
        assert handler._is_active is False

    def test_handler_session(self):
        """Test handler session management."""
        from voice_soundboard.llm.interruption import InterruptionHandler

        handler = InterruptionHandler()
        handler.start_session()

        assert handler._is_active is True

        handler.end_session()

        assert handler._is_active is False

    def test_handler_stats(self):
        """Test handler statistics."""
        from voice_soundboard.llm.interruption import InterruptionHandler

        handler = InterruptionHandler()
        stats = handler.stats

        assert "is_active" in stats
        assert "total_interruptions" in stats
        assert "strategy" in stats


# =============================================================================
# Module: conversation.py
# =============================================================================

class TestMessage:
    """Tests for conversation messages."""

    def test_message_creation(self):
        """Test creating a message."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        msg = Message(role=MessageRole.USER, content="Hello")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.id is not None

    def test_message_to_dict(self):
        """Test message serialization."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        msg = Message(role=MessageRole.ASSISTANT, content="Hi there!", emotion="friendly")
        data = msg.to_dict()

        assert data["role"] == "assistant"
        assert data["content"] == "Hi there!"
        assert data["emotion"] == "friendly"

    def test_message_from_dict(self):
        """Test message deserialization."""
        from voice_soundboard.llm.conversation import Message

        data = {
            "role": "user",
            "content": "Test message",
        }
        msg = Message.from_dict(data)

        assert msg.content == "Test message"


class TestConversationManager:
    """Tests for conversation manager."""

    def test_manager_init(self):
        """Test manager initialization."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()

        assert manager.state == ConversationState.IDLE
        assert len(manager.messages) == 0

    def test_manager_start_end(self):
        """Test starting and ending conversation."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()

        assert manager.state == ConversationState.LISTENING
        assert manager.started_at is not None

        manager.end()

        assert manager.state == ConversationState.ENDED

    def test_manager_add_messages(self):
        """Test adding messages to conversation."""
        from voice_soundboard.llm.conversation import ConversationManager

        manager = ConversationManager()
        manager.start()

        user_msg = manager.add_user_message("Hello!")
        assistant_msg = manager.add_assistant_message("Hi there!")

        assert len(manager.messages) == 2
        assert user_msg.content == "Hello!"
        assert assistant_msg.content == "Hi there!"

    def test_manager_get_llm_context(self):
        """Test getting LLM context."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationConfig

        config = ConversationConfig(system_prompt="You are helpful.")
        manager = ConversationManager(config=config)
        manager.start()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi!")

        context = manager.get_llm_context()

        # Should include system message
        assert any(m["role"] == "system" for m in context)
        assert any(m["role"] == "user" for m in context)
        assert any(m["role"] == "assistant" for m in context)

    def test_manager_stats(self):
        """Test conversation statistics."""
        from voice_soundboard.llm.conversation import ConversationManager

        manager = ConversationManager()
        manager.start()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there!")

        stats = manager.stats

        assert stats["message_count"] == 2
        assert stats["user_message_count"] == 1
        assert stats["assistant_message_count"] == 1
        assert "duration_seconds" in stats


class TestTurnTakingController:
    """Tests for turn-taking controller."""

    def test_controller_init(self):
        """Test controller initialization."""
        from voice_soundboard.llm.conversation import TurnTakingController, TurnTakingStrategy

        controller = TurnTakingController(strategy=TurnTakingStrategy.SILENCE)

        assert controller.strategy == TurnTakingStrategy.SILENCE
        assert controller._is_user_speaking is False

    def test_controller_force_end_turn(self):
        """Test forcing end of turn."""
        from voice_soundboard.llm.conversation import TurnTakingController

        controller = TurnTakingController()
        controller._is_user_speaking = True

        controller.force_end_turn()

        assert controller._is_user_speaking is False

    def test_controller_reset(self):
        """Test controller reset."""
        from voice_soundboard.llm.conversation import TurnTakingController

        controller = TurnTakingController()
        controller._is_user_speaking = True
        controller._speech_start = time.time()

        controller.reset()

        assert controller._is_user_speaking is False
        assert controller._speech_start is None


# =============================================================================
# Module: pipeline.py
# =============================================================================

class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_default_config(self):
        """Test default pipeline configuration."""
        from voice_soundboard.llm.pipeline import PipelineConfig

        config = PipelineConfig()

        assert config.stt_backend == "whisper"
        assert config.llm_backend == "ollama"
        assert config.tts_backend == "kokoro"
        assert config.auto_emotion is True

    def test_custom_config(self):
        """Test custom pipeline configuration."""
        from voice_soundboard.llm.pipeline import PipelineConfig

        config = PipelineConfig(
            llm_backend="openai",
            llm_model="gpt-4",
            system_prompt="Be concise.",
        )

        assert config.llm_backend == "openai"
        assert config.llm_model == "gpt-4"
        assert config.system_prompt == "Be concise."


class TestConversationTurn:
    """Tests for conversation turns."""

    def test_turn_creation(self):
        """Test creating a conversation turn."""
        from voice_soundboard.llm.pipeline import ConversationTurn, TurnType

        turn = ConversationTurn(
            type=TurnType.USER,
            content="Hello!",
        )

        assert turn.type == TurnType.USER
        assert turn.content == "Hello!"
        assert turn.timestamp is not None


class TestSpeechPipeline:
    """Tests for speech pipeline."""

    def test_pipeline_init(self):
        """Test pipeline initialization."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, PipelineState

        pipeline = SpeechPipeline(llm="mock")

        assert pipeline.state == PipelineState.IDLE
        assert len(pipeline.conversation_history) == 0

    def test_pipeline_reset(self):
        """Test pipeline reset."""
        from voice_soundboard.llm.pipeline import SpeechPipeline, ConversationTurn, TurnType

        pipeline = SpeechPipeline(llm="mock")
        pipeline.conversation_history.append(
            ConversationTurn(type=TurnType.USER, content="Test")
        )

        pipeline.reset()

        assert len(pipeline.conversation_history) == 0

    def test_pipeline_stats(self):
        """Test pipeline statistics."""
        from voice_soundboard.llm.pipeline import SpeechPipeline

        pipeline = SpeechPipeline(llm="mock")
        stats = pipeline.stats

        assert "state" in stats
        assert "turn_count" in stats
        assert "config" in stats


# =============================================================================
# Integration Tests
# =============================================================================

class TestLLMIntegration:
    """Integration tests for LLM module."""

    def test_full_conversation_flow(self):
        """Test complete conversation flow."""
        from voice_soundboard.llm import (
            ConversationManager,
            EmotionSelector,
        )

        # Start conversation
        manager = ConversationManager()
        manager.start()

        # User speaks - use explicit frustration keywords
        user_msg = "This is so frustrating! The software doesn't work and I can't believe how annoying this is!"
        manager.add_user_message(user_msg)

        # Detect emotion
        selector = EmotionSelector()
        sentiment, emotion = selector.detect_user_sentiment(user_msg)

        # Should detect negative sentiment
        assert sentiment == "negative"
        assert emotion in ["frustrated", "angry"]

        # Generate context-aware response emotion
        # Create a context object for the selector
        from voice_soundboard.llm import ConversationContext
        context = ConversationContext()
        context.user_emotion = emotion
        context.user_sentiment = sentiment

        response = "I understand how frustrating that can be. Let me help you."
        response_emotion, _ = selector.select_emotion(response, context)

        assert response_emotion in ["sympathetic", "calm", "neutral"]

        # Add assistant response
        manager.add_assistant_message(response, emotion=response_emotion)

        # Verify conversation
        assert len(manager.messages) == 2
        stats = manager.stats
        assert stats["turn_count"] == 2

    def test_context_aware_emotion_selection(self):
        """Test context-aware emotion selection flow."""
        from voice_soundboard.llm import (
            ConversationContext,
            EmotionSelector,
        )

        # Set up context with frustrated user
        context = ConversationContext()
        context.user_emotion = "frustrated"
        context.user_sentiment = "negative"

        # Select emotion for response
        selector = EmotionSelector()
        emotion, confidence = selector.select_emotion(
            "I'm sorry you're having trouble. Let me help.",
            context,
        )

        # Should respond with sympathy
        assert emotion in ["sympathetic", "calm"]
        assert confidence > 0

    @pytest.mark.asyncio
    async def test_mock_llm_conversation(self):
        """Test conversation with mock LLM."""
        from voice_soundboard.llm import MockLLMProvider

        provider = MockLLMProvider(
            responses={
                "hello": "Hi there! How can I help you today?",
                "weather": "I don't have access to weather information.",
            },
            default_response="I'm here to help!",
        )

        # Test greeting
        response = await provider.generate("hello there!")
        assert "Hi there" in response.content

        # Test unknown
        response = await provider.generate("something random")
        assert "I'm here to help" in response.content


class TestPackageExports:
    """Tests for LLM package exports."""

    def test_all_exports_available(self):
        """Test all expected symbols are exported."""
        from voice_soundboard import llm

        expected = [
            "SpeechPipeline",
            "StreamingLLMSpeaker",
            "ContextAwareSpeaker",
            "ConversationManager",
            "EmotionSelector",
            "InterruptionHandler",
            "MockLLMProvider",
            "create_provider",
        ]

        for name in expected:
            assert hasattr(llm, name), f"Missing export: {name}"

    def test_imports_work(self):
        """Test imports don't raise errors."""
        from voice_soundboard.llm import (
            SpeechPipeline,
            PipelineConfig,
            StreamingLLMSpeaker,
            StreamConfig,
            ContextAwareSpeaker,
            ContextConfig,
            ConversationManager,
            ConversationConfig,
            create_provider,
            LLMConfig,
            MockLLMProvider,
            InterruptionHandler,
            InterruptionStrategy,
            BargeInDetector,
            EmotionSelector,
            ConversationContext,
            Message,
            MessageRole,
        )

        # All imports should work
        assert SpeechPipeline is not None
        assert MockLLMProvider is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
