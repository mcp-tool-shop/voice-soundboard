"""
Batch 16: Additional coverage tests for the LLM module.

Tests for:
- voice_soundboard/llm/context.py
- voice_soundboard/llm/providers.py
- voice_soundboard/llm/conversation.py
"""

import pytest
import asyncio
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import tempfile


# ============================================================================
# Tests for llm/context.py
# ============================================================================

class TestProsodyHint:
    """Tests for ProsodyHint enum."""

    def test_prosody_hint_values(self):
        """Test ProsodyHint enum has expected values."""
        from voice_soundboard.llm.context import ProsodyHint

        assert ProsodyHint.NEUTRAL.value == "neutral"
        assert ProsodyHint.EMPATHETIC.value == "empathetic"
        assert ProsodyHint.EXCITED.value == "excited"
        assert ProsodyHint.CALM.value == "calm"
        assert ProsodyHint.SERIOUS.value == "serious"
        assert ProsodyHint.PLAYFUL.value == "playful"
        assert ProsodyHint.APOLOGETIC.value == "apologetic"
        assert ProsodyHint.ENCOURAGING.value == "encouraging"
        assert ProsodyHint.CONCERNED.value == "concerned"
        assert ProsodyHint.PROFESSIONAL.value == "professional"


class TestContextConfig:
    """Tests for ContextConfig dataclass."""

    def test_default_values(self):
        """Test ContextConfig default values."""
        from voice_soundboard.llm.context import ContextConfig

        config = ContextConfig()
        assert config.enable_auto_emotion is True
        assert config.emotion_confidence_threshold == 0.5
        assert config.analyze_user_sentiment is True
        assert config.history_window == 5
        assert config.default_emotion == "neutral"
        assert config.default_preset == "assistant"

    def test_custom_values(self):
        """Test ContextConfig with custom values."""
        from voice_soundboard.llm.context import ContextConfig

        config = ContextConfig(
            enable_auto_emotion=False,
            default_emotion="happy",
            empathy_speed_factor=0.8,
        )
        assert config.enable_auto_emotion is False
        assert config.default_emotion == "happy"
        assert config.empathy_speed_factor == 0.8


class TestConversationContext:
    """Tests for ConversationContext dataclass."""

    def test_default_values(self):
        """Test ConversationContext default values."""
        from voice_soundboard.llm.context import ConversationContext

        ctx = ConversationContext()
        assert ctx.messages == []
        assert ctx.turn_count == 0
        assert ctx.user_sentiment is None

    def test_add_message(self):
        """Test add_message method."""
        from voice_soundboard.llm.context import ConversationContext

        ctx = ConversationContext()
        ctx.add_message("user", "Hello!")
        ctx.add_message("assistant", "Hi there!")

        assert len(ctx.messages) == 2
        assert ctx.turn_count == 2
        assert ctx.messages[0]["role"] == "user"
        assert ctx.messages[0]["content"] == "Hello!"

    def test_get_recent_messages(self):
        """Test get_recent_messages method."""
        from voice_soundboard.llm.context import ConversationContext

        ctx = ConversationContext()
        for i in range(10):
            ctx.add_message("user", f"Message {i}")

        recent = ctx.get_recent_messages(3)
        assert len(recent) == 3
        assert recent[-1]["content"] == "Message 9"

    def test_get_recent_messages_empty(self):
        """Test get_recent_messages with empty context."""
        from voice_soundboard.llm.context import ConversationContext

        ctx = ConversationContext()
        assert ctx.get_recent_messages() == []

    def test_get_last_user_message(self):
        """Test get_last_user_message method."""
        from voice_soundboard.llm.context import ConversationContext

        ctx = ConversationContext()
        ctx.add_message("user", "First message")
        ctx.add_message("assistant", "Response")
        ctx.add_message("user", "Second message")

        assert ctx.get_last_user_message() == "Second message"

    def test_get_last_user_message_none(self):
        """Test get_last_user_message with no user messages."""
        from voice_soundboard.llm.context import ConversationContext

        ctx = ConversationContext()
        ctx.add_message("assistant", "Hello")
        assert ctx.get_last_user_message() is None


class TestEmotionSelector:
    """Tests for EmotionSelector class."""

    def test_creation(self):
        """Test EmotionSelector creation."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        assert selector.config is not None

    def test_select_emotion_keyword_based(self):
        """Test select_emotion with keyword detection."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        emotion, confidence = selector.select_emotion("That's wonderful news!")
        assert emotion in ["happy", "neutral"]

    def test_select_emotion_with_hint(self):
        """Test select_emotion with prosody hint."""
        from voice_soundboard.llm.context import EmotionSelector, ProsodyHint

        selector = EmotionSelector()
        emotion, confidence = selector.select_emotion(
            "Let me help you with that.",
            hint=ProsodyHint.EMPATHETIC,
        )
        assert emotion == "sympathetic"

    def test_select_emotion_with_context(self):
        """Test select_emotion with conversation context."""
        from voice_soundboard.llm.context import EmotionSelector, ConversationContext

        selector = EmotionSelector()
        ctx = ConversationContext()
        ctx.user_emotion = "frustrated"

        emotion, confidence = selector.select_emotion(
            "I understand your concern.",
            context=ctx,
        )
        assert emotion == "sympathetic"

    def test_analyze_keywords(self):
        """Test _analyze_keywords method."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()

        # "fantastic" and "amazing" are in the happy keywords list
        emotion, score = selector._analyze_keywords("This is fantastic and amazing!")
        assert emotion == "happy"
        assert score > 0

        emotion, score = selector._analyze_keywords("I'm sorry about that.")
        assert emotion == "sympathetic"

    def test_detect_content_type_question(self):
        """Test _detect_content_type for questions."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        assert selector._detect_content_type("What time is it?") == "question"
        assert selector._detect_content_type("How does this work?") == "question"

    def test_detect_content_type_greeting(self):
        """Test _detect_content_type for greetings."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        assert selector._detect_content_type("Hello there!") == "greeting"
        assert selector._detect_content_type("Good morning!") == "greeting"

    def test_detect_content_type_farewell(self):
        """Test _detect_content_type for farewells."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        assert selector._detect_content_type("Goodbye for now!") == "farewell"
        assert selector._detect_content_type("See you later!") == "farewell"

    def test_detect_content_type_apology(self):
        """Test _detect_content_type for apologies."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        assert selector._detect_content_type("I'm sorry about the delay.") == "apology"

    def test_detect_content_type_warning(self):
        """Test _detect_content_type for warnings."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        assert selector._detect_content_type("Warning: This is important!") == "warning"
        assert selector._detect_content_type("Be careful with that.") == "warning"

    def test_detect_content_type_encouragement(self):
        """Test _detect_content_type for encouragement."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        assert selector._detect_content_type("You can do it!") == "encouragement"
        assert selector._detect_content_type("Great job!") == "encouragement"

    def test_detect_content_type_none(self):
        """Test _detect_content_type for unrecognized content."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        assert selector._detect_content_type("The weather is nice today.") is None

    def test_hint_to_emotion(self):
        """Test _hint_to_emotion method."""
        from voice_soundboard.llm.context import EmotionSelector, ProsodyHint

        selector = EmotionSelector()
        assert selector._hint_to_emotion(ProsodyHint.NEUTRAL) == "neutral"
        assert selector._hint_to_emotion(ProsodyHint.EMPATHETIC) == "sympathetic"
        assert selector._hint_to_emotion(ProsodyHint.PROFESSIONAL) == "confident"

    def test_detect_user_sentiment_frustrated(self):
        """Test detect_user_sentiment for frustration."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        # "frustrated" is in the keyword list, not "frustrating"
        sentiment, emotion = selector.detect_user_sentiment("I'm so frustrated with this!")
        assert sentiment == "negative"
        assert emotion == "frustrated"

    def test_detect_user_sentiment_confused(self):
        """Test detect_user_sentiment for confusion."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        sentiment, emotion = selector.detect_user_sentiment("I don't understand what you mean.")
        assert sentiment == "negative"
        assert emotion == "confused"

    def test_detect_user_sentiment_happy(self):
        """Test detect_user_sentiment for happiness."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        sentiment, emotion = selector.detect_user_sentiment("Thank you so much! This is great!")
        assert sentiment == "positive"
        assert emotion == "happy"

    def test_detect_user_sentiment_anxious(self):
        """Test detect_user_sentiment for anxiety."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        sentiment, emotion = selector.detect_user_sentiment("I'm worried about this.")
        assert sentiment == "negative"
        assert emotion == "anxious"

    def test_detect_user_sentiment_neutral(self):
        """Test detect_user_sentiment for neutral."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()
        sentiment, emotion = selector.detect_user_sentiment("The sky is blue.")
        assert sentiment == "neutral"
        assert emotion == "neutral"


class TestContextAwareSpeaker:
    """Tests for ContextAwareSpeaker class."""

    def test_creation(self):
        """Test ContextAwareSpeaker creation."""
        from voice_soundboard.llm.context import ContextAwareSpeaker

        speaker = ContextAwareSpeaker()
        assert speaker.config is not None
        assert speaker.context is not None

    def test_update_context(self):
        """Test update_context method."""
        from voice_soundboard.llm.context import ContextAwareSpeaker

        speaker = ContextAwareSpeaker()
        # Use "frustrated" which is in the keyword list
        speaker.update_context("I'm so frustrated with this!")

        assert len(speaker.context.messages) == 1
        assert speaker.context.user_sentiment == "negative"
        assert speaker.context.user_emotion == "frustrated"

    def test_reset_context(self):
        """Test reset_context method."""
        from voice_soundboard.llm.context import ContextAwareSpeaker

        speaker = ContextAwareSpeaker()
        speaker.update_context("Hello")
        speaker.reset_context()

        assert len(speaker.context.messages) == 0
        assert speaker.context.turn_count == 0

    @patch('voice_soundboard.llm.context.ContextAwareSpeaker._get_engine')
    def test_speak(self, mock_get_engine):
        """Test speak method."""
        from voice_soundboard.llm.context import ContextAwareSpeaker

        mock_engine = MagicMock()
        mock_engine.speak.return_value = {"audio_path": "/fake/path.wav"}
        mock_get_engine.return_value = mock_engine

        speaker = ContextAwareSpeaker()
        result = speaker.speak("Hello there!", auto_emotion=True)

        assert "emotion" in result
        assert "confidence" in result
        assert "final_speed" in result


# ============================================================================
# Tests for llm/providers.py
# ============================================================================

class TestProviderType:
    """Tests for ProviderType enum."""

    def test_provider_type_values(self):
        """Test ProviderType enum values."""
        from voice_soundboard.llm.providers import ProviderType

        assert ProviderType.OLLAMA.value == "ollama"
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.VLLM.value == "vllm"
        assert ProviderType.MOCK.value == "mock"


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self):
        """Test LLMConfig default values."""
        from voice_soundboard.llm.providers import LLMConfig

        config = LLMConfig()
        assert config.model == "llama3.2"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.stream is True

    def test_custom_values(self):
        """Test LLMConfig with custom values."""
        from voice_soundboard.llm.providers import LLMConfig

        config = LLMConfig(
            model="gpt-4",
            temperature=0.5,
            api_key="test_key",
        )
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.api_key == "test_key"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_creation(self):
        """Test LLMResponse creation."""
        from voice_soundboard.llm.providers import LLMResponse

        response = LLMResponse(
            content="Hello world",
            model="llama3.2",
            finish_reason="stop",
            latency_ms=100.0,
        )
        assert response.content == "Hello world"
        assert response.model == "llama3.2"
        assert response.latency_ms == 100.0


class TestMockLLMProvider:
    """Tests for MockLLMProvider class."""

    def test_creation(self):
        """Test MockLLMProvider creation."""
        from voice_soundboard.llm.providers import MockLLMProvider, ProviderType

        provider = MockLLMProvider()
        assert provider.provider_type == ProviderType.MOCK

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test generate method."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(default_response="Test response")
        response = await provider.generate("Hello")

        assert response.content == "Test response"
        assert response.model == "mock"
        assert response.provider == "mock"

    @pytest.mark.asyncio
    async def test_generate_with_pattern_match(self):
        """Test generate with pattern matching."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(responses={
            "hello": "Hi there!",
            "weather": "It's sunny today.",
        })

        response = await provider.generate("Hello, how are you?")
        assert response.content == "Hi there!"

        response = await provider.generate("What's the weather like?")
        assert response.content == "It's sunny today."

    @pytest.mark.asyncio
    async def test_stream(self):
        """Test stream method."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            default_response="Hello world",
            token_delay_ms=1.0,
        )

        tokens = []
        async for token in provider.stream("Hi"):
            tokens.append(token)

        assert len(tokens) == 2  # "Hello " and "world"
        assert "".join(tokens) == "Hello world"

    @pytest.mark.asyncio
    async def test_chat(self):
        """Test chat method."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        response = await provider.chat(messages)
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_chat_stream(self):
        """Test chat_stream method."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(token_delay_ms=1.0)
        messages = [{"role": "user", "content": "Hi"}]

        tokens = []
        async for token in provider.chat_stream(messages):
            tokens.append(token)

        assert len(tokens) > 0


class TestOllamaProvider:
    """Tests for OllamaProvider class."""

    def test_creation(self):
        """Test OllamaProvider creation."""
        from voice_soundboard.llm.providers import OllamaProvider, ProviderType

        provider = OllamaProvider()
        assert provider.provider_type == ProviderType.OLLAMA
        assert "localhost:11434" in provider.base_url

    def test_custom_base_url(self):
        """Test OllamaProvider with custom base URL."""
        from voice_soundboard.llm.providers import OllamaProvider, LLMConfig

        config = LLMConfig(base_url="http://custom:1234")
        provider = OllamaProvider(config)
        assert provider.base_url == "http://custom:1234"


class TestOpenAIProvider:
    """Tests for OpenAIProvider class."""

    def test_creation(self):
        """Test OpenAIProvider creation."""
        from voice_soundboard.llm.providers import OpenAIProvider, ProviderType

        provider = OpenAIProvider()
        assert provider.provider_type == ProviderType.OPENAI
        assert "openai.com" in provider.base_url

    def test_api_key_from_env(self):
        """Test OpenAIProvider gets API key from environment."""
        from voice_soundboard.llm.providers import OpenAIProvider, LLMConfig
        import os

        # Set environment variable
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = LLMConfig(api_key=None)
            provider = OpenAIProvider(config)
            assert provider.config.api_key == "test_key"


class TestVLLMProvider:
    """Tests for VLLMProvider class."""

    def test_creation(self):
        """Test VLLMProvider creation."""
        from voice_soundboard.llm.providers import VLLMProvider, ProviderType

        provider = VLLMProvider()
        assert provider.provider_type == ProviderType.VLLM
        assert "localhost:8000" in provider.base_url


class TestCreateProvider:
    """Tests for create_provider factory function."""

    def test_create_mock_provider(self):
        """Test create_provider for mock."""
        from voice_soundboard.llm.providers import create_provider, MockLLMProvider

        provider = create_provider("mock")
        assert isinstance(provider, MockLLMProvider)

    def test_create_ollama_provider(self):
        """Test create_provider for ollama."""
        from voice_soundboard.llm.providers import create_provider, OllamaProvider

        provider = create_provider("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_create_provider_with_enum(self):
        """Test create_provider with ProviderType enum."""
        from voice_soundboard.llm.providers import create_provider, ProviderType, MockLLMProvider

        provider = create_provider(ProviderType.MOCK)
        assert isinstance(provider, MockLLMProvider)

    def test_create_provider_unknown(self):
        """Test create_provider with unknown type."""
        from voice_soundboard.llm.providers import create_provider

        with pytest.raises(ValueError):
            create_provider("unknown_provider")


# ============================================================================
# Tests for llm/conversation.py
# ============================================================================

class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_message_role_values(self):
        """Test MessageRole enum values."""
        from voice_soundboard.llm.conversation import MessageRole

        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"


class TestConversationStateEnum:
    """Tests for ConversationState enum."""

    def test_conversation_state_values(self):
        """Test ConversationState enum values."""
        from voice_soundboard.llm.conversation import ConversationState

        assert ConversationState.IDLE.value == "idle"
        assert ConversationState.LISTENING.value == "listening"
        assert ConversationState.PROCESSING.value == "processing"
        assert ConversationState.SPEAKING.value == "speaking"
        assert ConversationState.INTERRUPTED.value == "interrupted"
        assert ConversationState.ENDED.value == "ended"


class TestTurnTakingStrategy:
    """Tests for TurnTakingStrategy enum."""

    def test_turn_taking_strategy_values(self):
        """Test TurnTakingStrategy enum values."""
        from voice_soundboard.llm.conversation import TurnTakingStrategy

        assert TurnTakingStrategy.STRICT.value == "strict"
        assert TurnTakingStrategy.SILENCE.value == "silence"
        assert TurnTakingStrategy.PUSH_TO_TALK.value == "push_to_talk"
        assert TurnTakingStrategy.HOTWORD.value == "hotword"
        assert TurnTakingStrategy.CONTINUOUS.value == "continuous"


class TestMessage:
    """Tests for Message dataclass."""

    def test_creation(self):
        """Test Message creation."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.id is not None

    def test_to_dict(self):
        """Test to_dict method."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        msg = Message(
            role=MessageRole.USER,
            content="Hello",
            emotion="happy",
            voice="default",
        )
        data = msg.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Hello"
        assert data["emotion"] == "happy"
        assert "timestamp" in data

    def test_from_dict(self):
        """Test from_dict method."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        data = {
            "role": "user",
            "content": "Hello",
            "emotion": "happy",
        }
        msg = Message.from_dict(data)

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.emotion == "happy"


class TestConversationConfig:
    """Tests for ConversationConfig dataclass."""

    def test_default_values(self):
        """Test ConversationConfig default values."""
        from voice_soundboard.llm.conversation import ConversationConfig, TurnTakingStrategy

        config = ConversationConfig()
        assert config.turn_taking_strategy == TurnTakingStrategy.SILENCE
        assert config.silence_threshold_ms == 1500.0
        assert config.max_history_messages == 100
        assert config.auto_save is False


class TestConversationManager:
    """Tests for ConversationManager class."""

    def test_creation(self):
        """Test ConversationManager creation."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        assert manager.state == ConversationState.IDLE
        assert manager.messages == []

    def test_creation_with_system_prompt(self):
        """Test ConversationManager with system prompt."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationConfig

        config = ConversationConfig(system_prompt="You are helpful.")
        manager = ConversationManager(config=config)

        assert len(manager.messages) == 1
        assert manager.messages[0].content == "You are helpful."

    def test_start(self):
        """Test start method."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()

        assert manager.state == ConversationState.LISTENING
        assert manager.started_at is not None

    def test_end(self):
        """Test end method."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()
        manager.end()

        assert manager.state == ConversationState.ENDED
        assert manager.ended_at is not None

    def test_add_user_message(self):
        """Test add_user_message method."""
        from voice_soundboard.llm.conversation import ConversationManager, MessageRole, ConversationState

        manager = ConversationManager()
        manager.start()
        msg = manager.add_user_message("Hello!")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert len(manager.messages) == 1
        assert manager.state == ConversationState.PROCESSING

    def test_add_assistant_message(self):
        """Test add_assistant_message method."""
        from voice_soundboard.llm.conversation import ConversationManager, MessageRole, ConversationState

        manager = ConversationManager()
        manager.start()
        manager.add_user_message("Hello!")
        msg = manager.add_assistant_message("Hi there!", emotion="friendly")

        assert msg.role == MessageRole.ASSISTANT
        assert msg.emotion == "friendly"
        assert manager.state == ConversationState.LISTENING

    def test_set_speaking(self):
        """Test set_speaking method."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()
        manager.set_speaking()
        assert manager.state == ConversationState.SPEAKING

    def test_set_interrupted(self):
        """Test set_interrupted method."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()
        manager.set_interrupted()
        assert manager.state == ConversationState.INTERRUPTED

    def test_get_llm_context(self):
        """Test get_llm_context method."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationConfig

        config = ConversationConfig(system_prompt="You are helpful.")
        manager = ConversationManager(config=config)
        manager.start()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi!")
        manager.add_user_message("How are you?")

        context = manager.get_llm_context()
        assert context[0]["role"] == "system"
        assert context[0]["content"] == "You are helpful."
        assert len(context) == 4  # system + 3 messages

    def test_get_last_message(self):
        """Test get_last_message method."""
        from voice_soundboard.llm.conversation import ConversationManager, MessageRole

        manager = ConversationManager()
        manager.start()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi!")

        last = manager.get_last_message()
        assert last.role == MessageRole.ASSISTANT

        last_user = manager.get_last_message(MessageRole.USER)
        assert last_user.content == "Hello"

    def test_trim_history(self):
        """Test history trimming."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationConfig

        config = ConversationConfig(max_history_messages=5)
        manager = ConversationManager(config=config)
        manager.start()

        for i in range(10):
            manager.add_user_message(f"Message {i}")

        assert len(manager.messages) <= 5

    def test_save_and_load(self):
        """Test save and load methods."""
        from voice_soundboard.llm.conversation import ConversationManager

        manager = ConversationManager()
        manager.start()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi!")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = Path(f.name)

        try:
            manager.save(path)
            loaded = ConversationManager.load(path)

            assert loaded.id == manager.id
            assert len(loaded.messages) == 2
        finally:
            path.unlink()

    def test_duration_seconds(self):
        """Test duration_seconds property."""
        from voice_soundboard.llm.conversation import ConversationManager

        manager = ConversationManager()
        assert manager.duration_seconds == 0.0

        manager.start()
        time.sleep(0.1)
        assert manager.duration_seconds > 0

    def test_stats(self):
        """Test stats property."""
        from voice_soundboard.llm.conversation import ConversationManager

        manager = ConversationManager()
        manager.start()
        manager.add_user_message("Hello there!")
        manager.add_assistant_message("Hi! How are you?")

        stats = manager.stats
        assert stats["message_count"] == 2
        assert stats["user_message_count"] == 1
        assert stats["assistant_message_count"] == 1
        assert stats["user_word_count"] == 2
        assert stats["assistant_word_count"] == 4

    def test_state_change_callback(self):
        """Test on_state_change callback."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        callback = Mock()
        manager = ConversationManager()
        manager.on_state_change = callback

        manager.start()
        callback.assert_called_with(ConversationState.LISTENING)

    def test_turn_end_callback(self):
        """Test on_turn_end callback."""
        from voice_soundboard.llm.conversation import ConversationManager, MessageRole

        callback = Mock()
        manager = ConversationManager()
        manager.on_turn_end = callback

        manager.start()
        manager.add_user_message("Hello")
        callback.assert_called_with(MessageRole.USER)


class TestTurnTakingController:
    """Tests for TurnTakingController class."""

    def test_creation(self):
        """Test TurnTakingController creation."""
        from voice_soundboard.llm.conversation import TurnTakingController, TurnTakingStrategy

        controller = TurnTakingController()
        assert controller.strategy == TurnTakingStrategy.SILENCE

    def test_process_audio_speech_detected(self):
        """Test process_audio when speech is detected."""
        from voice_soundboard.llm.conversation import TurnTakingController

        controller = TurnTakingController()
        result = controller.process_audio(is_speech=True)
        assert result is False  # Turn hasn't ended yet

    def test_process_audio_silence_turn_end(self):
        """Test process_audio with silence-based turn ending."""
        from voice_soundboard.llm.conversation import TurnTakingController

        controller = TurnTakingController(silence_threshold_ms=100.0)

        # Start speaking
        controller.process_audio(is_speech=True)

        # Silence begins
        controller.process_audio(is_speech=False)

        # Wait for threshold
        time.sleep(0.15)

        # Should detect turn end
        result = controller.process_audio(is_speech=False)
        assert result is True

    def test_process_audio_strict_strategy(self):
        """Test process_audio with strict strategy."""
        from voice_soundboard.llm.conversation import TurnTakingController, TurnTakingStrategy

        controller = TurnTakingController(strategy=TurnTakingStrategy.STRICT)
        result = controller.process_audio(is_speech=False)
        assert result is False

    def test_process_audio_continuous_strategy(self):
        """Test process_audio with continuous strategy."""
        from voice_soundboard.llm.conversation import TurnTakingController, TurnTakingStrategy

        controller = TurnTakingController(strategy=TurnTakingStrategy.CONTINUOUS)
        result = controller.process_audio(is_speech=False)
        assert result is False

    def test_force_end_turn(self):
        """Test force_end_turn method."""
        from voice_soundboard.llm.conversation import TurnTakingController

        callback = Mock()
        controller = TurnTakingController()
        controller.on_user_end = callback

        controller.process_audio(is_speech=True)  # Start speaking
        controller.force_end_turn()

        callback.assert_called_once()

    def test_reset(self):
        """Test reset method."""
        from voice_soundboard.llm.conversation import TurnTakingController

        controller = TurnTakingController()
        controller.process_audio(is_speech=True)
        controller.reset()

        assert controller._is_user_speaking is False
        assert controller._silence_start is None

    def test_user_start_callback(self):
        """Test on_user_start callback."""
        from voice_soundboard.llm.conversation import TurnTakingController

        callback = Mock()
        controller = TurnTakingController()
        controller.on_user_start = callback

        controller.process_audio(is_speech=True)
        callback.assert_called_once()

    def test_user_end_callback(self):
        """Test on_user_end callback."""
        from voice_soundboard.llm.conversation import TurnTakingController

        callback = Mock()
        controller = TurnTakingController(silence_threshold_ms=50.0)
        controller.on_user_end = callback

        controller.process_audio(is_speech=True)
        controller.process_audio(is_speech=False)

        time.sleep(0.1)
        controller.process_audio(is_speech=False)

        callback.assert_called_once()


# ============================================================================
# Additional edge case tests
# ============================================================================

class TestLLMEdgeCases:
    """Additional edge case tests for LLM module."""

    def test_emotion_selector_no_match(self):
        """Test EmotionSelector when no patterns match."""
        from voice_soundboard.llm.context import EmotionSelector, ContextConfig

        config = ContextConfig(emotion_confidence_threshold=0.9)
        selector = EmotionSelector(config)

        emotion, confidence = selector.select_emotion("Random text without keywords.")
        assert emotion == "neutral"
        assert confidence == 0.0

    def test_message_id_generation(self):
        """Test Message ID generation is unique."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        ids = set()
        for _ in range(100):
            msg = Message(role=MessageRole.USER, content="Test")
            ids.add(msg.id)

        assert len(ids) == 100

    def test_conversation_auto_save(self):
        """Test conversation auto_save feature."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationConfig

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = Path(f.name)

        try:
            config = ConversationConfig(auto_save=True, save_path=path)
            manager = ConversationManager(config=config)
            manager.start()
            manager.add_user_message("Hello")

            # File should be created
            assert path.exists()

            # Content should be valid JSON
            with open(path) as f:
                data = json.load(f)
            assert "messages" in data
        finally:
            if path.exists():
                path.unlink()

    def test_load_ended_conversation(self):
        """Test loading an ended conversation."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()
        manager.add_user_message("Hello")
        manager.end()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = Path(f.name)

        try:
            manager.save(path)
            loaded = ConversationManager.load(path)
            assert loaded.state == ConversationState.ENDED
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_mock_provider_latency(self):
        """Test MockLLMProvider includes latency measurement."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider()
        response = await provider.generate("Test")

        assert response.latency_ms > 0

    def test_context_window_limit(self):
        """Test context window respects limit."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationConfig

        config = ConversationConfig(context_window_messages=3)
        manager = ConversationManager(config=config)
        manager.start()

        for i in range(10):
            manager.add_user_message(f"Message {i}")
            manager.add_assistant_message(f"Response {i}")

        context = manager.get_llm_context()
        non_system = [m for m in context if m["role"] != "system"]
        assert len(non_system) == 3
