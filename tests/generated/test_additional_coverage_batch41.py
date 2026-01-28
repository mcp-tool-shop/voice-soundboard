"""
Additional coverage tests - Batch 41: LLM Integration Coverage.

Comprehensive tests for:
- voice_soundboard/llm/conversation.py
- voice_soundboard/llm/providers.py
- voice_soundboard/llm/context.py
- voice_soundboard/llm/streaming.py
"""

import pytest
import json
import time
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock


# =============================================================================
# Message Tests
# =============================================================================

class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test creating a Message."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        msg = Message(
            role=MessageRole.USER,
            content="Hello world",
        )
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello world"
        assert msg.id is not None

    def test_message_with_metadata(self):
        """Test Message with metadata."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hi there!",
            emotion="happy",
            voice="af_bella",
            duration_ms=1500.0,
        )
        assert msg.emotion == "happy"
        assert msg.voice == "af_bella"
        assert msg.duration_ms == 1500.0

    def test_message_to_dict(self):
        """Test Message to_dict method."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        msg = Message(
            role=MessageRole.USER,
            content="Test message",
            emotion="neutral",
        )
        data = msg.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Test message"
        assert data["emotion"] == "neutral"
        assert "id" in data
        assert "timestamp" in data

    def test_message_from_dict(self):
        """Test Message from_dict method."""
        from voice_soundboard.llm.conversation import Message, MessageRole

        data = {
            "role": "assistant",
            "content": "Response text",
            "emotion": "excited",
        }
        msg = Message.from_dict(data)

        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Response text"
        assert msg.emotion == "excited"


# =============================================================================
# ConversationManager Tests
# =============================================================================

class TestConversationManager:
    """Tests for ConversationManager class."""

    def test_manager_creation(self):
        """Test creating a ConversationManager."""
        from voice_soundboard.llm.conversation import ConversationManager

        manager = ConversationManager()
        assert manager is not None
        assert manager.id is not None

    def test_manager_start_conversation(self):
        """Test starting a conversation."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()

        assert manager.state == ConversationState.LISTENING
        assert manager.started_at is not None

    def test_manager_add_user_message(self):
        """Test adding a user message."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState, MessageRole

        manager = ConversationManager()
        manager.start()

        msg = manager.add_user_message("Hello!")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert manager.state == ConversationState.PROCESSING

    def test_manager_add_assistant_message(self):
        """Test adding an assistant message."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState, MessageRole

        manager = ConversationManager()
        manager.start()
        manager.add_user_message("Hello!")

        msg = manager.add_assistant_message("Hi there!")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there!"
        assert manager.state == ConversationState.LISTENING

    def test_manager_get_llm_context(self):
        """Test getting LLM context."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationConfig

        config = ConversationConfig(system_prompt="You are helpful.")
        manager = ConversationManager(config=config)
        manager.start()
        manager.add_user_message("Hello!")
        manager.add_assistant_message("Hi!")
        manager.add_user_message("How are you?")

        context = manager.get_llm_context()

        # Should include system and recent messages
        assert any(m["role"] == "system" for m in context)
        assert any(m["role"] == "user" for m in context)

    def test_manager_context_window(self):
        """Test context window limiting."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationConfig

        config = ConversationConfig(context_window_messages=2)
        manager = ConversationManager(config=config)
        manager.start()

        # Add more messages than window
        for i in range(5):
            manager.add_user_message(f"Message {i}")
            manager.add_assistant_message(f"Response {i}")

        context = manager.get_llm_context()
        non_system = [m for m in context if m["role"] != "system"]

        # Should only have recent messages
        assert len(non_system) <= 2

    def test_manager_end_conversation(self):
        """Test ending a conversation."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()
        manager.end()

        assert manager.state == ConversationState.ENDED
        assert manager.ended_at is not None

    def test_manager_get_last_message(self):
        """Test getting last message."""
        from voice_soundboard.llm.conversation import ConversationManager, MessageRole

        manager = ConversationManager()
        manager.start()
        manager.add_user_message("First")
        manager.add_assistant_message("Response")
        manager.add_user_message("Second")

        last = manager.get_last_message()
        assert last.content == "Second"

        last_assistant = manager.get_last_message(role=MessageRole.ASSISTANT)
        assert last_assistant.content == "Response"

    def test_manager_stats(self):
        """Test conversation statistics."""
        from voice_soundboard.llm.conversation import ConversationManager

        manager = ConversationManager()
        manager.start()
        manager.add_user_message("Hello world")
        manager.add_assistant_message("Hi there how are you")

        stats = manager.stats

        assert stats["message_count"] >= 2
        assert stats["user_message_count"] >= 1
        assert stats["assistant_message_count"] >= 1
        assert "duration_seconds" in stats

    def test_manager_save_load(self, tmp_path):
        """Test saving and loading conversation."""
        from voice_soundboard.llm.conversation import ConversationManager

        # Create and save
        manager1 = ConversationManager()
        manager1.start()
        manager1.add_user_message("Hello!")
        manager1.add_assistant_message("Hi!")

        save_path = tmp_path / "conversation.json"
        manager1.save(save_path)

        # Load
        manager2 = ConversationManager.load(save_path)

        assert manager2.id == manager1.id
        assert len(manager2.messages) == len(manager1.messages)

    def test_manager_state_callbacks(self):
        """Test state change callbacks."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        states = []

        def on_state_change(state):
            states.append(state)

        manager = ConversationManager()
        manager.on_state_change = on_state_change

        manager.start()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")

        assert ConversationState.LISTENING in states
        assert ConversationState.PROCESSING in states

    def test_manager_set_speaking(self):
        """Test set_speaking method."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()
        manager.set_speaking()

        assert manager.state == ConversationState.SPEAKING

    def test_manager_set_interrupted(self):
        """Test set_interrupted method."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationState

        manager = ConversationManager()
        manager.start()
        manager.set_interrupted()

        assert manager.state == ConversationState.INTERRUPTED


# =============================================================================
# TurnTakingController Tests
# =============================================================================

class TestTurnTakingController:
    """Tests for TurnTakingController class."""

    def test_controller_creation(self):
        """Test creating a TurnTakingController."""
        from voice_soundboard.llm.conversation import TurnTakingController, TurnTakingStrategy

        controller = TurnTakingController()
        assert controller.strategy == TurnTakingStrategy.SILENCE

    def test_controller_custom_strategy(self):
        """Test controller with custom strategy."""
        from voice_soundboard.llm.conversation import TurnTakingController, TurnTakingStrategy

        controller = TurnTakingController(strategy=TurnTakingStrategy.STRICT)
        assert controller.strategy == TurnTakingStrategy.STRICT

    def test_controller_silence_detection(self):
        """Test silence-based turn detection."""
        from voice_soundboard.llm.conversation import TurnTakingController, TurnTakingStrategy

        controller = TurnTakingController(
            strategy=TurnTakingStrategy.SILENCE,
            silence_threshold_ms=100,
        )

        # Start speaking
        controller.process_audio(is_speech=True)

        # Stop speaking but short silence
        controller.process_audio(is_speech=False)

        # Continue silence past threshold
        time.sleep(0.15)
        ended = controller.process_audio(is_speech=False)

        # Turn should have ended
        assert ended is True

    def test_controller_force_end_turn(self):
        """Test force ending turn."""
        from voice_soundboard.llm.conversation import TurnTakingController

        controller = TurnTakingController()

        # Start speaking
        controller.process_audio(is_speech=True)

        # Force end
        controller.force_end_turn()

        # Should not be speaking
        assert controller._is_user_speaking is False

    def test_controller_reset(self):
        """Test resetting controller state."""
        from voice_soundboard.llm.conversation import TurnTakingController

        controller = TurnTakingController()
        controller.process_audio(is_speech=True)

        controller.reset()

        assert controller._is_user_speaking is False
        assert controller._silence_start is None

    def test_controller_callbacks(self):
        """Test turn-taking callbacks."""
        from voice_soundboard.llm.conversation import TurnTakingController, TurnTakingStrategy

        started = []
        ended = []

        def on_start():
            started.append(True)

        def on_end():
            ended.append(True)

        controller = TurnTakingController(
            strategy=TurnTakingStrategy.SILENCE,
            silence_threshold_ms=50,
        )
        controller.on_user_start = on_start
        controller.on_user_end = on_end

        # Start speaking
        controller.process_audio(is_speech=True)
        assert len(started) == 1

        # Wait and end
        controller.process_audio(is_speech=False)
        time.sleep(0.1)
        controller.process_audio(is_speech=False)
        assert len(ended) == 1


# =============================================================================
# LLM Config Tests
# =============================================================================

class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_config_defaults(self):
        """Test LLMConfig default values."""
        from voice_soundboard.llm.providers import LLMConfig

        config = LLMConfig()
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.stream is True

    def test_config_custom_values(self):
        """Test LLMConfig with custom values."""
        from voice_soundboard.llm.providers import LLMConfig

        config = LLMConfig(
            model="gpt-4",
            temperature=0.5,
            max_tokens=2048,
            api_key="test-key",
        )
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.api_key == "test-key"


# =============================================================================
# LLM Response Tests
# =============================================================================

class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_creation(self):
        """Test creating an LLMResponse."""
        from voice_soundboard.llm.providers import LLMResponse

        response = LLMResponse(
            content="Hello!",
            model="gpt-4",
            finish_reason="stop",
        )
        assert response.content == "Hello!"
        assert response.model == "gpt-4"

    def test_response_with_usage(self):
        """Test LLMResponse with usage data."""
        from voice_soundboard.llm.providers import LLMResponse

        response = LLMResponse(
            content="Response text",
            model="llama3",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
            },
            latency_ms=500.0,
        )
        assert response.usage["prompt_tokens"] == 10
        assert response.latency_ms == 500.0


# =============================================================================
# MockLLMProvider Tests
# =============================================================================

class TestMockLLMProvider:
    """Tests for MockLLMProvider class."""

    @pytest.mark.asyncio
    async def test_mock_generate(self):
        """Test mock generate method."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(default_response="Test response")
        response = await provider.generate("Hello")

        assert response.content == "Test response"
        assert response.provider == "mock"

    @pytest.mark.asyncio
    async def test_mock_with_patterns(self):
        """Test mock with response patterns."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            responses={
                "hello": "Hi there!",
                "weather": "It's sunny today.",
            },
            default_response="I don't understand.",
        )

        response1 = await provider.generate("Hello!")
        assert response1.content == "Hi there!"

        response2 = await provider.generate("What's the weather?")
        assert response2.content == "It's sunny today."

        response3 = await provider.generate("Random text")
        assert response3.content == "I don't understand."

    @pytest.mark.asyncio
    async def test_mock_stream(self):
        """Test mock streaming."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            default_response="Hello world",
            token_delay_ms=10,
        )

        tokens = []
        async for token in provider.stream("Test"):
            tokens.append(token)

        joined = "".join(tokens)
        assert "Hello" in joined
        assert "world" in joined

    @pytest.mark.asyncio
    async def test_mock_chat(self):
        """Test mock chat method."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(default_response="Chat response")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]

        response = await provider.chat(messages)
        assert response.content == "Chat response"


# =============================================================================
# Provider Factory Tests
# =============================================================================

class TestProviderFactory:
    """Tests for create_provider function."""

    def test_create_mock_provider(self):
        """Test creating mock provider."""
        from voice_soundboard.llm.providers import create_provider, ProviderType

        provider = create_provider(ProviderType.MOCK)
        assert provider.provider_type == ProviderType.MOCK

    def test_create_provider_by_string(self):
        """Test creating provider by string type."""
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
        from voice_soundboard.llm.providers import create_provider, ProviderType, LLMConfig

        config = LLMConfig(api_key="test-key")
        provider = create_provider(ProviderType.OPENAI, config=config)
        assert provider.provider_type == ProviderType.OPENAI

    def test_create_vllm_provider(self):
        """Test creating vLLM provider."""
        from voice_soundboard.llm.providers import create_provider, ProviderType

        provider = create_provider(ProviderType.VLLM)
        assert provider.provider_type == ProviderType.VLLM

    def test_create_unknown_provider(self):
        """Test creating unknown provider raises error."""
        from voice_soundboard.llm.providers import create_provider

        with pytest.raises(ValueError):
            create_provider("unknown")


# =============================================================================
# OpenAI Provider Tests (mocked)
# =============================================================================

class TestOpenAIProvider:
    """Tests for OpenAIProvider class."""

    def test_openai_provider_creation(self):
        """Test creating OpenAI provider."""
        from voice_soundboard.llm.providers import OpenAIProvider, LLMConfig

        config = LLMConfig(api_key="test-key", model="gpt-4")
        provider = OpenAIProvider(config)

        assert provider.provider_type.value == "openai"


# =============================================================================
# Ollama Provider Tests (mocked)
# =============================================================================

class TestOllamaProvider:
    """Tests for OllamaProvider class."""

    def test_ollama_provider_creation(self):
        """Test creating Ollama provider."""
        from voice_soundboard.llm.providers import OllamaProvider, LLMConfig

        config = LLMConfig(model="llama3.2")
        provider = OllamaProvider(config)

        assert provider.provider_type.value == "ollama"
        assert provider.base_url == "http://localhost:11434"


# =============================================================================
# ConversationState Tests
# =============================================================================

class TestConversationState:
    """Tests for ConversationState enum."""

    def test_conversation_states_exist(self):
        """Test all conversation states exist."""
        from voice_soundboard.llm.conversation import ConversationState

        assert ConversationState.IDLE.value == "idle"
        assert ConversationState.LISTENING.value == "listening"
        assert ConversationState.PROCESSING.value == "processing"
        assert ConversationState.SPEAKING.value == "speaking"
        assert ConversationState.INTERRUPTED.value == "interrupted"
        assert ConversationState.ENDED.value == "ended"


# =============================================================================
# TurnTakingStrategy Tests
# =============================================================================

class TestTurnTakingStrategy:
    """Tests for TurnTakingStrategy enum."""

    def test_strategies_exist(self):
        """Test all turn-taking strategies exist."""
        from voice_soundboard.llm.conversation import TurnTakingStrategy

        assert TurnTakingStrategy.STRICT.value == "strict"
        assert TurnTakingStrategy.SILENCE.value == "silence"
        assert TurnTakingStrategy.PUSH_TO_TALK.value == "push_to_talk"
        assert TurnTakingStrategy.HOTWORD.value == "hotword"
        assert TurnTakingStrategy.CONTINUOUS.value == "continuous"


# =============================================================================
# ConversationConfig Tests
# =============================================================================

class TestConversationConfig:
    """Tests for ConversationConfig dataclass."""

    def test_config_defaults(self):
        """Test ConversationConfig defaults."""
        from voice_soundboard.llm.conversation import ConversationConfig, TurnTakingStrategy

        config = ConversationConfig()

        assert config.turn_taking_strategy == TurnTakingStrategy.SILENCE
        assert config.silence_threshold_ms == 1500.0
        assert config.max_history_messages == 100

    def test_config_custom_values(self):
        """Test ConversationConfig with custom values."""
        from voice_soundboard.llm.conversation import ConversationConfig, TurnTakingStrategy

        config = ConversationConfig(
            turn_taking_strategy=TurnTakingStrategy.PUSH_TO_TALK,
            system_prompt="You are a helpful assistant.",
            auto_save=True,
        )

        assert config.turn_taking_strategy == TurnTakingStrategy.PUSH_TO_TALK
        assert config.system_prompt == "You are a helpful assistant."
        assert config.auto_save is True


# =============================================================================
# History Trimming Tests
# =============================================================================

class TestHistoryTrimming:
    """Tests for conversation history trimming."""

    def test_history_trimming(self):
        """Test that history is trimmed correctly."""
        from voice_soundboard.llm.conversation import ConversationManager, ConversationConfig

        config = ConversationConfig(max_history_messages=5)
        manager = ConversationManager(config=config)
        manager.start()

        # Add more messages than limit
        for i in range(10):
            manager.add_user_message(f"User {i}")
            manager.add_assistant_message(f"Assistant {i}")

        # History should be trimmed
        assert len(manager.messages) <= config.max_history_messages


# =============================================================================
# Duration and Timing Tests
# =============================================================================

class TestConversationTiming:
    """Tests for conversation timing."""

    def test_duration_calculation(self):
        """Test duration calculation."""
        from voice_soundboard.llm.conversation import ConversationManager

        manager = ConversationManager()
        manager.start()

        time.sleep(0.1)

        duration = manager.duration_seconds
        assert duration >= 0.1

    def test_duration_ended(self):
        """Test duration for ended conversation."""
        from voice_soundboard.llm.conversation import ConversationManager

        manager = ConversationManager()
        manager.start()
        time.sleep(0.05)
        manager.end()

        duration = manager.duration_seconds
        assert duration >= 0.05
