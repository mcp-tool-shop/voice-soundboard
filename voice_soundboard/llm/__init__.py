"""
LLM Integration Layer for Voice Soundboard.

.. warning:: **Experimental.** This module may change or be removed between
   minor releases. See docs/API_STABILITY.md and docs/FEATURE_FLAGS.md.

Provides integration between Large Language Models and text-to-speech,
enabling natural conversational AI experiences.

Features:
- Streaming LLM integration (speak as LLM generates)
- Context-aware prosody (automatic emotion selection)
- Speech-to-speech pipeline (STT → LLM → TTS)
- Interruption/barge-in handling
- Conversation state management
- Turn-taking logic

Example:
    # Basic streaming integration
    from voice_soundboard.llm import StreamingLLMSpeaker

    speaker = StreamingLLMSpeaker()
    async for chunk in llm.stream("Tell me a story"):
        await speaker.feed(chunk)
    await speaker.finish()

    # Full speech pipeline
    from voice_soundboard.llm import SpeechPipeline

    pipeline = SpeechPipeline(
        stt="whisper",
        llm="ollama",
        tts="kokoro"
    )

    response = await pipeline.converse(audio_input)

    # Context-aware speaking
    from voice_soundboard.llm import ContextAwareSpeaker

    speaker = ContextAwareSpeaker()
    await speaker.speak(
        "I understand how frustrating that must be.",
        context="User expressed frustration about a bug",
        auto_emotion=True
    )
"""

from voice_soundboard.llm.streaming import (
    StreamingLLMSpeaker,
    SentenceBoundaryDetector,
    StreamBuffer,
    StreamConfig,
    StreamState,
)
from voice_soundboard.llm.context import (
    ContextAwareSpeaker,
    EmotionSelector,
    ConversationContext,
    ContextConfig,
    ProsodyHint,
)
from voice_soundboard.llm.pipeline import (
    SpeechPipeline,
    PipelineConfig,
    PipelineState,
    ConversationTurn,
    TurnType,
)
from voice_soundboard.llm.providers import (
    LLMProvider,
    LLMConfig,
    OllamaProvider,
    OpenAIProvider,
    VLLMProvider,
    MockLLMProvider,
    create_provider,
)
from voice_soundboard.llm.interruption import (
    InterruptionHandler,
    InterruptionEvent,
    InterruptionStrategy,
    BargeInDetector,
)
from voice_soundboard.llm.conversation import (
    ConversationManager,
    ConversationConfig,
    ConversationState,
    Message,
    MessageRole,
    TurnTakingStrategy,
)

__all__ = [
    # Streaming
    "StreamingLLMSpeaker",
    "SentenceBoundaryDetector",
    "StreamBuffer",
    "StreamConfig",
    "StreamState",
    # Context
    "ContextAwareSpeaker",
    "EmotionSelector",
    "ConversationContext",
    "ContextConfig",
    "ProsodyHint",
    # Pipeline
    "SpeechPipeline",
    "PipelineConfig",
    "PipelineState",
    "ConversationTurn",
    "TurnType",
    # Providers
    "LLMProvider",
    "LLMConfig",
    "OllamaProvider",
    "OpenAIProvider",
    "VLLMProvider",
    "MockLLMProvider",
    "create_provider",
    # Interruption
    "InterruptionHandler",
    "InterruptionEvent",
    "InterruptionStrategy",
    "BargeInDetector",
    # Conversation
    "ConversationManager",
    "ConversationConfig",
    "ConversationState",
    "Message",
    "MessageRole",
    "TurnTakingStrategy",
]
