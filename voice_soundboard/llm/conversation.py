"""
Conversation state management and turn-taking logic.

Manages the flow of voice conversations including:
- Message history
- Turn-taking
- State persistence
- Conversation analytics
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path


class MessageRole(Enum):
    """Role of a message sender."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ConversationState(Enum):
    """State of the conversation."""
    IDLE = "idle"
    LISTENING = "listening"  # Waiting for user input
    PROCESSING = "processing"  # LLM generating response
    SPEAKING = "speaking"  # TTS playing response
    INTERRUPTED = "interrupted"
    ENDED = "ended"


class TurnTakingStrategy(Enum):
    """How to handle turn-taking."""
    STRICT = "strict"  # Wait for explicit end-of-turn
    SILENCE = "silence"  # End turn on silence detection
    PUSH_TO_TALK = "push_to_talk"  # Manual button press
    HOTWORD = "hotword"  # Wake word activation
    CONTINUOUS = "continuous"  # Always listening


@dataclass
class Message:
    """A single message in the conversation."""

    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Metadata
    emotion: Optional[str] = None
    voice: Optional[str] = None
    duration_ms: Optional[float] = None
    tokens: Optional[int] = None

    # Audio (optional)
    audio_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "emotion": self.emotion,
            "voice": self.voice,
            "duration_ms": self.duration_ms,
            "tokens": self.tokens,
            "audio_path": self.audio_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            emotion=data.get("emotion"),
            voice=data.get("voice"),
            duration_ms=data.get("duration_ms"),
            tokens=data.get("tokens"),
            audio_path=data.get("audio_path"),
        )


@dataclass
class ConversationConfig:
    """Configuration for conversation management."""

    # Turn-taking
    turn_taking_strategy: TurnTakingStrategy = TurnTakingStrategy.SILENCE
    silence_threshold_ms: float = 1500.0  # Silence before end of turn
    max_turn_duration_ms: float = 30000.0  # Maximum turn length

    # History
    max_history_messages: int = 100
    context_window_messages: int = 10  # Messages to send to LLM

    # System prompt
    system_prompt: Optional[str] = None

    # Persistence
    auto_save: bool = False
    save_path: Optional[Path] = None

    # Timeouts
    idle_timeout_ms: float = 300000.0  # 5 minutes
    processing_timeout_ms: float = 30000.0  # 30 seconds


class ConversationManager:
    """
    Manages conversation state and history.

    Usage:
        manager = ConversationManager()

        # Start conversation
        manager.start()

        # Add user message
        manager.add_user_message("Hello!")

        # Get context for LLM
        messages = manager.get_llm_context()

        # Add assistant response
        manager.add_assistant_message("Hi there! How can I help?")

        # End conversation
        manager.end()
    """

    def __init__(
        self,
        config: Optional[ConversationConfig] = None,
        conversation_id: Optional[str] = None,
    ):
        self.config = config or ConversationConfig()
        self.id = conversation_id or str(uuid.uuid4())[:12]

        # State
        self.state = ConversationState.IDLE
        self.messages: List[Message] = []
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None

        # Turn tracking
        self._current_turn_start: Optional[float] = None
        self._whose_turn: Optional[MessageRole] = None
        self._turn_count = 0

        # Callbacks
        self._on_state_change: Optional[Callable[[ConversationState], Any]] = None
        self._on_turn_end: Optional[Callable[[MessageRole], Any]] = None

        # Add system prompt if configured
        if self.config.system_prompt:
            self.messages.append(Message(
                role=MessageRole.SYSTEM,
                content=self.config.system_prompt,
            ))

    @property
    def on_state_change(self) -> Optional[Callable[[ConversationState], Any]]:
        """Get state change callback."""
        return self._on_state_change

    @on_state_change.setter
    def on_state_change(self, callback: Callable[[ConversationState], Any]) -> None:
        """Set state change callback."""
        self._on_state_change = callback

    @property
    def on_turn_end(self) -> Optional[Callable[[MessageRole], Any]]:
        """Get turn end callback."""
        return self._on_turn_end

    @on_turn_end.setter
    def on_turn_end(self, callback: Callable[[MessageRole], Any]) -> None:
        """Set turn end callback."""
        self._on_turn_end = callback

    def _set_state(self, state: ConversationState) -> None:
        """Set conversation state and notify."""
        old_state = self.state
        self.state = state

        if self._on_state_change and old_state != state:
            result = self._on_state_change(state)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)

    def start(self) -> None:
        """Start a new conversation."""
        self.started_at = time.time()
        self._set_state(ConversationState.LISTENING)
        self._whose_turn = MessageRole.USER

    def end(self) -> None:
        """End the conversation."""
        self.ended_at = time.time()
        self._set_state(ConversationState.ENDED)

        if self.config.auto_save and self.config.save_path:
            self.save(self.config.save_path)

    def add_user_message(
        self,
        content: str,
        emotion: Optional[str] = None,
        audio_path: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> Message:
        """Add a user message."""
        message = Message(
            role=MessageRole.USER,
            content=content,
            emotion=emotion,
            audio_path=audio_path,
            duration_ms=duration_ms,
        )

        self.messages.append(message)
        self._end_turn(MessageRole.USER)
        self._set_state(ConversationState.PROCESSING)

        # Trim history if needed
        self._trim_history()

        if self.config.auto_save and self.config.save_path:
            self.save(self.config.save_path)

        return message

    def add_assistant_message(
        self,
        content: str,
        emotion: Optional[str] = None,
        voice: Optional[str] = None,
        audio_path: Optional[str] = None,
        duration_ms: Optional[float] = None,
        tokens: Optional[int] = None,
    ) -> Message:
        """Add an assistant message."""
        message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            emotion=emotion,
            voice=voice,
            audio_path=audio_path,
            duration_ms=duration_ms,
            tokens=tokens,
        )

        self.messages.append(message)
        self._end_turn(MessageRole.ASSISTANT)
        self._set_state(ConversationState.LISTENING)

        if self.config.auto_save and self.config.save_path:
            self.save(self.config.save_path)

        return message

    def set_speaking(self) -> None:
        """Mark that the assistant is speaking."""
        self._set_state(ConversationState.SPEAKING)

    def set_interrupted(self) -> None:
        """Mark that the conversation was interrupted."""
        self._set_state(ConversationState.INTERRUPTED)

    def _end_turn(self, role: MessageRole) -> None:
        """End a turn and notify."""
        self._turn_count += 1
        self._current_turn_start = None

        # Switch turns
        if role == MessageRole.USER:
            self._whose_turn = MessageRole.ASSISTANT
        else:
            self._whose_turn = MessageRole.USER

        if self._on_turn_end:
            result = self._on_turn_end(role)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)

    def _trim_history(self) -> None:
        """Trim message history to configured limit."""
        if len(self.messages) > self.config.max_history_messages:
            # Keep system messages and recent messages
            system_messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
            other_messages = [m for m in self.messages if m.role != MessageRole.SYSTEM]

            keep_count = self.config.max_history_messages - len(system_messages)
            self.messages = system_messages + other_messages[-keep_count:]

    def get_llm_context(self) -> List[Dict[str, str]]:
        """
        Get messages formatted for LLM.

        Returns the most recent messages within the context window.
        """
        # Get system messages and recent conversation
        context = []

        for msg in self.messages:
            if msg.role == MessageRole.SYSTEM:
                context.append({"role": "system", "content": msg.content})

        # Add recent messages
        non_system = [m for m in self.messages if m.role != MessageRole.SYSTEM]
        recent = non_system[-self.config.context_window_messages:]

        for msg in recent:
            context.append({
                "role": msg.role.value,
                "content": msg.content,
            })

        return context

    def get_last_message(self, role: Optional[MessageRole] = None) -> Optional[Message]:
        """Get the last message, optionally filtered by role."""
        for msg in reversed(self.messages):
            if role is None or msg.role == role:
                return msg
        return None

    def save(self, path: Path) -> None:
        """Save conversation to file."""
        data = {
            "id": self.id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "turn_count": self._turn_count,
            "messages": [m.to_dict() for m in self.messages],
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path, config: Optional[ConversationConfig] = None) -> "ConversationManager":
        """Load conversation from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        manager = cls(config=config, conversation_id=data.get("id"))
        manager.started_at = data.get("started_at")
        manager.ended_at = data.get("ended_at")
        manager._turn_count = data.get("turn_count", 0)
        manager.messages = [Message.from_dict(m) for m in data.get("messages", [])]

        if manager.ended_at:
            manager.state = ConversationState.ENDED

        return manager

    @property
    def duration_seconds(self) -> float:
        """Get conversation duration in seconds."""
        if not self.started_at:
            return 0.0
        end = self.ended_at or time.time()
        return end - self.started_at

    @property
    def stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        user_messages = [m for m in self.messages if m.role == MessageRole.USER]
        assistant_messages = [m for m in self.messages if m.role == MessageRole.ASSISTANT]

        user_words = sum(len(m.content.split()) for m in user_messages)
        assistant_words = sum(len(m.content.split()) for m in assistant_messages)

        return {
            "id": self.id,
            "state": self.state.value,
            "duration_seconds": self.duration_seconds,
            "turn_count": self._turn_count,
            "message_count": len(self.messages),
            "user_message_count": len(user_messages),
            "assistant_message_count": len(assistant_messages),
            "user_word_count": user_words,
            "assistant_word_count": assistant_words,
            "whose_turn": self._whose_turn.value if self._whose_turn else None,
        }


class TurnTakingController:
    """
    Controls turn-taking in voice conversations.

    Determines when the user has finished speaking and
    when the assistant should start/stop.
    """

    def __init__(
        self,
        strategy: TurnTakingStrategy = TurnTakingStrategy.SILENCE,
        silence_threshold_ms: float = 1500.0,
    ):
        self.strategy = strategy
        self.silence_threshold_ms = silence_threshold_ms

        # State
        self._is_user_speaking = False
        self._silence_start: Optional[float] = None
        self._speech_start: Optional[float] = None

        # Callbacks
        self._on_user_start: Optional[Callable[[], Any]] = None
        self._on_user_end: Optional[Callable[[], Any]] = None

    @property
    def on_user_start(self) -> Optional[Callable[[], Any]]:
        """Get user start speaking callback."""
        return self._on_user_start

    @on_user_start.setter
    def on_user_start(self, callback: Callable[[], Any]) -> None:
        """Set user start speaking callback."""
        self._on_user_start = callback

    @property
    def on_user_end(self) -> Optional[Callable[[], Any]]:
        """Get user end speaking callback."""
        return self._on_user_end

    @on_user_end.setter
    def on_user_end(self, callback: Callable[[], Any]) -> None:
        """Set user end speaking callback."""
        self._on_user_end = callback

    def process_audio(self, is_speech: bool) -> bool:
        """
        Process audio frame for turn detection.

        Args:
            is_speech: Whether speech was detected in this frame

        Returns:
            True if turn ended, False otherwise
        """
        if self.strategy == TurnTakingStrategy.SILENCE:
            return self._process_silence_strategy(is_speech)
        elif self.strategy == TurnTakingStrategy.STRICT:
            return False  # Requires explicit end signal
        elif self.strategy == TurnTakingStrategy.PUSH_TO_TALK:
            return False  # Requires button release
        elif self.strategy == TurnTakingStrategy.CONTINUOUS:
            return False  # Never ends turn automatically

        return False

    def _process_silence_strategy(self, is_speech: bool) -> bool:
        """Process audio for silence-based turn-taking."""
        now = time.time()

        if is_speech:
            if not self._is_user_speaking:
                # User started speaking
                self._is_user_speaking = True
                self._speech_start = now
                self._silence_start = None

                if self._on_user_start:
                    result = self._on_user_start()
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
            else:
                # User continues speaking
                self._silence_start = None
        else:
            if self._is_user_speaking:
                if self._silence_start is None:
                    # Silence just started
                    self._silence_start = now
                else:
                    # Check if silence exceeded threshold
                    silence_duration = (now - self._silence_start) * 1000
                    if silence_duration >= self.silence_threshold_ms:
                        # Turn ended
                        self._is_user_speaking = False
                        self._silence_start = None

                        if self._on_user_end:
                            result = self._on_user_end()
                            if asyncio.iscoroutine(result):
                                asyncio.create_task(result)

                        return True

        return False

    def force_end_turn(self) -> None:
        """Force end of user turn."""
        if self._is_user_speaking:
            self._is_user_speaking = False
            self._silence_start = None

            if self._on_user_end:
                result = self._on_user_end()
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)

    def reset(self) -> None:
        """Reset turn-taking state."""
        self._is_user_speaking = False
        self._silence_start = None
        self._speech_start = None
