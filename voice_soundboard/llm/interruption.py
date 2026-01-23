"""
Interruption and barge-in handling for voice conversations.

Enables natural conversational flow by allowing users to interrupt
the AI while it's speaking.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Any, List
import threading


class InterruptionStrategy(Enum):
    """How to handle interruptions."""
    IGNORE = "ignore"  # Don't allow interruptions
    STOP_IMMEDIATE = "stop_immediate"  # Stop speaking immediately
    STOP_SENTENCE = "stop_sentence"  # Finish current sentence, then stop
    PAUSE = "pause"  # Pause and allow resume
    QUEUE = "queue"  # Queue the interruption for later


@dataclass
class InterruptionEvent:
    """An interruption event."""

    timestamp: float = field(default_factory=time.time)
    trigger: str = "user"  # user, voice_activity, hotword, manual
    audio_level: Optional[float] = None  # dB level if voice activity
    transcript: Optional[str] = None  # Transcribed interruption
    handled: bool = False
    strategy_used: Optional[InterruptionStrategy] = None

    @property
    def age_ms(self) -> float:
        """Age of the event in milliseconds."""
        return (time.time() - self.timestamp) * 1000


@dataclass
class BargeInConfig:
    """Configuration for barge-in detection."""

    # Enable/disable
    enabled: bool = True

    # Voice activity detection
    vad_threshold_db: float = -35.0  # dB threshold for voice activity
    vad_duration_ms: float = 200.0  # Minimum duration to trigger
    vad_cooldown_ms: float = 500.0  # Cooldown after interruption

    # Hotword detection
    hotword_enabled: bool = False
    hotwords: List[str] = field(default_factory=lambda: ["hey", "stop", "wait"])

    # Strategy
    default_strategy: InterruptionStrategy = InterruptionStrategy.STOP_IMMEDIATE

    # Timing
    ignore_initial_ms: float = 500.0  # Ignore interruptions in first N ms
    min_spoken_ms: float = 1000.0  # Minimum speaking before allowing interruption


class BargeInDetector:
    """
    Detects when the user wants to interrupt.

    Uses:
    - Voice activity detection (VAD)
    - Hotword detection
    - Manual trigger
    """

    def __init__(self, config: Optional[BargeInConfig] = None):
        self.config = config or BargeInConfig()
        self._is_listening = False
        self._last_trigger: Optional[float] = None
        self._callbacks: List[Callable[[InterruptionEvent], Any]] = []
        self._lock = threading.Lock()

        # State
        self._speaking_started: Optional[float] = None
        self._voice_active = False
        self._voice_start: Optional[float] = None

    def start_listening(self) -> None:
        """Start listening for interruptions."""
        with self._lock:
            self._is_listening = True
            self._speaking_started = time.time()

    def stop_listening(self) -> None:
        """Stop listening for interruptions."""
        with self._lock:
            self._is_listening = False
            self._speaking_started = None

    def on_interrupt(self, callback: Callable[[InterruptionEvent], Any]) -> None:
        """Register a callback for interruption events."""
        self._callbacks.append(callback)

    def check_audio_level(self, level_db: float) -> Optional[InterruptionEvent]:
        """
        Check audio level for voice activity.

        Args:
            level_db: Audio level in decibels

        Returns:
            InterruptionEvent if triggered, None otherwise
        """
        if not self.config.enabled or not self._is_listening:
            return None

        # Check if we're past the initial ignore period
        if self._speaking_started:
            elapsed = (time.time() - self._speaking_started) * 1000
            if elapsed < self.config.ignore_initial_ms:
                return None
            if elapsed < self.config.min_spoken_ms:
                return None

        # Check cooldown
        if self._last_trigger:
            cooldown_elapsed = (time.time() - self._last_trigger) * 1000
            if cooldown_elapsed < self.config.vad_cooldown_ms:
                return None

        # Voice activity detection
        if level_db >= self.config.vad_threshold_db:
            if not self._voice_active:
                self._voice_active = True
                self._voice_start = time.time()
            else:
                # Check if voice has been active long enough
                duration = (time.time() - self._voice_start) * 1000
                if duration >= self.config.vad_duration_ms:
                    return self._trigger_interrupt("voice_activity", audio_level=level_db)
        else:
            self._voice_active = False
            self._voice_start = None

        return None

    def check_transcript(self, text: str) -> Optional[InterruptionEvent]:
        """
        Check transcribed text for hotwords.

        Args:
            text: Transcribed text from STT

        Returns:
            InterruptionEvent if hotword detected, None otherwise
        """
        if not self.config.enabled or not self.config.hotword_enabled:
            return None
        if not self._is_listening:
            return None

        text_lower = text.lower()
        for hotword in self.config.hotwords:
            if hotword.lower() in text_lower:
                return self._trigger_interrupt("hotword", transcript=text)

        return None

    def trigger_manual(self) -> InterruptionEvent:
        """Manually trigger an interruption."""
        return self._trigger_interrupt("manual")

    def _trigger_interrupt(
        self,
        trigger: str,
        audio_level: Optional[float] = None,
        transcript: Optional[str] = None,
    ) -> InterruptionEvent:
        """Create and dispatch an interruption event."""
        with self._lock:
            self._last_trigger = time.time()

            event = InterruptionEvent(
                trigger=trigger,
                audio_level=audio_level,
                transcript=transcript,
            )

            # Dispatch to callbacks
            for callback in self._callbacks:
                try:
                    result = callback(event)
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                except Exception:
                    pass  # Don't let callback errors break the flow

            return event


class InterruptionHandler:
    """
    Handles interruptions during speech synthesis.

    Usage:
        handler = InterruptionHandler()
        handler.on_interrupt = lambda: engine.stop()

        # Start speaking
        handler.start_session()

        # ... speaking happens ...

        # User interrupts
        if handler.check_interrupt(audio_level=-30):
            # Speaking stopped automatically
            pass

        handler.end_session()
    """

    def __init__(
        self,
        strategy: InterruptionStrategy = InterruptionStrategy.STOP_IMMEDIATE,
        config: Optional[BargeInConfig] = None,
    ):
        self.strategy = strategy
        self.detector = BargeInDetector(config)

        # Callbacks
        self._on_interrupt: Optional[Callable[[], Any]] = None
        self._on_pause: Optional[Callable[[], Any]] = None
        self._on_resume: Optional[Callable[[], Any]] = None

        # State
        self._is_active = False
        self._is_paused = False
        self._interruptions: List[InterruptionEvent] = []
        self._queue: List[str] = []  # Queued user messages

        # Register detector callback
        self.detector.on_interrupt(self._handle_event)

    @property
    def on_interrupt(self) -> Optional[Callable[[], Any]]:
        """Get interrupt callback."""
        return self._on_interrupt

    @on_interrupt.setter
    def on_interrupt(self, callback: Callable[[], Any]) -> None:
        """Set interrupt callback."""
        self._on_interrupt = callback

    @property
    def on_pause(self) -> Optional[Callable[[], Any]]:
        """Get pause callback."""
        return self._on_pause

    @on_pause.setter
    def on_pause(self, callback: Callable[[], Any]) -> None:
        """Set pause callback."""
        self._on_pause = callback

    @property
    def on_resume(self) -> Optional[Callable[[], Any]]:
        """Get resume callback."""
        return self._on_resume

    @on_resume.setter
    def on_resume(self, callback: Callable[[], Any]) -> None:
        """Set resume callback."""
        self._on_resume = callback

    def start_session(self) -> None:
        """Start a speaking session."""
        self._is_active = True
        self._is_paused = False
        self.detector.start_listening()

    def end_session(self) -> None:
        """End the speaking session."""
        self._is_active = False
        self._is_paused = False
        self.detector.stop_listening()

    def check_interrupt(
        self,
        audio_level: Optional[float] = None,
        transcript: Optional[str] = None,
    ) -> bool:
        """
        Check for interruption conditions.

        Args:
            audio_level: Current audio level in dB
            transcript: Transcribed user speech

        Returns:
            True if interrupted, False otherwise
        """
        if not self._is_active:
            return False

        event = None

        if audio_level is not None:
            event = self.detector.check_audio_level(audio_level)

        if event is None and transcript is not None:
            event = self.detector.check_transcript(transcript)

        return event is not None and event.handled

    def force_interrupt(self) -> InterruptionEvent:
        """Force an immediate interruption."""
        event = self.detector.trigger_manual()
        self._handle_event(event)
        return event

    def _handle_event(self, event: InterruptionEvent) -> None:
        """Handle an interruption event."""
        if not self._is_active:
            return

        event.strategy_used = self.strategy
        self._interruptions.append(event)

        if self.strategy == InterruptionStrategy.IGNORE:
            return

        elif self.strategy == InterruptionStrategy.STOP_IMMEDIATE:
            if self._on_interrupt:
                result = self._on_interrupt()
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            event.handled = True

        elif self.strategy == InterruptionStrategy.STOP_SENTENCE:
            # This would need integration with the streaming speaker
            # to wait for sentence boundary
            if self._on_interrupt:
                result = self._on_interrupt()
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            event.handled = True

        elif self.strategy == InterruptionStrategy.PAUSE:
            if not self._is_paused:
                self._is_paused = True
                if self._on_pause:
                    result = self._on_pause()
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
            event.handled = True

        elif self.strategy == InterruptionStrategy.QUEUE:
            if event.transcript:
                self._queue.append(event.transcript)
            event.handled = True

    def resume(self) -> None:
        """Resume after a pause interruption."""
        if self._is_paused:
            self._is_paused = False
            if self._on_resume:
                result = self._on_resume()
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)

    def get_queued_messages(self) -> List[str]:
        """Get and clear queued messages."""
        messages = self._queue.copy()
        self._queue.clear()
        return messages

    @property
    def stats(self) -> dict:
        """Get interruption statistics."""
        return {
            "is_active": self._is_active,
            "is_paused": self._is_paused,
            "total_interruptions": len(self._interruptions),
            "queued_messages": len(self._queue),
            "strategy": self.strategy.value,
        }
