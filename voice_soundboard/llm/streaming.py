"""
Streaming LLM integration for real-time speech synthesis.

This module enables speaking as an LLM generates text, providing
low-latency conversational experiences.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable, Optional, List, Any
import numpy as np

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """State of the streaming speaker."""
    IDLE = "idle"
    BUFFERING = "buffering"
    SPEAKING = "speaking"
    FINISHING = "finishing"
    ERROR = "error"


@dataclass
class StreamConfig:
    """Configuration for streaming LLM speaker."""

    # Sentence detection
    sentence_end_chars: str = ".!?"
    min_sentence_length: int = 10
    max_buffer_length: int = 500

    # Timing
    flush_timeout_ms: float = 2000.0  # Force flush after this many ms
    inter_sentence_pause_ms: float = 200.0

    # Speech settings
    voice: Optional[str] = None
    preset: Optional[str] = "assistant"
    speed: float = 1.0
    emotion: Optional[str] = None

    # Behavior
    allow_partial_sentences: bool = True
    smart_punctuation: bool = True  # Handle abbreviations like "Dr.", "Mr."


@dataclass
class StreamBuffer:
    """Buffer for accumulating LLM output tokens."""

    content: str = ""
    last_update: float = field(default_factory=time.time)
    sentences_spoken: int = 0
    tokens_received: int = 0

    def append(self, text: str) -> None:
        """Append text to buffer."""
        self.content += text
        self.last_update = time.time()
        self.tokens_received += 1

    def clear(self) -> str:
        """Clear and return buffer content."""
        content = self.content
        self.content = ""
        return content

    def peek(self) -> str:
        """Get buffer content without clearing."""
        return self.content

    @property
    def age_ms(self) -> float:
        """Time since last update in milliseconds."""
        return (time.time() - self.last_update) * 1000

    @property
    def length(self) -> int:
        """Current buffer length."""
        return len(self.content)


class SentenceBoundaryDetector:
    """
    Detects sentence boundaries in streaming text.

    Handles edge cases like:
    - Abbreviations: "Dr.", "Mr.", "Mrs.", "vs.", "etc."
    - Numbers: "3.14", "$1.99"
    - URLs: "example.com"
    - Ellipsis: "..."
    - Multiple punctuation: "?!", "!!"
    """

    # Common abbreviations that don't end sentences
    ABBREVIATIONS = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
        "vs", "etc", "eg", "ie", "al", "et",
        "inc", "ltd", "corp", "co",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        "st", "nd", "rd", "th",  # Ordinals: 1st, 2nd
        "no", "vol", "pp", "pg",  # Numbers, volumes, pages
    }

    # Patterns that look like sentence endings but aren't
    FALSE_ENDINGS = [
        r'\d+\.\d+',  # Decimal numbers: 3.14
        r'\$\d+\.\d+',  # Currency: $1.99
        r'\w+\.\w+',  # Domain names: example.com
        r'\.{2,}',  # Ellipsis: ...
    ]

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self._false_ending_patterns = [
            re.compile(p) for p in self.FALSE_ENDINGS
        ]

    def find_boundary(self, text: str) -> Optional[int]:
        """
        Find the position of a sentence boundary.

        Returns:
            Index after the sentence-ending punctuation, or None if no boundary found.
        """
        if len(text) < self.config.min_sentence_length:
            return None

        # Look for sentence-ending punctuation
        for i, char in enumerate(text):
            if char in self.config.sentence_end_chars:
                # Check if this is a real sentence ending
                if self._is_real_ending(text, i):
                    # Include any trailing punctuation (e.g., "?!")
                    end_pos = i + 1
                    while end_pos < len(text) and text[end_pos] in self.config.sentence_end_chars:
                        end_pos += 1

                    # Skip trailing whitespace
                    while end_pos < len(text) and text[end_pos] in ' \t':
                        end_pos += 1

                    return end_pos

        return None

    def _is_real_ending(self, text: str, pos: int) -> bool:
        """Check if punctuation at position is a real sentence ending."""
        if pos == 0:
            return False

        # Get the word before the punctuation
        word_start = pos - 1
        while word_start > 0 and text[word_start - 1].isalnum():
            word_start -= 1

        word = text[word_start:pos].lower()

        # Check abbreviations
        if word in self.ABBREVIATIONS:
            return False

        # Check for false ending patterns
        # Look at context around the punctuation
        context_start = max(0, pos - 10)
        context_end = min(len(text), pos + 5)
        context = text[context_start:context_end]

        for pattern in self._false_ending_patterns:
            if pattern.search(context):
                # Check if the match includes our position
                for match in pattern.finditer(context):
                    match_pos = context_start + match.start()
                    if match_pos <= pos < context_start + match.end():
                        return False

        # Check if followed by lowercase (likely not a sentence end)
        if pos + 1 < len(text):
            next_char = text[pos + 1]
            if next_char.islower() and text[pos] == '.':
                return False

        return True

    def split_sentences(self, text: str) -> List[str]:
        """Split text into complete sentences."""
        sentences = []
        remaining = text

        while remaining:
            boundary = self.find_boundary(remaining)
            if boundary is not None:
                sentences.append(remaining[:boundary].strip())
                remaining = remaining[boundary:].strip()
            else:
                break

        return sentences

    def extract_complete(self, text: str) -> tuple[List[str], str]:
        """
        Extract complete sentences and return remaining text.

        Returns:
            Tuple of (complete_sentences, remaining_text)
        """
        sentences = []
        remaining = text

        while True:
            boundary = self.find_boundary(remaining)
            if boundary is not None and boundary <= len(remaining):
                sentence = remaining[:boundary].strip()
                if sentence:
                    sentences.append(sentence)
                remaining = remaining[boundary:].strip()
            else:
                break

        return sentences, remaining


class StreamingLLMSpeaker:
    """
    Speaks LLM output in real-time as tokens are generated.

    Usage:
        speaker = StreamingLLMSpeaker()

        async for token in llm.stream("Tell me a story"):
            await speaker.feed(token)

        await speaker.finish()
    """

    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        on_sentence: Optional[Callable[[str], Any]] = None,
        on_speak_start: Optional[Callable[[], Any]] = None,
        on_speak_end: Optional[Callable[[], Any]] = None,
    ):
        self.config = config or StreamConfig()
        self.buffer = StreamBuffer()
        self.detector = SentenceBoundaryDetector(self.config)
        self.state = StreamState.IDLE

        # Callbacks
        self._on_sentence = on_sentence
        self._on_speak_start = on_speak_start
        self._on_speak_end = on_speak_end

        # Engine reference (lazy loaded)
        self._engine = None
        self._streaming_engine = None

        # Stats
        self.sentences_spoken = 0
        self.total_tokens = 0
        self.start_time: Optional[float] = None

    def _get_engine(self):
        """Get or create TTS engine."""
        if self._engine is None:
            from voice_soundboard import VoiceEngine
            self._engine = VoiceEngine()
        return self._engine

    def _get_streaming_engine(self):
        """Get or create streaming TTS engine."""
        if self._streaming_engine is None:
            from voice_soundboard import StreamingEngine
            self._streaming_engine = StreamingEngine()
        return self._streaming_engine

    async def feed(self, token: str) -> Optional[str]:
        """
        Feed a token from the LLM.

        Returns the sentence if one was completed and spoken, else None.
        """
        if self.state == StreamState.IDLE:
            self.state = StreamState.BUFFERING
            self.start_time = time.time()

        self.buffer.append(token)
        self.total_tokens += 1

        # Check for complete sentences
        sentences, remaining = self.detector.extract_complete(self.buffer.content)

        if sentences:
            self.buffer.content = remaining

            for sentence in sentences:
                await self._speak_sentence(sentence)

            return sentences[-1]  # Return last spoken sentence

        # Check for timeout flush
        if self.buffer.age_ms > self.config.flush_timeout_ms:
            if self.buffer.length >= self.config.min_sentence_length:
                return await self._flush_buffer()

        # Check for max buffer length
        if self.buffer.length > self.config.max_buffer_length:
            return await self._flush_buffer()

        return None

    async def _speak_sentence(self, sentence: str) -> None:
        """Speak a complete sentence."""
        if not sentence.strip():
            return

        self.state = StreamState.SPEAKING

        if self._on_speak_start:
            try:
                callback_result = self._on_speak_start()
                if asyncio.iscoroutine(callback_result):
                    await callback_result
            except Exception as e:
                logger.warning("on_speak_start callback failed: %s", e)

        if self._on_sentence:
            try:
                callback_result = self._on_sentence(sentence)
                if asyncio.iscoroutine(callback_result):
                    await callback_result
            except Exception as e:
                logger.warning("on_sentence callback failed: %s", e)

        # Actually speak using TTS
        try:
            engine = self._get_streaming_engine()
            async for chunk in engine.stream(
                sentence,
                voice=self.config.voice,
                preset=self.config.preset,
                speed=self.config.speed,
            ):
                pass  # Streaming handles playback
        except ImportError as e:
            # Streaming engine not available, fall back to non-streaming
            logger.debug("Streaming engine not available, using non-streaming: %s", e)
            engine = self._get_engine()
            engine.speak(
                sentence,
                voice=self.config.voice,
                preset=self.config.preset,
                speed=self.config.speed,
            )
        except Exception as e:
            # Log the error and fall back to non-streaming TTS
            logger.warning("Streaming TTS failed, falling back to non-streaming: %s", e)
            try:
                engine = self._get_engine()
                engine.speak(
                    sentence,
                    voice=self.config.voice,
                    preset=self.config.preset,
                    speed=self.config.speed,
                )
            except Exception as fallback_error:
                logger.error("TTS fallback also failed: %s", fallback_error)
                self.state = StreamState.ERROR
                raise RuntimeError(
                    f"Failed to speak sentence: streaming failed ({e}), "
                    f"fallback also failed ({fallback_error})"
                ) from fallback_error

        self.sentences_spoken += 1

        if self._on_speak_end:
            try:
                callback_result = self._on_speak_end()
                if asyncio.iscoroutine(callback_result):
                    await callback_result
            except Exception as e:
                logger.warning("on_speak_end callback failed: %s", e)

        # Inter-sentence pause
        if self.config.inter_sentence_pause_ms > 0:
            await asyncio.sleep(self.config.inter_sentence_pause_ms / 1000)

        self.state = StreamState.BUFFERING

    async def _flush_buffer(self) -> Optional[str]:
        """Flush the current buffer as a partial sentence."""
        if not self.config.allow_partial_sentences:
            return None

        content = self.buffer.clear()
        if content.strip():
            await self._speak_sentence(content)
            return content
        return None

    async def finish(self) -> Optional[str]:
        """
        Finish speaking and flush any remaining buffer.

        Call this when the LLM stream is complete.
        """
        self.state = StreamState.FINISHING

        result = None
        if self.buffer.content.strip():
            result = await self._flush_buffer()

        self.state = StreamState.IDLE
        return result

    def reset(self) -> None:
        """Reset the speaker state."""
        self.buffer = StreamBuffer()
        self.state = StreamState.IDLE
        self.sentences_spoken = 0
        self.total_tokens = 0
        self.start_time = None

    @property
    def stats(self) -> dict:
        """Get streaming statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "state": self.state.value,
            "sentences_spoken": self.sentences_spoken,
            "total_tokens": self.total_tokens,
            "buffer_length": self.buffer.length,
            "elapsed_seconds": elapsed,
            "tokens_per_second": self.total_tokens / elapsed if elapsed > 0 else 0,
        }


async def stream_and_speak(
    token_iterator: AsyncIterator[str],
    config: Optional[StreamConfig] = None,
    on_sentence: Optional[Callable[[str], Any]] = None,
) -> dict:
    """
    Convenience function to stream LLM output and speak it.

    Args:
        token_iterator: Async iterator yielding LLM tokens
        config: Stream configuration
        on_sentence: Callback for each spoken sentence

    Returns:
        Statistics dictionary
    """
    speaker = StreamingLLMSpeaker(config=config, on_sentence=on_sentence)

    async for token in token_iterator:
        await speaker.feed(token)

    await speaker.finish()

    return speaker.stats
