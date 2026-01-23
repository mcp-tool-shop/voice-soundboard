"""
Speech-to-Speech Pipeline.

Complete voice conversation system that handles:
STT → LLM → TTS

Enables natural voice conversations with AI.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable, Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class TurnType(Enum):
    """Type of conversation turn."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class PipelineState(Enum):
    """State of the speech pipeline."""
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    type: TurnType
    content: str
    timestamp: float = field(default_factory=time.time)

    # Audio
    audio: Optional[np.ndarray] = None
    audio_path: Optional[Path] = None
    duration_ms: Optional[float] = None

    # Metadata
    emotion: Optional[str] = None
    voice: Optional[str] = None
    confidence: Optional[float] = None

    # Timing
    transcription_ms: Optional[float] = None
    llm_ms: Optional[float] = None
    tts_ms: Optional[float] = None


@dataclass
class PipelineConfig:
    """Configuration for speech pipeline."""

    # STT settings
    stt_backend: str = "whisper"
    stt_model: str = "base"
    stt_language: str = "en"

    # LLM settings
    llm_backend: str = "ollama"
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    # TTS settings
    tts_backend: str = "kokoro"
    tts_voice: Optional[str] = None
    tts_preset: str = "assistant"
    tts_speed: float = 1.0

    # System prompt
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational."

    # Context-aware settings
    auto_emotion: bool = True
    emotion_from_context: bool = True

    # Interruption settings
    allow_interruption: bool = True
    interruption_threshold_db: float = -35.0

    # Turn-taking
    silence_threshold_ms: float = 1500.0
    max_turn_duration_ms: float = 30000.0

    # Streaming
    stream_tts: bool = True
    speak_as_generated: bool = True


class SpeechPipeline:
    """
    Complete speech-to-speech pipeline.

    Usage:
        pipeline = SpeechPipeline(
            stt="whisper",
            llm="ollama",
            tts="kokoro"
        )

        # Full conversation
        response = await pipeline.converse(audio_input)

        # Streaming conversation
        async for event in pipeline.converse_stream(audio_input):
            print(event)
    """

    def __init__(
        self,
        stt: str = "whisper",
        llm: str = "ollama",
        tts: str = "kokoro",
        config: Optional[PipelineConfig] = None,
    ):
        self.config = config or PipelineConfig(
            stt_backend=stt,
            llm_backend=llm,
            tts_backend=tts,
        )

        # State
        self.state = PipelineState.IDLE
        self.conversation_history: List[ConversationTurn] = []

        # Components (lazy loaded)
        self._stt = None
        self._llm = None
        self._tts = None
        self._streaming_tts = None
        self._interruption_handler = None
        self._context_speaker = None
        self._streaming_speaker = None

        # Callbacks
        self._on_state_change: Optional[Callable[[PipelineState], Any]] = None
        self._on_transcription: Optional[Callable[[str], Any]] = None
        self._on_response: Optional[Callable[[str], Any]] = None
        self._on_audio: Optional[Callable[[np.ndarray], Any]] = None

    def _get_llm(self):
        """Get or create LLM provider."""
        if self._llm is None:
            from voice_soundboard.llm.providers import create_provider, LLMConfig

            config = LLMConfig(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
            self._llm = create_provider(self.config.llm_backend, config)

        return self._llm

    def _get_tts(self):
        """Get or create TTS engine."""
        if self._tts is None:
            from voice_soundboard import VoiceEngine
            self._tts = VoiceEngine()
        return self._tts

    def _get_streaming_tts(self):
        """Get or create streaming TTS engine."""
        if self._streaming_tts is None:
            from voice_soundboard import StreamingEngine
            self._streaming_tts = StreamingEngine()
        return self._streaming_tts

    def _get_interruption_handler(self):
        """Get or create interruption handler."""
        if self._interruption_handler is None:
            from voice_soundboard.llm.interruption import (
                InterruptionHandler,
                InterruptionStrategy,
                BargeInConfig,
            )

            config = BargeInConfig(
                enabled=self.config.allow_interruption,
                vad_threshold_db=self.config.interruption_threshold_db,
            )
            self._interruption_handler = InterruptionHandler(
                strategy=InterruptionStrategy.STOP_IMMEDIATE,
                config=config,
            )

        return self._interruption_handler

    def _get_context_speaker(self):
        """Get or create context-aware speaker."""
        if self._context_speaker is None:
            from voice_soundboard.llm.context import ContextAwareSpeaker, ContextConfig

            config = ContextConfig(
                enable_auto_emotion=self.config.auto_emotion,
            )
            self._context_speaker = ContextAwareSpeaker(config=config)

        return self._context_speaker

    def _get_streaming_speaker(self):
        """Get or create streaming LLM speaker."""
        if self._streaming_speaker is None:
            from voice_soundboard.llm.streaming import StreamingLLMSpeaker, StreamConfig

            config = StreamConfig(
                voice=self.config.tts_voice,
                preset=self.config.tts_preset,
                speed=self.config.tts_speed,
            )
            self._streaming_speaker = StreamingLLMSpeaker(config=config)

        return self._streaming_speaker

    def _set_state(self, state: PipelineState) -> None:
        """Set pipeline state and notify."""
        old_state = self.state
        self.state = state

        if self._on_state_change and old_state != state:
            try:
                result = self._on_state_change(state)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.warning("on_state_change callback failed: %s", e)

    @property
    def on_state_change(self) -> Optional[Callable[[PipelineState], Any]]:
        """Get state change callback."""
        return self._on_state_change

    @on_state_change.setter
    def on_state_change(self, callback: Callable[[PipelineState], Any]) -> None:
        """Set state change callback."""
        self._on_state_change = callback

    @property
    def on_transcription(self) -> Optional[Callable[[str], Any]]:
        """Get transcription callback."""
        return self._on_transcription

    @on_transcription.setter
    def on_transcription(self, callback: Callable[[str], Any]) -> None:
        """Set transcription callback."""
        self._on_transcription = callback

    @property
    def on_response(self) -> Optional[Callable[[str], Any]]:
        """Get response callback."""
        return self._on_response

    @on_response.setter
    def on_response(self, callback: Callable[[str], Any]) -> None:
        """Set response callback."""
        self._on_response = callback

    async def transcribe(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data or path to audio file
            sample_rate: Sample rate if audio is numpy array

        Returns:
            Transcribed text
        """
        self._set_state(PipelineState.TRANSCRIBING)

        # Mock transcription for now
        # In production, would integrate with Whisper or other STT
        try:
            # Try to use whisper if available
            import whisper

            model = whisper.load_model(self.config.stt_model)

            if isinstance(audio, (str, Path)):
                result = model.transcribe(str(audio), language=self.config.stt_language)
            else:
                # Save to temp file for whisper
                import tempfile
                import soundfile as sf

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio, sample_rate)
                    result = model.transcribe(f.name, language=self.config.stt_language)

            return result["text"].strip()

        except ImportError:
            # Fallback to mock
            return "[Mock transcription - install whisper for real STT]"

    async def generate_response(
        self,
        user_input: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Generate LLM response.

        Args:
            user_input: User's transcribed input
            context: Optional additional context

        Returns:
            LLM response text

        Raises:
            RuntimeError: If LLM fails to generate response
        """
        self._set_state(PipelineState.THINKING)

        # Build messages
        messages = [
            {"role": "system", "content": self.config.system_prompt}
        ]

        # Add conversation history
        for turn in self.conversation_history[-10:]:  # Last 10 turns
            messages.append({
                "role": turn.type.value,
                "content": turn.content,
            })

        # Add current input
        messages.append({"role": "user", "content": user_input})

        # Generate response
        try:
            llm = self._get_llm()
            response = await llm.chat(messages)
            return response.content
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            self._set_state(PipelineState.ERROR)
            raise RuntimeError(
                f"Failed to generate LLM response using {self.config.llm_backend}: {e}"
            ) from e

    async def speak(
        self,
        text: str,
        emotion: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Speak text using TTS.

        Args:
            text: Text to speak
            emotion: Optional emotion
            voice: Optional voice override

        Returns:
            Audio data if not playing immediately
        """
        self._set_state(PipelineState.SPEAKING)

        if self.config.auto_emotion and emotion is None:
            speaker = self._get_context_speaker()
            result = speaker.speak(
                text,
                auto_emotion=True,
                voice=voice or self.config.tts_voice,
                preset=self.config.tts_preset,
                speed=self.config.tts_speed,
            )
            return None  # Audio played by speaker

        tts = self._get_tts()
        result = tts.speak(
            text,
            voice=voice or self.config.tts_voice,
            preset=self.config.tts_preset,
            speed=self.config.tts_speed,
        )

        return result.audio if hasattr(result, 'audio') else None

    async def converse(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
    ) -> ConversationTurn:
        """
        Complete a single conversation turn.

        1. Transcribe user audio (STT)
        2. Generate LLM response
        3. Speak response (TTS)

        Args:
            audio: User audio input
            sample_rate: Audio sample rate

        Returns:
            ConversationTurn with response

        Raises:
            RuntimeError: If any stage of the pipeline fails
        """
        start_time = time.time()

        try:
            # 1. Transcribe
            self._set_state(PipelineState.LISTENING)
            transcription_start = time.time()
            try:
                user_text = await self.transcribe(audio, sample_rate)
            except Exception as e:
                logger.error("Transcription failed: %s", e)
                self._set_state(PipelineState.ERROR)
                raise RuntimeError(f"STT failed: {e}") from e
            transcription_ms = (time.time() - transcription_start) * 1000

            if self._on_transcription:
                try:
                    callback_result = self._on_transcription(user_text)
                    if asyncio.iscoroutine(callback_result):
                        await callback_result
                except Exception as e:
                    logger.warning("on_transcription callback failed: %s", e)

            # Record user turn
            user_turn = ConversationTurn(
                type=TurnType.USER,
                content=user_text,
                transcription_ms=transcription_ms,
            )
            self.conversation_history.append(user_turn)

            # 2. Generate response
            llm_start = time.time()
            response_text = await self.generate_response(user_text)
            llm_ms = (time.time() - llm_start) * 1000

            if self._on_response:
                try:
                    callback_result = self._on_response(response_text)
                    if asyncio.iscoroutine(callback_result):
                        await callback_result
                except Exception as e:
                    logger.warning("on_response callback failed: %s", e)

            # 3. Speak response
            tts_start = time.time()
            try:
                await self.speak(response_text)
            except Exception as e:
                logger.error("TTS failed: %s", e)
                # Don't fail the whole turn if TTS fails - we still have the text
                logger.warning("Continuing without speech due to TTS error")
            tts_ms = (time.time() - tts_start) * 1000

            # Record assistant turn
            assistant_turn = ConversationTurn(
                type=TurnType.ASSISTANT,
                content=response_text,
                llm_ms=llm_ms,
                tts_ms=tts_ms,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self.conversation_history.append(assistant_turn)

            self._set_state(PipelineState.IDLE)
            return assistant_turn

        except Exception as e:
            self._set_state(PipelineState.ERROR)
            logger.error("Conversation turn failed: %s", e)
            raise

    async def converse_stream(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a conversation turn.

        Yields events as they happen for real-time feedback.

        Args:
            audio: User audio input
            sample_rate: Audio sample rate

        Yields:
            Event dictionaries with type and data
        """
        start_time = time.time()

        # 1. Transcribe
        self._set_state(PipelineState.LISTENING)
        yield {"type": "state", "state": "listening"}

        transcription_start = time.time()
        user_text = await self.transcribe(audio, sample_rate)
        transcription_ms = (time.time() - transcription_start) * 1000

        yield {
            "type": "transcription",
            "text": user_text,
            "duration_ms": transcription_ms,
        }

        # Record user turn
        user_turn = ConversationTurn(
            type=TurnType.USER,
            content=user_text,
            transcription_ms=transcription_ms,
        )
        self.conversation_history.append(user_turn)

        # 2. Generate and stream response
        self._set_state(PipelineState.THINKING)
        yield {"type": "state", "state": "thinking"}

        llm = self._get_llm()

        # Build messages
        messages = [{"role": "system", "content": self.config.system_prompt}]
        for turn in self.conversation_history[-10:]:
            messages.append({"role": turn.type.value, "content": turn.content})
        messages.append({"role": "user", "content": user_text})

        # Stream LLM response
        full_response = ""
        llm_start = time.time()

        if self.config.speak_as_generated:
            # Use streaming speaker
            speaker = self._get_streaming_speaker()

            async for token in llm.chat_stream(messages):
                full_response += token
                yield {"type": "token", "token": token}

                sentence = await speaker.feed(token)
                if sentence:
                    yield {"type": "sentence", "text": sentence}

            # Finish any remaining text
            await speaker.finish()
        else:
            # Collect full response first
            async for token in llm.chat_stream(messages):
                full_response += token
                yield {"type": "token", "token": token}

        llm_ms = (time.time() - llm_start) * 1000

        yield {
            "type": "response",
            "text": full_response,
            "duration_ms": llm_ms,
        }

        # 3. Speak if not already spoken
        if not self.config.speak_as_generated:
            self._set_state(PipelineState.SPEAKING)
            yield {"type": "state", "state": "speaking"}

            tts_start = time.time()
            await self.speak(full_response)
            tts_ms = (time.time() - tts_start) * 1000

            yield {"type": "tts_complete", "duration_ms": tts_ms}
        else:
            tts_ms = 0  # Already spoken during streaming

        # Record assistant turn
        assistant_turn = ConversationTurn(
            type=TurnType.ASSISTANT,
            content=full_response,
            llm_ms=llm_ms,
            tts_ms=tts_ms,
            duration_ms=(time.time() - start_time) * 1000,
        )
        self.conversation_history.append(assistant_turn)

        self._set_state(PipelineState.IDLE)
        yield {"type": "complete", "turn": assistant_turn}

    def interrupt(self) -> None:
        """Interrupt current speech."""
        if self.state == PipelineState.SPEAKING:
            self._set_state(PipelineState.INTERRUPTED)

            # Stop TTS
            try:
                from voice_soundboard import stop_playback
                stop_playback()
            except ImportError as e:
                logger.debug("stop_playback not available: %s", e)
            except Exception as e:
                logger.warning("Failed to stop playback during interrupt: %s", e)

            # Stop streaming speaker
            if self._streaming_speaker:
                try:
                    self._streaming_speaker.reset()
                except Exception as e:
                    logger.warning("Failed to reset streaming speaker: %s", e)

    def reset(self) -> None:
        """Reset the pipeline state."""
        self.interrupt()
        self.conversation_history.clear()
        self._set_state(PipelineState.IDLE)

        if self._streaming_speaker:
            self._streaming_speaker.reset()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        total_user_ms = sum(
            t.transcription_ms or 0
            for t in self.conversation_history
            if t.type == TurnType.USER
        )
        total_llm_ms = sum(
            t.llm_ms or 0
            for t in self.conversation_history
            if t.type == TurnType.ASSISTANT
        )
        total_tts_ms = sum(
            t.tts_ms or 0
            for t in self.conversation_history
            if t.type == TurnType.ASSISTANT
        )

        return {
            "state": self.state.value,
            "turn_count": len(self.conversation_history),
            "total_transcription_ms": total_user_ms,
            "total_llm_ms": total_llm_ms,
            "total_tts_ms": total_tts_ms,
            "config": {
                "stt": self.config.stt_backend,
                "llm": self.config.llm_backend,
                "tts": self.config.tts_backend,
            },
        }


async def quick_converse(
    audio: Union[np.ndarray, Path, str],
    llm: str = "ollama",
    system_prompt: Optional[str] = None,
) -> str:
    """
    Quick one-shot voice conversation.

    Args:
        audio: User audio input
        llm: LLM backend to use
        system_prompt: Optional system prompt

    Returns:
        Assistant's response text
    """
    config = PipelineConfig(llm_backend=llm)
    if system_prompt:
        config.system_prompt = system_prompt

    pipeline = SpeechPipeline(config=config)
    turn = await pipeline.converse(audio)
    return turn.content
