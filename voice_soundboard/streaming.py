"""
Streaming TTS - Real-time audio generation for long text.

Generates audio chunk by chunk, enabling playback to start
before the entire text is processed.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional, AsyncGenerator, Callable
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from voice_soundboard.config import Config, VOICE_PRESETS


@dataclass
class StreamChunk:
    """A chunk of streamed audio."""
    samples: np.ndarray
    sample_rate: int
    chunk_index: int
    is_final: bool
    text_segment: str  # The text this chunk represents


@dataclass
class StreamResult:
    """Final result from streaming synthesis."""
    audio_path: Optional[Path]
    total_duration: float
    total_chunks: int
    generation_time: float
    voice_used: str


class StreamingEngine:
    """
    Streaming TTS engine for real-time audio generation.

    Generates audio in chunks, allowing playback to start immediately
    while the rest of the text is still being processed.

    Example:
        engine = StreamingEngine()

        async for chunk in engine.stream("Long text here...", voice="af_bella"):
            play_chunk(chunk.samples)
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._kokoro = None
        self._model_loaded = False

        self._model_dir = Path("F:/AI/voice-soundboard/models")
        self._model_path = self._model_dir / "kokoro-v1.0.onnx"
        self._voices_path = self._model_dir / "voices-v1.0.bin"

    def _ensure_model_loaded(self):
        """Lazy-load the Kokoro model."""
        if self._model_loaded:
            return

        from kokoro_onnx import Kokoro

        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {self._model_path}")

        self._kokoro = Kokoro(
            str(self._model_path),
            str(self._voices_path)
        )
        self._model_loaded = True

    async def stream(
        self,
        text: str,
        voice: Optional[str] = None,
        preset: Optional[str] = None,
        speed: float = 1.0,
        on_chunk: Optional[Callable[[StreamChunk], None]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream audio generation chunk by chunk.

        Args:
            text: Text to synthesize
            voice: Voice ID
            preset: Voice preset name
            speed: Speed multiplier
            on_chunk: Optional callback for each chunk

        Yields:
            StreamChunk objects with audio samples
        """
        self._ensure_model_loaded()

        # Resolve voice from preset
        if preset and preset in VOICE_PRESETS:
            preset_config = VOICE_PRESETS[preset]
            voice = voice or preset_config["voice"]
            speed = preset_config.get("speed", speed)

        voice = voice or self.config.default_voice
        speed = max(0.5, min(2.0, speed))

        chunk_index = 0

        async for samples, sample_rate in self._kokoro.create_stream(
            text, voice=voice, speed=speed
        ):
            chunk = StreamChunk(
                samples=samples,
                sample_rate=sample_rate,
                chunk_index=chunk_index,
                is_final=False,
                text_segment=""  # Kokoro doesn't provide segment info
            )

            if on_chunk:
                on_chunk(chunk)

            yield chunk
            chunk_index += 1

        # Mark final chunk
        if chunk_index > 0:
            yield StreamChunk(
                samples=np.array([], dtype=np.float32),
                sample_rate=24000,
                chunk_index=chunk_index,
                is_final=True,
                text_segment=""
            )

    async def stream_to_file(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        preset: Optional[str] = None,
        speed: float = 1.0,
        on_progress: Optional[Callable[[int, float], None]] = None,
    ) -> StreamResult:
        """
        Stream audio to a file, useful for long text.

        Args:
            text: Text to synthesize
            output_path: Where to save the audio
            voice: Voice ID
            preset: Voice preset name
            speed: Speed multiplier
            on_progress: Callback(chunk_index, duration_so_far)

        Returns:
            StreamResult with final statistics
        """
        self._ensure_model_loaded()

        # Resolve voice
        if preset and preset in VOICE_PRESETS:
            preset_config = VOICE_PRESETS[preset]
            voice = voice or preset_config["voice"]
            speed = preset_config.get("speed", speed)

        voice = voice or self.config.default_voice

        start_time = time.time()
        all_samples = []
        sample_rate = 24000
        chunk_count = 0

        async for samples, sr in self._kokoro.create_stream(
            text, voice=voice, speed=speed
        ):
            all_samples.append(samples)
            sample_rate = sr
            chunk_count += 1

            if on_progress:
                total_samples = sum(len(s) for s in all_samples)
                duration = total_samples / sample_rate
                on_progress(chunk_count, duration)

        # Concatenate and save
        if all_samples:
            combined = np.concatenate(all_samples)
            sf.write(str(output_path), combined, sample_rate)
            total_duration = len(combined) / sample_rate
        else:
            total_duration = 0.0

        return StreamResult(
            audio_path=output_path,
            total_duration=total_duration,
            total_chunks=chunk_count,
            generation_time=time.time() - start_time,
            voice_used=voice,
        )


@dataclass
class RealtimeStreamResult:
    """Result from real-time streaming playback."""
    total_duration: float
    total_chunks: int
    generation_time: float
    playback_started_at_chunk: int  # When playback began (0 = immediate)
    voice_used: str


class RealtimePlayer:
    """
    Real-time audio player with queue-based streaming.

    Starts playback as soon as first chunk arrives, continues
    playing while generation happens in parallel.
    """

    def __init__(self, sample_rate: int = 24000, buffer_chunks: int = 2):
        """
        Args:
            sample_rate: Audio sample rate
            buffer_chunks: Chunks to buffer before starting playback
        """
        self.sample_rate = sample_rate
        self.buffer_chunks = buffer_chunks
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._playback_task: asyncio.Task | None = None
        self._is_playing = False
        self._stop_event = asyncio.Event()

    async def _playback_loop(self):
        """Background task that plays audio from queue."""
        import sounddevice as sd

        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=2048,  # Lower latency
        )
        stream.start()
        self._is_playing = True

        try:
            while not self._stop_event.is_set():
                try:
                    # Get chunk with timeout
                    chunk = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=0.1
                    )

                    if chunk is None:  # Sentinel for end
                        break

                    # Play chunk (blocking but in thread)
                    await asyncio.to_thread(stream.write, chunk)

                except asyncio.TimeoutError:
                    continue

        finally:
            stream.stop()
            stream.close()
            self._is_playing = False

    async def start(self):
        """Start the playback loop."""
        self._stop_event.clear()
        self._playback_task = asyncio.create_task(self._playback_loop())

    async def add_chunk(self, samples: np.ndarray):
        """Add audio chunk to playback queue."""
        await self._queue.put(samples)

    async def stop(self):
        """Stop playback and wait for completion."""
        await self._queue.put(None)  # Sentinel
        if self._playback_task:
            await self._playback_task

    async def stop_immediate(self):
        """Stop playback immediately."""
        self._stop_event.set()
        await self._queue.put(None)
        if self._playback_task:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass


async def stream_realtime(
    text: str,
    voice: str = "af_bella",
    preset: str | None = None,
    speed: float = 1.0,
    on_chunk: Callable[[int, float], None] | None = None,
) -> RealtimeStreamResult:
    """
    Stream text-to-speech with real-time playback.

    Playback starts as soon as the first chunk is generated,
    while the rest of the text continues generating in parallel.

    Args:
        text: Text to speak
        voice: Voice ID
        preset: Voice preset name
        speed: Speed multiplier (0.5-2.0)
        on_chunk: Callback(chunk_index, duration_so_far)

    Returns:
        RealtimeStreamResult with statistics

    Example:
        await stream_realtime(
            "Hello, this is a test of real-time streaming.",
            voice="af_bella",
            on_chunk=lambda i, d: print(f"Chunk {i}: {d:.1f}s")
        )
    """
    import time

    engine = StreamingEngine()
    player = RealtimePlayer()

    start_time = time.time()
    chunk_count = 0
    total_samples = 0
    sample_rate = 24000
    playback_started = False
    playback_start_chunk = 0
    voice_used = voice or "af_bella"

    # Start playback loop
    await player.start()

    try:
        async for chunk in engine.stream(
            text,
            voice=voice,
            preset=preset,
            speed=speed
        ):
            if chunk.is_final:
                break

            # Update voice from engine
            sample_rate = chunk.sample_rate

            # Add to playback queue
            await player.add_chunk(chunk.samples)

            if not playback_started:
                playback_started = True
                playback_start_chunk = chunk_count

            total_samples += len(chunk.samples)
            chunk_count += 1

            if on_chunk:
                duration = total_samples / sample_rate
                on_chunk(chunk_count, duration)

    finally:
        # Wait for playback to finish
        await player.stop()

    return RealtimeStreamResult(
        total_duration=total_samples / sample_rate,
        total_chunks=chunk_count,
        generation_time=time.time() - start_time,
        playback_started_at_chunk=playback_start_chunk,
        voice_used=voice_used,
    )


async def stream_and_play(
    text: str,
    voice: str = "af_bella",
    speed: float = 1.0,
) -> None:
    """
    Stream text and play audio in real-time.

    Simple wrapper around stream_realtime for backwards compatibility.
    """
    await stream_realtime(text, voice=voice, speed=speed)


if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing streaming TTS...")

        engine = StreamingEngine()

        text = """
        This is a test of the streaming text to speech system.
        It generates audio chunk by chunk, allowing playback to start
        before the entire text is processed. This is especially useful
        for long documents or real-time applications.
        """

        output = Path("F:/AI/voice-soundboard/output/stream_test.wav")

        def progress(chunk, duration):
            print(f"  Chunk {chunk}: {duration:.2f}s generated")

        result = await engine.stream_to_file(
            text,
            output,
            preset="narrator",
            on_progress=progress
        )

        print(f"\nResult:")
        print(f"  File: {result.audio_path}")
        print(f"  Duration: {result.total_duration:.2f}s")
        print(f"  Chunks: {result.total_chunks}")
        print(f"  Gen time: {result.generation_time:.2f}s")
        print(f"  Voice: {result.voice_used}")

    asyncio.run(test())
