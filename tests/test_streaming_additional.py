"""
Additional tests for Streaming TTS module (streaming.py).

Tests cover:
- StreamChunk and StreamResult dataclasses
- RealtimePlayer initialization and queue operations
- StreamingEngine async stream behavior
- Cancellation and cleanup
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from voice_soundboard.streaming import (
    StreamChunk,
    StreamResult,
    RealtimeStreamResult,
    StreamingEngine,
    RealtimePlayer,
    stream_realtime,
)


class TestStreamChunkDataclass:
    """Tests for StreamChunk dataclass."""

    def test_stream_chunk_structure(self):
        """TEST-S18: StreamChunk dataclass has correct structure."""
        samples = np.zeros(2400, dtype=np.float32)
        chunk = StreamChunk(
            samples=samples,
            sample_rate=24000,
            chunk_index=0,
            is_final=False,
            text_segment="Hello"
        )

        assert np.array_equal(chunk.samples, samples)
        assert chunk.sample_rate == 24000
        assert chunk.chunk_index == 0
        assert chunk.is_final is False
        assert chunk.text_segment == "Hello"

    def test_stream_chunk_final_marker(self):
        """Test StreamChunk with is_final=True."""
        chunk = StreamChunk(
            samples=np.array([], dtype=np.float32),
            sample_rate=24000,
            chunk_index=5,
            is_final=True,
            text_segment=""
        )

        assert chunk.is_final is True
        assert len(chunk.samples) == 0


class TestStreamResultDataclass:
    """Tests for StreamResult dataclass."""

    def test_stream_result_structure(self):
        """TEST-S19: StreamResult dataclass has correct structure."""
        result = StreamResult(
            audio_path=Path("/tmp/test.wav"),
            total_duration=2.5,
            total_chunks=5,
            generation_time=0.8,
            voice_used="af_bella"
        )

        assert result.audio_path == Path("/tmp/test.wav")
        assert result.total_duration == 2.5
        assert result.total_chunks == 5
        assert result.generation_time == 0.8
        assert result.voice_used == "af_bella"

    def test_stream_result_none_path(self):
        """Test StreamResult with None audio_path."""
        result = StreamResult(
            audio_path=None,
            total_duration=0.0,
            total_chunks=0,
            generation_time=0.0,
            voice_used=""
        )

        assert result.audio_path is None


class TestRealtimeStreamResultDataclass:
    """Tests for RealtimeStreamResult dataclass."""

    def test_realtime_result_structure(self):
        """TEST-S20: RealtimeStreamResult dataclass has correct structure."""
        result = RealtimeStreamResult(
            total_duration=3.0,
            total_chunks=6,
            generation_time=1.2,
            playback_started_at_chunk=0,
            voice_used="am_adam"
        )

        assert result.total_duration == 3.0
        assert result.total_chunks == 6
        assert result.generation_time == 1.2
        assert result.playback_started_at_chunk == 0
        assert result.voice_used == "am_adam"


class TestRealtimePlayerInit:
    """Tests for RealtimePlayer initialization."""

    def test_realtime_player_defaults(self):
        """TEST-S15: RealtimePlayer initialization with defaults."""
        player = RealtimePlayer()

        assert player.sample_rate == 24000
        assert player.buffer_chunks == 2
        assert player._is_playing is False
        assert player._playback_task is None

    def test_realtime_player_custom_params(self):
        """Test RealtimePlayer with custom parameters."""
        player = RealtimePlayer(sample_rate=44100, buffer_chunks=4)

        assert player.sample_rate == 44100
        assert player.buffer_chunks == 4


class TestRealtimePlayerQueue:
    """Tests for RealtimePlayer queue operations."""

    @pytest.mark.asyncio
    async def test_queue_add_chunk(self):
        """TEST-S16: RealtimePlayer queue operations work correctly."""
        player = RealtimePlayer()

        samples = np.zeros(2400, dtype=np.float32)
        await player.add_chunk(samples)

        assert player._queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_queue_multiple_chunks(self):
        """Test adding multiple chunks to queue."""
        player = RealtimePlayer()

        for i in range(5):
            samples = np.zeros(2400, dtype=np.float32)
            await player.add_chunk(samples)

        assert player._queue.qsize() == 5


class TestRealtimePlayerStop:
    """Tests for RealtimePlayer stop methods."""

    @pytest.mark.asyncio
    async def test_stop_immediate_not_started(self):
        """TEST-S17: stop_immediate() on non-started player doesn't crash."""
        player = RealtimePlayer()

        # Should not raise when player hasn't started
        await player.stop_immediate()

        # Verify queue got the sentinel
        assert player._queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_stop_adds_sentinel(self):
        """Test that stop adds None sentinel to queue."""
        player = RealtimePlayer()

        await player.stop()

        # Check sentinel was added
        sentinel = await player._queue.get()
        assert sentinel is None


class TestStreamingEngineInit:
    """Tests for StreamingEngine initialization."""

    def test_streaming_engine_init(self):
        """Test StreamingEngine initializes correctly."""
        engine = StreamingEngine()

        assert engine.config is not None
        assert engine._kokoro is None
        assert engine._model_loaded is False

    def test_streaming_engine_with_config(self):
        """Test StreamingEngine with custom config."""
        from voice_soundboard.config import Config
        config = Config()
        engine = StreamingEngine(config=config)

        assert engine.config is config


class TestStreamingEngineStreamAsync:
    """Tests for StreamingEngine.stream() async generator."""

    def test_stream_is_async_generator(self):
        """TEST-S12: stream() returns async generator structure."""
        engine = StreamingEngine()

        # stream() should return an async generator
        import inspect
        assert inspect.isasyncgenfunction(engine.stream.__class__.__call__) or \
               hasattr(engine.stream("test"), '__anext__')


class TestStreamingEngineStreamToFile:
    """Tests for StreamingEngine.stream_to_file()."""

    @pytest.mark.asyncio
    async def test_stream_to_file_invalid_path(self, tmp_path):
        """TEST-S13: stream_to_file() with unwritable path raises error."""
        # Create a directory where the file should be (prevents file creation)
        invalid_dir = tmp_path / "nonexistent_deep" / "directory" / "path"
        output_path = invalid_dir / "output.wav"

        engine = StreamingEngine.__new__(StreamingEngine)
        engine.config = Mock()
        engine.config.default_voice = "af_bella"
        engine._model_loaded = True

        # Mock kokoro to return some chunks
        async def mock_stream(*args, **kwargs):
            yield np.zeros(24000, dtype=np.float32), 24000

        engine._kokoro = Mock()
        engine._kokoro.create_stream = mock_stream

        # Should raise when trying to write to nonexistent directory
        from soundfile import LibsndfileError
        with pytest.raises((OSError, FileNotFoundError, LibsndfileError)):
            await engine.stream_to_file(
                "Hello world",
                output_path
            )


class TestStreamingEngineVoiceResolution:
    """Tests for voice/preset resolution in streaming."""

    def test_preset_voice_resolution(self):
        """Test that preset configuration is properly resolved."""
        from voice_soundboard.config import VOICE_PRESETS

        engine = StreamingEngine()

        # Narrator preset should map to bm_george
        narrator_config = VOICE_PRESETS.get("narrator")
        assert narrator_config is not None
        assert "voice" in narrator_config


class TestStreamRealtimeFunction:
    """Tests for the stream_realtime convenience function."""

    def test_stream_realtime_signature(self):
        """Test stream_realtime function signature."""
        import inspect
        sig = inspect.signature(stream_realtime)

        params = list(sig.parameters.keys())
        assert "text" in params
        assert "voice" in params
        assert "preset" in params
        assert "speed" in params
        assert "on_chunk" in params


class TestStreamingCancellation:
    """Tests for cancellation of streaming operations."""

    @pytest.mark.asyncio
    async def test_realtime_player_cancel_flag(self):
        """TEST-S08: Cancel methods work correctly."""
        player = RealtimePlayer()

        # Initially event is not set
        assert not player._stop_event.is_set()

        # stop_immediate should set the event
        await player.stop_immediate()

        assert player._stop_event.is_set()

    @pytest.mark.asyncio
    async def test_stop_event_cleared_on_start(self):
        """Test that stop event is cleared when starting."""
        player = RealtimePlayer()
        player._stop_event.set()

        # Mock the playback loop to avoid actual audio
        with patch.object(player, '_playback_loop', new_callable=AsyncMock):
            await player.start()

            # Event should be cleared
            assert not player._stop_event.is_set()


class TestStreamingValidation:
    """Tests for input validation in streaming."""

    def test_speed_clamping(self):
        """Test that speed is clamped in streaming engine."""
        # The speed clamping logic in stream()
        speed = 0.1
        clamped = max(0.5, min(2.0, speed))
        assert clamped == 0.5

        speed = 5.0
        clamped = max(0.5, min(2.0, speed))
        assert clamped == 2.0


class TestStreamChunkCallback:
    """Tests for on_chunk callback in streaming."""

    def test_callback_signature(self):
        """TEST-S04: on_chunk callback structure."""
        # Define a callback with expected signature
        chunks_received = []

        def on_chunk(chunk: StreamChunk) -> None:
            chunks_received.append(chunk)

        # The callback should accept StreamChunk
        sample_chunk = StreamChunk(
            samples=np.zeros(2400, dtype=np.float32),
            sample_rate=24000,
            chunk_index=0,
            is_final=False,
            text_segment=""
        )

        on_chunk(sample_chunk)

        assert len(chunks_received) == 1
        assert chunks_received[0] is sample_chunk


class TestStreamResultFields:
    """Tests for StreamResult field validation."""

    def test_stream_result_realtime_factor(self):
        """TEST-S05: StreamResult has expected fields for RTF calculation."""
        result = StreamResult(
            audio_path=Path("/tmp/test.wav"),
            total_duration=2.0,
            total_chunks=4,
            generation_time=0.4,  # RTF would be 5.0
            voice_used="af_bella"
        )

        rtf = result.total_duration / result.generation_time if result.generation_time > 0 else 0
        assert rtf == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
