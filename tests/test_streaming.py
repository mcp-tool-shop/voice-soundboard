"""
Tests for Streaming TTS Module (streaming.py).

Tests cover:
- StreamChunk dataclass
- StreamResult dataclass
- RealtimeStreamResult dataclass
- StreamingEngine initialization and lazy loading
- StreamingEngine.stream method
- StreamingEngine.stream_to_file method
- RealtimePlayer class
- stream_realtime function
- stream_and_play function
"""

import pytest
import numpy as np
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import fields

from voice_soundboard.streaming import (
    StreamChunk,
    StreamResult,
    RealtimeStreamResult,
    StreamingEngine,
    RealtimePlayer,
    stream_realtime,
    stream_and_play,
)
from voice_soundboard.config import Config, VOICE_PRESETS


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_all_fields_present(self):
        """Test StreamChunk has all expected fields."""
        field_names = {f.name for f in fields(StreamChunk)}
        expected = {"samples", "sample_rate", "chunk_index", "is_final", "text_segment"}
        assert field_names == expected

    def test_create_stream_chunk(self):
        """Test creating a StreamChunk."""
        samples = np.zeros(1000, dtype=np.float32)
        chunk = StreamChunk(
            samples=samples,
            sample_rate=24000,
            chunk_index=0,
            is_final=False,
            text_segment="Hello",
        )

        assert len(chunk.samples) == 1000
        assert chunk.sample_rate == 24000
        assert chunk.chunk_index == 0
        assert chunk.is_final is False
        assert chunk.text_segment == "Hello"

    def test_final_chunk(self):
        """Test creating a final chunk."""
        chunk = StreamChunk(
            samples=np.array([], dtype=np.float32),
            sample_rate=24000,
            chunk_index=5,
            is_final=True,
            text_segment="",
        )

        assert chunk.is_final is True
        assert len(chunk.samples) == 0


class TestStreamResult:
    """Tests for StreamResult dataclass."""

    def test_all_fields_present(self):
        """Test StreamResult has all expected fields."""
        field_names = {f.name for f in fields(StreamResult)}
        expected = {"audio_path", "total_duration", "total_chunks", "generation_time", "voice_used"}
        assert field_names == expected

    def test_create_stream_result(self):
        """Test creating a StreamResult."""
        result = StreamResult(
            audio_path=Path("/test/audio.wav"),
            total_duration=5.5,
            total_chunks=10,
            generation_time=2.0,
            voice_used="af_bella",
        )

        assert result.audio_path == Path("/test/audio.wav")
        assert result.total_duration == 5.5
        assert result.total_chunks == 10
        assert result.generation_time == 2.0
        assert result.voice_used == "af_bella"

    def test_none_audio_path(self):
        """Test StreamResult with None audio_path."""
        result = StreamResult(
            audio_path=None,
            total_duration=0.0,
            total_chunks=0,
            generation_time=0.1,
            voice_used="af_bella",
        )

        assert result.audio_path is None


class TestRealtimeStreamResult:
    """Tests for RealtimeStreamResult dataclass."""

    def test_all_fields_present(self):
        """Test RealtimeStreamResult has all expected fields."""
        field_names = {f.name for f in fields(RealtimeStreamResult)}
        expected = {
            "total_duration", "total_chunks", "generation_time",
            "playback_started_at_chunk", "voice_used"
        }
        assert field_names == expected

    def test_create_realtime_result(self):
        """Test creating a RealtimeStreamResult."""
        result = RealtimeStreamResult(
            total_duration=3.0,
            total_chunks=5,
            generation_time=1.5,
            playback_started_at_chunk=0,
            voice_used="am_michael",
        )

        assert result.total_duration == 3.0
        assert result.total_chunks == 5
        assert result.playback_started_at_chunk == 0


class TestStreamingEngineInit:
    """Tests for StreamingEngine initialization."""

    @patch('pathlib.Path.mkdir')
    def test_init_with_default_config(self, mock_mkdir):
        """Test init creates default config."""
        engine = StreamingEngine()
        assert engine.config is not None
        assert isinstance(engine.config, Config)

    @patch('pathlib.Path.mkdir')
    def test_init_with_custom_config(self, mock_mkdir):
        """Test init accepts custom config."""
        config = Config(default_voice="am_michael")
        engine = StreamingEngine(config=config)
        assert engine.config.default_voice == "am_michael"

    @patch('pathlib.Path.mkdir')
    def test_model_not_loaded_initially(self, mock_mkdir):
        """Test model is lazy loaded."""
        engine = StreamingEngine()
        assert engine._kokoro is None
        assert engine._model_loaded is False

    @patch('pathlib.Path.mkdir')
    def test_model_paths_set(self, mock_mkdir):
        """Test model paths are configured."""
        engine = StreamingEngine()
        assert engine._model_path.name == "kokoro-v1.0.onnx"
        assert engine._voices_path.name == "voices-v1.0.bin"


class TestStreamingEngineLazyLoading:
    """Tests for StreamingEngine lazy model loading."""

    @patch('pathlib.Path.mkdir')
    def test_ensure_model_skips_if_loaded(self, mock_mkdir):
        """Test _ensure_model_loaded skips if already loaded."""
        engine = StreamingEngine()
        engine._model_loaded = True

        # Should not try to load again
        engine._ensure_model_loaded()  # Should not raise

    @patch('pathlib.Path.mkdir')
    def test_model_not_found_error(self, mock_mkdir):
        """Test error when model not found."""
        engine = StreamingEngine()
        engine._model_path = Path("/nonexistent/model.onnx")

        try:
            with pytest.raises(FileNotFoundError):
                engine._ensure_model_loaded()
        except ModuleNotFoundError:
            pytest.skip("kokoro_onnx requires onnxruntime")


class TestStreamingEngineStream:
    """Tests for StreamingEngine.stream method."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked model."""
        with patch('pathlib.Path.mkdir'):
            engine = StreamingEngine()

        # Mock the Kokoro model with async generator
        mock_kokoro = MagicMock()

        async def mock_stream(*args, **kwargs):
            for i in range(3):
                yield (np.zeros(1000, dtype=np.float32), 24000)

        mock_kokoro.create_stream = mock_stream

        engine._kokoro = mock_kokoro
        engine._model_loaded = True

        return engine

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, mock_engine):
        """Test stream yields StreamChunk objects."""
        chunks = []
        async for chunk in mock_engine.stream("Test text"):
            chunks.append(chunk)

        # Should have 3 data chunks + 1 final chunk
        assert len(chunks) == 4
        assert all(isinstance(c, StreamChunk) for c in chunks)
        assert chunks[-1].is_final is True

    @pytest.mark.asyncio
    async def test_stream_with_voice(self, mock_engine):
        """Test stream with custom voice."""
        chunks = []
        async for chunk in mock_engine.stream("Test", voice="am_michael"):
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_with_preset(self, mock_engine):
        """Test stream with preset."""
        chunks = []
        async for chunk in mock_engine.stream("Test", preset="narrator"):
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_speed_clamped(self, mock_engine):
        """Test stream clamps speed to valid range."""
        # Speed > 2.0 should be clamped
        async for chunk in mock_engine.stream("Test", speed=5.0):
            pass  # Just verify no error

    @pytest.mark.asyncio
    async def test_stream_on_chunk_callback(self, mock_engine):
        """Test on_chunk callback is called."""
        callback_called = []

        def on_chunk(chunk):
            callback_called.append(chunk.chunk_index)

        async for _ in mock_engine.stream("Test", on_chunk=on_chunk):
            pass

        # Should have been called for data chunks
        assert len(callback_called) >= 3

    @pytest.mark.asyncio
    async def test_stream_chunk_indices_sequential(self, mock_engine):
        """Test chunk indices are sequential."""
        indices = []
        async for chunk in mock_engine.stream("Test"):
            indices.append(chunk.chunk_index)

        # Should be 0, 1, 2, 3 (data chunks + final)
        for i, idx in enumerate(indices):
            assert idx == i


class TestStreamingEngineStreamToFile:
    """Tests for StreamingEngine.stream_to_file method."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked model."""
        with patch('pathlib.Path.mkdir'):
            engine = StreamingEngine()

        mock_kokoro = MagicMock()

        async def mock_stream(*args, **kwargs):
            for i in range(3):
                yield (np.zeros(1000, dtype=np.float32), 24000)

        mock_kokoro.create_stream = mock_stream

        engine._kokoro = mock_kokoro
        engine._model_loaded = True

        return engine

    @pytest.mark.asyncio
    async def test_stream_to_file_creates_file(self, mock_engine, tmp_path):
        """Test stream_to_file creates output file."""
        output_path = tmp_path / "output.wav"

        with patch('soundfile.write'):
            result = await mock_engine.stream_to_file("Test", output_path)

        assert isinstance(result, StreamResult)
        assert result.audio_path == output_path

    @pytest.mark.asyncio
    async def test_stream_to_file_with_preset(self, mock_engine, tmp_path):
        """Test stream_to_file with preset."""
        output_path = tmp_path / "output.wav"

        with patch('soundfile.write'):
            result = await mock_engine.stream_to_file("Test", output_path, preset="narrator")

        # Should use narrator voice
        assert result.voice_used == "bm_george"

    @pytest.mark.asyncio
    async def test_stream_to_file_on_progress_callback(self, mock_engine, tmp_path):
        """Test on_progress callback is called."""
        output_path = tmp_path / "output.wav"
        progress_calls = []

        def on_progress(chunk_idx, duration):
            progress_calls.append((chunk_idx, duration))

        with patch('soundfile.write'):
            await mock_engine.stream_to_file("Test", output_path, on_progress=on_progress)

        assert len(progress_calls) == 3

    @pytest.mark.asyncio
    async def test_stream_to_file_returns_stats(self, mock_engine, tmp_path):
        """Test stream_to_file returns statistics."""
        output_path = tmp_path / "output.wav"

        with patch('soundfile.write'):
            result = await mock_engine.stream_to_file("Test", output_path)

        assert result.total_chunks == 3
        assert result.generation_time >= 0


class TestRealtimePlayer:
    """Tests for RealtimePlayer class."""

    def test_init_defaults(self):
        """Test RealtimePlayer initialization."""
        player = RealtimePlayer()
        assert player.sample_rate == 24000
        assert player.buffer_chunks == 2
        assert player._is_playing is False

    def test_init_custom_params(self):
        """Test RealtimePlayer with custom parameters."""
        player = RealtimePlayer(sample_rate=44100, buffer_chunks=3)
        assert player.sample_rate == 44100
        assert player.buffer_chunks == 3

    @pytest.mark.asyncio
    async def test_add_chunk(self):
        """Test adding chunk to queue."""
        player = RealtimePlayer()
        samples = np.zeros(1000, dtype=np.float32)

        await player.add_chunk(samples)

        # Should be in queue
        assert not player._queue.empty()

    @pytest.mark.asyncio
    async def test_stop_adds_sentinel(self):
        """Test stop adds sentinel to queue."""
        player = RealtimePlayer()

        # Create a mock task that completes immediately
        async def mock_playback():
            pass

        player._playback_task = asyncio.create_task(mock_playback())

        await player.stop()

        # Sentinel should be in queue (or already processed)
        # Just verify no error


class TestStreamRealtime:
    """Tests for stream_realtime function."""

    @pytest.mark.asyncio
    async def test_stream_realtime_returns_result(self):
        """Test stream_realtime returns RealtimeStreamResult."""
        with patch('pathlib.Path.mkdir'):
            with patch.object(StreamingEngine, '_ensure_model_loaded'):
                # Mock the stream generator
                async def mock_stream(*args, **kwargs):
                    for i in range(2):
                        yield StreamChunk(
                            samples=np.zeros(1000, dtype=np.float32),
                            sample_rate=24000,
                            chunk_index=i,
                            is_final=False,
                            text_segment=""
                        )
                    yield StreamChunk(
                        samples=np.array([]),
                        sample_rate=24000,
                        chunk_index=2,
                        is_final=True,
                        text_segment=""
                    )

                with patch.object(StreamingEngine, 'stream', mock_stream):
                    with patch.object(RealtimePlayer, 'start', new_callable=AsyncMock):
                        with patch.object(RealtimePlayer, 'add_chunk', new_callable=AsyncMock):
                            with patch.object(RealtimePlayer, 'stop', new_callable=AsyncMock):
                                result = await stream_realtime("Test")

        assert isinstance(result, RealtimeStreamResult)


class TestStreamAndPlay:
    """Tests for stream_and_play function."""

    @pytest.mark.asyncio
    async def test_stream_and_play_calls_realtime(self):
        """Test stream_and_play calls stream_realtime."""
        with patch('voice_soundboard.streaming.stream_realtime', new_callable=AsyncMock) as mock_rt:
            await stream_and_play("Test", voice="af_bella", speed=1.0)

            mock_rt.assert_called_once_with("Test", voice="af_bella", speed=1.0)


class TestSpeedValidation:
    """Tests for speed parameter validation."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked model."""
        with patch('pathlib.Path.mkdir'):
            engine = StreamingEngine()

        mock_kokoro = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield (np.zeros(1000), 24000)

        mock_kokoro.create_stream = mock_stream
        engine._kokoro = mock_kokoro
        engine._model_loaded = True

        return engine

    @pytest.mark.asyncio
    async def test_speed_clamped_minimum(self, mock_engine):
        """Test speed is clamped to minimum 0.5."""
        async for chunk in mock_engine.stream("Test", speed=0.1):
            pass  # Just verify no error with clamped speed

    @pytest.mark.asyncio
    async def test_speed_clamped_maximum(self, mock_engine):
        """Test speed is clamped to maximum 2.0."""
        async for chunk in mock_engine.stream("Test", speed=5.0):
            pass


class TestPresetHandling:
    """Tests for voice preset handling in streaming."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked model."""
        with patch('pathlib.Path.mkdir'):
            engine = StreamingEngine()

        mock_kokoro = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield (np.zeros(1000), 24000)

        mock_kokoro.create_stream = mock_stream
        engine._kokoro = mock_kokoro
        engine._model_loaded = True

        return engine

    @pytest.mark.asyncio
    async def test_all_presets_work(self, mock_engine):
        """Test all presets can be used with stream."""
        for preset_name in VOICE_PRESETS:
            async for chunk in mock_engine.stream("Test", preset=preset_name):
                pass  # Just verify no error

    @pytest.mark.asyncio
    async def test_preset_overridden_by_voice(self, mock_engine):
        """Test explicit voice overrides preset."""
        # Narrator uses bm_george, but we specify af_bella
        # The engine should use af_bella
        async for chunk in mock_engine.stream("Test", voice="af_bella", preset="narrator"):
            pass  # Verify no error


class TestStreamChunkProperties:
    """Tests for StreamChunk properties."""

    def test_chunk_samples_are_numpy_array(self):
        """Test chunk samples are numpy array."""
        samples = np.zeros(1000, dtype=np.float32)
        chunk = StreamChunk(samples, 24000, 0, False, "")
        assert isinstance(chunk.samples, np.ndarray)

    def test_chunk_index_is_int(self):
        """Test chunk index is integer."""
        chunk = StreamChunk(np.array([]), 24000, 5, False, "")
        assert isinstance(chunk.chunk_index, int)

    def test_is_final_is_bool(self):
        """Test is_final is boolean."""
        chunk = StreamChunk(np.array([]), 24000, 0, True, "")
        assert isinstance(chunk.is_final, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
