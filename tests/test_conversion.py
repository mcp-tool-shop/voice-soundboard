"""
Tests for Real-Time Voice Conversion Module.

Tests the voice conversion system including:
- VoiceConverter base class and MockVoiceConverter
- ConversionConfig and LatencyMode
- AudioBuffer circular buffer
- StreamingConverter pipeline
- AudioDevice management
- RealtimeConverter integration
"""

import pytest
import numpy as np
import threading
import time

from voice_soundboard.conversion import (
    # Base
    VoiceConverter,
    ConversionConfig,
    ConversionResult,
    LatencyMode,
    ConversionState,
    # Streaming
    StreamingConverter,
    AudioBuffer,
    ConversionPipeline,
    PipelineStage,
    # Devices
    AudioDevice,
    DeviceType,
    list_audio_devices,
    get_default_input_device,
    get_default_output_device,
    AudioDeviceManager,
    # Realtime
    RealtimeConverter,
    RealtimeSession,
    start_realtime_conversion,
)
from voice_soundboard.conversion.base import MockVoiceConverter, ConversionStats


# =============================================================================
# ConversionConfig Tests
# =============================================================================

class TestConversionConfig:
    """Tests for ConversionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ConversionConfig()

        assert config.latency_mode == LatencyMode.BALANCED
        assert config.sample_rate == 24000
        assert config.channels == 1
        assert config.preserve_pitch is True

    def test_latency_modes(self):
        """Test latency mode calculations."""
        for mode, expected_ms in [
            (LatencyMode.ULTRA_LOW, 60.0),
            (LatencyMode.LOW, 100.0),
            (LatencyMode.BALANCED, 150.0),
            (LatencyMode.HIGH_QUALITY, 300.0),
        ]:
            config = ConversionConfig(latency_mode=mode)
            assert config.get_latency_ms() == expected_ms

    def test_target_latency_override(self):
        """Test that target_latency_ms overrides latency_mode."""
        config = ConversionConfig(
            latency_mode=LatencyMode.HIGH_QUALITY,
            target_latency_ms=75.0,
        )
        assert config.get_latency_ms() == 75.0

    def test_chunk_size_calculation(self):
        """Test chunk size in samples calculation."""
        config = ConversionConfig(
            sample_rate=24000,
            chunk_size_ms=20.0,
        )
        # 20ms at 24kHz = 480 samples
        assert config.get_chunk_samples() == 480

    def test_buffer_size_calculation(self):
        """Test buffer size in samples calculation."""
        config = ConversionConfig(sample_rate=24000)
        # 100ms at 24kHz = 2400 samples
        assert config.get_buffer_samples(100.0) == 2400


# =============================================================================
# ConversionResult Tests
# =============================================================================

class TestConversionResult:
    """Tests for ConversionResult dataclass."""

    def test_realtime_factor(self):
        """Test realtime factor calculation."""
        result = ConversionResult(
            audio=np.zeros(24000, dtype=np.float32),
            sample_rate=24000,
            input_duration_ms=1000.0,
            output_duration_ms=1000.0,
            processing_time_ms=500.0,
            latency_ms=500.0,
        )

        # 500ms processing for 1000ms audio = 0.5x
        assert result.realtime_factor == 0.5
        assert result.is_realtime is True

    def test_slower_than_realtime(self):
        """Test detection of slower-than-realtime processing."""
        result = ConversionResult(
            audio=np.zeros(24000, dtype=np.float32),
            sample_rate=24000,
            input_duration_ms=1000.0,
            output_duration_ms=1000.0,
            processing_time_ms=1500.0,
            latency_ms=1500.0,
        )

        assert result.realtime_factor == 1.5
        assert result.is_realtime is False


# =============================================================================
# ConversionStats Tests
# =============================================================================

class TestConversionStats:
    """Tests for ConversionStats tracking."""

    def test_initial_values(self):
        """Test initial statistics values."""
        stats = ConversionStats()

        assert stats.chunks_processed == 0
        assert stats.chunks_dropped == 0
        assert stats.avg_latency_ms == 0.0

    def test_latency_tracking(self):
        """Test latency statistics updates."""
        stats = ConversionStats()

        stats.update_latency(10.0)
        stats.update_latency(20.0)
        stats.update_latency(30.0)

        assert stats.min_latency_ms == 10.0
        assert stats.max_latency_ms == 30.0
        assert stats.avg_latency_ms == 20.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = ConversionStats()
        stats.chunks_processed = 100
        stats.update_latency(15.0)

        data = stats.to_dict()

        assert data["chunks_processed"] == 100
        assert "avg_latency_ms" in data
        assert "realtime_factor" in data


# =============================================================================
# MockVoiceConverter Tests
# =============================================================================

class TestMockVoiceConverter:
    """Tests for MockVoiceConverter implementation."""

    def test_initialization(self):
        """Test mock converter initialization."""
        converter = MockVoiceConverter()

        assert converter.name == "mock"
        assert converter.state == ConversionState.IDLE

    def test_load(self):
        """Test loading the converter."""
        converter = MockVoiceConverter()
        converter.load()

        assert converter._loaded is True

    def test_convert_audio(self):
        """Test converting audio."""
        converter = MockVoiceConverter()
        audio = np.random.randn(24000).astype(np.float32)

        result = converter.convert(audio, sample_rate=24000)

        assert isinstance(result, ConversionResult)
        assert isinstance(result.audio, np.ndarray)
        assert result.sample_rate == 24000
        assert result.is_realtime  # Mock should be fast

    def test_convert_chunk(self):
        """Test converting a streaming chunk."""
        converter = MockVoiceConverter()
        converter.load()
        chunk = np.random.randn(480).astype(np.float32)

        result = converter.convert_chunk(chunk)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_set_target_voice_string(self):
        """Test setting target voice by ID."""
        converter = MockVoiceConverter()
        converter.set_target_voice("test_voice")

        assert converter._target_voice_id == "test_voice"
        assert converter._target_embedding is not None

    def test_set_target_voice_array(self):
        """Test setting target voice by embedding."""
        converter = MockVoiceConverter()
        embedding = np.random.randn(256).astype(np.float32)

        converter.set_target_voice(embedding)

        assert converter._target_voice_id == "custom"
        np.testing.assert_array_equal(converter._target_embedding, embedding)

    def test_stats_updated(self):
        """Test that stats are updated during conversion."""
        converter = MockVoiceConverter(simulate_latency=False)
        audio = np.random.randn(24000).astype(np.float32)

        converter.convert(audio, sample_rate=24000)
        converter.convert(audio, sample_rate=24000)

        assert converter.stats.chunks_processed == 2
        assert converter.stats.total_input_ms > 0

    def test_reset_stats(self):
        """Test resetting statistics."""
        converter = MockVoiceConverter()
        audio = np.random.randn(24000).astype(np.float32)

        converter.convert(audio, sample_rate=24000)
        converter.reset_stats()

        assert converter.stats.chunks_processed == 0


# =============================================================================
# AudioBuffer Tests
# =============================================================================

class TestAudioBuffer:
    """Tests for AudioBuffer circular buffer."""

    def test_basic_write_read(self):
        """Test basic write and read operations."""
        buffer = AudioBuffer(capacity_samples=1000)

        # Write data
        data = np.arange(100, dtype=np.float32)
        written = buffer.write(data)

        assert written == 100
        assert buffer.available == 100

        # Read data
        read = buffer.read(100)

        np.testing.assert_array_equal(read, data)
        assert buffer.available == 0

    def test_wrap_around(self):
        """Test buffer wrap-around behavior."""
        buffer = AudioBuffer(capacity_samples=100)

        # Fill buffer partially
        data1 = np.arange(80, dtype=np.float32)
        buffer.write(data1)

        # Read most of it
        buffer.read(60)

        # Write more (will wrap around)
        data2 = np.arange(50, dtype=np.float32)
        buffer.write(data2)

        # Read all
        result = buffer.read(70, block=False)

        assert len(result) == 70

    def test_non_blocking_read_empty(self):
        """Test non-blocking read on empty buffer."""
        buffer = AudioBuffer(capacity_samples=100)

        result = buffer.read(50, block=False)

        assert result is None

    def test_non_blocking_write_full(self):
        """Test non-blocking write on full buffer."""
        buffer = AudioBuffer(capacity_samples=100)

        # Fill buffer
        data = np.zeros(100, dtype=np.float32)
        buffer.write(data)

        # Try to write more
        more_data = np.zeros(50, dtype=np.float32)
        written = buffer.write(more_data, block=False)

        assert written == 0

    def test_peek(self):
        """Test peeking without consuming."""
        buffer = AudioBuffer(capacity_samples=100)

        data = np.arange(50, dtype=np.float32)
        buffer.write(data)

        # Peek
        peeked = buffer.peek(30)
        assert len(peeked) == 30
        assert buffer.available == 50  # Not consumed

        # Read
        read = buffer.read(30)
        np.testing.assert_array_equal(peeked, read)

    def test_clear(self):
        """Test clearing the buffer."""
        buffer = AudioBuffer(capacity_samples=100)

        data = np.zeros(50, dtype=np.float32)
        buffer.write(data)
        buffer.clear()

        assert buffer.available == 0

    def test_thread_safety(self):
        """Test thread-safe access."""
        buffer = AudioBuffer(capacity_samples=1000)
        written_count = [0]
        read_count = [0]

        def writer():
            for _ in range(100):
                data = np.zeros(10, dtype=np.float32)
                buffer.write(data, block=True, timeout=0.1)
                written_count[0] += 10
                time.sleep(0.001)

        def reader():
            for _ in range(100):
                data = buffer.read(10, block=True, timeout=0.1)
                if data is not None:
                    read_count[0] += len(data)
                time.sleep(0.001)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Should have processed most of the data
        assert written_count[0] == 1000


# =============================================================================
# StreamingConverter Tests
# =============================================================================

class TestStreamingConverter:
    """Tests for StreamingConverter."""

    def test_start_stop(self):
        """Test starting and stopping the converter."""
        converter = MockVoiceConverter()
        streaming = StreamingConverter(converter, chunk_size=480)

        streaming.start()
        assert streaming.is_running

        streaming.stop()
        assert not streaming.is_running

    def test_push_pull(self):
        """Test pushing and pulling audio."""
        converter = MockVoiceConverter()
        streaming = StreamingConverter(converter, chunk_size=480)

        output_chunks = []

        def on_output(chunk):
            output_chunks.append(chunk)

        streaming.start(on_output=on_output)

        # Push some audio
        audio = np.random.randn(480).astype(np.float32)
        streaming.push(audio)

        # Wait for processing
        time.sleep(0.2)

        streaming.stop()

        # Should have processed the chunk
        assert len(output_chunks) > 0

    def test_latency_tracking(self):
        """Test latency tracking."""
        converter = MockVoiceConverter(simulate_latency=False)
        streaming = StreamingConverter(converter, chunk_size=480)

        streaming.start()

        # Push audio
        for _ in range(5):
            audio = np.random.randn(480).astype(np.float32)
            streaming.push(audio)
            time.sleep(0.05)

        streaming.stop()

        # Should have measured latency
        assert streaming.avg_latency_ms >= 0


# =============================================================================
# ConversionPipeline Tests
# =============================================================================

class TestConversionPipeline:
    """Tests for ConversionPipeline."""

    def test_single_stage(self):
        """Test pipeline with single stage."""

        def processor(audio):
            return audio * 2  # Simple amplification

        pipeline = ConversionPipeline(
            stages=[(PipelineStage.CONVERT, processor)],
            chunk_size=480,
        )

        pipeline.start()

        # Push audio
        audio = np.ones(480, dtype=np.float32)
        pipeline.push(audio)

        # Pull result
        result = pipeline.pull(timeout=0.5)

        pipeline.stop()

        assert result is not None
        np.testing.assert_array_almost_equal(result, audio * 2)

    def test_multi_stage(self):
        """Test pipeline with multiple stages."""

        def stage1(audio):
            return audio + 1

        def stage2(audio):
            return audio * 2

        pipeline = ConversionPipeline(
            stages=[
                (PipelineStage.PREPROCESS, stage1),
                (PipelineStage.CONVERT, stage2),
            ],
            chunk_size=480,
        )

        pipeline.start()

        audio = np.ones(480, dtype=np.float32)
        pipeline.push(audio)

        result = pipeline.pull(timeout=0.5)

        pipeline.stop()

        # (1 + 1) * 2 = 4
        assert result is not None
        np.testing.assert_array_almost_equal(result, np.full(480, 4.0))


# =============================================================================
# AudioDevice Tests
# =============================================================================

class TestAudioDevice:
    """Tests for AudioDevice dataclass."""

    def test_create_device(self):
        """Test creating an audio device."""
        device = AudioDevice(
            id=0,
            name="Test Microphone",
            device_type=DeviceType.INPUT,
            max_input_channels=2,
            is_default=True,
        )

        assert device.id == 0
        assert device.name == "Test Microphone"
        assert device.device_type == DeviceType.INPUT
        assert device.is_default is True

    def test_supports_sample_rate(self):
        """Test sample rate support check."""
        device = AudioDevice(
            id=0,
            name="Test",
            device_type=DeviceType.INPUT,
        )

        assert device.supports_sample_rate(24000) is True
        assert device.supports_sample_rate(44100) is True
        assert device.supports_sample_rate(12345) is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        device = AudioDevice(
            id=1,
            name="Speakers",
            device_type=DeviceType.OUTPUT,
            max_output_channels=2,
        )

        data = device.to_dict()

        assert data["id"] == 1
        assert data["name"] == "Speakers"
        assert data["type"] == "output"


class TestDeviceFunctions:
    """Tests for device utility functions."""

    def test_list_audio_devices(self):
        """Test listing audio devices."""
        devices = list_audio_devices()

        # Should return mock devices if sounddevice not available
        assert isinstance(devices, list)
        assert len(devices) > 0

    def test_list_input_devices(self):
        """Test listing input devices only."""
        devices = list_audio_devices(DeviceType.INPUT)

        for device in devices:
            assert device.max_input_channels > 0 or device.device_type == DeviceType.DUPLEX

    def test_list_output_devices(self):
        """Test listing output devices only."""
        devices = list_audio_devices(DeviceType.OUTPUT)

        for device in devices:
            assert device.max_output_channels > 0 or device.device_type == DeviceType.DUPLEX

    def test_get_default_input(self):
        """Test getting default input device."""
        device = get_default_input_device()

        assert device is not None
        assert device.max_input_channels > 0 or device.device_type == DeviceType.DUPLEX

    def test_get_default_output(self):
        """Test getting default output device."""
        device = get_default_output_device()

        assert device is not None
        assert device.max_output_channels > 0 or device.device_type == DeviceType.DUPLEX


# =============================================================================
# AudioDeviceManager Tests
# =============================================================================

class TestAudioDeviceManager:
    """Tests for AudioDeviceManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = AudioDeviceManager(
            sample_rate=24000,
            channels=1,
            chunk_size=480,
        )

        assert manager.sample_rate == 24000
        assert manager.channels == 1
        assert manager.chunk_size == 480

    def test_set_devices(self):
        """Test setting input/output devices."""
        manager = AudioDeviceManager()

        input_device = manager.set_input_device()
        output_device = manager.set_output_device()

        assert input_device is not None
        assert output_device is not None

    def test_context_manager(self):
        """Test context manager behavior."""
        with AudioDeviceManager() as manager:
            assert manager is not None

        # Should be stopped after exit


# =============================================================================
# RealtimeConverter Tests
# =============================================================================

class TestRealtimeConverter:
    """Tests for RealtimeConverter."""

    def test_initialization(self):
        """Test realtime converter initialization."""
        converter = RealtimeConverter()

        assert converter.is_running is False
        assert converter.current_session is None

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = ConversionConfig(latency_mode=LatencyMode.LOW)
        converter = RealtimeConverter(config=config)

        assert converter.config.latency_mode == LatencyMode.LOW

    def test_start_stop(self):
        """Test starting and stopping conversion."""
        converter = RealtimeConverter()

        session = converter.start(target_voice="test_voice")

        assert converter.is_running is True
        assert session is not None
        assert session.session_id is not None
        assert session.target_voice == "test_voice"

        result_session = converter.stop()

        assert converter.is_running is False
        assert result_session.duration_seconds >= 0

    def test_set_target_voice_while_running(self):
        """Test changing target voice during conversion."""
        converter = RealtimeConverter()

        converter.start(target_voice="voice1")
        converter.set_target_voice("voice2")

        assert converter.current_session.target_voice == "voice2"

        converter.stop()

    def test_get_latency(self):
        """Test getting current latency."""
        converter = RealtimeConverter()

        converter.start(target_voice="test")
        latency = converter.get_latency()

        assert latency >= 0

        converter.stop()


class TestRealtimeSession:
    """Tests for RealtimeSession."""

    def test_session_creation(self):
        """Test session creation."""
        session = RealtimeSession(
            session_id="test_session",
            target_voice="my_voice",
        )

        assert session.session_id == "test_session"
        assert session.target_voice == "my_voice"

    def test_duration_calculation(self):
        """Test session duration calculation."""
        session = RealtimeSession(
            session_id="test",
            started_at=time.time() - 5.0,  # Started 5 seconds ago
        )

        assert session.duration_seconds >= 5.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        session = RealtimeSession(
            session_id="test",
            target_voice="voice1",
            started_at=time.time(),
        )

        data = session.to_dict()

        assert data["session_id"] == "test"
        assert data["target_voice"] == "voice1"
        assert "stats" in data


# =============================================================================
# Integration Tests
# =============================================================================

class TestConversionIntegration:
    """Integration tests for the voice conversion system."""

    def test_full_conversion_pipeline(self):
        """Test complete conversion pipeline."""
        # Create converter
        config = ConversionConfig(latency_mode=LatencyMode.LOW)
        converter = MockVoiceConverter(config, simulate_latency=False)

        # Set target voice
        converter.set_target_voice("celebrity_clone")

        # Generate test audio
        audio = np.random.randn(24000).astype(np.float32)

        # Convert
        result = converter.convert(audio, sample_rate=24000)

        # Verify result
        assert result.audio is not None
        assert len(result.audio) > 0
        assert result.similarity_score > 0
        assert result.naturalness_score > 0

    def test_streaming_conversion(self):
        """Test streaming conversion end-to-end."""
        converter = MockVoiceConverter(simulate_latency=False)
        streaming = StreamingConverter(converter, chunk_size=480)

        results = []

        def on_output(chunk):
            results.append(chunk)

        streaming.start(on_output=on_output)

        # Simulate audio input
        for _ in range(10):
            chunk = np.random.randn(480).astype(np.float32)
            streaming.push(chunk)
            time.sleep(0.02)

        # Wait for processing
        time.sleep(0.3)

        streaming.stop()

        # Should have processed chunks
        assert len(results) > 0

    def test_start_realtime_convenience(self):
        """Test start_realtime_conversion convenience function."""
        converter = start_realtime_conversion(
            target_voice="test_voice",
            latency_mode=LatencyMode.BALANCED,
        )

        assert converter.is_running is True

        converter.stop()

        assert converter.is_running is False


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_convert_empty_audio(self):
        """Test converting empty audio."""
        converter = MockVoiceConverter()
        audio = np.array([], dtype=np.float32)

        result = converter.convert(audio, sample_rate=24000)

        assert result.input_duration_ms == 0

    def test_convert_very_short_audio(self):
        """Test converting very short audio."""
        converter = MockVoiceConverter()
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        result = converter.convert(audio, sample_rate=24000)

        assert result.audio is not None

    def test_double_start(self):
        """Test starting converter twice."""
        converter = RealtimeConverter()

        converter.start(target_voice="voice1")

        with pytest.raises(RuntimeError):
            converter.start(target_voice="voice2")

        converter.stop()

    def test_stop_not_running(self):
        """Test stopping converter that's not running."""
        converter = RealtimeConverter()

        result = converter.stop()

        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
