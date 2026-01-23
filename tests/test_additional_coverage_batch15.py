"""
Batch 15: Additional coverage tests for the conversion module.

Tests for:
- voice_soundboard/conversion/base.py
- voice_soundboard/conversion/streaming.py
- voice_soundboard/conversion/devices.py
- voice_soundboard/conversion/realtime.py
"""

import pytest
import numpy as np
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Tests for conversion/base.py
# ============================================================================

class TestLatencyMode:
    """Tests for LatencyMode enum."""

    def test_latency_mode_values(self):
        """Test LatencyMode enum has expected values."""
        from voice_soundboard.conversion.base import LatencyMode

        assert hasattr(LatencyMode, "ULTRA_LOW")
        assert hasattr(LatencyMode, "LOW")
        assert hasattr(LatencyMode, "BALANCED")
        assert hasattr(LatencyMode, "HIGH_QUALITY")

    def test_latency_mode_are_distinct(self):
        """Test LatencyMode values are distinct."""
        from voice_soundboard.conversion.base import LatencyMode

        modes = [LatencyMode.ULTRA_LOW, LatencyMode.LOW, LatencyMode.BALANCED, LatencyMode.HIGH_QUALITY]
        values = [m.value for m in modes]
        assert len(set(values)) == 4


class TestConversionState:
    """Tests for ConversionState enum."""

    def test_conversion_state_values(self):
        """Test ConversionState enum has expected values."""
        from voice_soundboard.conversion.base import ConversionState

        assert hasattr(ConversionState, "IDLE")
        assert hasattr(ConversionState, "STARTING")
        assert hasattr(ConversionState, "RUNNING")
        assert hasattr(ConversionState, "STOPPING")
        assert hasattr(ConversionState, "ERROR")


class TestConversionConfig:
    """Tests for ConversionConfig dataclass."""

    def test_default_values(self):
        """Test ConversionConfig default values."""
        from voice_soundboard.conversion.base import ConversionConfig, LatencyMode

        config = ConversionConfig()
        assert config.latency_mode == LatencyMode.BALANCED
        assert config.sample_rate == 24000
        assert config.channels == 1
        assert config.chunk_size_ms == 20.0
        assert config.preserve_pitch is True
        assert config.use_gpu is True

    def test_get_latency_ms_with_override(self):
        """Test get_latency_ms with target_latency_ms override."""
        from voice_soundboard.conversion.base import ConversionConfig

        config = ConversionConfig(target_latency_ms=200.0)
        assert config.get_latency_ms() == 200.0

    def test_get_latency_ms_from_mode(self):
        """Test get_latency_ms from latency mode."""
        from voice_soundboard.conversion.base import ConversionConfig, LatencyMode

        config = ConversionConfig(latency_mode=LatencyMode.ULTRA_LOW)
        assert config.get_latency_ms() == 60.0

        config = ConversionConfig(latency_mode=LatencyMode.LOW)
        assert config.get_latency_ms() == 100.0

        config = ConversionConfig(latency_mode=LatencyMode.HIGH_QUALITY)
        assert config.get_latency_ms() == 300.0

    def test_get_chunk_samples(self):
        """Test get_chunk_samples calculation."""
        from voice_soundboard.conversion.base import ConversionConfig

        config = ConversionConfig(sample_rate=24000, chunk_size_ms=20.0)
        assert config.get_chunk_samples() == 480  # 24000 * 20 / 1000

    def test_get_buffer_samples(self):
        """Test get_buffer_samples calculation."""
        from voice_soundboard.conversion.base import ConversionConfig

        config = ConversionConfig(sample_rate=24000)
        assert config.get_buffer_samples(100.0) == 2400  # 24000 * 100 / 1000


class TestConversionResult:
    """Tests for ConversionResult dataclass."""

    def test_creation(self):
        """Test ConversionResult creation."""
        from voice_soundboard.conversion.base import ConversionResult

        audio = np.zeros(1000, dtype=np.float32)
        result = ConversionResult(
            audio=audio,
            sample_rate=24000,
            input_duration_ms=41.67,
            output_duration_ms=41.67,
            processing_time_ms=10.0,
            latency_ms=10.0,
        )
        assert result.sample_rate == 24000
        assert result.processing_time_ms == 10.0

    def test_realtime_factor(self):
        """Test realtime_factor calculation."""
        from voice_soundboard.conversion.base import ConversionResult

        audio = np.zeros(1000, dtype=np.float32)
        result = ConversionResult(
            audio=audio,
            sample_rate=24000,
            input_duration_ms=100.0,
            output_duration_ms=100.0,
            processing_time_ms=50.0,
            latency_ms=50.0,
        )
        assert result.realtime_factor == 0.5

    def test_realtime_factor_zero_input(self):
        """Test realtime_factor with zero input duration."""
        from voice_soundboard.conversion.base import ConversionResult

        audio = np.zeros(0, dtype=np.float32)
        result = ConversionResult(
            audio=audio,
            sample_rate=24000,
            input_duration_ms=0.0,
            output_duration_ms=0.0,
            processing_time_ms=10.0,
            latency_ms=10.0,
        )
        assert result.realtime_factor == 0.0

    def test_is_realtime(self):
        """Test is_realtime property."""
        from voice_soundboard.conversion.base import ConversionResult

        audio = np.zeros(1000, dtype=np.float32)

        # Faster than realtime
        result = ConversionResult(
            audio=audio,
            sample_rate=24000,
            input_duration_ms=100.0,
            output_duration_ms=100.0,
            processing_time_ms=50.0,
            latency_ms=50.0,
        )
        assert result.is_realtime is True

        # Slower than realtime
        result2 = ConversionResult(
            audio=audio,
            sample_rate=24000,
            input_duration_ms=100.0,
            output_duration_ms=100.0,
            processing_time_ms=150.0,
            latency_ms=150.0,
        )
        assert result2.is_realtime is False


class TestConversionStats:
    """Tests for ConversionStats dataclass."""

    def test_default_values(self):
        """Test ConversionStats default values."""
        from voice_soundboard.conversion.base import ConversionStats

        stats = ConversionStats()
        assert stats.chunks_processed == 0
        assert stats.chunks_dropped == 0
        assert stats.avg_latency_ms == 0.0

    def test_update_latency(self):
        """Test update_latency method."""
        from voice_soundboard.conversion.base import ConversionStats

        stats = ConversionStats()
        stats.update_latency(10.0)
        stats.update_latency(20.0)
        stats.update_latency(30.0)

        assert stats.min_latency_ms == 10.0
        assert stats.max_latency_ms == 30.0
        assert stats.avg_latency_ms == 20.0

    def test_update_similarity(self):
        """Test update_similarity method."""
        from voice_soundboard.conversion.base import ConversionStats

        stats = ConversionStats()
        stats.chunks_processed = 2
        stats.update_similarity(0.8)
        stats.update_similarity(0.9)

        assert stats.avg_similarity == pytest.approx(0.85, abs=0.01)

    def test_to_dict(self):
        """Test to_dict method."""
        from voice_soundboard.conversion.base import ConversionStats

        stats = ConversionStats()
        stats.chunks_processed = 10
        stats.total_input_ms = 1000.0
        stats.total_processing_ms = 500.0
        stats.update_latency(50.0)

        result = stats.to_dict()
        assert isinstance(result, dict)
        assert result["chunks_processed"] == 10
        assert result["total_input_ms"] == 1000.0
        assert result["realtime_factor"] == 0.5

    def test_to_dict_with_inf_latency(self):
        """Test to_dict handles inf min_latency."""
        from voice_soundboard.conversion.base import ConversionStats

        stats = ConversionStats()
        result = stats.to_dict()
        assert result["min_latency_ms"] == 0.0


class TestMockVoiceConverter:
    """Tests for MockVoiceConverter class."""

    def test_creation(self):
        """Test MockVoiceConverter creation."""
        from voice_soundboard.conversion.base import MockVoiceConverter, ConversionConfig

        converter = MockVoiceConverter()
        assert converter.name == "mock"
        assert converter._loaded is False

    def test_load(self):
        """Test load method."""
        from voice_soundboard.conversion.base import MockVoiceConverter, ConversionState

        converter = MockVoiceConverter()
        converter.load()
        assert converter._loaded is True
        assert converter.state == ConversionState.IDLE

    def test_convert(self):
        """Test convert method."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)
        audio = np.random.randn(1000).astype(np.float32)

        result = converter.convert(audio, 24000)
        assert result.audio is not None
        assert len(result.audio) > 0
        assert result.sample_rate == 24000
        assert result.similarity_score == 0.85

    def test_convert_with_target_voice(self):
        """Test convert with target voice."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)
        audio = np.random.randn(1000).astype(np.float32)

        result = converter.convert(audio, 24000, target_voice="test_voice")
        assert result.target_voice == "test_voice"

    def test_convert_chunk(self):
        """Test convert_chunk method."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)
        chunk = np.random.randn(480).astype(np.float32)

        result = converter.convert_chunk(chunk)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_mock_convert_with_pitch_shift(self):
        """Test mock conversion with pitch shift."""
        from voice_soundboard.conversion.base import MockVoiceConverter, ConversionConfig

        config = ConversionConfig(pitch_shift_semitones=2.0)
        converter = MockVoiceConverter(config=config, simulate_latency=False)
        audio = np.random.randn(1000).astype(np.float32)

        result = converter._mock_convert(audio, 24000)
        assert isinstance(result, np.ndarray)

    def test_mock_convert_with_formant_shift(self):
        """Test mock conversion with formant shift."""
        from voice_soundboard.conversion.base import MockVoiceConverter, ConversionConfig

        config = ConversionConfig(formant_shift_ratio=1.2)
        converter = MockVoiceConverter(config=config, simulate_latency=False)
        audio = np.random.randn(1000).astype(np.float32)

        result = converter._mock_convert(audio, 24000)
        assert isinstance(result, np.ndarray)

    def test_mock_convert_with_target_embedding(self):
        """Test mock conversion with target embedding."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)
        converter._target_embedding = np.random.randn(256).astype(np.float32)

        audio = np.random.randn(1000).astype(np.float32)
        result = converter._mock_convert(audio, 24000)
        assert isinstance(result, np.ndarray)

    def test_stats_update_on_convert(self):
        """Test stats are updated on convert."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)
        audio = np.random.randn(1000).astype(np.float32)

        converter.convert(audio, 24000)
        converter.convert(audio, 24000)

        assert converter.stats.chunks_processed == 2


class TestVoiceConverterBase:
    """Tests for VoiceConverter base class."""

    def test_properties(self):
        """Test VoiceConverter properties."""
        from voice_soundboard.conversion.base import MockVoiceConverter, ConversionState

        converter = MockVoiceConverter()
        assert converter.state == ConversionState.IDLE
        assert converter.is_running is False

    def test_set_target_voice_with_string(self):
        """Test set_target_voice with string."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter()
        converter.set_target_voice("test_voice")
        assert converter._target_voice_id == "test_voice"

    def test_set_target_voice_with_path(self):
        """Test set_target_voice with Path."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter()
        # Mock the _extract_embedding method to avoid file read
        with patch.object(converter, '_extract_embedding', return_value=np.random.randn(256).astype(np.float32)):
            converter.set_target_voice(Path("/fake/audio.wav"))
        assert converter._target_voice_id == "audio"

    def test_set_target_voice_with_array(self):
        """Test set_target_voice with numpy array."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter()
        embedding = np.random.randn(256).astype(np.float32)
        converter.set_target_voice(embedding)
        assert converter._target_voice_id == "custom"
        np.testing.assert_array_equal(converter._target_embedding, embedding)

    def test_reset_stats(self):
        """Test reset_stats method."""
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)
        audio = np.random.randn(1000).astype(np.float32)
        converter.convert(audio, 24000)

        assert converter.stats.chunks_processed > 0
        converter.reset_stats()
        assert converter.stats.chunks_processed == 0


# ============================================================================
# Tests for conversion/streaming.py
# ============================================================================

class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_pipeline_stage_values(self):
        """Test PipelineStage enum values."""
        from voice_soundboard.conversion.streaming import PipelineStage

        assert hasattr(PipelineStage, "INPUT")
        assert hasattr(PipelineStage, "PREPROCESS")
        assert hasattr(PipelineStage, "ENCODE")
        assert hasattr(PipelineStage, "CONVERT")
        assert hasattr(PipelineStage, "DECODE")
        assert hasattr(PipelineStage, "POSTPROCESS")
        assert hasattr(PipelineStage, "OUTPUT")


class TestAudioChunk:
    """Tests for AudioChunk dataclass."""

    def test_creation(self):
        """Test AudioChunk creation."""
        from voice_soundboard.conversion.streaming import AudioChunk, PipelineStage

        data = np.random.randn(480).astype(np.float32)
        chunk = AudioChunk(data=data, sample_rate=24000, timestamp_ms=100.0)

        assert chunk.sample_rate == 24000
        assert chunk.timestamp_ms == 100.0
        assert chunk.stage == PipelineStage.INPUT

    def test_duration_ms(self):
        """Test duration_ms property."""
        from voice_soundboard.conversion.streaming import AudioChunk

        data = np.zeros(480, dtype=np.float32)
        chunk = AudioChunk(data=data, sample_rate=24000, timestamp_ms=0.0)

        assert chunk.duration_ms == 20.0  # 480 / 24000 * 1000

    def test_processing_time_ms(self):
        """Test processing_time_ms property."""
        from voice_soundboard.conversion.streaming import AudioChunk

        data = np.zeros(480, dtype=np.float32)
        chunk = AudioChunk(
            data=data,
            sample_rate=24000,
            timestamp_ms=0.0,
            processing_started_ms=100.0,
            processing_completed_ms=150.0,
        )

        assert chunk.processing_time_ms == 50.0

    def test_processing_time_ms_not_completed(self):
        """Test processing_time_ms when not completed."""
        from voice_soundboard.conversion.streaming import AudioChunk

        data = np.zeros(480, dtype=np.float32)
        chunk = AudioChunk(data=data, sample_rate=24000, timestamp_ms=0.0)

        assert chunk.processing_time_ms == 0.0

    def test_copy(self):
        """Test copy method."""
        from voice_soundboard.conversion.streaming import AudioChunk

        data = np.random.randn(480).astype(np.float32)
        original = AudioChunk(data=data, sample_rate=24000, timestamp_ms=100.0)

        copy = original.copy()
        assert copy.sample_rate == original.sample_rate
        np.testing.assert_array_equal(copy.data, original.data)
        assert copy.data is not original.data  # Different objects


class TestAudioBuffer:
    """Tests for AudioBuffer class."""

    def test_creation(self):
        """Test AudioBuffer creation."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000, channels=1)
        assert buffer.capacity == 1000
        assert buffer.channels == 1
        assert buffer.available == 0

    def test_write_and_read(self):
        """Test write and read operations."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000, channels=1)
        data = np.random.randn(500).astype(np.float32)

        written = buffer.write(data)
        assert written == 500
        assert buffer.available == 500

        read_data = buffer.read(500, block=False)
        assert read_data is not None
        np.testing.assert_array_almost_equal(read_data, data)

    def test_free_space(self):
        """Test free_space property."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000, channels=1)
        assert buffer.free_space == 1000

        data = np.zeros(300, dtype=np.float32)
        buffer.write(data, block=False)
        assert buffer.free_space == 700

    def test_write_nonblocking_overflow(self):
        """Test write with non-blocking overflow."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=100, channels=1)
        data = np.zeros(150, dtype=np.float32)

        written = buffer.write(data, block=False)
        assert written == 100

    def test_read_nonblocking_empty(self):
        """Test read from empty buffer non-blocking."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000, channels=1)
        result = buffer.read(100, block=False)
        assert result is None

    def test_wrap_around(self):
        """Test circular buffer wrap-around."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=100, channels=1)

        # Write 80 samples
        data1 = np.ones(80, dtype=np.float32)
        buffer.write(data1, block=False)

        # Read 50 samples
        buffer.read(50, block=False)

        # Write 60 more (should wrap around)
        data2 = np.ones(60, dtype=np.float32) * 2
        written = buffer.write(data2, block=False)
        assert written == 60

        # Read remaining
        remaining = buffer.read(90, block=False)
        assert remaining is not None
        assert len(remaining) == 90

    def test_peek(self):
        """Test peek method."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000, channels=1)
        data = np.random.randn(500).astype(np.float32)
        buffer.write(data, block=False)

        # Peek should not remove data
        peeked = buffer.peek(100)
        assert peeked is not None
        assert len(peeked) == 100
        assert buffer.available == 500

    def test_peek_empty(self):
        """Test peek on empty buffer."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000, channels=1)
        result = buffer.peek(100)
        assert result is None

    def test_clear(self):
        """Test clear method."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=1000, channels=1)
        data = np.zeros(500, dtype=np.float32)
        buffer.write(data, block=False)

        buffer.clear()
        assert buffer.available == 0


class TestStreamingConverter:
    """Tests for StreamingConverter class."""

    def test_creation(self):
        """Test StreamingConverter creation."""
        from voice_soundboard.conversion.streaming import StreamingConverter
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter()
        streaming = StreamingConverter(converter=converter)

        assert streaming.chunk_size == 480
        assert streaming.sample_rate == 24000
        assert streaming.is_running is False

    def test_start_and_stop(self):
        """Test start and stop methods."""
        from voice_soundboard.conversion.streaming import StreamingConverter
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)
        streaming = StreamingConverter(converter=converter)

        streaming.start()
        assert streaming.is_running is True

        streaming.stop()
        assert streaming.is_running is False

    def test_push_and_process(self):
        """Test push and processing."""
        from voice_soundboard.conversion.streaming import StreamingConverter
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)
        streaming = StreamingConverter(converter=converter, chunk_size=100)

        output_chunks = []

        def on_output(audio):
            output_chunks.append(audio)

        streaming.start(on_output=on_output)

        # Push some audio
        audio = np.random.randn(200).astype(np.float32)
        streaming.push(audio)

        # Wait for processing
        time.sleep(0.2)

        streaming.stop()

        # Should have processed some chunks
        assert len(output_chunks) > 0

    def test_input_and_output_available(self):
        """Test input_available and output_available properties."""
        from voice_soundboard.conversion.streaming import StreamingConverter
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)
        streaming = StreamingConverter(converter=converter)

        audio = np.zeros(100, dtype=np.float32)
        streaming.push(audio)

        assert streaming.input_available >= 0
        assert streaming.output_available >= 0

    def test_avg_latency_ms(self):
        """Test avg_latency_ms property."""
        from voice_soundboard.conversion.streaming import StreamingConverter
        from voice_soundboard.conversion.base import MockVoiceConverter

        converter = MockVoiceConverter(simulate_latency=False)
        streaming = StreamingConverter(converter=converter, chunk_size=100)

        streaming.start()
        audio = np.random.randn(200).astype(np.float32)
        streaming.push(audio)

        time.sleep(0.2)
        streaming.stop()

        # Should have some latency recorded
        assert streaming.avg_latency_ms >= 0


class TestConversionPipeline:
    """Tests for ConversionPipeline class."""

    def test_creation(self):
        """Test ConversionPipeline creation."""
        from voice_soundboard.conversion.streaming import ConversionPipeline, PipelineStage

        stages = [
            (PipelineStage.PREPROCESS, lambda x: x),
            (PipelineStage.CONVERT, lambda x: x),
        ]
        pipeline = ConversionPipeline(stages=stages)

        assert len(pipeline.stages) == 2
        assert pipeline.is_running is False

    def test_start_and_stop(self):
        """Test start and stop methods."""
        from voice_soundboard.conversion.streaming import ConversionPipeline, PipelineStage

        stages = [
            (PipelineStage.PREPROCESS, lambda x: x),
        ]
        pipeline = ConversionPipeline(stages=stages)

        pipeline.start()
        assert pipeline.is_running is True

        pipeline.stop()
        assert pipeline.is_running is False

    def test_push_and_pull(self):
        """Test push and pull operations."""
        from voice_soundboard.conversion.streaming import ConversionPipeline, PipelineStage

        def double(x):
            return x * 2

        stages = [
            (PipelineStage.CONVERT, double),
        ]
        pipeline = ConversionPipeline(stages=stages)
        pipeline.start()

        audio = np.ones(100, dtype=np.float32)
        pipeline.push(audio)

        time.sleep(0.2)

        result = pipeline.pull(timeout=0.2)
        pipeline.stop()

        assert result is not None
        np.testing.assert_array_almost_equal(result, audio * 2)

    def test_get_stage_latency(self):
        """Test get_stage_latency method."""
        from voice_soundboard.conversion.streaming import ConversionPipeline, PipelineStage

        stages = [
            (PipelineStage.CONVERT, lambda x: x),
        ]
        pipeline = ConversionPipeline(stages=stages)

        # Before any processing
        latency = pipeline.get_stage_latency(PipelineStage.CONVERT)
        assert latency == 0.0

    def test_get_total_latency(self):
        """Test get_total_latency method."""
        from voice_soundboard.conversion.streaming import ConversionPipeline, PipelineStage

        stages = [
            (PipelineStage.PREPROCESS, lambda x: x),
            (PipelineStage.CONVERT, lambda x: x),
        ]
        pipeline = ConversionPipeline(stages=stages)

        latency = pipeline.get_total_latency()
        assert latency >= 0.0


# ============================================================================
# Tests for conversion/devices.py
# ============================================================================

class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_device_type_values(self):
        """Test DeviceType enum values."""
        from voice_soundboard.conversion.devices import DeviceType

        assert hasattr(DeviceType, "INPUT")
        assert hasattr(DeviceType, "OUTPUT")
        assert hasattr(DeviceType, "DUPLEX")


class TestAudioDevice:
    """Tests for AudioDevice dataclass."""

    def test_creation(self):
        """Test AudioDevice creation."""
        from voice_soundboard.conversion.devices import AudioDevice, DeviceType

        device = AudioDevice(
            id=0,
            name="Test Microphone",
            device_type=DeviceType.INPUT,
            max_input_channels=2,
        )
        assert device.name == "Test Microphone"
        assert device.device_type == DeviceType.INPUT

    def test_supports_sample_rate(self):
        """Test supports_sample_rate method."""
        from voice_soundboard.conversion.devices import AudioDevice, DeviceType

        device = AudioDevice(id=0, name="Test", device_type=DeviceType.INPUT)
        assert device.supports_sample_rate(44100) is True
        assert device.supports_sample_rate(48000) is True
        assert device.supports_sample_rate(12345) is False

    def test_to_dict(self):
        """Test to_dict method."""
        from voice_soundboard.conversion.devices import AudioDevice, DeviceType

        device = AudioDevice(
            id=0,
            name="Test",
            device_type=DeviceType.INPUT,
            is_default=True,
        )
        result = device.to_dict()

        assert isinstance(result, dict)
        assert result["id"] == 0
        assert result["name"] == "Test"
        assert result["type"] == "input"
        assert result["is_default"] is True


class TestListAudioDevices:
    """Tests for list_audio_devices function."""

    def test_list_all_devices(self):
        """Test list all audio devices."""
        from voice_soundboard.conversion.devices import list_audio_devices

        devices = list_audio_devices()
        assert isinstance(devices, list)
        assert len(devices) > 0

    def test_list_input_devices(self):
        """Test list input devices."""
        from voice_soundboard.conversion.devices import list_audio_devices, DeviceType

        devices = list_audio_devices(DeviceType.INPUT)
        assert isinstance(devices, list)
        for device in devices:
            assert device.max_input_channels > 0

    def test_list_output_devices(self):
        """Test list output devices."""
        from voice_soundboard.conversion.devices import list_audio_devices, DeviceType

        devices = list_audio_devices(DeviceType.OUTPUT)
        assert isinstance(devices, list)
        for device in devices:
            assert device.max_output_channels > 0


class TestGetDefaultDevices:
    """Tests for get_default_input_device and get_default_output_device."""

    def test_get_default_input_device(self):
        """Test get_default_input_device."""
        from voice_soundboard.conversion.devices import get_default_input_device

        device = get_default_input_device()
        assert device is not None
        assert device.max_input_channels > 0

    def test_get_default_output_device(self):
        """Test get_default_output_device."""
        from voice_soundboard.conversion.devices import get_default_output_device

        device = get_default_output_device()
        assert device is not None
        assert device.max_output_channels > 0


class TestAudioDeviceManager:
    """Tests for AudioDeviceManager class."""

    def test_creation(self):
        """Test AudioDeviceManager creation."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager(sample_rate=24000, channels=1)
        assert manager.sample_rate == 24000
        assert manager.channels == 1

    def test_set_input_device(self):
        """Test set_input_device method."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()
        device = manager.set_input_device(None)
        assert device is not None

    def test_set_output_device(self):
        """Test set_output_device method."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()
        device = manager.set_output_device(None)
        assert device is not None

    def test_resolve_device_by_id(self):
        """Test _resolve_device by device ID."""
        from voice_soundboard.conversion.devices import AudioDeviceManager, DeviceType

        manager = AudioDeviceManager()
        device = manager._resolve_device(0, DeviceType.INPUT)
        assert device is not None
        assert device.id == 0

    def test_resolve_device_by_name(self):
        """Test _resolve_device by name."""
        from voice_soundboard.conversion.devices import AudioDeviceManager, DeviceType

        manager = AudioDeviceManager()
        device = manager._resolve_device("Microphone", DeviceType.INPUT)
        assert device is not None
        assert "microphone" in device.name.lower()

    def test_resolve_device_invalid_id(self):
        """Test _resolve_device with invalid ID."""
        from voice_soundboard.conversion.devices import AudioDeviceManager, DeviceType

        manager = AudioDeviceManager()
        with pytest.raises(ValueError, match="not found"):
            manager._resolve_device(999, DeviceType.INPUT)

    def test_resolve_device_invalid_name(self):
        """Test _resolve_device with invalid name."""
        from voice_soundboard.conversion.devices import AudioDeviceManager, DeviceType

        manager = AudioDeviceManager()
        with pytest.raises(ValueError, match="not found"):
            manager._resolve_device("NonexistentDevice12345", DeviceType.INPUT)

    def test_resolve_device_invalid_type(self):
        """Test _resolve_device with invalid type."""
        from voice_soundboard.conversion.devices import AudioDeviceManager, DeviceType

        manager = AudioDeviceManager()
        with pytest.raises(TypeError, match="Invalid device type"):
            manager._resolve_device(3.14, DeviceType.INPUT)

    def test_start_stop_capture_mock(self):
        """Test start_capture and stop_capture with mock devices."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()
        callback = Mock()

        manager.start_capture(callback)
        assert manager.is_capturing is True

        manager.stop_capture()
        assert manager.is_capturing is False

    def test_start_stop_playback_mock(self):
        """Test start_playback and stop_playback with mock devices."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()

        manager.start_playback()
        assert manager.is_playing is True

        manager.stop_playback()
        assert manager.is_playing is False

    def test_play_chunk_when_not_playing(self):
        """Test play_chunk when not playing."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()
        audio = np.zeros(100, dtype=np.float32)

        # Should not raise
        manager.play_chunk(audio)

    def test_stop_all(self):
        """Test stop_all method."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()
        manager.start_capture(lambda x: None)
        manager.start_playback()

        manager.stop_all()

        assert manager.is_capturing is False
        assert manager.is_playing is False

    def test_context_manager(self):
        """Test context manager protocol."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        with AudioDeviceManager() as manager:
            manager.start_playback()
            assert manager.is_playing is True

        # After context, should be stopped
        assert manager.is_playing is False


# ============================================================================
# Tests for conversion/realtime.py
# ============================================================================

class TestRealtimeSession:
    """Tests for RealtimeSession dataclass."""

    def test_creation(self):
        """Test RealtimeSession creation."""
        from voice_soundboard.conversion.realtime import RealtimeSession

        session = RealtimeSession(session_id="session_1")
        assert session.session_id == "session_1"
        assert session.started_at == 0.0

    def test_duration_seconds_not_started(self):
        """Test duration_seconds when not started."""
        from voice_soundboard.conversion.realtime import RealtimeSession

        session = RealtimeSession(session_id="session_1")
        assert session.duration_seconds == 0.0

    def test_duration_seconds_running(self):
        """Test duration_seconds when running."""
        from voice_soundboard.conversion.realtime import RealtimeSession

        session = RealtimeSession(session_id="session_1", started_at=time.time() - 5)
        assert 4.5 < session.duration_seconds < 6.0

    def test_duration_seconds_stopped(self):
        """Test duration_seconds when stopped."""
        from voice_soundboard.conversion.realtime import RealtimeSession

        start = time.time() - 10
        session = RealtimeSession(
            session_id="session_1",
            started_at=start,
            stopped_at=start + 5,
        )
        assert session.duration_seconds == pytest.approx(5.0, abs=0.1)

    def test_to_dict(self):
        """Test to_dict method."""
        from voice_soundboard.conversion.realtime import RealtimeSession
        from voice_soundboard.conversion.devices import AudioDevice, DeviceType

        session = RealtimeSession(
            session_id="session_1",
            input_device=AudioDevice(id=0, name="Mic", device_type=DeviceType.INPUT),
            target_voice="test_voice",
        )
        result = session.to_dict()

        assert isinstance(result, dict)
        assert result["session_id"] == "session_1"
        assert result["input_device"] == "Mic"
        assert result["target_voice"] == "test_voice"


class TestRealtimeConverter:
    """Tests for RealtimeConverter class."""

    def test_creation(self):
        """Test RealtimeConverter creation."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()
        assert converter.is_running is False
        assert converter.current_session is None

    def test_creation_with_config(self):
        """Test RealtimeConverter creation with config."""
        from voice_soundboard.conversion.realtime import RealtimeConverter
        from voice_soundboard.conversion.base import ConversionConfig, LatencyMode

        config = ConversionConfig(latency_mode=LatencyMode.LOW)
        converter = RealtimeConverter(config=config)
        assert converter.config.latency_mode == LatencyMode.LOW

    def test_start_creates_session(self):
        """Test start creates a session."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()
        session = converter.start()

        assert session is not None
        assert session.session_id == "session_1"
        assert converter.is_running is True

        converter.stop()

    def test_start_twice_raises(self):
        """Test starting twice raises error."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()
        converter.start()

        with pytest.raises(RuntimeError, match="already running"):
            converter.start()

        converter.stop()

    def test_stop_returns_session(self):
        """Test stop returns session."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()
        converter.start()
        session = converter.stop()

        assert session is not None
        assert session.stopped_at > 0
        assert converter.is_running is False

    def test_stop_when_not_running(self):
        """Test stop when not running."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()
        result = converter.stop()
        assert result is None

    def test_set_target_voice(self):
        """Test set_target_voice method."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()
        converter.start()
        converter.set_target_voice("new_voice")

        assert converter.current_session.target_voice == "new_voice"
        converter.stop()

    def test_set_latency_mode(self):
        """Test set_latency_mode method."""
        from voice_soundboard.conversion.realtime import RealtimeConverter
        from voice_soundboard.conversion.base import LatencyMode

        converter = RealtimeConverter()
        converter.set_latency_mode(LatencyMode.ULTRA_LOW)
        assert converter.config.latency_mode == LatencyMode.ULTRA_LOW

    def test_get_latency(self):
        """Test get_latency method."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()
        latency = converter.get_latency()
        assert latency >= 0.0

    def test_stats_property(self):
        """Test stats property."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()
        stats = converter.stats
        assert stats.chunks_processed >= 0


class TestStartRealtimeConversion:
    """Tests for start_realtime_conversion function."""

    def test_start_with_defaults(self):
        """Test start_realtime_conversion with defaults."""
        from voice_soundboard.conversion.realtime import start_realtime_conversion

        converter = start_realtime_conversion()
        assert converter.is_running is True
        converter.stop()

    def test_start_with_target_voice(self):
        """Test start_realtime_conversion with target voice."""
        from voice_soundboard.conversion.realtime import start_realtime_conversion

        converter = start_realtime_conversion(target_voice="test_voice")
        assert converter.is_running is True
        converter.stop()

    def test_start_with_latency_mode(self):
        """Test start_realtime_conversion with latency mode."""
        from voice_soundboard.conversion.realtime import start_realtime_conversion
        from voice_soundboard.conversion.base import LatencyMode

        converter = start_realtime_conversion(latency_mode=LatencyMode.ULTRA_LOW)
        assert converter.config.latency_mode == LatencyMode.ULTRA_LOW
        converter.stop()

    def test_start_with_callback(self):
        """Test start_realtime_conversion with callback."""
        from voice_soundboard.conversion.realtime import start_realtime_conversion

        callback = Mock()
        converter = start_realtime_conversion(on_converted=callback)
        assert converter.is_running is True
        converter.stop()


# ============================================================================
# Additional edge case tests
# ============================================================================

class TestConversionEdgeCases:
    """Additional edge case tests for conversion module."""

    def test_buffer_wrap_read(self):
        """Test buffer read with wrap-around."""
        from voice_soundboard.conversion.streaming import AudioBuffer

        buffer = AudioBuffer(capacity_samples=100, channels=1)

        # Write to end of buffer
        data1 = np.ones(80, dtype=np.float32)
        buffer.write(data1, block=False)

        # Read some
        buffer.read(50, block=False)

        # Write more (wraps)
        data2 = np.ones(50, dtype=np.float32) * 2
        buffer.write(data2, block=False)

        # Read all (should handle wrap)
        result = buffer.read(80, block=False)
        assert result is not None
        assert len(result) == 80

    def test_conversion_result_with_all_fields(self):
        """Test ConversionResult with all optional fields."""
        from voice_soundboard.conversion.base import ConversionResult, ConversionConfig

        config = ConversionConfig()
        audio = np.zeros(1000, dtype=np.float32)

        result = ConversionResult(
            audio=audio,
            sample_rate=24000,
            input_duration_ms=41.67,
            output_duration_ms=41.67,
            processing_time_ms=10.0,
            latency_ms=10.0,
            similarity_score=0.9,
            naturalness_score=0.85,
            source_voice="original",
            target_voice="clone",
            config=config,
        )

        assert result.source_voice == "original"
        assert result.config == config

    def test_mock_devices_with_duplex(self):
        """Test mock devices returns duplex device."""
        from voice_soundboard.conversion.devices import _get_mock_devices, DeviceType

        devices = _get_mock_devices()
        duplex_devices = [d for d in devices if d.device_type == DeviceType.DUPLEX]
        assert len(duplex_devices) > 0

    def test_mock_devices_filter_input(self):
        """Test mock devices filter for input."""
        from voice_soundboard.conversion.devices import _get_mock_devices, DeviceType

        devices = _get_mock_devices(DeviceType.INPUT)
        for device in devices:
            assert device.max_input_channels > 0

    def test_audio_device_manager_with_audio_device(self):
        """Test AudioDeviceManager with AudioDevice object."""
        from voice_soundboard.conversion.devices import AudioDeviceManager, AudioDevice, DeviceType

        manager = AudioDeviceManager()
        device = AudioDevice(id=0, name="Custom", device_type=DeviceType.INPUT)

        result = manager._resolve_device(device, DeviceType.INPUT)
        assert result == device
