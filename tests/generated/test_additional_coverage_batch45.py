"""
Test Additional Coverage Batch 45: Conversion & Real-time Tests

Tests for:
- LatencyMode enum
- ConversionState enum
- ConversionConfig dataclass
- ConversionResult dataclass
- ConversionStats dataclass
- VoiceConverter base class
- MockVoiceConverter implementation
- RealtimeSession dataclass
- RealtimeConverter class
- start_realtime_conversion helper
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


# ============== LatencyMode Enum Tests ==============

class TestLatencyModeEnum:
    """Tests for LatencyMode enum."""

    def test_latency_mode_low_value(self):
        """Test LatencyMode.LOW has correct value."""
        from voice_soundboard.conversion.base import LatencyMode
        assert LatencyMode.LOW.value == "low"

    def test_latency_mode_medium_value(self):
        """Test LatencyMode.MEDIUM has correct value."""
        from voice_soundboard.conversion.base import LatencyMode
        assert LatencyMode.MEDIUM.value == "medium"

    def test_latency_mode_high_value(self):
        """Test LatencyMode.HIGH has correct value."""
        from voice_soundboard.conversion.base import LatencyMode
        assert LatencyMode.HIGH.value == "high"

    def test_latency_mode_all_values(self):
        """Test all LatencyMode values are available."""
        from voice_soundboard.conversion.base import LatencyMode
        modes = list(LatencyMode)
        assert len(modes) == 3
        values = [m.value for m in modes]
        assert "low" in values
        assert "medium" in values
        assert "high" in values


# ============== ConversionState Enum Tests ==============

class TestConversionStateEnum:
    """Tests for ConversionState enum."""

    def test_conversion_state_idle(self):
        """Test ConversionState.IDLE value."""
        from voice_soundboard.conversion.base import ConversionState
        assert ConversionState.IDLE.value == "idle"

    def test_conversion_state_converting(self):
        """Test ConversionState.CONVERTING value."""
        from voice_soundboard.conversion.base import ConversionState
        assert ConversionState.CONVERTING.value == "converting"

    def test_conversion_state_paused(self):
        """Test ConversionState.PAUSED value."""
        from voice_soundboard.conversion.base import ConversionState
        assert ConversionState.PAUSED.value == "paused"

    def test_conversion_state_error(self):
        """Test ConversionState.ERROR value."""
        from voice_soundboard.conversion.base import ConversionState
        assert ConversionState.ERROR.value == "error"


# ============== ConversionConfig Tests ==============

class TestConversionConfig:
    """Tests for ConversionConfig dataclass."""

    def test_conversion_config_default_values(self):
        """Test ConversionConfig has correct defaults."""
        from voice_soundboard.conversion.base import ConversionConfig, LatencyMode
        config = ConversionConfig()
        assert config.latency_mode == LatencyMode.MEDIUM
        assert config.sample_rate == 22050
        assert config.chunk_size == 1024

    def test_conversion_config_custom_latency_mode(self):
        """Test ConversionConfig with custom latency mode."""
        from voice_soundboard.conversion.base import ConversionConfig, LatencyMode
        config = ConversionConfig(latency_mode=LatencyMode.LOW)
        assert config.latency_mode == LatencyMode.LOW

    def test_conversion_config_custom_sample_rate(self):
        """Test ConversionConfig with custom sample rate."""
        from voice_soundboard.conversion.base import ConversionConfig
        config = ConversionConfig(sample_rate=44100)
        assert config.sample_rate == 44100

    def test_conversion_config_custom_chunk_size(self):
        """Test ConversionConfig with custom chunk size."""
        from voice_soundboard.conversion.base import ConversionConfig
        config = ConversionConfig(chunk_size=2048)
        assert config.chunk_size == 2048


# ============== ConversionResult Tests ==============

class TestConversionResult:
    """Tests for ConversionResult dataclass."""

    def test_conversion_result_creation(self):
        """Test ConversionResult basic creation."""
        from voice_soundboard.conversion.base import ConversionResult
        audio = np.zeros(1000, dtype=np.float32)
        result = ConversionResult(audio=audio, sample_rate=22050)
        assert result.sample_rate == 22050
        assert len(result.audio) == 1000

    def test_conversion_result_with_latency(self):
        """Test ConversionResult with latency value."""
        from voice_soundboard.conversion.base import ConversionResult
        audio = np.zeros(1000, dtype=np.float32)
        result = ConversionResult(audio=audio, sample_rate=22050, latency_ms=15.5)
        assert result.latency_ms == 15.5

    def test_conversion_result_duration_property(self):
        """Test ConversionResult duration calculation."""
        from voice_soundboard.conversion.base import ConversionResult
        # 22050 samples at 22050 Hz = 1 second
        audio = np.zeros(22050, dtype=np.float32)
        result = ConversionResult(audio=audio, sample_rate=22050)
        assert abs(result.duration - 1.0) < 0.001


# ============== ConversionStats Tests ==============

class TestConversionStats:
    """Tests for ConversionStats dataclass."""

    def test_conversion_stats_creation(self):
        """Test ConversionStats basic creation."""
        from voice_soundboard.conversion.base import ConversionStats
        stats = ConversionStats()
        assert stats.total_chunks == 0
        assert stats.total_samples == 0
        assert stats.avg_latency_ms == 0.0

    def test_conversion_stats_with_values(self):
        """Test ConversionStats with custom values."""
        from voice_soundboard.conversion.base import ConversionStats
        stats = ConversionStats(
            total_chunks=100,
            total_samples=102400,
            avg_latency_ms=12.5,
            min_latency_ms=8.0,
            max_latency_ms=25.0
        )
        assert stats.total_chunks == 100
        assert stats.total_samples == 102400
        assert stats.avg_latency_ms == 12.5
        assert stats.min_latency_ms == 8.0
        assert stats.max_latency_ms == 25.0


# ============== VoiceConverter Base Class Tests ==============

class TestVoiceConverterBase:
    """Tests for VoiceConverter abstract base class."""

    def test_voice_converter_is_abstract(self):
        """Test VoiceConverter cannot be instantiated directly."""
        from voice_soundboard.conversion.base import VoiceConverter
        with pytest.raises(TypeError):
            VoiceConverter()

    def test_voice_converter_subclass_must_implement_convert(self):
        """Test subclass must implement convert method."""
        from voice_soundboard.conversion.base import VoiceConverter

        class IncompleteConverter(VoiceConverter):
            pass

        with pytest.raises(TypeError):
            IncompleteConverter()


# ============== MockVoiceConverter Tests ==============

class TestMockVoiceConverter:
    """Tests for MockVoiceConverter implementation."""

    def test_mock_converter_creation(self):
        """Test MockVoiceConverter instantiation."""
        from voice_soundboard.conversion.base import MockVoiceConverter
        converter = MockVoiceConverter()
        assert converter is not None

    def test_mock_converter_convert_returns_audio(self):
        """Test MockVoiceConverter.convert returns audio array."""
        from voice_soundboard.conversion.base import MockVoiceConverter, ConversionConfig
        converter = MockVoiceConverter()
        config = ConversionConfig()
        audio = np.random.randn(1024).astype(np.float32)
        result = converter.convert(audio, config)
        assert result is not None
        assert hasattr(result, 'audio')

    def test_mock_converter_preserves_length(self):
        """Test MockVoiceConverter preserves audio length."""
        from voice_soundboard.conversion.base import MockVoiceConverter, ConversionConfig
        converter = MockVoiceConverter()
        config = ConversionConfig()
        audio = np.random.randn(2048).astype(np.float32)
        result = converter.convert(audio, config)
        assert len(result.audio) == len(audio)

    def test_mock_converter_with_different_configs(self):
        """Test MockVoiceConverter with various configurations."""
        from voice_soundboard.conversion.base import MockVoiceConverter, ConversionConfig, LatencyMode
        converter = MockVoiceConverter()

        for mode in LatencyMode:
            config = ConversionConfig(latency_mode=mode)
            audio = np.random.randn(512).astype(np.float32)
            result = converter.convert(audio, config)
            assert result is not None


# ============== RealtimeSession Tests ==============

class TestRealtimeSession:
    """Tests for RealtimeSession dataclass."""

    def test_realtime_session_creation(self):
        """Test RealtimeSession basic creation."""
        from voice_soundboard.conversion.realtime import RealtimeSession
        session = RealtimeSession(session_id="test-123")
        assert session.session_id == "test-123"

    def test_realtime_session_with_voice_id(self):
        """Test RealtimeSession with voice_id."""
        from voice_soundboard.conversion.realtime import RealtimeSession
        session = RealtimeSession(session_id="test-456", voice_id="voice-abc")
        assert session.voice_id == "voice-abc"

    def test_realtime_session_default_state(self):
        """Test RealtimeSession default state is idle."""
        from voice_soundboard.conversion.realtime import RealtimeSession
        from voice_soundboard.conversion.base import ConversionState
        session = RealtimeSession(session_id="test-789")
        assert session.state == ConversionState.IDLE

    def test_realtime_session_to_dict(self):
        """Test RealtimeSession to_dict method."""
        from voice_soundboard.conversion.realtime import RealtimeSession
        session = RealtimeSession(session_id="test-dict", voice_id="voice-123")
        d = session.to_dict()
        assert d["session_id"] == "test-dict"
        assert d["voice_id"] == "voice-123"


# ============== RealtimeConverter Tests ==============

class TestRealtimeConverter:
    """Tests for RealtimeConverter class."""

    def test_realtime_converter_creation(self):
        """Test RealtimeConverter instantiation."""
        from voice_soundboard.conversion.realtime import RealtimeConverter
        converter = RealtimeConverter()
        assert converter is not None

    def test_realtime_converter_create_session(self):
        """Test RealtimeConverter.create_session."""
        from voice_soundboard.conversion.realtime import RealtimeConverter
        converter = RealtimeConverter()
        session = converter.create_session(voice_id="test-voice")
        assert session is not None
        assert session.voice_id == "test-voice"

    def test_realtime_converter_get_session(self):
        """Test RealtimeConverter.get_session."""
        from voice_soundboard.conversion.realtime import RealtimeConverter
        converter = RealtimeConverter()
        session = converter.create_session(voice_id="voice-get")
        retrieved = converter.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_realtime_converter_get_nonexistent_session(self):
        """Test RealtimeConverter.get_session returns None for unknown session."""
        from voice_soundboard.conversion.realtime import RealtimeConverter
        converter = RealtimeConverter()
        result = converter.get_session("nonexistent-id")
        assert result is None

    def test_realtime_converter_stop_session(self):
        """Test RealtimeConverter.stop_session."""
        from voice_soundboard.conversion.realtime import RealtimeConverter
        converter = RealtimeConverter()
        session = converter.create_session(voice_id="voice-stop")
        converter.stop_session(session.session_id)
        # Session should be stopped/removed
        result = converter.get_session(session.session_id)
        # Depending on implementation, might be None or state changed
        assert result is None or result.state.value in ["idle", "error"]

    def test_realtime_converter_list_sessions(self):
        """Test RealtimeConverter.list_sessions."""
        from voice_soundboard.conversion.realtime import RealtimeConverter
        converter = RealtimeConverter()
        converter.create_session(voice_id="voice-1")
        converter.create_session(voice_id="voice-2")
        sessions = converter.list_sessions()
        assert len(sessions) >= 2


# ============== start_realtime_conversion Helper Tests ==============

class TestStartRealtimeConversion:
    """Tests for start_realtime_conversion helper function."""

    @patch('voice_soundboard.conversion.realtime.RealtimeConverter')
    def test_start_realtime_conversion_creates_session(self, mock_converter_class):
        """Test start_realtime_conversion creates a session."""
        from voice_soundboard.conversion.realtime import start_realtime_conversion

        mock_converter = Mock()
        mock_session = Mock()
        mock_session.session_id = "new-session"
        mock_converter.create_session.return_value = mock_session
        mock_converter_class.return_value = mock_converter

        result = start_realtime_conversion(voice_id="test-voice")
        assert result is not None

    @patch('voice_soundboard.conversion.realtime.RealtimeConverter')
    def test_start_realtime_conversion_with_config(self, mock_converter_class):
        """Test start_realtime_conversion with custom config."""
        from voice_soundboard.conversion.realtime import start_realtime_conversion
        from voice_soundboard.conversion.base import ConversionConfig, LatencyMode

        mock_converter = Mock()
        mock_session = Mock()
        mock_converter.create_session.return_value = mock_session
        mock_converter_class.return_value = mock_converter

        config = ConversionConfig(latency_mode=LatencyMode.LOW)
        result = start_realtime_conversion(voice_id="voice-config", config=config)
        assert result is not None
