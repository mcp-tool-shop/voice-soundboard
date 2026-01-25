"""
Additional test coverage batch 9.

Tests for:
- conversion/devices.py (DeviceType, AudioDevice, AudioDeviceManager, list_audio_devices)
- conversion/realtime.py (RealtimeSession, RealtimeConverter, start_realtime_conversion)
- llm/providers.py (ProviderType, LLMConfig, LLMResponse, MockLLMProvider, create_provider)
"""

import pytest
import numpy as np
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock


# =============================================================================
# Tests for conversion/devices.py
# =============================================================================

class TestDeviceTypeEnum:
    """Tests for DeviceType enum."""

    def test_device_type_values(self):
        """Test DeviceType enum values."""
        from voice_soundboard.conversion.devices import DeviceType

        assert DeviceType.INPUT is not None
        assert DeviceType.OUTPUT is not None
        assert DeviceType.DUPLEX is not None

    def test_device_type_count(self):
        """Test DeviceType enum has 3 types."""
        from voice_soundboard.conversion.devices import DeviceType

        assert len(DeviceType) == 3


class TestAudioDevice:
    """Tests for AudioDevice dataclass."""

    def test_audio_device_creation(self):
        """Test AudioDevice creation."""
        from voice_soundboard.conversion.devices import AudioDevice, DeviceType

        device = AudioDevice(
            id=0,
            name="Test Microphone",
            device_type=DeviceType.INPUT
        )

        assert device.id == 0
        assert device.name == "Test Microphone"
        assert device.device_type == DeviceType.INPUT
        assert device.max_input_channels == 0
        assert device.max_output_channels == 0
        assert device.default_sample_rate == 44100.0

    def test_audio_device_with_channels(self):
        """Test AudioDevice with channel info."""
        from voice_soundboard.conversion.devices import AudioDevice, DeviceType

        device = AudioDevice(
            id=1,
            name="Test Speakers",
            device_type=DeviceType.OUTPUT,
            max_output_channels=2,
            default_sample_rate=48000.0
        )

        assert device.max_output_channels == 2
        assert device.default_sample_rate == 48000.0

    def test_audio_device_supports_sample_rate(self):
        """Test AudioDevice supports_sample_rate method."""
        from voice_soundboard.conversion.devices import AudioDevice, DeviceType

        device = AudioDevice(id=0, name="Test", device_type=DeviceType.INPUT)

        assert device.supports_sample_rate(44100) is True
        assert device.supports_sample_rate(48000) is True
        assert device.supports_sample_rate(24000) is True
        assert device.supports_sample_rate(12345) is False

    def test_audio_device_to_dict(self):
        """Test AudioDevice to_dict method."""
        from voice_soundboard.conversion.devices import AudioDevice, DeviceType

        device = AudioDevice(
            id=0,
            name="Test Device",
            device_type=DeviceType.DUPLEX,
            max_input_channels=2,
            max_output_channels=2,
            is_default=True
        )

        data = device.to_dict()

        assert data["id"] == 0
        assert data["name"] == "Test Device"
        assert data["type"] == "duplex"
        assert data["is_default"] is True


class TestListAudioDevices:
    """Tests for list_audio_devices function."""

    def test_list_audio_devices_all(self):
        """Test listing all audio devices."""
        from voice_soundboard.conversion.devices import list_audio_devices

        devices = list_audio_devices()

        assert len(devices) >= 1
        assert all(hasattr(d, 'id') for d in devices)
        assert all(hasattr(d, 'name') for d in devices)

    def test_list_audio_devices_input(self):
        """Test listing input devices."""
        from voice_soundboard.conversion.devices import list_audio_devices, DeviceType

        devices = list_audio_devices(DeviceType.INPUT)

        assert len(devices) >= 1
        for device in devices:
            assert device.max_input_channels > 0 or device.device_type == DeviceType.DUPLEX

    def test_list_audio_devices_output(self):
        """Test listing output devices."""
        from voice_soundboard.conversion.devices import list_audio_devices, DeviceType

        devices = list_audio_devices(DeviceType.OUTPUT)

        assert len(devices) >= 1
        for device in devices:
            assert device.max_output_channels > 0 or device.device_type == DeviceType.DUPLEX


class TestGetDefaultDevices:
    """Tests for get_default_input/output_device functions."""

    def test_get_default_input_device(self):
        """Test getting default input device."""
        from voice_soundboard.conversion.devices import get_default_input_device

        device = get_default_input_device()

        assert device is not None
        assert device.max_input_channels > 0 or hasattr(device, 'is_default')

    def test_get_default_output_device(self):
        """Test getting default output device."""
        from voice_soundboard.conversion.devices import get_default_output_device

        device = get_default_output_device()

        assert device is not None
        assert device.max_output_channels > 0 or hasattr(device, 'is_default')


class TestAudioDeviceManager:
    """Tests for AudioDeviceManager class."""

    def test_device_manager_creation(self):
        """Test AudioDeviceManager creation."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()

        assert manager.sample_rate == 24000
        assert manager.channels == 1
        assert manager.chunk_size == 480

    def test_device_manager_custom_settings(self):
        """Test AudioDeviceManager with custom settings."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager(
            sample_rate=48000,
            channels=2,
            chunk_size=960
        )

        assert manager.sample_rate == 48000
        assert manager.channels == 2
        assert manager.chunk_size == 960

    def test_device_manager_set_input_device(self):
        """Test setting input device."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()
        device = manager.set_input_device()

        assert device is not None

    def test_device_manager_set_output_device(self):
        """Test setting output device."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()
        device = manager.set_output_device()

        assert device is not None

    def test_device_manager_set_device_by_name(self):
        """Test setting device by name."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()
        # This should match "Default Microphone" from mock devices
        # Note: Mock devices include "Default Microphone" so we use "microphone"
        device = manager.set_input_device("microphone")

        assert device is not None
        assert "Microphone" in device.name

    def test_device_manager_is_capturing_property(self):
        """Test is_capturing property."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()

        assert manager.is_capturing is False

    def test_device_manager_is_playing_property(self):
        """Test is_playing property."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()

        assert manager.is_playing is False

    def test_device_manager_context_manager(self):
        """Test AudioDeviceManager as context manager."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        with AudioDeviceManager() as manager:
            assert manager is not None

    def test_device_manager_start_stop_capture(self):
        """Test start and stop capture."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()
        callback = Mock()

        manager.start_capture(callback)
        assert manager.is_capturing is True

        manager.stop_capture()
        assert manager.is_capturing is False

    def test_device_manager_start_stop_playback(self):
        """Test start and stop playback."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()

        manager.start_playback()
        assert manager.is_playing is True

        manager.stop_playback()
        assert manager.is_playing is False

    def test_device_manager_stop_all(self):
        """Test stop_all method."""
        from voice_soundboard.conversion.devices import AudioDeviceManager

        manager = AudioDeviceManager()

        manager.start_capture(Mock())
        manager.start_playback()

        manager.stop_all()

        assert manager.is_capturing is False
        assert manager.is_playing is False

    def test_device_manager_resolve_device_invalid_type(self):
        """Test resolving device with invalid type."""
        from voice_soundboard.conversion.devices import AudioDeviceManager, DeviceType

        manager = AudioDeviceManager()

        with pytest.raises(TypeError):
            manager._resolve_device(3.14, DeviceType.INPUT)

    def test_device_manager_resolve_device_not_found(self):
        """Test resolving non-existent device ID."""
        from voice_soundboard.conversion.devices import AudioDeviceManager, DeviceType

        manager = AudioDeviceManager()

        with pytest.raises(ValueError, match="not found"):
            manager._resolve_device(999, DeviceType.INPUT)


# =============================================================================
# Tests for conversion/realtime.py
# =============================================================================

class TestRealtimeSession:
    """Tests for RealtimeSession dataclass."""

    def test_realtime_session_creation(self):
        """Test RealtimeSession creation."""
        from voice_soundboard.conversion.realtime import RealtimeSession

        session = RealtimeSession(session_id="test_1")

        assert session.session_id == "test_1"
        assert session.input_device is None
        assert session.output_device is None
        assert session.target_voice is None

    def test_realtime_session_duration(self):
        """Test RealtimeSession duration_seconds property."""
        from voice_soundboard.conversion.realtime import RealtimeSession

        session = RealtimeSession(
            session_id="test_1",
            started_at=time.time() - 5.0  # Started 5 seconds ago
        )

        assert session.duration_seconds >= 4.9

    def test_realtime_session_duration_stopped(self):
        """Test RealtimeSession duration when stopped."""
        from voice_soundboard.conversion.realtime import RealtimeSession

        now = time.time()
        session = RealtimeSession(
            session_id="test_1",
            started_at=now - 10.0,
            stopped_at=now - 5.0
        )

        assert abs(session.duration_seconds - 5.0) < 0.1

    def test_realtime_session_to_dict(self):
        """Test RealtimeSession to_dict method."""
        from voice_soundboard.conversion.realtime import RealtimeSession
        from voice_soundboard.conversion.devices import AudioDevice, DeviceType

        device = AudioDevice(id=0, name="Test", device_type=DeviceType.INPUT)
        session = RealtimeSession(
            session_id="test_1",
            input_device=device,
            target_voice="my_voice"
        )

        data = session.to_dict()

        assert data["session_id"] == "test_1"
        assert data["input_device"] == "Test"
        assert data["target_voice"] == "my_voice"


class TestRealtimeConverter:
    """Tests for RealtimeConverter class."""

    def test_realtime_converter_creation(self):
        """Test RealtimeConverter creation."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()

        assert converter.is_running is False
        assert converter.current_session is None

    def test_realtime_converter_with_config(self):
        """Test RealtimeConverter with custom config."""
        from voice_soundboard.conversion.realtime import RealtimeConverter
        from voice_soundboard.conversion.base import ConversionConfig, LatencyMode

        config = ConversionConfig(latency_mode=LatencyMode.LOW)
        converter = RealtimeConverter(config=config)

        assert converter.config.latency_mode == LatencyMode.LOW

    def test_realtime_converter_start_stop(self):
        """Test RealtimeConverter start and stop."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()

        session = converter.start()
        assert converter.is_running is True
        assert session is not None
        assert session.session_id == "session_1"

        final_session = converter.stop()
        assert converter.is_running is False
        assert final_session is not None
        assert final_session.stopped_at > 0

    def test_realtime_converter_cannot_start_twice(self):
        """Test that starting twice raises error."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()
        converter.start()

        with pytest.raises(RuntimeError, match="already running"):
            converter.start()

        converter.stop()

    def test_realtime_converter_stop_when_not_running(self):
        """Test stopping when not running returns None."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()
        result = converter.stop()

        assert result is None

    def test_realtime_converter_set_target_voice(self):
        """Test setting target voice."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()
        converter.start()

        converter.set_target_voice("new_voice")

        assert converter.current_session.target_voice == "new_voice"

        converter.stop()

    def test_realtime_converter_set_latency_mode(self):
        """Test setting latency mode."""
        from voice_soundboard.conversion.realtime import RealtimeConverter
        from voice_soundboard.conversion.base import LatencyMode

        converter = RealtimeConverter()

        converter.set_latency_mode(LatencyMode.ULTRA_LOW)

        assert converter.config.latency_mode == LatencyMode.ULTRA_LOW

    def test_realtime_converter_get_latency(self):
        """Test getting latency."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()

        latency = converter.get_latency()

        assert latency == 0.0  # No processing yet

    def test_realtime_converter_stats(self):
        """Test getting stats."""
        from voice_soundboard.conversion.realtime import RealtimeConverter

        converter = RealtimeConverter()
        stats = converter.stats

        assert hasattr(stats, 'chunks_processed')
        assert stats.chunks_processed == 0


class TestStartRealtimeConversion:
    """Tests for start_realtime_conversion function."""

    def test_start_realtime_conversion(self):
        """Test start_realtime_conversion convenience function."""
        from voice_soundboard.conversion.realtime import start_realtime_conversion

        converter = start_realtime_conversion()

        assert converter.is_running is True

        converter.stop()

    def test_start_realtime_conversion_with_latency_mode(self):
        """Test start_realtime_conversion with latency mode."""
        from voice_soundboard.conversion.realtime import start_realtime_conversion
        from voice_soundboard.conversion.base import LatencyMode

        converter = start_realtime_conversion(latency_mode=LatencyMode.HIGH_QUALITY)

        assert converter.config.latency_mode == LatencyMode.HIGH_QUALITY

        converter.stop()

    def test_start_realtime_conversion_with_target_voice(self):
        """Test start_realtime_conversion with target voice."""
        from voice_soundboard.conversion.realtime import start_realtime_conversion

        converter = start_realtime_conversion(target_voice="test_voice")

        assert converter.current_session is not None

        converter.stop()


# =============================================================================
# Tests for llm/providers.py
# =============================================================================

class TestProviderTypeEnum:
    """Tests for ProviderType enum."""

    def test_provider_type_values(self):
        """Test ProviderType enum values."""
        from voice_soundboard.llm.providers import ProviderType

        assert ProviderType.OLLAMA.value == "ollama"
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.VLLM.value == "vllm"
        assert ProviderType.MOCK.value == "mock"

    def test_provider_type_count(self):
        """Test ProviderType enum has 4 types."""
        from voice_soundboard.llm.providers import ProviderType

        assert len(ProviderType) == 4


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_llm_config_defaults(self):
        """Test LLMConfig default values."""
        from voice_soundboard.llm.providers import LLMConfig

        config = LLMConfig()

        assert config.model == "llama3.2"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.stream is True

    def test_llm_config_custom_values(self):
        """Test LLMConfig with custom values."""
        from voice_soundboard.llm.providers import LLMConfig

        config = LLMConfig(
            model="gpt-4",
            temperature=0.5,
            max_tokens=2048,
            api_key="test_key"
        )

        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.api_key == "test_key"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_creation(self):
        """Test LLMResponse creation."""
        from voice_soundboard.llm.providers import LLMResponse

        response = LLMResponse(
            content="Hello, world!",
            model="test-model"
        )

        assert response.content == "Hello, world!"
        assert response.model == "test-model"
        assert response.finish_reason is None
        assert response.latency_ms == 0.0

    def test_llm_response_with_usage(self):
        """Test LLMResponse with usage info."""
        from voice_soundboard.llm.providers import LLMResponse

        response = LLMResponse(
            content="Response text",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5}
        )

        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5


class TestMockLLMProvider:
    """Tests for MockLLMProvider class."""

    def test_mock_provider_creation(self):
        """Test MockLLMProvider creation."""
        from voice_soundboard.llm.providers import MockLLMProvider, ProviderType

        provider = MockLLMProvider()

        assert provider.provider_type == ProviderType.MOCK
        assert provider.default_response == "This is a mock response."

    def test_mock_provider_with_custom_response(self):
        """Test MockLLMProvider with custom response."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            default_response="Custom response",
            token_delay_ms=10.0
        )

        assert provider.default_response == "Custom response"
        assert provider.token_delay_ms == 10.0

    @pytest.mark.asyncio
    async def test_mock_provider_generate(self):
        """Test MockLLMProvider generate method."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider()

        response = await provider.generate("Hello")

        assert response.content == "This is a mock response."
        assert response.model == "mock"
        assert response.provider == "mock"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_mock_provider_generate_with_pattern(self):
        """Test MockLLMProvider generate with pattern matching."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            responses={
                "hello": "Hi there!",
                "bye": "Goodbye!"
            }
        )

        response1 = await provider.generate("Hello world")
        assert response1.content == "Hi there!"

        response2 = await provider.generate("bye bye")
        assert response2.content == "Goodbye!"

    @pytest.mark.asyncio
    async def test_mock_provider_stream(self):
        """Test MockLLMProvider stream method."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            default_response="Hello world",
            token_delay_ms=1.0
        )

        tokens = []
        async for token in provider.stream("Test"):
            tokens.append(token)

        # Should have streamed "Hello" and "world"
        assert len(tokens) == 2
        assert "Hello" in tokens[0]
        assert "world" in tokens[1]

    @pytest.mark.asyncio
    async def test_mock_provider_chat(self):
        """Test MockLLMProvider chat method."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider()

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]

        response = await provider.chat(messages)

        assert response.content is not None

    @pytest.mark.asyncio
    async def test_mock_provider_chat_stream(self):
        """Test MockLLMProvider chat_stream method."""
        from voice_soundboard.llm.providers import MockLLMProvider

        provider = MockLLMProvider(
            default_response="Chat response",
            token_delay_ms=1.0
        )

        messages = [{"role": "user", "content": "Hello"}]

        tokens = []
        async for token in provider.chat_stream(messages):
            tokens.append(token)

        assert len(tokens) >= 1


class TestOllamaProvider:
    """Tests for OllamaProvider class."""

    def test_ollama_provider_creation(self):
        """Test OllamaProvider creation."""
        from voice_soundboard.llm.providers import OllamaProvider, ProviderType

        provider = OllamaProvider()

        assert provider.provider_type == ProviderType.OLLAMA
        assert provider.base_url == "http://localhost:11434"

    def test_ollama_provider_custom_url(self):
        """Test OllamaProvider with custom URL."""
        from voice_soundboard.llm.providers import OllamaProvider, LLMConfig

        config = LLMConfig(base_url="http://custom:8080")
        provider = OllamaProvider(config)

        assert provider.base_url == "http://custom:8080"


class TestOpenAIProvider:
    """Tests for OpenAIProvider class."""

    def test_openai_provider_creation(self):
        """Test OpenAIProvider creation."""
        from voice_soundboard.llm.providers import OpenAIProvider, ProviderType

        provider = OpenAIProvider()

        assert provider.provider_type == ProviderType.OPENAI
        assert provider.base_url == "https://api.openai.com/v1"

    def test_openai_provider_with_api_key(self):
        """Test OpenAIProvider with API key."""
        from voice_soundboard.llm.providers import OpenAIProvider, LLMConfig

        config = LLMConfig(api_key="sk-test-key")
        provider = OpenAIProvider(config)

        assert provider.config.api_key == "sk-test-key"


class TestVLLMProvider:
    """Tests for VLLMProvider class."""

    def test_vllm_provider_creation(self):
        """Test VLLMProvider creation."""
        from voice_soundboard.llm.providers import VLLMProvider, ProviderType

        provider = VLLMProvider()

        assert provider.provider_type == ProviderType.VLLM
        assert provider.base_url == "http://localhost:8000/v1"

    def test_vllm_provider_custom_url(self):
        """Test VLLMProvider with custom URL."""
        from voice_soundboard.llm.providers import VLLMProvider, LLMConfig

        config = LLMConfig(base_url="http://vllm-server:8000/v1")
        provider = VLLMProvider(config)

        assert provider.base_url == "http://vllm-server:8000/v1"


class TestCreateProvider:
    """Tests for create_provider factory function."""

    def test_create_mock_provider(self):
        """Test creating mock provider."""
        from voice_soundboard.llm.providers import create_provider, ProviderType

        provider = create_provider(ProviderType.MOCK)

        assert provider.provider_type == ProviderType.MOCK

    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        from voice_soundboard.llm.providers import create_provider, ProviderType

        provider = create_provider(ProviderType.OLLAMA)

        assert provider.provider_type == ProviderType.OLLAMA

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        from voice_soundboard.llm.providers import create_provider, ProviderType

        provider = create_provider(ProviderType.OPENAI)

        assert provider.provider_type == ProviderType.OPENAI

    def test_create_vllm_provider(self):
        """Test creating vLLM provider."""
        from voice_soundboard.llm.providers import create_provider, ProviderType

        provider = create_provider(ProviderType.VLLM)

        assert provider.provider_type == ProviderType.VLLM

    def test_create_provider_from_string(self):
        """Test creating provider from string."""
        from voice_soundboard.llm.providers import create_provider, ProviderType

        provider = create_provider("mock")

        assert provider.provider_type == ProviderType.MOCK

    def test_create_provider_with_config(self):
        """Test creating provider with config."""
        from voice_soundboard.llm.providers import create_provider, LLMConfig

        config = LLMConfig(model="custom-model")
        provider = create_provider("mock", config=config)

        assert provider.config.model == "custom-model"

    def test_create_provider_with_kwargs(self):
        """Test creating provider with kwargs."""
        from voice_soundboard.llm.providers import create_provider

        provider = create_provider(
            "mock",
            default_response="Custom",
            token_delay_ms=5.0
        )

        assert provider.default_response == "Custom"
        assert provider.token_delay_ms == 5.0

    def test_create_provider_invalid_type(self):
        """Test creating provider with invalid type."""
        from voice_soundboard.llm.providers import create_provider

        with pytest.raises(ValueError):
            create_provider("invalid")


class TestMockDevices:
    """Tests for _get_mock_devices function."""

    def test_get_mock_devices_all(self):
        """Test getting all mock devices."""
        from voice_soundboard.conversion.devices import _get_mock_devices

        devices = _get_mock_devices()

        assert len(devices) == 3
        assert any(d.name == "Default Microphone" for d in devices)
        assert any(d.name == "Default Speakers" for d in devices)
        assert any(d.name == "USB Headset" for d in devices)

    def test_get_mock_devices_input(self):
        """Test getting mock input devices."""
        from voice_soundboard.conversion.devices import _get_mock_devices, DeviceType

        devices = _get_mock_devices(DeviceType.INPUT)

        # Should include devices with input channels
        assert len(devices) >= 1
        for device in devices:
            assert device.max_input_channels > 0 or device.device_type == DeviceType.DUPLEX

    def test_get_mock_devices_output(self):
        """Test getting mock output devices."""
        from voice_soundboard.conversion.devices import _get_mock_devices, DeviceType

        devices = _get_mock_devices(DeviceType.OUTPUT)

        # Should include devices with output channels
        assert len(devices) >= 1
        for device in devices:
            assert device.max_output_channels > 0 or device.device_type == DeviceType.DUPLEX


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
