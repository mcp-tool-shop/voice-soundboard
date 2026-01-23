"""
Audio Device Management.

Provides utilities for selecting and managing audio input/output devices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable
import threading

import numpy as np

logger = logging.getLogger(__name__)

# Try to import sounddevice for device management
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class DeviceType(Enum):
    """Type of audio device."""

    INPUT = auto()   # Microphone
    OUTPUT = auto()  # Speakers
    DUPLEX = auto()  # Both input and output


@dataclass
class AudioDevice:
    """Represents an audio device."""

    # Identification
    id: int
    name: str
    device_type: DeviceType

    # Capabilities
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_sample_rate: float = 44100.0

    # Status
    is_default: bool = False
    is_available: bool = True

    # Host API
    host_api: str = ""
    host_api_index: int = 0

    def supports_sample_rate(self, sample_rate: int) -> bool:
        """Check if device supports the given sample rate."""
        # Common supported rates
        common_rates = [8000, 16000, 22050, 24000, 44100, 48000, 96000]
        return sample_rate in common_rates

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.device_type.name.lower(),
            "max_input_channels": self.max_input_channels,
            "max_output_channels": self.max_output_channels,
            "default_sample_rate": self.default_sample_rate,
            "is_default": self.is_default,
            "is_available": self.is_available,
            "host_api": self.host_api,
        }


def list_audio_devices(
    device_type: Optional[DeviceType] = None,
) -> List[AudioDevice]:
    """
    List available audio devices.

    Args:
        device_type: Filter by device type (INPUT, OUTPUT, or None for all)

    Returns:
        List of AudioDevice objects
    """
    if not SOUNDDEVICE_AVAILABLE:
        # Return mock devices
        return _get_mock_devices(device_type)

    devices = []

    try:
        device_info = sd.query_devices()
        host_apis = sd.query_hostapis()
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]

        for i, info in enumerate(device_info):
            # Determine device type
            has_input = info['max_input_channels'] > 0
            has_output = info['max_output_channels'] > 0

            if has_input and has_output:
                dtype = DeviceType.DUPLEX
            elif has_input:
                dtype = DeviceType.INPUT
            elif has_output:
                dtype = DeviceType.OUTPUT
            else:
                continue  # Skip devices with no channels

            # Filter by type if specified
            if device_type is not None:
                if device_type == DeviceType.INPUT and not has_input:
                    continue
                if device_type == DeviceType.OUTPUT and not has_output:
                    continue

            # Get host API name
            host_api_idx = info.get('hostapi', 0)
            host_api_name = ""
            if host_api_idx < len(host_apis):
                host_api_name = host_apis[host_api_idx].get('name', '')

            device = AudioDevice(
                id=i,
                name=info['name'],
                device_type=dtype,
                max_input_channels=info['max_input_channels'],
                max_output_channels=info['max_output_channels'],
                default_sample_rate=info['default_samplerate'],
                is_default=(i == default_input or i == default_output),
                is_available=True,
                host_api=host_api_name,
                host_api_index=host_api_idx,
            )
            devices.append(device)

    except Exception as e:
        # Log the error and fall back to mock devices
        logger.warning("Failed to query audio devices, using mock devices: %s", e)
        return _get_mock_devices(device_type)

    return devices


def _get_mock_devices(
    device_type: Optional[DeviceType] = None,
) -> List[AudioDevice]:
    """Get mock devices for testing."""
    mock_devices = [
        AudioDevice(
            id=0,
            name="Default Microphone",
            device_type=DeviceType.INPUT,
            max_input_channels=2,
            max_output_channels=0,
            default_sample_rate=44100.0,
            is_default=True,
            host_api="Mock",
        ),
        AudioDevice(
            id=1,
            name="Default Speakers",
            device_type=DeviceType.OUTPUT,
            max_input_channels=0,
            max_output_channels=2,
            default_sample_rate=44100.0,
            is_default=True,
            host_api="Mock",
        ),
        AudioDevice(
            id=2,
            name="USB Headset",
            device_type=DeviceType.DUPLEX,
            max_input_channels=1,
            max_output_channels=2,
            default_sample_rate=48000.0,
            is_default=False,
            host_api="Mock",
        ),
    ]

    if device_type is None:
        return mock_devices

    return [d for d in mock_devices if d.device_type == device_type or
            (device_type == DeviceType.INPUT and d.max_input_channels > 0) or
            (device_type == DeviceType.OUTPUT and d.max_output_channels > 0)]


def get_default_input_device() -> Optional[AudioDevice]:
    """Get the default input (microphone) device."""
    devices = list_audio_devices(DeviceType.INPUT)
    for device in devices:
        if device.is_default:
            return device
    return devices[0] if devices else None


def get_default_output_device() -> Optional[AudioDevice]:
    """Get the default output (speakers) device."""
    devices = list_audio_devices(DeviceType.OUTPUT)
    for device in devices:
        if device.is_default:
            return device
    return devices[0] if devices else None


class AudioDeviceManager:
    """
    Manages audio device selection and streaming.

    Provides a unified interface for capturing from input devices
    and playing to output devices.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        chunk_size: int = 480,  # 20ms at 24kHz
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

        self._input_device: Optional[AudioDevice] = None
        self._output_device: Optional[AudioDevice] = None

        self._input_stream = None
        self._output_stream = None

        self._is_capturing = False
        self._is_playing = False

        self._lock = threading.Lock()

        # Callbacks
        self._on_input_chunk: Optional[Callable[[np.ndarray], None]] = None

    def set_input_device(
        self,
        device: Optional[Union[int, str, AudioDevice]] = None,
    ) -> AudioDevice:
        """
        Set the input device.

        Args:
            device: Device ID, name, AudioDevice, or None for default

        Returns:
            Selected AudioDevice
        """
        self._input_device = self._resolve_device(device, DeviceType.INPUT)
        return self._input_device

    def set_output_device(
        self,
        device: Optional[Union[int, str, AudioDevice]] = None,
    ) -> AudioDevice:
        """
        Set the output device.

        Args:
            device: Device ID, name, AudioDevice, or None for default

        Returns:
            Selected AudioDevice
        """
        self._output_device = self._resolve_device(device, DeviceType.OUTPUT)
        return self._output_device

    def _resolve_device(
        self,
        device: Optional[Union[int, str, AudioDevice]],
        device_type: DeviceType,
    ) -> AudioDevice:
        """
        Resolve device specification to AudioDevice.

        Args:
            device: Device ID, name, AudioDevice, or None for default
            device_type: Type of device to resolve

        Returns:
            AudioDevice matching the specification

        Raises:
            ValueError: If device not found or no default available
            TypeError: If device is an invalid type
        """
        if device is None:
            if device_type == DeviceType.INPUT:
                resolved = get_default_input_device()
            else:
                resolved = get_default_output_device()

            if resolved is None:
                raise ValueError(
                    f"No default {device_type.name.lower()} device available. "
                    "Check that audio devices are connected and drivers are installed."
                )
            return resolved

        if isinstance(device, AudioDevice):
            return device

        devices = list_audio_devices(device_type)

        if not devices:
            raise ValueError(
                f"No {device_type.name.lower()} devices available. "
                "Check that audio devices are connected and drivers are installed."
            )

        if isinstance(device, int):
            for d in devices:
                if d.id == device:
                    return d
            available_ids = [d.id for d in devices]
            raise ValueError(
                f"Device ID {device} not found. Available IDs: {available_ids}"
            )

        if isinstance(device, str):
            # Match by name (case-insensitive partial match)
            device_lower = device.lower()
            for d in devices:
                if device_lower in d.name.lower():
                    return d
            available_names = [d.name for d in devices]
            raise ValueError(
                f"Device '{device}' not found. Available devices: {available_names}"
            )

        raise TypeError(
            f"Invalid device type: {type(device).__name__}. "
            "Expected int (device ID), str (device name), AudioDevice, or None."
        )

    def start_capture(
        self,
        callback: Callable[[np.ndarray], None],
    ) -> None:
        """
        Start capturing audio from input device.

        Args:
            callback: Function called with each audio chunk
        """
        if not SOUNDDEVICE_AVAILABLE:
            self._is_capturing = True
            self._on_input_chunk = callback
            return

        if self._input_device is None:
            self._input_device = get_default_input_device()

        self._on_input_chunk = callback

        def audio_callback(indata, frames, time_info, status):
            if status:
                pass  # Handle errors if needed
            if self._on_input_chunk:
                # Convert to mono float32
                audio = indata[:, 0].astype(np.float32)
                self._on_input_chunk(audio)

        try:
            self._input_stream = sd.InputStream(
                device=self._input_device.id,
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                callback=audio_callback,
            )
            self._input_stream.start()
            self._is_capturing = True
        except Exception as e:
            self._is_capturing = False
            raise RuntimeError(f"Failed to start capture: {e}")

    def stop_capture(self) -> None:
        """Stop audio capture."""
        self._is_capturing = False
        if self._input_stream is not None:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception as e:
                logger.debug("Error stopping input stream (may already be stopped): %s", e)
            finally:
                self._input_stream = None

    def start_playback(self) -> None:
        """Start playback stream."""
        if not SOUNDDEVICE_AVAILABLE:
            self._is_playing = True
            return

        if self._output_device is None:
            self._output_device = get_default_output_device()

        try:
            self._output_stream = sd.OutputStream(
                device=self._output_device.id,
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
            )
            self._output_stream.start()
            self._is_playing = True
        except Exception as e:
            self._is_playing = False
            raise RuntimeError(f"Failed to start playback: {e}")

    def play_chunk(self, audio: np.ndarray) -> None:
        """
        Play an audio chunk.

        Args:
            audio: Audio data as float32 array
        """
        if not self._is_playing:
            return

        if not SOUNDDEVICE_AVAILABLE:
            return

        if self._output_stream is not None:
            try:
                # Ensure correct shape
                if audio.ndim == 1:
                    audio = audio.reshape(-1, 1)
                self._output_stream.write(audio.astype(np.float32))
            except Exception as e:
                logger.debug("Error writing audio chunk: %s", e)

    def stop_playback(self) -> None:
        """Stop audio playback."""
        self._is_playing = False
        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception as e:
                logger.debug("Error stopping output stream (may already be stopped): %s", e)
            finally:
                self._output_stream = None

    def stop_all(self) -> None:
        """Stop all audio streams."""
        self.stop_capture()
        self.stop_playback()

    @property
    def is_capturing(self) -> bool:
        """Whether capture is active."""
        return self._is_capturing

    @property
    def is_playing(self) -> bool:
        """Whether playback is active."""
        return self._is_playing

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop_all()


# Type alias for Union
from typing import Union
