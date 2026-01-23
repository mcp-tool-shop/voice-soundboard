"""
Real-Time Voice Conversion.

Provides a high-level interface for real-time voice conversion
from microphone to speakers with minimal latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Union, Any
from pathlib import Path
import threading
import time

import numpy as np

from voice_soundboard.conversion.base import (
    VoiceConverter,
    MockVoiceConverter,
    ConversionConfig,
    ConversionResult,
    ConversionStats,
    LatencyMode,
    ConversionState,
)
from voice_soundboard.conversion.streaming import (
    StreamingConverter,
    AudioBuffer,
)
from voice_soundboard.conversion.devices import (
    AudioDevice,
    AudioDeviceManager,
    list_audio_devices,
    get_default_input_device,
    get_default_output_device,
    DeviceType,
)


# Type alias for callbacks
ConversionCallback = Callable[[np.ndarray, ConversionResult], None]


@dataclass
class RealtimeSession:
    """Information about a real-time conversion session."""

    # Session ID
    session_id: str

    # Devices
    input_device: Optional[AudioDevice] = None
    output_device: Optional[AudioDevice] = None

    # Voice
    target_voice: Optional[str] = None

    # Timing
    started_at: float = 0.0
    stopped_at: float = 0.0

    # Statistics
    stats: ConversionStats = field(default_factory=ConversionStats)

    @property
    def duration_seconds(self) -> float:
        """Session duration in seconds."""
        end_time = self.stopped_at if self.stopped_at > 0 else time.time()
        return end_time - self.started_at if self.started_at > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "input_device": self.input_device.name if self.input_device else None,
            "output_device": self.output_device.name if self.output_device else None,
            "target_voice": self.target_voice,
            "duration_seconds": self.duration_seconds,
            "stats": self.stats.to_dict(),
        }


class RealtimeConverter:
    """
    Real-time voice converter for live audio streams.

    Handles microphone input, voice conversion, and speaker output
    with minimal latency.

    Example:
        converter = RealtimeConverter()
        converter.start(
            source="microphone",
            target_voice="my_clone",
            output="speakers"
        )

        # Later...
        converter.stop()
    """

    def __init__(
        self,
        converter: Optional[VoiceConverter] = None,
        config: Optional[ConversionConfig] = None,
    ):
        self.config = config or ConversionConfig()

        # Use provided converter or create mock
        if converter is None:
            self._converter = MockVoiceConverter(self.config)
        else:
            self._converter = converter

        # Device manager
        self._device_manager = AudioDeviceManager(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            chunk_size=self.config.get_chunk_samples(),
        )

        # Streaming converter
        self._streaming = StreamingConverter(
            converter=self._converter,
            chunk_size=self.config.get_chunk_samples(),
            sample_rate=self.config.sample_rate,
        )

        # Session tracking
        self._session: Optional[RealtimeSession] = None
        self._session_counter = 0

        # State
        self._running = False
        self._lock = threading.Lock()

        # Callbacks
        self._on_converted: Optional[ConversionCallback] = None

    def start(
        self,
        source: Optional[Union[str, int, AudioDevice]] = None,
        target_voice: Optional[Union[str, Path, np.ndarray]] = None,
        output: Optional[Union[str, int, AudioDevice]] = None,
        on_converted: Optional[ConversionCallback] = None,
    ) -> RealtimeSession:
        """
        Start real-time voice conversion.

        Args:
            source: Input device ("microphone", device ID, or AudioDevice)
            target_voice: Target voice (ID, path, or embedding)
            output: Output device ("speakers", device ID, or AudioDevice)
            on_converted: Callback for each converted chunk

        Returns:
            RealtimeSession with session info
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Converter already running")

            # Resolve source
            if source is None or source == "microphone":
                input_device = get_default_input_device()
            else:
                input_device = self._device_manager._resolve_device(
                    source, DeviceType.INPUT
                )

            # Resolve output
            if output is None or output == "speakers":
                output_device = get_default_output_device()
            else:
                output_device = self._device_manager._resolve_device(
                    output, DeviceType.OUTPUT
                )

            # Set target voice
            if target_voice is not None:
                self._converter.set_target_voice(target_voice)

            # Set up devices
            self._device_manager.set_input_device(input_device)
            self._device_manager.set_output_device(output_device)

            # Store callback
            self._on_converted = on_converted

            # Create session
            self._session_counter += 1
            self._session = RealtimeSession(
                session_id=f"session_{self._session_counter}",
                input_device=input_device,
                output_device=output_device,
                target_voice=self._converter._target_voice_id,
                started_at=time.time(),
            )

            # Start streaming
            self._running = True
            self._streaming.start(on_output=self._handle_output)

            # Start device capture
            self._device_manager.start_capture(self._handle_input)
            self._device_manager.start_playback()

            return self._session

    def stop(self) -> Optional[RealtimeSession]:
        """
        Stop real-time voice conversion.

        Returns:
            Final RealtimeSession with statistics
        """
        with self._lock:
            if not self._running:
                return None

            self._running = False

            # Stop streaming
            self._streaming.stop()

            # Stop devices
            self._device_manager.stop_all()

            # Finalize session
            if self._session:
                self._session.stopped_at = time.time()
                self._session.stats = self._converter.stats

            return self._session

    def _handle_input(self, audio: np.ndarray) -> None:
        """Handle audio input from microphone."""
        if self._running:
            self._streaming.push(audio)

    def _handle_output(self, audio: np.ndarray) -> None:
        """Handle converted audio output."""
        if not self._running:
            return

        # Play through speakers
        self._device_manager.play_chunk(audio)

        # Call callback if set
        if self._on_converted is not None:
            result = ConversionResult(
                audio=audio,
                sample_rate=self.config.sample_rate,
                input_duration_ms=len(audio) / self.config.sample_rate * 1000,
                output_duration_ms=len(audio) / self.config.sample_rate * 1000,
                processing_time_ms=self._streaming.avg_latency_ms,
                latency_ms=self._streaming.avg_latency_ms,
                target_voice=self._converter._target_voice_id,
            )
            self._on_converted(audio, result)

    def set_target_voice(
        self,
        voice: Union[str, Path, np.ndarray],
    ) -> None:
        """
        Change target voice during conversion.

        Args:
            voice: New target voice
        """
        self._converter.set_target_voice(voice)
        if self._session:
            self._session.target_voice = self._converter._target_voice_id

    def set_latency_mode(self, mode: LatencyMode) -> None:
        """
        Change latency mode during conversion.

        Args:
            mode: New latency mode
        """
        self.config.latency_mode = mode

    @property
    def is_running(self) -> bool:
        """Whether converter is running."""
        return self._running

    @property
    def current_session(self) -> Optional[RealtimeSession]:
        """Current conversion session."""
        return self._session

    @property
    def stats(self) -> ConversionStats:
        """Current conversion statistics."""
        return self._converter.stats

    def get_latency(self) -> float:
        """Get current processing latency in ms."""
        return self._streaming.avg_latency_ms


def start_realtime_conversion(
    target_voice: Optional[Union[str, Path, np.ndarray]] = None,
    latency_mode: LatencyMode = LatencyMode.BALANCED,
    source: Optional[Union[str, int]] = None,
    output: Optional[Union[str, int]] = None,
    on_converted: Optional[ConversionCallback] = None,
) -> RealtimeConverter:
    """
    Convenience function to start real-time voice conversion.

    Args:
        target_voice: Target voice for conversion
        latency_mode: Latency/quality trade-off
        source: Input device
        output: Output device
        on_converted: Callback for converted audio

    Returns:
        Running RealtimeConverter instance

    Example:
        converter = start_realtime_conversion(
            target_voice="celebrity_clone",
            latency_mode=LatencyMode.LOW,
        )

        # Later...
        converter.stop()
    """
    config = ConversionConfig(latency_mode=latency_mode)
    converter = RealtimeConverter(config=config)
    converter.start(
        source=source,
        target_voice=target_voice,
        output=output,
        on_converted=on_converted,
    )
    return converter
