"""
Real-Time Voice Conversion Module.

Provides low-latency voice conversion for live audio streams,
enabling real-time voice changing and voice cloning applications.

Key Features:
- Sub-100ms latency voice conversion
- Multiple latency/quality trade-offs
- Streaming pipeline with circular buffers
- Audio device selection (input/output)
- Integration with voice cloning library

Example:
    from voice_soundboard.conversion import VoiceConverter

    converter = VoiceConverter()
    converter.start(
        source="microphone",
        target_voice="my_clone",
        output="speakers"
    )

    # Stop conversion
    converter.stop()
"""

from voice_soundboard.conversion.base import (
    VoiceConverter,
    MockVoiceConverter,
    ConversionConfig,
    ConversionResult,
    LatencyMode,
    ConversionState,
)
from voice_soundboard.conversion.streaming import (
    StreamingConverter,
    AudioBuffer,
    ConversionPipeline,
    PipelineStage,
)
from voice_soundboard.conversion.devices import (
    AudioDevice,
    DeviceType,
    list_audio_devices,
    get_default_input_device,
    get_default_output_device,
    AudioDeviceManager,
)
from voice_soundboard.conversion.realtime import (
    RealtimeConverter,
    RealtimeSession,
    start_realtime_conversion,
    ConversionCallback,
)

__all__ = [
    # Base
    "VoiceConverter",
    "MockVoiceConverter",
    "ConversionConfig",
    "ConversionResult",
    "LatencyMode",
    "ConversionState",
    # Streaming
    "StreamingConverter",
    "AudioBuffer",
    "ConversionPipeline",
    "PipelineStage",
    # Devices
    "AudioDevice",
    "DeviceType",
    "list_audio_devices",
    "get_default_input_device",
    "get_default_output_device",
    "AudioDeviceManager",
    # Realtime
    "RealtimeConverter",
    "RealtimeSession",
    "start_realtime_conversion",
    "ConversionCallback",
]
