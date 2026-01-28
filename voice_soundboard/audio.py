"""
Audio playback and file handling utilities.

SECURITY: This module restricts file access to the configured output directory
to prevent arbitrary file access through path manipulation.
"""

import logging
from pathlib import Path
from typing import Optional, Union, List, Dict
import numpy as np

from voice_soundboard.config import Config

logger = logging.getLogger(__name__)


def _validate_audio_path(path: Union[str, Path]) -> Path:
    """
    Validate that an audio path is within allowed directories.

    SECURITY: Prevents arbitrary file access by restricting paths to:
    - The configured output directory
    - Standard audio file extensions only

    Args:
        path: Path to validate

    Returns:
        Resolved, validated Path

    Raises:
        ValueError: If path is outside allowed directories or has invalid extension
    """
    path = Path(path).resolve()
    config = Config()

    # Get allowed directories (output dir and any parent temp dirs)
    allowed_dirs = [
        config.output_dir.resolve(),
        Path.home().resolve(),  # Allow home directory for user files
    ]

    # Check if path is within an allowed directory
    is_allowed = False
    for allowed_dir in allowed_dirs:
        try:
            path.relative_to(allowed_dir)
            is_allowed = True
            break
        except ValueError:
            continue

    if not is_allowed:
        raise ValueError(f"Access denied: {path} is outside allowed directories")

    # Validate file extension for audio files
    allowed_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'}
    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Invalid audio file extension: {path.suffix}")

    return path


def play_audio(
    source: Union[str, Path, np.ndarray],
    sample_rate: int = 24000,
    blocking: bool = True,
) -> None:
    """
    Play audio from file or numpy array.

    Args:
        source: Path to audio file, or numpy array of samples
        sample_rate: Sample rate (only used if source is numpy array)
        blocking: If True, wait for playback to complete

    Raises:
        ValueError: If file path is outside allowed directories
        FileNotFoundError: If audio file doesn't exist
        ImportError: If sounddevice is not installed
        RuntimeError: If audio playback fails
    """
    try:
        import sounddevice as sd
    except ImportError as e:
        raise ImportError(
            "sounddevice is required for audio playback. "
            "Install with: pip install sounddevice"
        ) from e

    if isinstance(source, (str, Path)):
        try:
            import soundfile as sf
        except ImportError as e:
            raise ImportError(
                "soundfile is required for audio file playback. "
                "Install with: pip install soundfile"
            ) from e

        # SECURITY: Validate path before accessing
        validated_path = _validate_audio_path(source)
        if not validated_path.exists():
            raise FileNotFoundError(f"Audio file not found: {validated_path}")

        try:
            data, sample_rate = sf.read(str(validated_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to read audio file {validated_path}: {e}"
            ) from e
    else:
        data = source

    try:
        # Use OutputStream directly to avoid CFFI callback cleanup bug in sounddevice 0.5.x on Windows
        # The bug causes "AttributeError: '_CallbackContext' object has no attribute 'out'"
        # when the finished_callback tries to access self.out after stream cleanup
        import sys
        if sys.platform == 'win32':
            # Suppress the CFFI error by using a stream without finished_callback
            with sd.OutputStream(samplerate=sample_rate, channels=data.shape[1] if data.ndim > 1 else 1) as stream:
                stream.write(data.astype(np.float32) if data.dtype != np.float32 else data)
        else:
            sd.play(data, sample_rate)
            if blocking:
                sd.wait()
    except Exception as e:
        logger.error("Audio playback failed: %s", e)
        raise RuntimeError(f"Failed to play audio: {e}") from e


def stop_playback() -> None:
    """
    Stop any currently playing audio.

    Raises:
        ImportError: If sounddevice is not installed
    """
    try:
        import sounddevice as sd
        sd.stop()
    except ImportError as e:
        raise ImportError(
            "sounddevice is required for audio control. "
            "Install with: pip install sounddevice"
        ) from e
    except Exception as e:
        logger.debug("Error stopping playback (may not be playing): %s", e)


def get_audio_duration(path: Union[str, Path]) -> float:
    """
    Get duration of an audio file in seconds.

    Args:
        path: Path to audio file

    Returns:
        Duration in seconds

    Raises:
        ValueError: If file path is outside allowed directories
        FileNotFoundError: If audio file doesn't exist
        ImportError: If soundfile is not installed
        RuntimeError: If file cannot be read
    """
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required for audio file info. "
            "Install with: pip install soundfile"
        ) from e

    # SECURITY: Validate path before accessing
    validated_path = _validate_audio_path(path)
    if not validated_path.exists():
        raise FileNotFoundError(f"Audio file not found: {validated_path}")

    try:
        info = sf.info(str(validated_path))
        return info.duration
    except Exception as e:
        raise RuntimeError(
            f"Failed to read audio file info for {validated_path}: {e}"
        ) from e


def list_audio_devices() -> List[Dict]:
    """
    List available audio output devices.

    Returns:
        List of device dictionaries with index, name, and channels

    Raises:
        ImportError: If sounddevice is not installed
        RuntimeError: If device query fails
    """
    try:
        import sounddevice as sd
    except ImportError as e:
        raise ImportError(
            "sounddevice is required for audio device listing. "
            "Install with: pip install sounddevice"
        ) from e

    try:
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "channels": d["max_output_channels"]}
            for i, d in enumerate(devices)
            if d["max_output_channels"] > 0
        ]
    except Exception as e:
        logger.error("Failed to query audio devices: %s", e)
        raise RuntimeError(f"Failed to list audio devices: {e}") from e


def set_output_device(device_index: int) -> None:
    """
    Set the default audio output device.

    Args:
        device_index: Index of the device to use (from list_audio_devices)

    Raises:
        ImportError: If sounddevice is not installed
        ValueError: If device_index is invalid
    """
    if not isinstance(device_index, int):
        raise TypeError(
            f"device_index must be an integer, got {type(device_index).__name__}"
        )

    if device_index < 0:
        raise ValueError(f"device_index must be non-negative, got {device_index}")

    try:
        import sounddevice as sd
    except ImportError as e:
        raise ImportError(
            "sounddevice is required for audio device selection. "
            "Install with: pip install sounddevice"
        ) from e

    try:
        # Verify the device exists
        devices = sd.query_devices()
        if device_index >= len(devices):
            raise ValueError(
                f"device_index {device_index} out of range. "
                f"Available devices: 0-{len(devices) - 1}"
            )

        if devices[device_index]["max_output_channels"] == 0:
            raise ValueError(
                f"Device {device_index} ({devices[device_index]['name']}) "
                "is not an output device"
            )

        sd.default.device = (None, device_index)
        logger.debug("Set output device to %d: %s", device_index, devices[device_index]["name"])
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to set output device: {e}") from e
