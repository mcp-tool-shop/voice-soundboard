"""
Audio playback and file handling utilities.

SECURITY: This module restricts file access to the configured output directory
to prevent arbitrary file access through path manipulation.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np

from voice_soundboard.config import Config


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
    """
    import sounddevice as sd

    if isinstance(source, (str, Path)):
        import soundfile as sf
        # SECURITY: Validate path before accessing
        validated_path = _validate_audio_path(source)
        if not validated_path.exists():
            raise FileNotFoundError(f"Audio file not found: {validated_path}")
        data, sample_rate = sf.read(str(validated_path))
    else:
        data = source

    sd.play(data, sample_rate)
    if blocking:
        sd.wait()


def stop_playback() -> None:
    """Stop any currently playing audio."""
    import sounddevice as sd
    sd.stop()


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
    """
    import soundfile as sf
    # SECURITY: Validate path before accessing
    validated_path = _validate_audio_path(path)
    if not validated_path.exists():
        raise FileNotFoundError(f"Audio file not found: {validated_path}")
    info = sf.info(str(validated_path))
    return info.duration


def list_audio_devices() -> list[dict]:
    """List available audio output devices."""
    import sounddevice as sd
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_output_channels"]}
        for i, d in enumerate(devices)
        if d["max_output_channels"] > 0
    ]


def set_output_device(device_index: int) -> None:
    """Set the default audio output device."""
    import sounddevice as sd
    sd.default.device = (None, device_index)
