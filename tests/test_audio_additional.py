"""
Additional tests for Audio module (audio.py).

Tests cover:
- Audio path validation
- play_audio with different inputs
- Audio device management
- Error handling
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from voice_soundboard.audio import (
    play_audio,
    stop_playback,
    get_audio_duration,
    list_audio_devices,
    set_output_device,
    _validate_audio_path,
)
from voice_soundboard.config import Config


class TestValidateAudioPath:
    """Tests for _validate_audio_path function."""

    def test_validate_allows_output_dir(self, tmp_path):
        """TEST-AUD-SEC01: _validate_audio_path allows output dir."""
        # Create a file in output directory
        with patch('voice_soundboard.audio.Config') as mock_config:
            config = Mock()
            config.output_dir = tmp_path
            mock_config.return_value = config

            audio_file = tmp_path / "test.wav"
            audio_file.touch()

            # Should not raise
            result = _validate_audio_path(audio_file)
            assert result.exists()

    def test_validate_allows_home_dir(self, tmp_path):
        """TEST-A13: _validate_audio_path allows home directory paths."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config = Mock()
            config.output_dir = tmp_path
            mock_config.return_value = config

            # Create a file in home directory
            home_audio = Path.home() / "test_audio_validate.wav"
            try:
                home_audio.touch()
                result = _validate_audio_path(home_audio)
                assert result is not None
            finally:
                if home_audio.exists():
                    home_audio.unlink()

    def test_validate_rejects_invalid_extension(self, tmp_path):
        """TEST-AUD-SEC02: _validate_audio_path rejects non-audio extensions."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config = Mock()
            config.output_dir = tmp_path
            mock_config.return_value = config

            # Try with .exe extension
            exe_file = tmp_path / "malicious.exe"
            exe_file.touch()

            with pytest.raises(ValueError) as exc_info:
                _validate_audio_path(exe_file)

            assert "Invalid audio file extension" in str(exc_info.value)

    def test_validate_rejects_system_paths(self, tmp_path):
        """TEST-AUD-SEC03: _validate_audio_path rejects system paths."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config = Mock()
            config.output_dir = tmp_path
            mock_config.return_value = config

            # Try with a system path
            system_path = Path("C:/Windows/System32/test.wav")

            with pytest.raises(ValueError) as exc_info:
                _validate_audio_path(system_path)

            assert "Access denied" in str(exc_info.value)

    def test_validate_allowed_extensions(self, tmp_path):
        """Test all allowed audio extensions."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config = Mock()
            config.output_dir = tmp_path
            mock_config.return_value = config

            allowed = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']

            for ext in allowed:
                audio_file = tmp_path / f"test{ext}"
                audio_file.touch()

                # Should not raise
                result = _validate_audio_path(audio_file)
                assert result.suffix.lower() == ext


class TestPlayAudioInput:
    """Tests for play_audio with different input types."""

    def test_play_audio_with_file_path(self, tmp_path):
        """TEST-A01: play_audio with file path."""
        with patch('sounddevice.play') as mock_play, \
             patch('sounddevice.wait') as mock_wait, \
             patch('soundfile.read') as mock_read, \
             patch.object(Config, '__init__', lambda self: None):

            audio_file = tmp_path / "test.wav"
            audio_file.touch()

            # Setup mocks
            mock_read.return_value = (np.zeros(24000), 24000)

            # Need to also mock _validate_audio_path
            with patch('voice_soundboard.audio._validate_audio_path', return_value=audio_file):
                play_audio(audio_file, blocking=False)

            mock_play.assert_called_once()

    def test_play_audio_with_numpy_array(self):
        """TEST-A02: play_audio with numpy array input."""
        with patch('sounddevice.play') as mock_play, \
             patch('sounddevice.wait') as mock_wait:

            samples = np.zeros(24000, dtype=np.float32)

            play_audio(samples, sample_rate=24000, blocking=False)

            mock_play.assert_called_once()

    def test_play_audio_blocking_waits(self):
        """Test play_audio with blocking=True waits for playback."""
        with patch('sounddevice.play') as mock_play, \
             patch('sounddevice.wait') as mock_wait:

            samples = np.zeros(24000, dtype=np.float32)

            play_audio(samples, sample_rate=24000, blocking=True)

            mock_play.assert_called_once()
            mock_wait.assert_called_once()

    def test_play_audio_non_blocking(self):
        """TEST-A12 partial: play_audio with blocking=False returns immediately."""
        with patch('sounddevice.play') as mock_play, \
             patch('sounddevice.wait') as mock_wait:

            samples = np.zeros(24000, dtype=np.float32)

            play_audio(samples, sample_rate=24000, blocking=False)

            mock_play.assert_called_once()
            mock_wait.assert_not_called()

    def test_play_audio_file_not_found(self, tmp_path):
        """TEST-A06: play_audio with nonexistent file raises error."""
        nonexistent = tmp_path / "nonexistent.wav"

        with patch('voice_soundboard.audio._validate_audio_path', return_value=nonexistent):
            with pytest.raises(FileNotFoundError):
                play_audio(nonexistent)


class TestPlayAudioEmptyArray:
    """Tests for edge cases with empty arrays."""

    def test_play_audio_empty_array(self):
        """TEST-A08: play_audio with empty numpy array doesn't crash."""
        with patch('sounddevice.play') as mock_play, \
             patch('sounddevice.wait'):

            samples = np.array([], dtype=np.float32)

            # Should not raise
            play_audio(samples, sample_rate=24000, blocking=False)

            mock_play.assert_called_once()


class TestStopPlayback:
    """Tests for stop_playback function."""

    def test_stop_playback(self):
        """TEST-A03: stop_playback stops playing audio."""
        with patch('sounddevice.stop') as mock_stop:
            stop_playback()

            mock_stop.assert_called_once()


class TestGetAudioDuration:
    """Tests for get_audio_duration function."""

    def test_get_audio_duration_valid(self, tmp_path):
        """TEST-A04: get_audio_duration returns correct duration."""
        audio_file = tmp_path / "test.wav"
        audio_file.touch()

        mock_info = Mock()
        mock_info.duration = 1.5

        with patch('voice_soundboard.audio._validate_audio_path', return_value=audio_file), \
             patch('soundfile.info', return_value=mock_info):

            duration = get_audio_duration(audio_file)

            assert duration == 1.5

    def test_get_audio_duration_file_not_found(self, tmp_path):
        """TEST-A09: get_audio_duration with nonexistent file."""
        nonexistent = tmp_path / "nonexistent.wav"

        with patch('voice_soundboard.audio._validate_audio_path', return_value=nonexistent):
            with pytest.raises(FileNotFoundError):
                get_audio_duration(nonexistent)


class TestListAudioDevices:
    """Tests for list_audio_devices function."""

    def test_list_audio_devices(self):
        """TEST-A05: list_audio_devices returns device list."""
        with patch('sounddevice.query_devices') as mock_query:
            mock_query.return_value = [
                {"name": "Speakers", "max_output_channels": 2},
                {"name": "Headphones", "max_output_channels": 2},
                {"name": "Microphone", "max_output_channels": 0},  # Input device
            ]

            devices = list_audio_devices()

            # Should only return output devices
            assert len(devices) == 2
            assert all("name" in d for d in devices)
            assert all("channels" in d for d in devices)

    def test_list_audio_devices_empty(self):
        """Test list_audio_devices with no devices."""
        with patch('sounddevice.query_devices') as mock_query:
            mock_query.return_value = []

            devices = list_audio_devices()

            assert devices == []


class TestSetOutputDevice:
    """Tests for set_output_device function."""

    def test_set_output_device(self):
        """TEST-A11: set_output_device with valid index."""
        mock_devices = [
            {"name": "Output Device", "max_output_channels": 2},
        ]
        with patch('sounddevice.query_devices', return_value=mock_devices), \
             patch('sounddevice.default', create=True) as mock_default:

            set_output_device(0)

            # Should set default output device
            assert mock_default.device == (None, 0)

    def test_set_output_device_validation(self):
        """Test that device validation now happens immediately."""
        mock_devices = [
            {"name": "Output Device", "max_output_channels": 2},
        ]
        with patch('sounddevice.query_devices', return_value=mock_devices):
            # Invalid index now raises ValueError immediately
            with pytest.raises(ValueError) as exc_info:
                set_output_device(999)

            assert "out of range" in str(exc_info.value)


class TestAudioPathSecurity:
    """Tests for audio path security."""

    def test_path_traversal_blocked(self, tmp_path):
        """Test that path traversal is blocked."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config = Mock()
            config.output_dir = tmp_path
            mock_config.return_value = config

            # Path traversal attempt - use an absolute path that's definitely outside
            import os
            if os.name == 'nt':  # Windows
                malicious = Path("C:/Windows/System32/test.wav")
            else:  # Unix
                malicious = Path("/etc/passwd.wav")

            with pytest.raises(ValueError):
                _validate_audio_path(malicious)


class TestAudioDurationValidation:
    """Tests for get_audio_duration path validation."""

    def test_get_duration_validates_path(self, tmp_path):
        """TEST-AUD-SEC04: get_audio_duration validates path first."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config = Mock()
            config.output_dir = tmp_path
            mock_config.return_value = config

            # Try to get duration of system file
            system_path = Path("C:/Windows/System32/config.wav")

            with pytest.raises(ValueError) as exc_info:
                get_audio_duration(system_path)

            assert "Access denied" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
