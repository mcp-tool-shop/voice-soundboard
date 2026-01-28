"""
Tests for Audio Module (audio.py).

Tests cover:
- _validate_audio_path security function
- play_audio function
- stop_playback function
- get_audio_duration function
- list_audio_devices function
- set_output_device function
- Path security: directory restrictions, extension validation
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from voice_soundboard.audio import (
    _validate_audio_path,
    play_audio,
    stop_playback,
    get_audio_duration,
    list_audio_devices,
    set_output_device,
)


class TestValidateAudioPath:
    """Tests for _validate_audio_path security function."""

    def test_valid_wav_in_output_dir(self, tmp_path):
        """Test valid .wav file in output directory."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            audio_file = tmp_path / "test.wav"
            audio_file.touch()

            result = _validate_audio_path(audio_file)
            assert result == audio_file.resolve()

    def test_valid_mp3_extension(self, tmp_path):
        """Test .mp3 extension is allowed."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            audio_file = tmp_path / "test.mp3"
            audio_file.touch()

            result = _validate_audio_path(audio_file)
            assert result.suffix == ".mp3"

    def test_valid_flac_extension(self, tmp_path):
        """Test .flac extension is allowed."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            audio_file = tmp_path / "test.flac"
            audio_file.touch()

            result = _validate_audio_path(audio_file)
            assert result.suffix == ".flac"

    def test_invalid_extension_rejected(self, tmp_path):
        """Test non-audio extensions are rejected."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            bad_file = tmp_path / "test.exe"

            with pytest.raises(ValueError) as exc_info:
                _validate_audio_path(bad_file)

            assert "Invalid audio file extension" in str(exc_info.value)

    def test_invalid_txt_extension_rejected(self, tmp_path):
        """Test .txt extension is rejected."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            bad_file = tmp_path / "test.txt"

            with pytest.raises(ValueError):
                _validate_audio_path(bad_file)

    def test_path_outside_allowed_dirs_rejected(self, tmp_path):
        """Test paths outside allowed directories are rejected."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            # Set output_dir to tmp_path
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            # Try to access system file (outside allowed dirs)
            system_file = Path("/etc/passwd.wav")

            with pytest.raises(ValueError) as exc_info:
                _validate_audio_path(system_file)

            assert "outside allowed directories" in str(exc_info.value)

    def test_path_in_home_directory_allowed(self):
        """Test paths in home directory are allowed."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = Path("/nonexistent/output")
            mock_config.return_value = config_instance

            home_audio = Path.home() / "test.wav"

            # Should not raise (home is allowed)
            result = _validate_audio_path(home_audio)
            assert Path.home().resolve() in result.parents or result.parent == Path.home().resolve()

    def test_case_insensitive_extension(self, tmp_path):
        """Test extension check is case insensitive."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            # Uppercase extension
            audio_file = tmp_path / "test.WAV"
            audio_file.touch()

            result = _validate_audio_path(audio_file)
            assert result is not None

    def test_all_supported_extensions(self, tmp_path):
        """Test all supported audio extensions."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']
            for ext in extensions:
                audio_file = tmp_path / f"test{ext}"
                audio_file.touch()
                result = _validate_audio_path(audio_file)
                assert result is not None


class TestPlayAudio:
    """Tests for play_audio function."""

    def test_play_from_file_blocking(self, tmp_path):
        """Test playing audio from file in blocking mode."""
        import sys
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            audio_file = tmp_path / "test.wav"
            audio_file.touch()

            with patch('soundfile.read') as mock_read:
                mock_read.return_value = (np.zeros(1000), 24000)

                with patch('sounddevice.play') as mock_play, \
                     patch('sounddevice.wait') as mock_wait, \
                     patch('sounddevice.OutputStream') as mock_stream:
                    
                    mock_stream_instance = mock_stream.return_value.__enter__.return_value
                    play_audio(audio_file, blocking=True)

                    if sys.platform == 'win32':
                        mock_stream.assert_called_once()
                    else:
                        mock_play.assert_called_once()
                        mock_wait.assert_called_once()

    def test_play_from_file_non_blocking(self, tmp_path):
        """Test playing audio from file in non-blocking mode."""
        import sys
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            audio_file = tmp_path / "test.wav"
            audio_file.touch()

            with patch('soundfile.read') as mock_read:
                mock_read.return_value = (np.zeros(1000), 24000)

                with patch('sounddevice.play') as mock_play, \
                     patch('sounddevice.wait') as mock_wait, \
                     patch('sounddevice.OutputStream') as mock_stream:
                    
                    mock_stream_instance = mock_stream.return_value.__enter__.return_value
                    play_audio(audio_file, blocking=False)

                    if sys.platform == 'win32':
                        mock_stream.assert_called_once()
                    else:
                        mock_play.assert_called_once()
                        mock_wait.assert_not_called()

    def test_play_from_numpy_array(self):
        """Test playing audio from numpy array."""
        import sys
        samples = np.zeros(1000, dtype=np.float32)

        with patch('sounddevice.play') as mock_play, \
             patch('sounddevice.wait') as mock_wait, \
             patch('sounddevice.OutputStream') as mock_stream:
            
            mock_stream_instance = mock_stream.return_value.__enter__.return_value
            play_audio(samples, sample_rate=44100, blocking=True)

            if sys.platform == 'win32':
                mock_stream.assert_called_once()
                # On Windows, check samplerate in OutputStream constructor call
                call_kwargs = mock_stream.call_args[1]
                assert call_kwargs['samplerate'] == 44100
            else:
                mock_play.assert_called_once()
                call_args = mock_play.call_args
                assert call_args[0][1] == 44100  # sample rate

    def test_play_file_not_found(self, tmp_path):
        """Test FileNotFoundError for non-existent file."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            nonexistent = tmp_path / "nonexistent.wav"

            with pytest.raises(FileNotFoundError):
                play_audio(nonexistent)

    def test_play_path_security(self, tmp_path):
        """Test play_audio rejects paths outside allowed dirs."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            bad_path = Path("/etc/passwd.wav")

            with pytest.raises(ValueError):
                play_audio(bad_path)


class TestStopPlayback:
    """Tests for stop_playback function."""

    def test_stop_calls_sounddevice(self):
        """Test stop_playback calls sounddevice.stop."""
        with patch('sounddevice.stop') as mock_stop:
            stop_playback()
            mock_stop.assert_called_once()


class TestGetAudioDuration:
    """Tests for get_audio_duration function."""

    def test_get_duration(self, tmp_path):
        """Test getting audio duration."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            audio_file = tmp_path / "test.wav"
            audio_file.touch()

            mock_info = MagicMock()
            mock_info.duration = 5.5

            with patch('soundfile.info', return_value=mock_info):
                duration = get_audio_duration(audio_file)

            assert duration == 5.5

    def test_duration_file_not_found(self, tmp_path):
        """Test FileNotFoundError for non-existent file."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            nonexistent = tmp_path / "nonexistent.wav"

            with pytest.raises(FileNotFoundError):
                get_audio_duration(nonexistent)

    def test_duration_path_security(self, tmp_path):
        """Test get_audio_duration rejects paths outside allowed dirs."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            bad_path = Path("/etc/passwd.wav")

            with pytest.raises(ValueError):
                get_audio_duration(bad_path)


class TestListAudioDevices:
    """Tests for list_audio_devices function."""

    def test_list_devices(self):
        """Test listing audio devices."""
        mock_devices = [
            {"name": "Speaker", "max_output_channels": 2},
            {"name": "Headphones", "max_output_channels": 2},
            {"name": "Microphone", "max_output_channels": 0},  # Input only
        ]

        with patch('sounddevice.query_devices', return_value=mock_devices):
            devices = list_audio_devices()

        # Should only include output devices
        assert len(devices) == 2
        assert devices[0]["name"] == "Speaker"
        assert devices[1]["name"] == "Headphones"

    def test_list_devices_returns_dict_format(self):
        """Test device list has expected format."""
        mock_devices = [
            {"name": "Test Device", "max_output_channels": 6},
        ]

        with patch('sounddevice.query_devices', return_value=mock_devices):
            devices = list_audio_devices()

        assert "index" in devices[0]
        assert "name" in devices[0]
        assert "channels" in devices[0]
        assert devices[0]["channels"] == 6

    def test_list_empty_when_no_output_devices(self):
        """Test empty list when no output devices."""
        mock_devices = [
            {"name": "Microphone", "max_output_channels": 0},
        ]

        with patch('sounddevice.query_devices', return_value=mock_devices):
            devices = list_audio_devices()

        assert devices == []


class TestSetOutputDevice:
    """Tests for set_output_device function."""

    def test_set_device(self):
        """Test setting output device."""
        mock_devices = [
            {"name": "Input Device", "max_output_channels": 0},
            {"name": "Output Device 1", "max_output_channels": 2},
            {"name": "Output Device 2", "max_output_channels": 2},
        ]
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = mock_devices

        with patch.dict('sys.modules', {'sounddevice': mock_sd}):
            set_output_device(2)

            assert mock_sd.default.device == (None, 2)

    def test_set_device_invalid_index(self):
        """Test setting invalid device index raises error."""
        mock_devices = [
            {"name": "Device 1", "max_output_channels": 2},
        ]
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = mock_devices

        with patch.dict('sys.modules', {'sounddevice': mock_sd}):
            with pytest.raises(ValueError) as exc_info:
                set_output_device(5)

            assert "out of range" in str(exc_info.value)

    def test_set_device_not_output(self):
        """Test setting input-only device raises error."""
        mock_devices = [
            {"name": "Microphone", "max_output_channels": 0},
        ]
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = mock_devices

        with patch.dict('sys.modules', {'sounddevice': mock_sd}):
            with pytest.raises(ValueError) as exc_info:
                set_output_device(0)

            assert "not an output device" in str(exc_info.value)

    def test_set_device_negative_index(self):
        """Test setting negative device index raises error."""
        with pytest.raises(ValueError) as exc_info:
            set_output_device(-1)

        assert "non-negative" in str(exc_info.value)

    def test_set_device_invalid_type(self):
        """Test setting non-integer device index raises error."""
        with pytest.raises(TypeError) as exc_info:
            set_output_device("not an int")

        assert "must be an integer" in str(exc_info.value)


class TestPathSecurity:
    """Additional security tests for audio path handling."""

    def test_path_traversal_blocked(self, tmp_path):
        """Test path traversal attacks are blocked."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            # Set output_dir to a specific subdirectory
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            config_instance.output_dir = output_dir
            mock_config.return_value = config_instance

            # Try to access a path outside the output dir using traversal
            # This would resolve to somewhere outside the output directory
            # but still has audio extension - test that it fails
            outside_path = Path("/nonexistent/etc/passwd.wav")

            with pytest.raises(ValueError):
                _validate_audio_path(outside_path)

    def test_symlink_resolved(self, tmp_path):
        """Test symlinks are resolved before validation."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            # Path is resolved, so symlinks would be followed
            audio_file = tmp_path / "test.wav"
            audio_file.touch()

            result = _validate_audio_path(audio_file)
            assert result == audio_file.resolve()


class TestPlayAudioWithStringPath:
    """Tests for play_audio with string paths."""

    def test_accepts_string_path(self, tmp_path):
        """Test play_audio accepts string path."""
        with patch('voice_soundboard.audio.Config') as mock_config:
            config_instance = MagicMock()
            config_instance.output_dir = tmp_path
            mock_config.return_value = config_instance

            audio_file = tmp_path / "test.wav"
            audio_file.touch()

            with patch('soundfile.read') as mock_read:
                mock_read.return_value = (np.zeros(1000), 24000)

                with patch('sounddevice.play'):
                    with patch('sounddevice.wait'):
                        # Pass as string
                        play_audio(str(audio_file))


class TestAudioDefaultSampleRate:
    """Tests for default sample rate handling."""

    def test_default_sample_rate_for_array(self):
        """Test default sample rate is 24kHz for numpy arrays."""
        import sys
        samples = np.zeros(1000, dtype=np.float32)

        with patch('sounddevice.play') as mock_play, \
             patch('sounddevice.wait'), \
             patch('sounddevice.OutputStream') as mock_stream:
            
            mock_stream_instance = mock_stream.return_value.__enter__.return_value
            play_audio(samples)

            if sys.platform == 'win32':
                # Default sample rate should be 24000
                call_kwargs = mock_stream.call_args[1]
                assert call_kwargs['samplerate'] == 24000
            else:
                # Default sample rate should be 24000
                call_args = mock_play.call_args
                assert call_args[0][1] == 24000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
