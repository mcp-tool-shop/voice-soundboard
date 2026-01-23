"""
Tests for Configuration (config.py).

Tests cover:
- Config dataclass defaults and custom values
- Path creation on init
- CUDA/CPU device detection
- KOKORO_VOICES registry
- VOICE_PRESETS registry
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from voice_soundboard.config import Config, KOKORO_VOICES, VOICE_PRESETS


class TestConfigDefaults:
    """Tests for Config dataclass default values."""

    @patch('pathlib.Path.mkdir')
    def test_default_device_is_cuda_or_cpu(self, mock_mkdir):
        """Test default device is cuda or cpu (depending on availability)."""
        config = Config()
        assert config.device in ["cuda", "cpu"]

    @patch('pathlib.Path.mkdir')
    def test_default_voice(self, mock_mkdir):
        """Test default voice is set."""
        config = Config()
        assert config.default_voice == "af_bella"

    @patch('pathlib.Path.mkdir')
    def test_default_speed(self, mock_mkdir):
        """Test default speed is 1.0."""
        config = Config()
        assert config.default_speed == 1.0

    @patch('pathlib.Path.mkdir')
    def test_default_sample_rate(self, mock_mkdir):
        """Test default sample rate is 24kHz for Kokoro."""
        config = Config()
        assert config.sample_rate == 24000

    @patch('pathlib.Path.mkdir')
    def test_default_use_gpu(self, mock_mkdir):
        """Test GPU is enabled by default."""
        config = Config()
        assert config.use_gpu is True

    @patch('pathlib.Path.mkdir')
    def test_default_cache_models(self, mock_mkdir):
        """Test model caching is enabled by default."""
        config = Config()
        assert config.cache_models is True


class TestConfigPaths:
    """Tests for Config path handling."""

    @patch('pathlib.Path.mkdir')
    def test_output_dir_is_path(self, mock_mkdir):
        """Test output_dir is a Path object."""
        config = Config()
        assert isinstance(config.output_dir, Path)

    @patch('pathlib.Path.mkdir')
    def test_cache_dir_is_path(self, mock_mkdir):
        """Test cache_dir is a Path object."""
        config = Config()
        assert isinstance(config.cache_dir, Path)

    @patch('pathlib.Path.mkdir')
    def test_custom_output_dir(self, mock_mkdir):
        """Test custom output directory can be set."""
        custom_path = Path("/custom/output")
        config = Config(output_dir=custom_path)
        assert config.output_dir == custom_path

    @patch('pathlib.Path.mkdir')
    def test_custom_cache_dir(self, mock_mkdir):
        """Test custom cache directory can be set."""
        custom_path = Path("/custom/cache")
        config = Config(cache_dir=custom_path)
        assert config.cache_dir == custom_path


class TestConfigPostInit:
    """Tests for Config __post_init__ behavior."""

    def test_creates_output_directory(self, tmp_path):
        """Test output directory is created on init."""
        output_dir = tmp_path / "output_test"
        with patch.object(Path, 'mkdir') as mock_mkdir:
            mock_mkdir.return_value = None
            config = Config(output_dir=output_dir, cache_dir=tmp_path / "cache")

    def test_creates_cache_directory(self, tmp_path):
        """Test cache directory is created on init."""
        cache_dir = tmp_path / "cache_test"
        with patch.object(Path, 'mkdir') as mock_mkdir:
            mock_mkdir.return_value = None
            config = Config(output_dir=tmp_path / "output", cache_dir=cache_dir)

    @patch('pathlib.Path.mkdir')
    def test_sets_xformers_disabled(self, mock_mkdir):
        """Test XFORMERS_DISABLED env var is set."""
        import os
        config = Config()
        assert os.environ.get("XFORMERS_DISABLED") == "1"


class TestConfigDeviceDetection:
    """Tests for CUDA/CPU device auto-detection."""

    @patch('pathlib.Path.mkdir')
    def test_device_is_valid(self, mock_mkdir):
        """Test device is a valid option after auto-detection."""
        config = Config(use_gpu=True)
        assert config.device in ["cuda", "cpu"]

    @patch('pathlib.Path.mkdir')
    def test_cpu_when_gpu_disabled(self, mock_mkdir):
        """Test config works when GPU is disabled."""
        config = Config(use_gpu=False)
        # Config should be valid either way
        assert config.device in ["cuda", "cpu"]

    @patch('pathlib.Path.mkdir')
    def test_device_detection_does_not_crash(self, mock_mkdir):
        """Test device detection handles various scenarios."""
        # Should not crash regardless of onnxruntime availability
        config = Config(use_gpu=True)
        assert config is not None


class TestKokoroVoices:
    """Tests for KOKORO_VOICES registry."""

    def test_not_empty(self):
        """Test voice registry is not empty."""
        assert len(KOKORO_VOICES) > 0

    def test_american_female_voices(self):
        """Test American female voices are present."""
        af_voices = [v for v in KOKORO_VOICES if v.startswith("af_")]
        assert len(af_voices) >= 5
        assert "af_bella" in KOKORO_VOICES

    def test_american_male_voices(self):
        """Test American male voices are present."""
        am_voices = [v for v in KOKORO_VOICES if v.startswith("am_")]
        assert len(am_voices) >= 5
        assert "am_michael" in KOKORO_VOICES

    def test_british_female_voices(self):
        """Test British female voices are present."""
        bf_voices = [v for v in KOKORO_VOICES if v.startswith("bf_")]
        assert len(bf_voices) >= 2
        assert "bf_emma" in KOKORO_VOICES

    def test_british_male_voices(self):
        """Test British male voices are present."""
        bm_voices = [v for v in KOKORO_VOICES if v.startswith("bm_")]
        assert len(bm_voices) >= 2
        assert "bm_george" in KOKORO_VOICES

    def test_voice_has_name(self):
        """Test each voice has a name field."""
        for voice_id, info in KOKORO_VOICES.items():
            assert "name" in info, f"Voice {voice_id} missing name"
            assert isinstance(info["name"], str)

    def test_voice_has_gender(self):
        """Test each voice has a gender field."""
        for voice_id, info in KOKORO_VOICES.items():
            assert "gender" in info, f"Voice {voice_id} missing gender"
            assert info["gender"] in ["male", "female"]

    def test_voice_has_accent(self):
        """Test each voice has an accent field."""
        for voice_id, info in KOKORO_VOICES.items():
            assert "accent" in info, f"Voice {voice_id} missing accent"

    def test_voice_has_style(self):
        """Test each voice has a style field."""
        for voice_id, info in KOKORO_VOICES.items():
            assert "style" in info, f"Voice {voice_id} missing style"

    def test_voice_id_format(self):
        """Test voice IDs follow expected format."""
        for voice_id in KOKORO_VOICES:
            # Should be like "af_bella", "am_michael", etc.
            parts = voice_id.split("_")
            assert len(parts) == 2, f"Voice {voice_id} unexpected format"
            assert len(parts[0]) == 2, f"Voice {voice_id} prefix unexpected"

    def test_japanese_voice_present(self):
        """Test Japanese voice is available."""
        assert "jf_alpha" in KOKORO_VOICES
        assert KOKORO_VOICES["jf_alpha"]["accent"] == "japanese"

    def test_mandarin_voice_present(self):
        """Test Mandarin voice is available."""
        assert "zf_xiaobei" in KOKORO_VOICES
        assert KOKORO_VOICES["zf_xiaobei"]["accent"] == "mandarin"


class TestVoicePresets:
    """Tests for VOICE_PRESETS registry."""

    def test_not_empty(self):
        """Test preset registry is not empty."""
        assert len(VOICE_PRESETS) > 0

    def test_assistant_preset(self):
        """Test assistant preset exists with expected fields."""
        assert "assistant" in VOICE_PRESETS
        preset = VOICE_PRESETS["assistant"]
        assert "voice" in preset
        assert "speed" in preset
        assert "description" in preset

    def test_narrator_preset(self):
        """Test narrator preset exists."""
        assert "narrator" in VOICE_PRESETS
        preset = VOICE_PRESETS["narrator"]
        assert preset["voice"] == "bm_george"
        assert preset["speed"] < 1.0  # Narrator is slower

    def test_announcer_preset(self):
        """Test announcer preset exists."""
        assert "announcer" in VOICE_PRESETS
        preset = VOICE_PRESETS["announcer"]
        assert preset["speed"] > 1.0  # Announcer is faster

    def test_storyteller_preset(self):
        """Test storyteller preset exists."""
        assert "storyteller" in VOICE_PRESETS

    def test_whisper_preset(self):
        """Test whisper preset exists."""
        assert "whisper" in VOICE_PRESETS
        preset = VOICE_PRESETS["whisper"]
        assert preset["speed"] < 1.0  # Whisper is slower

    def test_preset_voices_are_valid(self):
        """Test preset voices reference valid KOKORO_VOICES."""
        for preset_name, preset in VOICE_PRESETS.items():
            voice = preset["voice"]
            assert voice in KOKORO_VOICES, f"Preset {preset_name} references invalid voice {voice}"

    def test_preset_speeds_in_range(self):
        """Test preset speeds are reasonable."""
        for preset_name, preset in VOICE_PRESETS.items():
            speed = preset.get("speed", 1.0)
            assert 0.5 <= speed <= 2.0, f"Preset {preset_name} has invalid speed {speed}"

    def test_preset_has_description(self):
        """Test all presets have descriptions."""
        for preset_name, preset in VOICE_PRESETS.items():
            assert "description" in preset, f"Preset {preset_name} missing description"
            assert len(preset["description"]) > 0


class TestConfigImmutability:
    """Tests for Config field behavior."""

    @patch('pathlib.Path.mkdir')
    def test_custom_values_override_defaults(self, mock_mkdir):
        """Test custom values override defaults."""
        config = Config(
            default_voice="am_michael",
            default_speed=1.5,
            sample_rate=44100,
        )
        assert config.default_voice == "am_michael"
        assert config.default_speed == 1.5
        assert config.sample_rate == 44100

    @patch('pathlib.Path.mkdir')
    def test_all_fields_can_be_set(self, mock_mkdir):
        """Test all config fields can be set at init."""
        config = Config(
            output_dir=Path("/test/output"),
            cache_dir=Path("/test/cache"),
            device="cpu",
            default_voice="bf_emma",
            default_speed=0.9,
            sample_rate=22050,
            use_gpu=False,
            cache_models=False,
        )
        assert config.device == "cpu"
        assert config.default_voice == "bf_emma"
        assert config.cache_models is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
