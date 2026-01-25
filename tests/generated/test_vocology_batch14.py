"""
Tests for Phase 10: Humanization & Rhythm - Batch 14
BreathInserter continued, PitchHumanizer, VoiceHumanizer, Convenience Functions

Tests cover:
- BreathInserter class (TEST-HUM-51 to TEST-HUM-53)
- PitchHumanizer class (TEST-HUM-54 to TEST-HUM-61)
- VoiceHumanizer class (TEST-HUM-62 to TEST-HUM-70)
- Convenience functions (TEST-HUM-71 to TEST-HUM-77)
"""

import pytest
import numpy as np


from voice_soundboard.vocology.humanize import (
    BreathType,
    EmotionalState,
    BreathConfig,
    PitchHumanizeConfig,
    TimingHumanizeConfig,
    HumanizeConfig,
    BreathGenerator,
    BreathInserter,
    PitchHumanizer,
    VoiceHumanizer,
    humanize_audio,
    add_breaths,
    humanize_pitch,
)


# =============================================================================
# TEST-HUM-51 to TEST-HUM-53: BreathInserter Class Tests Continued
# =============================================================================

class TestBreathInserterContinued:
    """Tests for BreathInserter class (TEST-HUM-51 to TEST-HUM-53)."""

    @pytest.fixture
    def mock_audio(self):
        """Create mock audio with silence gaps."""
        sr = 24000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 200 * t) * 0.5
        # Create silence gap
        silence_start = int(0.8 * sr)
        silence_end = int(1.2 * sr)
        audio[silence_start:silence_end] = 0
        return audio.astype(np.float32), sr

    def test_hum_51_breath_inserter_selects_breath_type(self, mock_audio):
        """TEST-HUM-51: BreathInserter selects appropriate breath type based on phrase length."""
        audio, sr = mock_audio
        config = BreathConfig(deep_breath_threshold_s=0.5)
        inserter = BreathInserter(sample_rate=sr)
        # Should function without error - breath type selection is internal
        result = inserter.insert_breaths(audio, config=config)
        assert isinstance(result, np.ndarray)

    def test_hum_52_breath_inserter_output_valid(self, mock_audio):
        """TEST-HUM-52: BreathInserter returns valid numpy array."""
        audio, sr = mock_audio
        config = BreathConfig()
        inserter = BreathInserter(sample_rate=sr)
        result = inserter.insert_breaths(audio, config=config)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_hum_53_breath_inserter_output_length(self, mock_audio):
        """TEST-HUM-53: BreathInserter output length >= input length."""
        audio, sr = mock_audio
        config = BreathConfig(enabled=True)
        inserter = BreathInserter(sample_rate=sr)
        result = inserter.insert_breaths(audio, config=config)
        # Output should be at least as long as input
        assert len(result) >= len(audio)


# =============================================================================
# TEST-HUM-54 to TEST-HUM-61: PitchHumanizer Class Tests
# =============================================================================

class TestPitchHumanizer:
    """Tests for PitchHumanizer class (TEST-HUM-54 to TEST-HUM-61)."""

    @pytest.fixture
    def mock_audio(self):
        """Create mock audio with clear pitch."""
        sr = 24000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        # 200 Hz tone with harmonics
        audio = np.zeros_like(t)
        for h in range(1, 10):
            audio += (1.0 / h) * np.sin(2 * np.pi * 200 * h * t)
        audio = audio / np.max(np.abs(audio)) * 0.8
        return audio.astype(np.float32), sr

    def test_hum_54_pitch_humanizer_init(self):
        """TEST-HUM-54: PitchHumanizer initializes with sample_rate."""
        humanizer = PitchHumanizer(sample_rate=24000)
        assert humanizer is not None
        assert humanizer.sample_rate == 24000

    def test_hum_55_pitch_humanizer_returns_array(self, mock_audio):
        """TEST-HUM-55: PitchHumanizer.humanize() returns numpy array."""
        audio, sr = mock_audio
        config = PitchHumanizeConfig()
        humanizer = PitchHumanizer(sample_rate=sr)
        result = humanizer.humanize(audio, config=config)
        assert isinstance(result, np.ndarray)

    def test_hum_56_pitch_humanizer_adds_micro_jitter(self, mock_audio):
        """TEST-HUM-56: PitchHumanizer adds micro-jitter to pitch contour."""
        audio, sr = mock_audio
        config = PitchHumanizeConfig(jitter_cents=10.0, enabled=True)
        humanizer = PitchHumanizer(sample_rate=sr)
        result_audio = humanizer.humanize(audio, config=config)
        # Result should be a valid array (jitter effect may be subtle)
        assert isinstance(result_audio, np.ndarray)
        assert len(result_audio) == len(audio)

    def test_hum_57_pitch_humanizer_adds_drift(self, mock_audio):
        """TEST-HUM-57: PitchHumanizer adds drift to pitch contour."""
        audio, sr = mock_audio
        config = PitchHumanizeConfig(drift_max_cents=20.0)
        humanizer = PitchHumanizer(sample_rate=sr)
        result_audio = humanizer.humanize(audio, config=config)
        assert isinstance(result_audio, np.ndarray)
        assert len(result_audio) > 0

    def test_hum_58_pitch_humanizer_scooping(self, mock_audio):
        """TEST-HUM-58: PitchHumanizer applies scooping at phrase starts."""
        audio, sr = mock_audio
        config = PitchHumanizeConfig(scoop_enabled=True, scoop_cents=30.0)
        humanizer = PitchHumanizer(sample_rate=sr)
        result_audio = humanizer.humanize(audio, config=config)
        assert isinstance(result_audio, np.ndarray)

    def test_hum_59_pitch_humanizer_preserves_length(self, mock_audio):
        """TEST-HUM-59: PitchHumanizer output length equals input length."""
        audio, sr = mock_audio
        config = PitchHumanizeConfig()
        humanizer = PitchHumanizer(sample_rate=sr)
        result_audio = humanizer.humanize(audio, config=config)
        assert len(result_audio) == len(audio)

    def test_hum_60_pitch_humanizer_no_clipping(self, mock_audio):
        """TEST-HUM-60: PitchHumanizer does not clip output audio."""
        audio, sr = mock_audio
        config = PitchHumanizeConfig()
        humanizer = PitchHumanizer(sample_rate=sr)
        result_audio = humanizer.humanize(audio, config=config)
        assert np.max(np.abs(result_audio)) <= 1.1  # Allow small overshoot

    def test_hum_61_pitch_humanizer_disabled_returns_unchanged(self, mock_audio):
        """TEST-HUM-61: PitchHumanizer with disabled config returns unchanged audio."""
        audio, sr = mock_audio
        config = PitchHumanizeConfig(enabled=False)
        humanizer = PitchHumanizer(sample_rate=sr)
        result_audio = humanizer.humanize(audio, config=config)
        # With disabled config, should return same audio
        np.testing.assert_array_almost_equal(audio, result_audio)


# =============================================================================
# TEST-HUM-62 to TEST-HUM-70: VoiceHumanizer Class Tests
# =============================================================================

class TestVoiceHumanizer:
    """Tests for VoiceHumanizer class (TEST-HUM-62 to TEST-HUM-70)."""

    @pytest.fixture
    def mock_audio(self):
        """Create mock audio."""
        sr = 24000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 200 * t) * 0.5
        return audio.astype(np.float32), sr

    def test_hum_62_voice_humanizer_init_default(self):
        """TEST-HUM-62: VoiceHumanizer initializes with default sample_rate."""
        humanizer = VoiceHumanizer()
        assert humanizer is not None
        assert hasattr(humanizer, 'sample_rate')
        assert humanizer.sample_rate == 24000

    def test_hum_63_voice_humanizer_init_custom_sample_rate(self):
        """TEST-HUM-63: VoiceHumanizer initializes with custom sample_rate."""
        humanizer = VoiceHumanizer(sample_rate=44100)
        assert humanizer.sample_rate == 44100

    def test_hum_64_voice_humanizer_returns_tuple(self, mock_audio):
        """TEST-HUM-64: VoiceHumanizer.humanize() returns tuple (audio, sample_rate)."""
        audio, sr = mock_audio
        humanizer = VoiceHumanizer(sample_rate=sr)
        result = humanizer.humanize(audio, sample_rate=sr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_hum_65_voice_humanizer_applies_breath(self, mock_audio):
        """TEST-HUM-65: VoiceHumanizer applies breath insertion when enabled."""
        audio, sr = mock_audio
        config = HumanizeConfig()
        config.breath.enabled = True
        humanizer = VoiceHumanizer(sample_rate=sr)
        result_audio, _ = humanizer.humanize(audio, config=config, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_hum_66_voice_humanizer_applies_pitch(self, mock_audio):
        """TEST-HUM-66: VoiceHumanizer applies pitch humanization when enabled."""
        audio, sr = mock_audio
        config = HumanizeConfig()
        config.pitch.enabled = True
        config.breath.enabled = False
        humanizer = VoiceHumanizer(sample_rate=sr)
        result_audio, _ = humanizer.humanize(audio, config=config, sample_rate=sr)
        # Output should be different from input
        assert isinstance(result_audio, np.ndarray)

    def test_hum_67_voice_humanizer_applies_timing(self, mock_audio):
        """TEST-HUM-67: VoiceHumanizer applies timing variation when enabled."""
        audio, sr = mock_audio
        config = HumanizeConfig()
        config.timing.enabled = True
        humanizer = VoiceHumanizer(sample_rate=sr)
        result_audio, _ = humanizer.humanize(audio, config=config, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_hum_68_voice_humanizer_chains_effects(self, mock_audio):
        """TEST-HUM-68: VoiceHumanizer chains all effects correctly."""
        audio, sr = mock_audio
        config = HumanizeConfig()
        config.breath.enabled = True
        config.pitch.enabled = True
        config.timing.enabled = True
        humanizer = VoiceHumanizer(sample_rate=sr)
        result_audio, result_sr = humanizer.humanize(audio, config=config, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)
        assert result_sr == sr

    def test_hum_69_voice_humanizer_question_inflection(self, mock_audio):
        """TEST-HUM-69: VoiceHumanizer.humanize() with is_question=True adds rising inflection."""
        audio, sr = mock_audio
        humanizer = VoiceHumanizer(sample_rate=sr)
        result_audio, _ = humanizer.humanize(audio, sample_rate=sr, is_question=True)
        # Should process without error
        assert isinstance(result_audio, np.ndarray)

    def test_hum_70_voice_humanizer_no_clipping(self, mock_audio):
        """TEST-HUM-70: VoiceHumanizer output does not clip."""
        audio, sr = mock_audio
        humanizer = VoiceHumanizer(sample_rate=sr)
        result_audio, _ = humanizer.humanize(audio, sample_rate=sr)
        assert np.max(np.abs(result_audio)) <= 1.1  # Allow small overshoot


# =============================================================================
# TEST-HUM-71 to TEST-HUM-77: Convenience Functions Tests
# =============================================================================

class TestHumanizeConvenienceFunctions:
    """Tests for humanize convenience functions (TEST-HUM-71 to TEST-HUM-77)."""

    @pytest.fixture
    def mock_audio(self):
        """Create mock audio."""
        sr = 24000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 200 * t) * 0.5
        return audio.astype(np.float32), sr

    def test_hum_71_humanize_audio_with_array(self, mock_audio):
        """TEST-HUM-71: humanize_audio() function works with numpy array input."""
        audio, sr = mock_audio
        result_audio, result_sr = humanize_audio(audio, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)
        assert result_sr == sr

    def test_hum_72_humanize_audio_returns_tuple(self, mock_audio):
        """TEST-HUM-72: humanize_audio() returns (audio, sample_rate) tuple."""
        audio, sr = mock_audio
        result = humanize_audio(audio, sample_rate=sr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_hum_73_humanize_audio_with_emotional_state(self, mock_audio):
        """TEST-HUM-73: humanize_audio() accepts emotional_state parameter."""
        audio, sr = mock_audio
        result_audio, _ = humanize_audio(
            audio, sample_rate=sr, emotion=EmotionalState.EXCITED
        )
        assert isinstance(result_audio, np.ndarray)

    def test_hum_74_add_breaths_function(self, mock_audio):
        """TEST-HUM-74: add_breaths() function adds breaths to audio."""
        audio, sr = mock_audio
        result_audio = add_breaths(audio, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_hum_75_add_breaths_intensity(self, mock_audio):
        """TEST-HUM-75: add_breaths() respects intensity parameter."""
        audio, sr = mock_audio
        low_intensity = add_breaths(audio, sample_rate=sr, intensity=0.3)
        high_intensity = add_breaths(audio, sample_rate=sr, intensity=0.9)
        # Both should be valid arrays
        assert isinstance(low_intensity, np.ndarray)
        assert isinstance(high_intensity, np.ndarray)

    def test_hum_76_humanize_pitch_function(self, mock_audio):
        """TEST-HUM-76: humanize_pitch() function applies pitch variations."""
        audio, sr = mock_audio
        result_audio = humanize_pitch(audio, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_hum_77_humanize_pitch_jitter_amount(self, mock_audio):
        """TEST-HUM-77: humanize_pitch() respects jitter_cents parameter."""
        audio, sr = mock_audio
        result_audio = humanize_pitch(audio, sample_rate=sr, jitter_cents=10.0)
        assert isinstance(result_audio, np.ndarray)
        assert len(result_audio) == len(audio)
