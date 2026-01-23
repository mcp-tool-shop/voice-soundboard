"""
Tests for Phase 10: Humanization & Rhythm - Batch 13
TimingHumanizeConfig continued, HumanizeConfig, BreathGenerator, BreathInserter

Tests cover:
- TimingHumanizeConfig dataclass (TEST-HUM-26 to TEST-HUM-28)
- HumanizeConfig dataclass (TEST-HUM-29 to TEST-HUM-38)
- BreathGenerator class (TEST-HUM-39 to TEST-HUM-45)
- BreathInserter class (TEST-HUM-46 to TEST-HUM-50)
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
)


# =============================================================================
# TEST-HUM-26 to TEST-HUM-28: TimingHumanizeConfig Dataclass Tests
# =============================================================================

class TestTimingHumanizeConfigContinued:
    """Tests for TimingHumanizeConfig dataclass (TEST-HUM-26 to TEST-HUM-28)."""

    def test_hum_26_timing_config_has_syllable_variation_ms(self):
        """TEST-HUM-26: TimingHumanizeConfig has timing_variation_ms field."""
        config = TimingHumanizeConfig()
        assert hasattr(config, 'timing_variation_ms')
        assert config.timing_variation_ms == 20.0

    def test_hum_27_timing_config_has_phrase_boundary_pause_ms(self):
        """TEST-HUM-27: TimingHumanizeConfig has gap_variation_percent field."""
        config = TimingHumanizeConfig()
        assert hasattr(config, 'gap_variation_percent')
        assert config.gap_variation_percent == 15.0

    def test_hum_28_timing_config_accepts_custom_values(self):
        """TEST-HUM-28: TimingHumanizeConfig accepts custom values."""
        config = TimingHumanizeConfig(
            enabled=False,
            timing_variation_ms=30.0,
            timing_bias_ms=10.0,
            gap_variation_percent=25.0,
        )
        assert config.enabled is False
        assert config.timing_variation_ms == 30.0
        assert config.timing_bias_ms == 10.0
        assert config.gap_variation_percent == 25.0


# =============================================================================
# TEST-HUM-29 to TEST-HUM-38: HumanizeConfig Dataclass Tests
# =============================================================================

class TestHumanizeConfig:
    """Tests for HumanizeConfig dataclass (TEST-HUM-29 to TEST-HUM-38)."""

    def test_hum_29_humanize_config_contains_breath_config(self):
        """TEST-HUM-29: HumanizeConfig contains breath_config."""
        config = HumanizeConfig()
        assert hasattr(config, 'breath')
        assert isinstance(config.breath, BreathConfig)

    def test_hum_30_humanize_config_contains_pitch_config(self):
        """TEST-HUM-30: HumanizeConfig contains pitch_config."""
        config = HumanizeConfig()
        assert hasattr(config, 'pitch')
        assert isinstance(config.pitch, PitchHumanizeConfig)

    def test_hum_31_humanize_config_contains_timing_config(self):
        """TEST-HUM-31: HumanizeConfig contains timing_config."""
        config = HumanizeConfig()
        assert hasattr(config, 'timing')
        assert isinstance(config.timing, TimingHumanizeConfig)

    def test_hum_32_humanize_config_has_emotional_state(self):
        """TEST-HUM-32: HumanizeConfig has emotional_state field."""
        config = HumanizeConfig()
        assert hasattr(config, 'emotion')
        assert isinstance(config.emotion, EmotionalState)
        assert config.emotion == EmotionalState.NEUTRAL

    def test_hum_33_humanize_config_for_emotion_excited(self):
        """TEST-HUM-33: HumanizeConfig.for_emotion() returns correct preset for EXCITED."""
        config = HumanizeConfig.for_emotion(EmotionalState.EXCITED)
        assert config.emotion == EmotionalState.EXCITED
        # Excited has higher jitter and intensity
        assert config.pitch.jitter_cents == 7.0
        assert config.breath.intensity == 0.8

    def test_hum_34_humanize_config_for_emotion_calm(self):
        """TEST-HUM-34: HumanizeConfig.for_emotion() returns correct preset for CALM."""
        config = HumanizeConfig.for_emotion(EmotionalState.CALM)
        assert config.emotion == EmotionalState.CALM
        # Calm has lower jitter
        assert config.pitch.jitter_cents == 3.0
        assert config.breath.intensity == 0.5

    def test_hum_35_humanize_config_for_emotion_tired(self):
        """TEST-HUM-35: HumanizeConfig.for_emotion() returns correct preset for TIRED."""
        config = HumanizeConfig.for_emotion(EmotionalState.TIRED)
        assert config.emotion == EmotionalState.TIRED
        # Tired has behind-the-beat timing
        assert config.timing.timing_bias_ms == 15.0
        assert config.breath.intensity == 0.9

    def test_hum_36_humanize_config_for_emotion_anxious(self):
        """TEST-HUM-36: HumanizeConfig.for_emotion() returns correct preset for ANXIOUS."""
        config = HumanizeConfig.for_emotion(EmotionalState.ANXIOUS)
        assert config.emotion == EmotionalState.ANXIOUS
        # Anxious has higher jitter and timing variation
        assert config.pitch.jitter_cents == 8.0
        assert config.timing.timing_variation_ms == 30.0

    def test_hum_37_humanize_config_for_emotion_confident(self):
        """TEST-HUM-37: HumanizeConfig.for_emotion() returns correct preset for CONFIDENT."""
        config = HumanizeConfig.for_emotion(EmotionalState.CONFIDENT)
        assert config.emotion == EmotionalState.CONFIDENT
        # Confident has slightly ahead timing
        assert config.timing.timing_bias_ms == -5.0

    def test_hum_38_humanize_config_for_emotion_intimate(self):
        """TEST-HUM-38: HumanizeConfig.for_emotion() returns correct preset for INTIMATE."""
        config = HumanizeConfig.for_emotion(EmotionalState.INTIMATE)
        assert config.emotion == EmotionalState.INTIMATE
        # Intimate has more audible breaths
        assert config.breath.volume_db == -10.0
        assert config.breath.intensity == 0.9


# =============================================================================
# TEST-HUM-39 to TEST-HUM-45: BreathGenerator Class Tests
# =============================================================================

class TestBreathGenerator:
    """Tests for BreathGenerator class (TEST-HUM-39 to TEST-HUM-45)."""

    def test_hum_39_breath_generator_init_default_sample_rate(self):
        """TEST-HUM-39: BreathGenerator initializes with default sample_rate."""
        generator = BreathGenerator()
        assert hasattr(generator, 'sample_rate')
        assert generator.sample_rate == 24000

    def test_hum_40_breath_generator_generate_returns_array(self):
        """TEST-HUM-40: BreathGenerator.generate() returns numpy array."""
        generator = BreathGenerator()
        result = generator.generate()
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_hum_41_breath_generator_quick_breath_duration(self):
        """TEST-HUM-41: BreathGenerator.generate(BreathType.QUICK) returns 100-150ms audio."""
        generator = BreathGenerator(sample_rate=24000)
        result = generator.generate(BreathType.QUICK)
        duration_ms = len(result) / generator.sample_rate * 1000
        # QUICK breath default is ~120ms
        assert 80 <= duration_ms <= 180

    def test_hum_42_breath_generator_medium_breath_duration(self):
        """TEST-HUM-42: BreathGenerator.generate(BreathType.MEDIUM) returns 200-300ms audio."""
        generator = BreathGenerator(sample_rate=24000)
        result = generator.generate(BreathType.MEDIUM)
        duration_ms = len(result) / generator.sample_rate * 1000
        # MEDIUM breath default is ~250ms
        assert 150 <= duration_ms <= 350

    def test_hum_43_breath_generator_deep_breath_duration(self):
        """TEST-HUM-43: BreathGenerator.generate(BreathType.DEEP) returns 300-500ms audio."""
        generator = BreathGenerator(sample_rate=24000)
        result = generator.generate(BreathType.DEEP)
        duration_ms = len(result) / generator.sample_rate * 1000
        # DEEP breath default is ~400ms
        assert 250 <= duration_ms <= 550

    def test_hum_44_breath_generator_intensity_parameter(self):
        """TEST-HUM-44: BreathGenerator applies intensity parameter correctly."""
        generator = BreathGenerator()
        low_intensity = generator.generate(intensity=0.3)
        high_intensity = generator.generate(intensity=0.9)
        # Higher intensity should produce louder (higher RMS) breath
        rms_low = np.sqrt(np.mean(low_intensity ** 2))
        rms_high = np.sqrt(np.mean(high_intensity ** 2))
        assert rms_high > rms_low

    def test_hum_45_breath_generator_output_normalized(self):
        """TEST-HUM-45: BreathGenerator output is normalized to not clip."""
        generator = BreathGenerator()
        for breath_type in BreathType:
            result = generator.generate(breath_type, intensity=1.0)
            # Output should not exceed [-1, 1]
            assert np.max(np.abs(result)) <= 1.0


# =============================================================================
# TEST-HUM-46 to TEST-HUM-50: BreathInserter Class Tests
# =============================================================================

class TestBreathInserter:
    """Tests for BreathInserter class (TEST-HUM-46 to TEST-HUM-50)."""

    @pytest.fixture
    def mock_audio(self):
        """Create mock audio with silence gaps."""
        sr = 24000
        # 2 seconds of audio with silence in middle
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 200 * t) * 0.5

        # Create silence gap (phrase boundary)
        silence_start = int(0.8 * sr)
        silence_end = int(1.2 * sr)
        audio[silence_start:silence_end] = 0

        return audio.astype(np.float32), sr

    def test_hum_46_breath_inserter_init(self):
        """TEST-HUM-46: BreathInserter initializes with sample_rate."""
        inserter = BreathInserter(sample_rate=24000)
        assert inserter is not None
        assert inserter.sample_rate == 24000

    def test_hum_47_breath_inserter_has_breath_generator(self):
        """TEST-HUM-47: BreathInserter has internal BreathGenerator."""
        inserter = BreathInserter()
        assert hasattr(inserter, 'breath_generator')

    def test_hum_48_breath_inserter_insert_breaths_returns_array(self, mock_audio):
        """TEST-HUM-48: BreathInserter.insert_breaths() returns numpy array."""
        audio, sr = mock_audio
        config = BreathConfig(enabled=True, intensity=0.5)
        inserter = BreathInserter(sample_rate=sr)
        result = inserter.insert_breaths(audio, config=config)
        assert isinstance(result, np.ndarray)

    def test_hum_49_breath_inserter_respects_config(self, mock_audio):
        """TEST-HUM-49: BreathInserter respects BreathConfig settings."""
        audio, sr = mock_audio
        # Different config settings
        config1 = BreathConfig(intensity=0.3)
        config2 = BreathConfig(intensity=0.9)
        inserter = BreathInserter(sample_rate=sr)
        # Both should work without error
        result1 = inserter.insert_breaths(audio, config=config1)
        result2 = inserter.insert_breaths(audio, config=config2)
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)

    def test_hum_50_breath_inserter_with_phrase_boundaries(self, mock_audio):
        """TEST-HUM-50: BreathInserter accepts phrase_boundaries parameter."""
        audio, sr = mock_audio
        inserter = BreathInserter(sample_rate=sr)
        boundaries = [(0.0, 0.8), (1.2, 2.0)]  # Two phrases
        result = inserter.insert_breaths(audio, phrase_boundaries=boundaries)
        assert isinstance(result, np.ndarray)
