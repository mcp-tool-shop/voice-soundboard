"""
Tests for Phase 10: Humanization & Rhythm - Batch 12
BreathType, EmotionalState Enums, BreathConfig, PitchHumanizeConfig, TimingHumanizeConfig

Tests cover:
- BreathType enum (TEST-HUM-01 to TEST-HUM-06)
- EmotionalState enum (TEST-HUM-07 to TEST-HUM-13)
- BreathConfig dataclass (TEST-HUM-14 to TEST-HUM-19)
- PitchHumanizeConfig dataclass (TEST-HUM-20 to TEST-HUM-24)
- TimingHumanizeConfig dataclass (TEST-HUM-25)
"""

import pytest
import numpy as np


# =============================================================================
# TEST-HUM-01 to TEST-HUM-06: BreathType Enum Tests
# =============================================================================

from voice_soundboard.vocology.humanize import (
    BreathType,
    EmotionalState,
    BreathConfig,
    PitchHumanizeConfig,
    TimingHumanizeConfig,
    HumanizeConfig,
)


class TestBreathTypeEnum:
    """Tests for BreathType enum (TEST-HUM-01 to TEST-HUM-06)."""

    def test_hum_01_breath_type_quick(self):
        """TEST-HUM-01: BreathType.QUICK has correct value 'quick'."""
        assert hasattr(BreathType, 'QUICK')
        assert BreathType.QUICK.value == "quick"

    def test_hum_02_breath_type_medium(self):
        """TEST-HUM-02: BreathType.MEDIUM has correct value 'medium'."""
        assert hasattr(BreathType, 'MEDIUM')
        assert BreathType.MEDIUM.value == "medium"

    def test_hum_03_breath_type_deep(self):
        """TEST-HUM-03: BreathType.DEEP has correct value 'deep'."""
        assert hasattr(BreathType, 'DEEP')
        assert BreathType.DEEP.value == "deep"

    def test_hum_04_breath_type_gasp(self):
        """TEST-HUM-04: BreathType.GASP has correct value 'gasp'."""
        assert hasattr(BreathType, 'GASP')
        assert BreathType.GASP.value == "gasp"

    def test_hum_05_breath_type_sigh(self):
        """TEST-HUM-05: BreathType.SIGH has correct value 'sigh'."""
        assert hasattr(BreathType, 'SIGH')
        assert BreathType.SIGH.value == "sigh"

    def test_hum_06_breath_type_nasal(self):
        """TEST-HUM-06: BreathType.NASAL has correct value 'nasal'."""
        assert hasattr(BreathType, 'NASAL')
        assert BreathType.NASAL.value == "nasal"


# =============================================================================
# TEST-HUM-07 to TEST-HUM-13: EmotionalState Enum Tests
# =============================================================================

class TestEmotionalStateEnum:
    """Tests for EmotionalState enum (TEST-HUM-07 to TEST-HUM-13)."""

    def test_hum_07_emotional_state_neutral(self):
        """TEST-HUM-07: EmotionalState.NEUTRAL has correct value 'neutral'."""
        assert hasattr(EmotionalState, 'NEUTRAL')
        assert EmotionalState.NEUTRAL.value == "neutral"

    def test_hum_08_emotional_state_excited(self):
        """TEST-HUM-08: EmotionalState.EXCITED has correct value 'excited'."""
        assert hasattr(EmotionalState, 'EXCITED')
        assert EmotionalState.EXCITED.value == "excited"

    def test_hum_09_emotional_state_calm(self):
        """TEST-HUM-09: EmotionalState.CALM has correct value 'calm'."""
        assert hasattr(EmotionalState, 'CALM')
        assert EmotionalState.CALM.value == "calm"

    def test_hum_10_emotional_state_tired(self):
        """TEST-HUM-10: EmotionalState.TIRED has correct value 'tired'."""
        assert hasattr(EmotionalState, 'TIRED')
        assert EmotionalState.TIRED.value == "tired"

    def test_hum_11_emotional_state_anxious(self):
        """TEST-HUM-11: EmotionalState.ANXIOUS has correct value 'anxious'."""
        assert hasattr(EmotionalState, 'ANXIOUS')
        assert EmotionalState.ANXIOUS.value == "anxious"

    def test_hum_12_emotional_state_confident(self):
        """TEST-HUM-12: EmotionalState.CONFIDENT has correct value 'confident'."""
        assert hasattr(EmotionalState, 'CONFIDENT')
        assert EmotionalState.CONFIDENT.value == "confident"

    def test_hum_13_emotional_state_intimate(self):
        """TEST-HUM-13: EmotionalState.INTIMATE has correct value 'intimate'."""
        assert hasattr(EmotionalState, 'INTIMATE')
        assert EmotionalState.INTIMATE.value == "intimate"


# =============================================================================
# TEST-HUM-14 to TEST-HUM-19: BreathConfig Dataclass Tests
# =============================================================================

class TestBreathConfig:
    """Tests for BreathConfig dataclass (TEST-HUM-14 to TEST-HUM-19)."""

    def test_hum_14_breath_config_default_enabled(self):
        """TEST-HUM-14: BreathConfig has default enabled=True."""
        config = BreathConfig()
        assert hasattr(config, 'enabled')
        assert config.enabled is True

    def test_hum_15_breath_config_default_volume_db(self):
        """TEST-HUM-15: BreathConfig has default volume_db=-24.0."""
        config = BreathConfig()
        assert hasattr(config, 'volume_db')
        assert config.volume_db == -24.0

    def test_hum_16_breath_config_default_pre_phrase_offset_ms(self):
        """TEST-HUM-16: BreathConfig has default pre_phrase_offset_ms=150."""
        config = BreathConfig()
        assert hasattr(config, 'pre_phrase_offset_ms')
        assert config.pre_phrase_offset_ms == 150

    def test_hum_17_breath_config_default_min_phrase_gap_ms(self):
        """TEST-HUM-17: BreathConfig has default min_phrase_gap_ms=300."""
        config = BreathConfig()
        assert hasattr(config, 'min_phrase_gap_ms')
        assert config.min_phrase_gap_ms == 300

    def test_hum_18_breath_config_default_intensity(self):
        """TEST-HUM-18: BreathConfig has default intensity=0.25."""
        config = BreathConfig()
        assert hasattr(config, 'intensity')
        assert config.intensity == 0.25

    def test_hum_19_breath_config_accepts_custom_values(self):
        """TEST-HUM-19: BreathConfig accepts custom values."""
        config = BreathConfig(
            enabled=False,
            volume_db=-18.0,
            pre_phrase_offset_ms=200,
            min_phrase_gap_ms=500,
            intensity=0.5,
        )
        assert config.enabled is False
        assert config.volume_db == -18.0
        assert config.pre_phrase_offset_ms == 200
        assert config.min_phrase_gap_ms == 500
        assert config.intensity == 0.5


# =============================================================================
# TEST-HUM-20 to TEST-HUM-24: PitchHumanizeConfig Dataclass Tests
# =============================================================================

class TestPitchHumanizeConfig:
    """Tests for PitchHumanizeConfig dataclass (TEST-HUM-20 to TEST-HUM-24)."""

    def test_hum_20_pitch_config_has_enabled(self):
        """TEST-HUM-20: PitchHumanizeConfig has enabled field."""
        config = PitchHumanizeConfig()
        assert hasattr(config, 'enabled')
        assert config.enabled is True

    def test_hum_21_pitch_config_has_jitter_amount(self):
        """TEST-HUM-21: PitchHumanizeConfig has jitter_cents field."""
        config = PitchHumanizeConfig()
        assert hasattr(config, 'jitter_cents')
        assert config.jitter_cents == 5.0

    def test_hum_22_pitch_config_has_drift_amount(self):
        """TEST-HUM-22: PitchHumanizeConfig has drift_max_cents field."""
        config = PitchHumanizeConfig()
        assert hasattr(config, 'drift_max_cents')
        assert config.drift_max_cents == 15.0

    def test_hum_23_pitch_config_has_scoop_probability(self):
        """TEST-HUM-23: PitchHumanizeConfig has scoop_enabled field."""
        config = PitchHumanizeConfig()
        assert hasattr(config, 'scoop_enabled')
        assert config.scoop_enabled is True

    def test_hum_24_pitch_config_accepts_custom_values(self):
        """TEST-HUM-24: PitchHumanizeConfig accepts custom values."""
        config = PitchHumanizeConfig(
            enabled=False,
            jitter_cents=10.0,
            drift_max_cents=20.0,
            scoop_enabled=False,
            scoop_cents=40.0,
        )
        assert config.enabled is False
        assert config.jitter_cents == 10.0
        assert config.drift_max_cents == 20.0
        assert config.scoop_enabled is False
        assert config.scoop_cents == 40.0


# =============================================================================
# TEST-HUM-25: TimingHumanizeConfig Dataclass Tests
# =============================================================================

class TestTimingHumanizeConfig:
    """Tests for TimingHumanizeConfig dataclass (TEST-HUM-25)."""

    def test_hum_25_timing_config_has_enabled(self):
        """TEST-HUM-25: TimingHumanizeConfig has enabled field."""
        config = TimingHumanizeConfig()
        assert hasattr(config, 'enabled')
        assert config.enabled is True
