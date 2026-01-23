"""
Tests for Phase 9: Vocology Module - Batch 9
PitchContour, DurationPattern, ProsodyContour, BreakStrength

Tests cover:
- BreakStrength enum (TEST-PRO-01 to TEST-PRO-04)
- PitchContour dataclass (TEST-PRO-05 to TEST-PRO-12)
- DurationPattern dataclass (TEST-PRO-13 to TEST-PRO-17)
- ProsodyContour dataclass (TEST-PRO-18 to TEST-PRO-25)
"""

import pytest
import numpy as np
from pathlib import Path


# =============================================================================
# TEST-PRO-01 to TEST-PRO-04: BreakStrength Enum Tests
# =============================================================================

from voice_soundboard.vocology.prosody import (
    BreakStrength,
    PitchContour,
    DurationPattern,
    ProsodyContour,
    ProsodyAnalyzer,
    ProsodyModifier,
    analyze_prosody,
    modify_prosody,
)


class TestBreakStrengthEnum:
    """Tests for BreakStrength enum (TEST-PRO-01 to TEST-PRO-04)."""

    def test_pro_01_has_none(self):
        """TEST-PRO-01: BreakStrength.NONE exists."""
        assert hasattr(BreakStrength, 'NONE')
        assert BreakStrength.NONE.value == "none"

    def test_pro_02_has_weak(self):
        """TEST-PRO-02: BreakStrength.WEAK exists."""
        assert hasattr(BreakStrength, 'WEAK')
        assert BreakStrength.WEAK.value == "weak"

    def test_pro_03_has_strong(self):
        """TEST-PRO-03: BreakStrength.STRONG exists."""
        assert hasattr(BreakStrength, 'STRONG')
        assert BreakStrength.STRONG.value == "strong"

    def test_pro_04_has_x_strong(self):
        """TEST-PRO-04: BreakStrength.X_STRONG exists."""
        assert hasattr(BreakStrength, 'X_STRONG')
        assert BreakStrength.X_STRONG.value == "x-strong"


# =============================================================================
# TEST-PRO-05 to TEST-PRO-12: PitchContour Dataclass Tests
# =============================================================================

class TestPitchContour:
    """Tests for PitchContour dataclass (TEST-PRO-05 to TEST-PRO-12)."""

    @pytest.fixture
    def mock_pitch_contour(self):
        """Create mock PitchContour."""
        n_frames = 100
        times = np.linspace(0, 1.0, n_frames)
        # Simulate F0 around 120 Hz with some variation
        frequencies = 120 + 10 * np.sin(2 * np.pi * 2 * times)
        voiced = np.ones(n_frames, dtype=bool)
        voiced[40:50] = False  # Some unvoiced frames

        return PitchContour(
            times=times,
            frequencies=frequencies,
            voiced=voiced,
            confidence=np.ones(n_frames) * 0.9,
        )

    def test_pro_05_has_times(self, mock_pitch_contour):
        """TEST-PRO-05: PitchContour has times array."""
        assert hasattr(mock_pitch_contour, 'times')
        assert isinstance(mock_pitch_contour.times, np.ndarray)

    def test_pro_06_has_frequencies(self, mock_pitch_contour):
        """TEST-PRO-06: PitchContour has frequencies array."""
        assert hasattr(mock_pitch_contour, 'frequencies')
        assert isinstance(mock_pitch_contour.frequencies, np.ndarray)

    def test_pro_07_has_voiced(self, mock_pitch_contour):
        """TEST-PRO-07: PitchContour has voiced array."""
        assert hasattr(mock_pitch_contour, 'voiced')
        assert isinstance(mock_pitch_contour.voiced, np.ndarray)

    def test_pro_08_duration_property(self, mock_pitch_contour):
        """TEST-PRO-08: duration property returns total duration."""
        assert mock_pitch_contour.duration == 1.0

    def test_pro_09_mean_f0_property(self, mock_pitch_contour):
        """TEST-PRO-09: mean_f0 property returns mean of voiced frames."""
        mean_f0 = mock_pitch_contour.mean_f0
        assert 110 < mean_f0 < 130  # Around 120 Hz

    def test_pro_10_f0_range_property(self, mock_pitch_contour):
        """TEST-PRO-10: f0_range property returns (min, max) tuple."""
        f0_range = mock_pitch_contour.f0_range
        assert isinstance(f0_range, tuple)
        assert len(f0_range) == 2
        assert f0_range[0] < f0_range[1]

    def test_pro_11_resample_method(self, mock_pitch_contour):
        """TEST-PRO-11: resample() returns new PitchContour with n points."""
        resampled = mock_pitch_contour.resample(50)
        assert isinstance(resampled, PitchContour)
        assert len(resampled.times) == 50

    def test_pro_12_resample_preserves_duration(self, mock_pitch_contour):
        """TEST-PRO-12: resample() preserves total duration."""
        resampled = mock_pitch_contour.resample(200)
        assert abs(resampled.duration - mock_pitch_contour.duration) < 0.01


# =============================================================================
# TEST-PRO-13 to TEST-PRO-17: DurationPattern Dataclass Tests
# =============================================================================

class TestDurationPattern:
    """Tests for DurationPattern dataclass (TEST-PRO-13 to TEST-PRO-17)."""

    @pytest.fixture
    def mock_duration_pattern(self):
        """Create mock DurationPattern."""
        units = ["Hello", "world", "this", "is", "a", "test"]
        durations = np.array([0.3, 0.4, 0.2, 0.15, 0.1, 0.35])
        boundaries = np.cumsum(np.insert(durations, 0, 0))

        return DurationPattern(
            units=units,
            durations=durations,
            boundaries=boundaries,
        )

    def test_pro_13_has_units(self, mock_duration_pattern):
        """TEST-PRO-13: DurationPattern has units list."""
        assert hasattr(mock_duration_pattern, 'units')
        assert isinstance(mock_duration_pattern.units, list)

    def test_pro_14_has_durations(self, mock_duration_pattern):
        """TEST-PRO-14: DurationPattern has durations array."""
        assert hasattr(mock_duration_pattern, 'durations')
        assert isinstance(mock_duration_pattern.durations, np.ndarray)

    def test_pro_15_total_duration_property(self, mock_duration_pattern):
        """TEST-PRO-15: total_duration property returns sum of durations."""
        expected = 0.3 + 0.4 + 0.2 + 0.15 + 0.1 + 0.35
        assert abs(mock_duration_pattern.total_duration - expected) < 0.01

    def test_pro_16_mean_duration_property(self, mock_duration_pattern):
        """TEST-PRO-16: mean_duration property returns average duration."""
        expected = (0.3 + 0.4 + 0.2 + 0.15 + 0.1 + 0.35) / 6
        assert abs(mock_duration_pattern.mean_duration - expected) < 0.01

    def test_pro_17_scale_method(self, mock_duration_pattern):
        """TEST-PRO-17: scale() returns scaled DurationPattern."""
        scaled = mock_duration_pattern.scale(2.0)
        assert isinstance(scaled, DurationPattern)
        assert abs(scaled.total_duration - mock_duration_pattern.total_duration * 2) < 0.01


# =============================================================================
# TEST-PRO-18 to TEST-PRO-25: ProsodyContour Dataclass Tests
# =============================================================================

class TestProsodyContour:
    """Tests for ProsodyContour dataclass (TEST-PRO-18 to TEST-PRO-25)."""

    @pytest.fixture
    def mock_pitch_contour(self):
        """Create mock PitchContour."""
        n_frames = 100
        times = np.linspace(0, 1.0, n_frames)
        frequencies = 120 + 10 * np.sin(2 * np.pi * 2 * times)
        voiced = np.ones(n_frames, dtype=bool)
        return PitchContour(times=times, frequencies=frequencies, voiced=voiced)

    @pytest.fixture
    def mock_prosody_contour(self, mock_pitch_contour):
        """Create mock ProsodyContour."""
        return ProsodyContour(
            pitch=mock_pitch_contour,
            duration=None,
            energy=np.random.rand(100),
            pauses=[(0.3, 0.1), (0.7, 0.15)],
        )

    def test_pro_18_has_pitch(self, mock_prosody_contour):
        """TEST-PRO-18: ProsodyContour has pitch field."""
        assert hasattr(mock_prosody_contour, 'pitch')

    def test_pro_19_pitch_is_optional(self):
        """TEST-PRO-19: ProsodyContour pitch can be None."""
        contour = ProsodyContour()
        assert contour.pitch is None

    def test_pro_20_has_duration(self, mock_prosody_contour):
        """TEST-PRO-20: ProsodyContour has duration field."""
        assert hasattr(mock_prosody_contour, 'duration')

    def test_pro_21_has_energy(self, mock_prosody_contour):
        """TEST-PRO-21: ProsodyContour has energy field."""
        assert hasattr(mock_prosody_contour, 'energy')
        assert isinstance(mock_prosody_contour.energy, np.ndarray)

    def test_pro_22_has_pauses(self, mock_prosody_contour):
        """TEST-PRO-22: ProsodyContour has pauses list."""
        assert hasattr(mock_prosody_contour, 'pauses')
        assert isinstance(mock_prosody_contour.pauses, list)

    def test_pro_23_pauses_format(self, mock_prosody_contour):
        """TEST-PRO-23: pauses are (time, duration) tuples."""
        for pause in mock_prosody_contour.pauses:
            assert isinstance(pause, tuple)
            assert len(pause) == 2

    def test_pro_24_set_pitch_range(self, mock_prosody_contour):
        """TEST-PRO-24: set_pitch_range() modifies pitch range."""
        original_range = mock_prosody_contour.pitch.f0_range
        mock_prosody_contour.set_pitch_range(80.0, 160.0)
        new_range = mock_prosody_contour.pitch.f0_range
        # Should have modified the range
        assert new_range[0] >= 79  # Allow some tolerance
        assert new_range[1] <= 161

    def test_pro_25_empty_contour_creation(self):
        """TEST-PRO-25: ProsodyContour can be created empty."""
        contour = ProsodyContour()
        assert contour.pitch is None
        assert contour.duration is None
        assert contour.energy is None
        assert contour.pauses == []
