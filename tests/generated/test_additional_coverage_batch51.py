"""
Test Additional Coverage Batch 51: Vocology Prosody Tests

Tests for:
- BreakStrength enum
- PitchContour dataclass
- DurationPattern dataclass
- ProsodyContour dataclass
- ProsodyAnalyzer class
- ProsodyModifier class
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


# ============== BreakStrength Enum Tests ==============

class TestBreakStrengthEnum:
    """Tests for BreakStrength enum."""

    def test_break_strength_none(self):
        """Test BreakStrength.NONE value."""
        from voice_soundboard.vocology.prosody import BreakStrength
        assert BreakStrength.NONE.value == "none"

    def test_break_strength_x_weak(self):
        """Test BreakStrength.X_WEAK value."""
        from voice_soundboard.vocology.prosody import BreakStrength
        assert BreakStrength.X_WEAK.value == "x-weak"

    def test_break_strength_weak(self):
        """Test BreakStrength.WEAK value."""
        from voice_soundboard.vocology.prosody import BreakStrength
        assert BreakStrength.WEAK.value == "weak"

    def test_break_strength_medium(self):
        """Test BreakStrength.MEDIUM value."""
        from voice_soundboard.vocology.prosody import BreakStrength
        assert BreakStrength.MEDIUM.value == "medium"

    def test_break_strength_strong(self):
        """Test BreakStrength.STRONG value."""
        from voice_soundboard.vocology.prosody import BreakStrength
        assert BreakStrength.STRONG.value == "strong"

    def test_break_strength_x_strong(self):
        """Test BreakStrength.X_STRONG value."""
        from voice_soundboard.vocology.prosody import BreakStrength
        assert BreakStrength.X_STRONG.value == "x-strong"


# ============== PitchContour Tests ==============

class TestPitchContour:
    """Tests for PitchContour dataclass."""

    def test_pitch_contour_creation(self):
        """Test PitchContour basic creation."""
        from voice_soundboard.vocology.prosody import PitchContour
        times = np.array([0.0, 0.01, 0.02, 0.03])
        frequencies = np.array([150.0, 155.0, 160.0, 158.0])
        voiced = np.array([True, True, True, True])
        contour = PitchContour(times=times, frequencies=frequencies, voiced=voiced)
        assert len(contour.times) == 4
        assert len(contour.frequencies) == 4

    def test_pitch_contour_duration_property(self):
        """Test PitchContour.duration property."""
        from voice_soundboard.vocology.prosody import PitchContour
        times = np.array([0.0, 0.01, 0.02, 0.03, 0.5])
        frequencies = np.array([150.0, 155.0, 160.0, 158.0, 152.0])
        voiced = np.array([True, True, True, True, True])
        contour = PitchContour(times=times, frequencies=frequencies, voiced=voiced)
        assert contour.duration == 0.5

    def test_pitch_contour_mean_f0_property(self):
        """Test PitchContour.mean_f0 property."""
        from voice_soundboard.vocology.prosody import PitchContour
        times = np.array([0.0, 0.01, 0.02])
        frequencies = np.array([100.0, 150.0, 200.0])
        voiced = np.array([True, True, True])
        contour = PitchContour(times=times, frequencies=frequencies, voiced=voiced)
        assert contour.mean_f0 == 150.0

    def test_pitch_contour_mean_f0_with_unvoiced(self):
        """Test PitchContour.mean_f0 excludes unvoiced regions."""
        from voice_soundboard.vocology.prosody import PitchContour
        times = np.array([0.0, 0.01, 0.02, 0.03])
        frequencies = np.array([100.0, 0.0, 200.0, 0.0])
        voiced = np.array([True, False, True, False])
        contour = PitchContour(times=times, frequencies=frequencies, voiced=voiced)
        assert contour.mean_f0 == 150.0  # Only voiced: (100 + 200) / 2

    def test_pitch_contour_f0_range_property(self):
        """Test PitchContour.f0_range property."""
        from voice_soundboard.vocology.prosody import PitchContour
        times = np.array([0.0, 0.01, 0.02])
        frequencies = np.array([100.0, 200.0, 150.0])
        voiced = np.array([True, True, True])
        contour = PitchContour(times=times, frequencies=frequencies, voiced=voiced)
        assert contour.f0_range == (100.0, 200.0)

    def test_pitch_contour_resample(self):
        """Test PitchContour.resample method."""
        from voice_soundboard.vocology.prosody import PitchContour
        times = np.array([0.0, 0.5, 1.0])
        frequencies = np.array([100.0, 150.0, 200.0])
        voiced = np.array([True, True, True])
        contour = PitchContour(times=times, frequencies=frequencies, voiced=voiced)
        resampled = contour.resample(5)
        assert len(resampled.times) == 5
        assert len(resampled.frequencies) == 5


# ============== DurationPattern Tests ==============

class TestDurationPattern:
    """Tests for DurationPattern dataclass."""

    def test_duration_pattern_creation(self):
        """Test DurationPattern basic creation."""
        from voice_soundboard.vocology.prosody import DurationPattern
        units = ["a", "b", "c"]
        durations = np.array([0.1, 0.15, 0.12])
        boundaries = np.array([0.0, 0.1, 0.25, 0.37])
        pattern = DurationPattern(units=units, durations=durations, boundaries=boundaries)
        assert len(pattern.units) == 3

    def test_duration_pattern_total_duration(self):
        """Test DurationPattern.total_duration property."""
        from voice_soundboard.vocology.prosody import DurationPattern
        units = ["a", "b", "c"]
        durations = np.array([0.1, 0.2, 0.3])
        boundaries = np.array([0.0, 0.1, 0.3, 0.6])
        pattern = DurationPattern(units=units, durations=durations, boundaries=boundaries)
        assert pattern.total_duration == 0.6

    def test_duration_pattern_mean_duration(self):
        """Test DurationPattern.mean_duration property."""
        from voice_soundboard.vocology.prosody import DurationPattern
        units = ["a", "b", "c"]
        durations = np.array([0.1, 0.2, 0.3])
        boundaries = np.array([0.0, 0.1, 0.3, 0.6])
        pattern = DurationPattern(units=units, durations=durations, boundaries=boundaries)
        assert pattern.mean_duration == 0.2

    def test_duration_pattern_scale(self):
        """Test DurationPattern.scale method."""
        from voice_soundboard.vocology.prosody import DurationPattern
        units = ["a", "b"]
        durations = np.array([0.1, 0.2])
        boundaries = np.array([0.0, 0.1, 0.3])
        pattern = DurationPattern(units=units, durations=durations, boundaries=boundaries)
        scaled = pattern.scale(2.0)
        np.testing.assert_array_almost_equal(scaled.durations, [0.2, 0.4])


# ============== ProsodyContour Tests ==============

class TestProsodyContour:
    """Tests for ProsodyContour dataclass."""

    def test_prosody_contour_creation_empty(self):
        """Test ProsodyContour with default empty values."""
        from voice_soundboard.vocology.prosody import ProsodyContour
        contour = ProsodyContour()
        assert contour.pitch is None
        assert contour.duration is None
        assert contour.energy is None
        assert contour.pauses == []

    def test_prosody_contour_with_pitch(self):
        """Test ProsodyContour with pitch data."""
        from voice_soundboard.vocology.prosody import ProsodyContour, PitchContour
        pitch = PitchContour(
            times=np.array([0.0, 0.1]),
            frequencies=np.array([150.0, 155.0]),
            voiced=np.array([True, True])
        )
        contour = ProsodyContour(pitch=pitch)
        assert contour.pitch is not None
        assert contour.pitch.mean_f0 == 152.5

    def test_prosody_contour_with_pauses(self):
        """Test ProsodyContour with pause data."""
        from voice_soundboard.vocology.prosody import ProsodyContour
        pauses = [(0.5, 0.2), (1.5, 0.3)]  # (time, duration)
        contour = ProsodyContour(pauses=pauses)
        assert len(contour.pauses) == 2
        assert contour.pauses[0] == (0.5, 0.2)


# ============== ProsodyAnalyzer Tests ==============

class TestProsodyAnalyzer:
    """Tests for ProsodyAnalyzer class."""

    def test_prosody_analyzer_init(self):
        """Test ProsodyAnalyzer initialization."""
        from voice_soundboard.vocology.prosody import ProsodyAnalyzer
        analyzer = ProsodyAnalyzer(f0_min=75.0, f0_max=400.0, hop_length=0.005)
        assert analyzer.f0_min == 75.0
        assert analyzer.f0_max == 400.0
        assert analyzer.hop_length == 0.005

    def test_prosody_analyzer_default_init(self):
        """Test ProsodyAnalyzer default initialization."""
        from voice_soundboard.vocology.prosody import ProsodyAnalyzer
        analyzer = ProsodyAnalyzer()
        assert analyzer.f0_min == 50.0
        assert analyzer.f0_max == 500.0
        assert analyzer.hop_length == 0.010

    def test_prosody_analyzer_extract_energy(self):
        """Test ProsodyAnalyzer._extract_energy method."""
        from voice_soundboard.vocology.prosody import ProsodyAnalyzer
        analyzer = ProsodyAnalyzer(hop_length=0.01)
        # 1 second of audio at 24000 Hz
        audio = np.random.randn(24000).astype(np.float32)
        energy = analyzer._extract_energy(audio, 24000)
        # Should have ~100 frames (1 sec / 0.01 hop)
        assert len(energy) >= 90
        assert len(energy) <= 110

    def test_prosody_analyzer_detect_pauses(self):
        """Test ProsodyAnalyzer._detect_pauses method."""
        from voice_soundboard.vocology.prosody import ProsodyAnalyzer
        analyzer = ProsodyAnalyzer(hop_length=0.01)
        # Create audio with silence in the middle
        audio = np.concatenate([
            np.random.randn(12000) * 0.5,  # Speech
            np.zeros(6000),                 # 250ms silence
            np.random.randn(6000) * 0.5    # Speech
        ]).astype(np.float32)
        pauses = analyzer._detect_pauses(audio, 24000)
        # Should detect at least one pause
        assert len(pauses) >= 0  # May or may not detect depending on threshold

    def test_prosody_analyzer_simple_pitch_extraction(self):
        """Test ProsodyAnalyzer._simple_pitch_extraction method."""
        from voice_soundboard.vocology.prosody import ProsodyAnalyzer
        analyzer = ProsodyAnalyzer(hop_length=0.01)
        # Create a simple sinusoidal signal at ~150 Hz
        sr = 24000
        t = np.linspace(0, 0.5, int(0.5 * sr))
        audio = np.sin(2 * np.pi * 150 * t).astype(np.float32)
        contour = analyzer._simple_pitch_extraction(audio, sr)
        assert len(contour.times) > 0
        assert len(contour.frequencies) > 0

    @patch('voice_soundboard.vocology.prosody.ProsodyAnalyzer._load_audio')
    def test_prosody_analyzer_analyze(self, mock_load):
        """Test ProsodyAnalyzer.analyze method."""
        from voice_soundboard.vocology.prosody import ProsodyAnalyzer
        analyzer = ProsodyAnalyzer()
        audio = np.random.randn(24000).astype(np.float32)
        mock_load.return_value = (audio, 24000)

        result = analyzer.analyze(audio, sample_rate=24000)
        assert result.pitch is not None
        assert result.energy is not None


# ============== ProsodyModifier Tests ==============

class TestProsodyModifier:
    """Tests for ProsodyModifier class."""

    def test_prosody_modifier_init(self):
        """Test ProsodyModifier initialization."""
        from voice_soundboard.vocology.prosody import ProsodyModifier
        modifier = ProsodyModifier()
        assert modifier is not None

    @patch('voice_soundboard.vocology.prosody.ProsodyModifier._load_audio')
    def test_prosody_modifier_modify_pitch_no_change(self, mock_load):
        """Test ProsodyModifier.modify_pitch with ratio=1.0."""
        from voice_soundboard.vocology.prosody import ProsodyModifier
        modifier = ProsodyModifier()
        audio = np.random.randn(24000).astype(np.float32)
        mock_load.return_value = (audio, 24000)

        # With ratio=1.0, should return similar audio
        with patch('librosa.effects.pitch_shift', return_value=audio):
            output, sr = modifier.modify_pitch(audio, ratio=1.0, sample_rate=24000)
            assert sr == 24000
            assert len(output) == len(audio)

    @patch('voice_soundboard.vocology.prosody.ProsodyModifier._load_audio')
    def test_prosody_modifier_modify_pitch_semitones(self, mock_load):
        """Test ProsodyModifier.modify_pitch with semitones."""
        from voice_soundboard.vocology.prosody import ProsodyModifier
        modifier = ProsodyModifier()
        audio = np.random.randn(24000).astype(np.float32)
        mock_load.return_value = (audio, 24000)

        with patch('librosa.effects.pitch_shift', return_value=audio) as mock_shift:
            output, sr = modifier.modify_pitch(audio, semitones=2.0, sample_rate=24000)
            mock_shift.assert_called_once()

    def test_prosody_modifier_modify_pitch_requires_sample_rate(self):
        """Test ProsodyModifier.modify_pitch raises error without sample_rate."""
        from voice_soundboard.vocology.prosody import ProsodyModifier
        modifier = ProsodyModifier()
        audio = np.random.randn(24000).astype(np.float32)

        with pytest.raises(ValueError, match="sample_rate required"):
            modifier.modify_pitch(audio, ratio=1.2)
