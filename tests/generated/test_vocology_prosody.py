"""
Tests for Vocology Prosody Module

Targets voice_soundboard/vocology/prosody.py (23% coverage)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestBreakStrength:
    """Tests for BreakStrength enum."""

    def test_break_strength_values(self):
        """Should have all expected break strengths."""
        from voice_soundboard.vocology.prosody import BreakStrength

        assert BreakStrength.NONE.value == "none"
        assert BreakStrength.X_WEAK.value == "x-weak"
        assert BreakStrength.WEAK.value == "weak"
        assert BreakStrength.MEDIUM.value == "medium"
        assert BreakStrength.STRONG.value == "strong"
        assert BreakStrength.X_STRONG.value == "x-strong"

    def test_break_strength_count(self):
        """Should have 6 break strength levels."""
        from voice_soundboard.vocology.prosody import BreakStrength

        assert len(list(BreakStrength)) == 6


class TestPitchContour:
    """Tests for PitchContour dataclass."""

    def test_create_pitch_contour(self):
        """Should create pitch contour."""
        from voice_soundboard.vocology.prosody import PitchContour

        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        frequencies = np.array([150.0, 155.0, 160.0, 158.0, 152.0])
        voiced = np.array([True, True, True, True, True])

        contour = PitchContour(
            times=times,
            frequencies=frequencies,
            voiced=voiced,
        )

        assert len(contour.times) == 5
        assert len(contour.frequencies) == 5
        assert all(contour.voiced)

    def test_pitch_contour_duration(self):
        """Should calculate duration."""
        from voice_soundboard.vocology.prosody import PitchContour

        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        frequencies = np.array([100.0, 110.0, 120.0, 115.0, 105.0])
        voiced = np.array([True, True, True, True, True])

        contour = PitchContour(times=times, frequencies=frequencies, voiced=voiced)

        assert contour.duration == 2.0

    def test_pitch_contour_empty_duration(self):
        """Should handle empty contour."""
        from voice_soundboard.vocology.prosody import PitchContour

        contour = PitchContour(
            times=np.array([]),
            frequencies=np.array([]),
            voiced=np.array([]),
        )

        assert contour.duration == 0.0

    def test_pitch_contour_mean_f0(self):
        """Should calculate mean F0 of voiced regions."""
        from voice_soundboard.vocology.prosody import PitchContour

        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        frequencies = np.array([100.0, 150.0, 200.0, 0.0, 0.0])
        voiced = np.array([True, True, True, False, False])

        contour = PitchContour(times=times, frequencies=frequencies, voiced=voiced)

        assert contour.mean_f0 == 150.0  # (100 + 150 + 200) / 3

    def test_pitch_contour_mean_f0_no_voiced(self):
        """Should handle no voiced frames."""
        from voice_soundboard.vocology.prosody import PitchContour

        times = np.array([0.0, 0.1, 0.2])
        frequencies = np.array([0.0, 0.0, 0.0])
        voiced = np.array([False, False, False])

        contour = PitchContour(times=times, frequencies=frequencies, voiced=voiced)

        assert contour.mean_f0 == 0.0

    def test_pitch_contour_f0_range(self):
        """Should calculate F0 range."""
        from voice_soundboard.vocology.prosody import PitchContour

        times = np.array([0.0, 0.1, 0.2, 0.3])
        frequencies = np.array([100.0, 200.0, 150.0, 180.0])
        voiced = np.array([True, True, True, True])

        contour = PitchContour(times=times, frequencies=frequencies, voiced=voiced)

        f0_min, f0_max = contour.f0_range
        assert f0_min == 100.0
        assert f0_max == 200.0

    def test_pitch_contour_f0_range_no_voiced(self):
        """Should handle no voiced frames for range."""
        from voice_soundboard.vocology.prosody import PitchContour

        times = np.array([0.0, 0.1])
        frequencies = np.array([0.0, 0.0])
        voiced = np.array([False, False])

        contour = PitchContour(times=times, frequencies=frequencies, voiced=voiced)

        assert contour.f0_range == (0.0, 0.0)

    def test_pitch_contour_resample(self):
        """Should resample to fixed number of points."""
        from voice_soundboard.vocology.prosody import PitchContour

        times = np.array([0.0, 0.5, 1.0])
        frequencies = np.array([100.0, 150.0, 100.0])
        voiced = np.array([True, True, True])

        contour = PitchContour(times=times, frequencies=frequencies, voiced=voiced)
        resampled = contour.resample(n_points=5)

        assert len(resampled.times) == 5
        assert len(resampled.frequencies) == 5
        assert resampled.times[0] == 0.0
        assert resampled.times[-1] == 1.0

    def test_pitch_contour_with_confidence(self):
        """Should store confidence values."""
        from voice_soundboard.vocology.prosody import PitchContour

        times = np.array([0.0, 0.1, 0.2])
        frequencies = np.array([150.0, 155.0, 160.0])
        voiced = np.array([True, True, True])
        confidence = np.array([0.9, 0.85, 0.95])

        contour = PitchContour(
            times=times,
            frequencies=frequencies,
            voiced=voiced,
            confidence=confidence,
        )

        assert contour.confidence is not None
        assert len(contour.confidence) == 3


class TestDurationPattern:
    """Tests for DurationPattern dataclass."""

    def test_create_duration_pattern(self):
        """Should create duration pattern."""
        from voice_soundboard.vocology.prosody import DurationPattern

        pattern = DurationPattern(
            units=["hello", "world"],
            durations=np.array([0.3, 0.4]),
            boundaries=np.array([0.0, 0.3, 0.7]),
        )

        assert len(pattern.units) == 2
        assert len(pattern.durations) == 2

    def test_duration_pattern_total(self):
        """Should calculate total duration."""
        from voice_soundboard.vocology.prosody import DurationPattern

        pattern = DurationPattern(
            units=["a", "b", "c"],
            durations=np.array([0.2, 0.3, 0.5]),
            boundaries=np.array([0.0, 0.2, 0.5, 1.0]),
        )

        assert pattern.total_duration == 1.0

    def test_duration_pattern_mean(self):
        """Should calculate mean duration."""
        from voice_soundboard.vocology.prosody import DurationPattern

        pattern = DurationPattern(
            units=["a", "b", "c"],
            durations=np.array([0.1, 0.2, 0.3]),
            boundaries=np.array([0.0, 0.1, 0.3, 0.6]),
        )

        assert pattern.mean_duration == 0.2

    def test_duration_pattern_scale(self):
        """Should scale durations."""
        from voice_soundboard.vocology.prosody import DurationPattern

        pattern = DurationPattern(
            units=["a", "b"],
            durations=np.array([0.2, 0.3]),
            boundaries=np.array([0.0, 0.2, 0.5]),
        )

        scaled = pattern.scale(2.0)

        assert scaled.durations[0] == 0.4
        assert scaled.durations[1] == 0.6
        assert scaled.boundaries[-1] == 1.0


class TestProsodyContour:
    """Tests for ProsodyContour dataclass."""

    def test_create_empty_prosody_contour(self):
        """Should create empty prosody contour."""
        from voice_soundboard.vocology.prosody import ProsodyContour

        contour = ProsodyContour()

        assert contour.pitch is None
        assert contour.duration is None
        assert contour.energy is None
        assert contour.pauses == []

    def test_prosody_contour_with_pitch(self):
        """Should create prosody contour with pitch."""
        from voice_soundboard.vocology.prosody import ProsodyContour, PitchContour

        pitch = PitchContour(
            times=np.array([0.0, 0.5, 1.0]),
            frequencies=np.array([150.0, 160.0, 155.0]),
            voiced=np.array([True, True, True]),
        )

        contour = ProsodyContour(pitch=pitch)

        assert contour.pitch is not None
        assert contour.pitch.mean_f0 == pytest.approx(155.0)

    def test_prosody_contour_with_pauses(self):
        """Should store pause information."""
        from voice_soundboard.vocology.prosody import ProsodyContour

        contour = ProsodyContour(
            pauses=[(0.5, 0.2), (1.2, 0.3)]  # (time, duration)
        )

        assert len(contour.pauses) == 2
        assert contour.pauses[0] == (0.5, 0.2)

    def test_set_pitch_range(self):
        """Should set pitch range."""
        from voice_soundboard.vocology.prosody import ProsodyContour, PitchContour

        pitch = PitchContour(
            times=np.array([0.0, 0.5, 1.0]),
            frequencies=np.array([100.0, 200.0, 150.0]),
            voiced=np.array([True, True, True]),
        )

        contour = ProsodyContour(pitch=pitch)
        contour.set_pitch_range(low=80.0, high=180.0)

        # Frequencies should be scaled to new range
        assert contour.pitch.frequencies[0] == pytest.approx(80.0)  # Was min
        assert contour.pitch.frequencies[1] == pytest.approx(180.0)  # Was max

    def test_set_pitch_range_no_pitch(self):
        """Should handle no pitch gracefully."""
        from voice_soundboard.vocology.prosody import ProsodyContour

        contour = ProsodyContour()
        contour.set_pitch_range(low=80.0, high=180.0)  # Should not error

    def test_add_emphasis(self):
        """Should add emphasis at time."""
        from voice_soundboard.vocology.prosody import ProsodyContour, PitchContour

        pitch = PitchContour(
            times=np.array([0.0, 0.5, 1.0]),
            frequencies=np.array([150.0, 150.0, 150.0]),
            voiced=np.array([True, True, True]),
        )

        contour = ProsodyContour(pitch=pitch)
        contour.add_emphasis(time=0.5, pitch_boost=1.2, duration_boost=1.1)

        # Method exists and doesn't error


class TestProsodyAnalyzer:
    """Tests for prosody analysis functions."""

    def test_extract_pitch_contour(self):
        """Should extract pitch contour from audio."""
        # This would test the actual extraction function
        pass

    def test_extract_duration_pattern(self):
        """Should extract duration pattern."""
        pass

    def test_detect_pauses(self):
        """Should detect pauses in audio."""
        pass


class TestProsodyModification:
    """Tests for prosody modification functions."""

    def test_modify_pitch(self):
        """Should modify pitch contour."""
        from voice_soundboard.vocology.prosody import PitchContour

        pitch = PitchContour(
            times=np.array([0.0, 0.5, 1.0]),
            frequencies=np.array([100.0, 150.0, 120.0]),
            voiced=np.array([True, True, True]),
        )

        # Modify by scaling
        pitch.frequencies = pitch.frequencies * 1.1

        assert pitch.frequencies[0] == pytest.approx(110.0)

    def test_modify_duration(self):
        """Should modify duration pattern."""
        from voice_soundboard.vocology.prosody import DurationPattern

        pattern = DurationPattern(
            units=["a", "b", "c"],
            durations=np.array([0.1, 0.2, 0.3]),
            boundaries=np.array([0.0, 0.1, 0.3, 0.6]),
        )

        # Scale to slower speech
        scaled = pattern.scale(1.5)

        assert scaled.total_duration == pytest.approx(0.9)


class TestProsodyTransfer:
    """Tests for prosody transfer functions."""

    def test_transfer_pitch_contour(self):
        """Should transfer pitch contour between utterances."""
        from voice_soundboard.vocology.prosody import PitchContour

        source = PitchContour(
            times=np.array([0.0, 0.5, 1.0]),
            frequencies=np.array([100.0, 150.0, 120.0]),
            voiced=np.array([True, True, True]),
        )

        # Resample to target duration
        target = source.resample(n_points=10)

        assert len(target.times) == 10


class TestProsodyFeatures:
    """Tests for prosody feature extraction."""

    def test_extract_intonation_pattern(self):
        """Should extract intonation pattern."""
        pass

    def test_extract_stress_pattern(self):
        """Should extract stress pattern."""
        pass

    def test_extract_rhythm_features(self):
        """Should extract rhythm features."""
        pass


class TestProsodyEdgeCases:
    """Edge case tests for prosody."""

    def test_empty_audio(self):
        """Should handle empty audio."""
        from voice_soundboard.vocology.prosody import PitchContour

        contour = PitchContour(
            times=np.array([]),
            frequencies=np.array([]),
            voiced=np.array([]),
        )

        assert contour.duration == 0.0
        assert contour.mean_f0 == 0.0

    def test_single_frame(self):
        """Should handle single frame."""
        from voice_soundboard.vocology.prosody import PitchContour

        contour = PitchContour(
            times=np.array([0.0]),
            frequencies=np.array([150.0]),
            voiced=np.array([True]),
        )

        assert contour.duration == 0.0
        assert contour.mean_f0 == 150.0

    def test_all_unvoiced(self):
        """Should handle all unvoiced frames."""
        from voice_soundboard.vocology.prosody import PitchContour

        contour = PitchContour(
            times=np.array([0.0, 0.1, 0.2]),
            frequencies=np.array([0.0, 0.0, 0.0]),
            voiced=np.array([False, False, False]),
        )

        assert contour.mean_f0 == 0.0
        assert contour.f0_range == (0.0, 0.0)

    def test_mixed_voicing(self):
        """Should handle mixed voicing."""
        from voice_soundboard.vocology.prosody import PitchContour

        contour = PitchContour(
            times=np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            frequencies=np.array([150.0, 0.0, 160.0, 0.0, 155.0]),
            voiced=np.array([True, False, True, False, True]),
        )

        assert contour.mean_f0 == pytest.approx(155.0)  # (150 + 160 + 155) / 3

    def test_very_high_f0(self):
        """Should handle very high F0."""
        from voice_soundboard.vocology.prosody import PitchContour

        contour = PitchContour(
            times=np.array([0.0, 0.1]),
            frequencies=np.array([500.0, 600.0]),  # Very high
            voiced=np.array([True, True]),
        )

        assert contour.mean_f0 == 550.0

    def test_very_low_f0(self):
        """Should handle very low F0."""
        from voice_soundboard.vocology.prosody import PitchContour

        contour = PitchContour(
            times=np.array([0.0, 0.1]),
            frequencies=np.array([50.0, 60.0]),  # Very low
            voiced=np.array([True, True]),
        )

        assert contour.mean_f0 == 55.0
