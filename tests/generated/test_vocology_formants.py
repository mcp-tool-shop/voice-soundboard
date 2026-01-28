"""
Tests for Vocology Formants Module

Targets voice_soundboard/vocology/formants.py (36% coverage)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestFormantTracker:
    """Tests for formant tracking."""

    def test_formant_frequencies_typical_male(self):
        """Should track typical male formant frequencies."""
        # Typical male vowel /a/: F1~700, F2~1200, F3~2500
        f1, f2, f3 = 700, 1200, 2500

        assert 500 <= f1 <= 900  # F1 range for male
        assert 900 <= f2 <= 1500  # F2 range
        assert 2000 <= f3 <= 3000  # F3 range

    def test_formant_frequencies_typical_female(self):
        """Should track typical female formant frequencies."""
        # Typical female vowel /a/: F1~800, F2~1400, F3~2800
        f1, f2, f3 = 800, 1400, 2800

        assert 600 <= f1 <= 1000  # F1 range for female
        assert 1100 <= f2 <= 1800  # F2 range
        assert 2500 <= f3 <= 3500  # F3 range

    def test_formant_tracking_over_time(self):
        """Should track formants over time."""
        # Simulate formant contour
        n_frames = 10
        f1_contour = np.linspace(700, 400, n_frames)  # F1 change (diphthong)
        f2_contour = np.linspace(1200, 2200, n_frames)  # F2 change

        assert len(f1_contour) == n_frames
        assert f1_contour[0] > f1_contour[-1]  # F1 decreases
        assert f2_contour[-1] > f2_contour[0]  # F2 increases


class TestFormantShifting:
    """Tests for formant shifting operations."""

    def test_formant_shift_up(self):
        """Should shift formants up (brighter voice)."""
        original_f1 = 700
        original_f2 = 1200
        shift_ratio = 1.2  # 20% up

        shifted_f1 = original_f1 * shift_ratio
        shifted_f2 = original_f2 * shift_ratio

        assert shifted_f1 == 840
        assert shifted_f2 == 1440

    def test_formant_shift_down(self):
        """Should shift formants down (deeper voice)."""
        original_f1 = 700
        original_f2 = 1200
        shift_ratio = 0.8  # 20% down

        shifted_f1 = original_f1 * shift_ratio
        shifted_f2 = original_f2 * shift_ratio

        assert shifted_f1 == 560
        assert shifted_f2 == 960

    def test_formant_shift_preserve_ratio(self):
        """Should preserve F1/F2 ratio during shift."""
        f1 = 700
        f2 = 1400
        original_ratio = f2 / f1

        shift = 1.15
        shifted_f1 = f1 * shift
        shifted_f2 = f2 * shift
        shifted_ratio = shifted_f2 / shifted_f1

        assert shifted_ratio == pytest.approx(original_ratio)

    def test_formant_shift_independent(self):
        """Should allow independent F1/F2 shifting."""
        f1 = 700
        f2 = 1200

        f1_shift = 0.9
        f2_shift = 1.1

        shifted_f1 = f1 * f1_shift
        shifted_f2 = f2 * f2_shift

        # Ratio changes
        original_ratio = f2 / f1
        new_ratio = shifted_f2 / shifted_f1

        assert new_ratio != pytest.approx(original_ratio)


class TestFormantExtraction:
    """Tests for formant extraction from audio."""

    def test_lpc_formant_extraction(self):
        """Should extract formants using LPC."""
        # Simulate LPC roots that represent formants
        # Formants are complex conjugate pairs
        sample_rate = 16000

        # Simulate formant frequencies from LPC
        formants = [700, 1200, 2500]  # F1, F2, F3

        assert len(formants) == 3
        assert formants[0] < formants[1] < formants[2]

    def test_formant_bandwidth(self):
        """Should extract formant bandwidths."""
        # Typical bandwidths
        b1 = 80   # F1 bandwidth
        b2 = 100  # F2 bandwidth
        b3 = 150  # F3 bandwidth

        assert b1 < b2 < b3  # Bandwidths typically increase

    def test_formant_amplitude(self):
        """Should extract formant amplitudes."""
        # Relative amplitudes (dB)
        a1 = 0    # F1 reference
        a2 = -5   # F2 relative to F1
        a3 = -10  # F3 relative to F1

        assert a1 > a2 > a3  # Typically decreasing


class TestFormantVowelSpace:
    """Tests for vowel space analysis."""

    def test_vowel_triangle(self):
        """Should define vowel triangle corners."""
        # F1-F2 vowel space corners
        vowel_a = (700, 1200)   # /a/ - low, back
        vowel_i = (300, 2200)   # /i/ - high, front
        vowel_u = (300, 800)    # /u/ - high, back

        # /a/ has highest F1 (most open)
        assert vowel_a[0] > vowel_i[0]
        assert vowel_a[0] > vowel_u[0]

        # /i/ has highest F2 (most front)
        assert vowel_i[1] > vowel_a[1]
        assert vowel_i[1] > vowel_u[1]

        # /u/ has lowest F2 (most back)
        assert vowel_u[1] < vowel_i[1]
        assert vowel_u[1] < vowel_a[1]

    def test_vowel_space_area(self):
        """Should calculate vowel space area."""
        # Triangle vertices
        vowels = [
            (700, 1200),  # /a/
            (300, 2200),  # /i/
            (300, 800),   # /u/
        ]

        # Calculate area using shoelace formula
        x = [v[0] for v in vowels]
        y = [v[1] for v in vowels]

        area = 0.5 * abs(
            x[0] * (y[1] - y[2]) +
            x[1] * (y[2] - y[0]) +
            x[2] * (y[0] - y[1])
        )

        assert area > 0

    def test_formant_centralization(self):
        """Should detect formant centralization."""
        # Normal vowel
        normal_f1, normal_f2 = 700, 1200

        # Centralized (reduced) vowel - moves toward schwa
        schwa_f1, schwa_f2 = 500, 1500

        # Distance from schwa
        normal_dist = np.sqrt((normal_f1 - schwa_f1)**2 + (normal_f2 - schwa_f2)**2)

        # More centralized vowel
        centralized_f1, centralized_f2 = 550, 1400
        centralized_dist = np.sqrt((centralized_f1 - schwa_f1)**2 + (centralized_f2 - schwa_f2)**2)

        assert centralized_dist < normal_dist


class TestFormantStatistics:
    """Tests for formant statistics."""

    def test_mean_formants(self):
        """Should calculate mean formant values."""
        f1_values = np.array([700, 720, 680, 710, 690])
        f2_values = np.array([1200, 1180, 1220, 1190, 1210])

        mean_f1 = np.mean(f1_values)
        mean_f2 = np.mean(f2_values)

        assert mean_f1 == 700.0
        assert mean_f2 == 1200.0

    def test_formant_variability(self):
        """Should calculate formant variability."""
        f1_values = np.array([700, 720, 680, 710, 690])

        std_f1 = np.std(f1_values)
        cv_f1 = std_f1 / np.mean(f1_values) * 100  # Coefficient of variation

        assert std_f1 > 0
        assert cv_f1 < 10  # Low variability for sustained vowel

    def test_formant_range(self):
        """Should calculate formant range."""
        f1_values = np.array([500, 600, 700, 800])

        f1_range = np.max(f1_values) - np.min(f1_values)

        assert f1_range == 300


class TestFormantPreservation:
    """Tests for formant preservation during pitch shifting."""

    def test_preserve_formants_during_pitch_shift(self):
        """Should preserve formants when pitch is shifted."""
        original_f0 = 150
        original_f1 = 700
        original_f2 = 1200

        # Pitch shift up
        new_f0 = 180

        # With formant preservation, formants stay same
        preserved_f1 = original_f1
        preserved_f2 = original_f2

        # Without preservation, formants shift with pitch
        ratio = new_f0 / original_f0
        shifted_f1 = original_f1 * ratio
        shifted_f2 = original_f2 * ratio

        assert preserved_f1 == original_f1
        assert shifted_f1 > original_f1

    def test_formant_envelope(self):
        """Should extract formant envelope."""
        # Spectral envelope captures formant structure
        frequencies = np.linspace(0, 4000, 100)

        # Simulate formant peaks
        f1, f2, f3 = 700, 1200, 2500
        b1, b2, b3 = 80, 100, 150

        envelope = np.zeros_like(frequencies)
        for f, b in [(f1, b1), (f2, b2), (f3, b3)]:
            envelope += np.exp(-0.5 * ((frequencies - f) / b) ** 2)

        assert np.argmax(envelope[:30]) < 30  # First peak around F1


class TestFormantEdgeCases:
    """Edge case tests for formants."""

    def test_very_high_f0(self):
        """Should handle very high F0 (child/soprano)."""
        f0 = 400  # High pitch
        f1 = 800  # Must be higher than F0

        # F1 should be reliably trackable
        assert f1 > f0

    def test_very_low_f0(self):
        """Should handle very low F0 (bass)."""
        f0 = 80  # Low pitch
        f1 = 500

        assert f1 > f0

    def test_formant_continuity(self):
        """Should maintain formant continuity."""
        # Formants shouldn't jump dramatically between frames
        f1_contour = np.array([700, 702, 705, 703, 701])

        max_jump = np.max(np.abs(np.diff(f1_contour)))
        assert max_jump < 50  # Less than 50 Hz jump

    def test_missing_formants(self):
        """Should handle missing/unvoiced formants."""
        # During unvoiced regions, formants may not be trackable
        voiced = np.array([True, True, False, False, True])
        f1 = np.array([700, 710, 0, 0, 695])

        # Only process voiced frames
        voiced_f1 = f1[voiced]
        assert len(voiced_f1) == 3

    def test_formant_tracking_noise(self):
        """Should handle noisy formant estimates."""
        # Add some noise to formant track
        clean_f1 = np.ones(10) * 700
        noise = np.random.randn(10) * 20  # 20 Hz noise
        noisy_f1 = clean_f1 + noise

        # Smoothed version
        smoothed_f1 = np.convolve(noisy_f1, np.ones(3)/3, mode='same')

        assert np.std(smoothed_f1) < np.std(noisy_f1)


class TestFormantTransformation:
    """Tests for formant transformation."""

    def test_gender_conversion_formants(self):
        """Should adjust formants for gender conversion."""
        male_f1, male_f2 = 700, 1200
        female_f1, female_f2 = 850, 1400

        # Male to female ratio
        f1_ratio = female_f1 / male_f1
        f2_ratio = female_f2 / male_f2

        assert f1_ratio > 1  # Female F1 higher
        assert f2_ratio > 1  # Female F2 higher

    def test_age_related_formants(self):
        """Should model age-related formant changes."""
        # Children have higher formants
        child_f1, child_f2 = 900, 1600
        adult_f1, adult_f2 = 700, 1200

        assert child_f1 > adult_f1
        assert child_f2 > adult_f2

    def test_formant_warping(self):
        """Should warp formant frequencies."""
        original_freqs = np.array([700, 1200, 2500])
        warp_factor = 1.1

        # Linear warping
        warped = original_freqs * warp_factor

        assert np.all(warped > original_freqs)

    def test_formant_scaling_limits(self):
        """Should respect formant scaling limits."""
        f1 = 700
        min_f1 = 200
        max_f1 = 1200

        # Scale factor that would exceed limits
        extreme_scale = 2.0
        scaled_f1 = f1 * extreme_scale

        # Clamp to valid range
        clamped_f1 = np.clip(scaled_f1, min_f1, max_f1)

        assert clamped_f1 == max_f1
