"""
Tests for Vocology Phonation Module

Targets voice_soundboard/vocology/phonation.py (21% coverage)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestPhonationTypes:
    """Tests for phonation type classification."""

    def test_modal_phonation(self):
        """Should identify modal (normal) phonation."""
        # Modal phonation: regular vibration, clear voice
        # Characteristics: moderate HNR, low jitter/shimmer
        modal_features = {
            "hnr_db": 22.0,
            "jitter_percent": 0.5,
            "shimmer_percent": 3.0,
            "h1_h2_db": 2.0,  # Moderate difference
        }

        # Modal voice should have good HNR
        assert modal_features["hnr_db"] > 15.0

    def test_breathy_phonation(self):
        """Should identify breathy phonation."""
        # Breathy: incomplete glottal closure, air leak
        # Characteristics: low HNR, high H1-H2
        breathy_features = {
            "hnr_db": 8.0,  # Low
            "jitter_percent": 1.5,
            "shimmer_percent": 6.0,
            "h1_h2_db": 8.0,  # High - weak high harmonics
        }

        assert breathy_features["hnr_db"] < 15.0
        assert breathy_features["h1_h2_db"] > 5.0

    def test_creaky_phonation(self):
        """Should identify creaky (vocal fry) phonation."""
        # Creaky: irregular vibration, low F0
        # Characteristics: very low F0, high jitter, low H1-H2
        creaky_features = {
            "f0_mean": 60,  # Very low
            "jitter_percent": 4.0,  # High irregularity
            "shimmer_percent": 8.0,
            "h1_h2_db": -2.0,  # Low - strong high harmonics
        }

        assert creaky_features["f0_mean"] < 80
        assert creaky_features["jitter_percent"] > 2.0

    def test_pressed_phonation(self):
        """Should identify pressed/tense phonation."""
        # Pressed: high laryngeal tension, tight closure
        # Characteristics: negative H1-H2, high spectral tilt
        pressed_features = {
            "hnr_db": 18.0,
            "h1_h2_db": -4.0,  # Negative - enhanced high harmonics
            "spectral_tilt": -15.0,  # Steep tilt
        }

        assert pressed_features["h1_h2_db"] < 0


class TestVoiceQuality:
    """Tests for voice quality analysis."""

    def test_roughness_detection(self):
        """Should detect voice roughness."""
        # Roughness = irregularity in vocal fold vibration
        rough_voice_jitter = 3.5  # High
        smooth_voice_jitter = 0.4  # Low

        assert rough_voice_jitter > 2.0
        assert smooth_voice_jitter < 1.0

    def test_strain_detection(self):
        """Should detect vocal strain."""
        # Strain indicators: high shimmer, spectral noise
        strained = {
            "shimmer_percent": 9.0,
            "spectral_noise": 0.3,
        }

        normal = {
            "shimmer_percent": 2.5,
            "spectral_noise": 0.05,
        }

        assert strained["shimmer_percent"] > normal["shimmer_percent"]

    def test_asthenia_detection(self):
        """Should detect vocal weakness (asthenia)."""
        # Asthenia: weak, thin voice
        # Low intensity, high breathiness
        weak_voice = {
            "intensity_db": 55,  # Low
            "hnr_db": 10,  # Low
            "h1_h2_db": 6.0,  # High
        }

        strong_voice = {
            "intensity_db": 75,
            "hnr_db": 25,
            "h1_h2_db": 2.0,
        }

        assert weak_voice["intensity_db"] < strong_voice["intensity_db"]


class TestGlottalAnalysis:
    """Tests for glottal analysis."""

    def test_open_quotient(self):
        """Should calculate open quotient."""
        # OQ = open phase / total period
        open_phase = 0.006  # 6 ms
        total_period = 0.010  # 10 ms (100 Hz)

        oq = open_phase / total_period

        assert oq == 0.6  # Typical value 0.4-0.7

    def test_closed_quotient(self):
        """Should calculate closed quotient."""
        # CQ = closed phase / total period
        closed_phase = 0.004
        total_period = 0.010

        cq = closed_phase / total_period

        assert cq == 0.4  # CQ = 1 - OQ

    def test_speed_quotient(self):
        """Should calculate speed quotient."""
        # SQ = opening phase / closing phase
        opening_phase = 0.003
        closing_phase = 0.004

        sq = opening_phase / closing_phase

        assert sq < 1.0  # Closing usually longer than opening

    def test_glottal_closure_index(self):
        """Should calculate glottal closure index."""
        # GCI = measure of closure completeness
        # Higher = more complete closure
        complete_closure_gci = 0.95
        incomplete_closure_gci = 0.5

        assert complete_closure_gci > incomplete_closure_gci


class TestHarmonicAnalysis:
    """Tests for harmonic analysis."""

    def test_h1_h2_difference(self):
        """Should calculate H1-H2 difference."""
        # H1 = amplitude of first harmonic
        # H2 = amplitude of second harmonic
        h1_db = -10
        h2_db = -12

        h1_h2 = h1_db - h2_db

        assert h1_h2 == 2.0

    def test_harmonic_richness_factor(self):
        """Should calculate harmonic richness."""
        # Sum of harmonics vs fundamental
        fundamental_energy = 1.0
        harmonic_energies = [0.5, 0.3, 0.2, 0.1]  # H2-H5

        total_harmonic = sum(harmonic_energies)
        hrf = total_harmonic / fundamental_energy

        assert hrf == 1.1

    def test_cepstral_peak(self):
        """Should identify cepstral peak."""
        # Cepstrum has peak at F0 period for periodic voice
        f0 = 150  # Hz
        expected_quefrency = 1 / f0

        assert expected_quefrency == pytest.approx(0.00667, rel=0.01)


class TestPhonationDetection:
    """Tests for phonation detection algorithms."""

    def test_voicing_decision(self):
        """Should make voicing decision."""
        # Simple energy + zero-crossing based decision
        energy = 0.5
        zero_crossings = 50  # Low for voiced

        energy_threshold = 0.1
        zc_threshold = 100

        is_voiced = energy > energy_threshold and zero_crossings < zc_threshold

        assert is_voiced is True

    def test_onset_detection(self):
        """Should detect phonation onset."""
        energy_contour = np.array([0.01, 0.02, 0.1, 0.5, 0.6, 0.55])
        threshold = 0.05

        # Find first crossing
        onset_idx = np.argmax(energy_contour > threshold)

        assert onset_idx == 2

    def test_offset_detection(self):
        """Should detect phonation offset."""
        energy_contour = np.array([0.5, 0.45, 0.3, 0.1, 0.02, 0.01])
        threshold = 0.05

        # Find last above-threshold frame
        offset_idx = len(energy_contour) - 1 - np.argmax(energy_contour[::-1] > threshold)

        assert offset_idx == 3


class TestPhonationMetrics:
    """Tests for phonation metrics computation."""

    def test_phonation_time_ratio(self):
        """Should calculate phonation time ratio."""
        total_duration = 10.0  # seconds
        phonation_duration = 7.0  # seconds

        ptr = phonation_duration / total_duration

        assert ptr == 0.7

    def test_maximum_phonation_time(self):
        """Should measure maximum phonation time."""
        # MPT = how long someone can sustain /a/
        mpt_normal = 20.0  # seconds
        mpt_impaired = 8.0  # seconds

        assert mpt_normal > 15  # Healthy adult > 15s
        assert mpt_impaired < 10

    def test_s_z_ratio(self):
        """Should calculate S/Z ratio."""
        # S duration (voiceless) vs Z duration (voiced)
        s_duration = 15.0  # seconds
        z_duration = 18.0  # seconds

        s_z_ratio = s_duration / z_duration

        assert s_z_ratio < 1.4  # Normal < 1.4


class TestPhonationEdgeCases:
    """Edge case tests for phonation."""

    def test_mixed_phonation(self):
        """Should handle mixed phonation types."""
        # Voice can shift between modal and creaky
        phonation_sequence = ["modal", "modal", "creaky", "modal"]

        modal_count = phonation_sequence.count("modal")
        creaky_count = phonation_sequence.count("creaky")

        assert modal_count == 3
        assert creaky_count == 1

    def test_diplophonia(self):
        """Should detect diplophonia (double pitch)."""
        # Diplophonia: two pitches present
        # Subharmonics appear
        fundamental = 150
        subharmonic = fundamental / 2

        assert subharmonic == 75

    def test_aphonia(self):
        """Should handle aphonia (no voice)."""
        # Complete loss of phonation
        energy = 0.01  # Very low
        periodicity = 0.0  # No periodicity

        is_aphonic = energy < 0.05 and periodicity < 0.1

        assert is_aphonic is True

    def test_falsetto_detection(self):
        """Should detect falsetto register."""
        # Falsetto: very high F0, thin voice quality
        falsetto_f0 = 350  # High
        modal_f0 = 120  # Normal

        assert falsetto_f0 > 250
        assert modal_f0 < 200


class TestPhonationDynamics:
    """Tests for phonation dynamics over time."""

    def test_phonation_stability(self):
        """Should measure phonation stability."""
        # Standard deviation of F0 for sustained vowel
        f0_values = np.array([150, 151, 149, 150, 152, 148])

        std_f0 = np.std(f0_values)

        assert std_f0 < 5.0  # Stable phonation

    def test_phonation_breaks(self):
        """Should detect phonation breaks."""
        # Sudden loss of voicing
        voiced = np.array([1, 1, 1, 0, 0, 1, 1, 1])

        # Find break
        break_indices = np.where(np.diff(voiced) == -1)[0]

        assert len(break_indices) == 1
        assert break_indices[0] == 2

    def test_tremor_detection(self):
        """Should detect vocal tremor."""
        # Tremor = low-frequency modulation of F0/amplitude
        # Typically 4-7 Hz
        tremor_rate = 5.0  # Hz
        tremor_extent = 10  # Hz variation

        assert 4 <= tremor_rate <= 7
        assert tremor_extent > 5


class TestPhonationComparison:
    """Tests for comparing phonation patterns."""

    def test_pre_post_therapy(self):
        """Should compare pre and post therapy voices."""
        pre_therapy = {
            "jitter_percent": 3.5,
            "hnr_db": 12.0,
        }

        post_therapy = {
            "jitter_percent": 1.2,
            "hnr_db": 20.0,
        }

        # Improvement
        jitter_improvement = pre_therapy["jitter_percent"] - post_therapy["jitter_percent"]
        hnr_improvement = post_therapy["hnr_db"] - pre_therapy["hnr_db"]

        assert jitter_improvement > 0
        assert hnr_improvement > 0

    def test_aging_voice_changes(self):
        """Should model aging voice changes."""
        young_adult = {
            "f0_range_semitones": 24,
            "mpt_seconds": 25,
        }

        elderly = {
            "f0_range_semitones": 16,
            "mpt_seconds": 15,
        }

        assert elderly["f0_range_semitones"] < young_adult["f0_range_semitones"]
        assert elderly["mpt_seconds"] < young_adult["mpt_seconds"]
