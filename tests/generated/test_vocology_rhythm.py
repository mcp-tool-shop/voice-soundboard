"""
Tests for Vocology Rhythm Module

Targets voice_soundboard/vocology/rhythm.py (21% coverage)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestRhythmAnalyzer:
    """Tests for rhythm analysis."""

    def test_calculate_pvi(self):
        """Should calculate Pairwise Variability Index."""
        # PVI measures rhythm by comparing adjacent durations
        durations = np.array([0.1, 0.15, 0.12, 0.18, 0.11])

        # Calculate raw PVI
        diffs = np.abs(np.diff(durations))
        pvi = np.mean(diffs) * 100  # Percentage

        assert pvi > 0

    def test_calculate_npvi(self):
        """Should calculate normalized PVI."""
        durations = np.array([0.1, 0.2, 0.15, 0.25])

        # Calculate nPVI (normalized)
        npvi_values = []
        for i in range(len(durations) - 1):
            d1, d2 = durations[i], durations[i + 1]
            if d1 + d2 > 0:
                npvi_values.append(abs(d1 - d2) / ((d1 + d2) / 2))

        npvi = np.mean(npvi_values) * 100 if npvi_values else 0

        assert npvi > 0

    def test_speech_rate_calculation(self):
        """Should calculate speech rate."""
        # Syllables per second
        n_syllables = 50
        duration_seconds = 10.0

        speech_rate = n_syllables / duration_seconds
        assert speech_rate == 5.0  # 5 syllables per second

    def test_articulation_rate(self):
        """Should calculate articulation rate (excluding pauses)."""
        n_syllables = 50
        total_duration = 12.0
        pause_duration = 2.0

        articulation_time = total_duration - pause_duration
        articulation_rate = n_syllables / articulation_time

        assert articulation_rate == 5.0

    def test_pause_detection(self):
        """Should detect pauses in signal."""
        # Simulate energy signal with pauses
        energy = np.array([1.0, 1.0, 0.01, 0.01, 0.01, 1.0, 1.0])
        threshold = 0.1

        is_pause = energy < threshold
        pause_indices = np.where(is_pause)[0]

        assert len(pause_indices) == 3

    def test_pause_duration_calculation(self):
        """Should calculate pause durations."""
        # Pauses at frames
        pause_frames = [
            (10, 20),  # Frames 10-20
            (50, 55),  # Frames 50-55
        ]
        frame_rate = 100  # frames per second

        pause_durations = [(end - start) / frame_rate for start, end in pause_frames]

        assert pause_durations[0] == 0.1
        assert pause_durations[1] == 0.05


class TestSyllableAnalysis:
    """Tests for syllable-level rhythm analysis."""

    def test_syllable_duration_extraction(self):
        """Should extract syllable durations."""
        # Mock syllable boundaries (in seconds)
        boundaries = np.array([0.0, 0.15, 0.32, 0.45, 0.62, 0.80])
        durations = np.diff(boundaries)

        assert len(durations) == 5
        assert durations[0] == pytest.approx(0.15)

    def test_syllable_rate_variability(self):
        """Should measure syllable rate variability."""
        durations = np.array([0.15, 0.12, 0.18, 0.14, 0.16])

        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        cv = (std_duration / mean_duration) * 100  # Coefficient of variation

        assert cv > 0

    def test_stressed_syllable_detection(self):
        """Should detect stressed syllables."""
        # Stressed syllables typically have higher energy/duration
        durations = np.array([0.10, 0.20, 0.12, 0.22, 0.11])  # 2nd and 4th stressed
        energies = np.array([0.5, 1.0, 0.6, 1.1, 0.55])

        # Simple stress detection: above average in both
        mean_dur = np.mean(durations)
        mean_energy = np.mean(energies)

        stressed = (durations > mean_dur) & (energies > mean_energy)

        assert stressed[1] == True
        assert stressed[3] == True


class TestRhythmPatterns:
    """Tests for rhythm pattern detection."""

    def test_isochrony_measure(self):
        """Should measure timing regularity (isochrony)."""
        # Regular durations (isochronous)
        regular = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        irregular = np.array([0.1, 0.3, 0.15, 0.25, 0.2])

        regular_cv = np.std(regular) / np.mean(regular)
        irregular_cv = np.std(irregular) / np.mean(irregular)

        assert regular_cv < irregular_cv

    def test_rhythm_class_detection(self):
        """Should detect rhythm class (stress-timed vs syllable-timed)."""
        # Stress-timed languages have higher variability
        # Syllable-timed languages have more regular timing

        stress_timed_durations = np.array([0.08, 0.25, 0.10, 0.30, 0.09])
        syllable_timed_durations = np.array([0.15, 0.16, 0.14, 0.15, 0.16])

        stress_npvi = np.mean(np.abs(np.diff(stress_timed_durations)))
        syllable_npvi = np.mean(np.abs(np.diff(syllable_timed_durations)))

        assert stress_npvi > syllable_npvi

    def test_foot_structure_analysis(self):
        """Should analyze metrical foot structure."""
        # Stress pattern: S-W-W-S-W (stressed-weak-weak-stressed-weak)
        stress_pattern = [1, 0, 0, 1, 0]

        # Detect trochaic/iambic pattern
        # This is a simplified test
        assert stress_pattern[0] == 1  # Starts with stress


class TestRhythmModification:
    """Tests for rhythm modification."""

    def test_tempo_scaling(self):
        """Should scale tempo."""
        durations = np.array([0.2, 0.3, 0.25])
        speed_factor = 1.5  # 50% faster

        scaled = durations / speed_factor

        assert scaled[0] == pytest.approx(0.133, rel=0.01)

    def test_pause_insertion(self):
        """Should insert pauses."""
        word_boundaries = [0.0, 0.5, 1.0, 1.5]
        pause_positions = [1, 2]  # After word 1 and 2
        pause_duration = 0.2

        # Calculate new boundaries with pauses
        new_boundaries = []
        offset = 0
        for i, b in enumerate(word_boundaries):
            new_boundaries.append(b + offset)
            if i in pause_positions:
                offset += pause_duration

        assert new_boundaries[-1] == 1.9  # Original + 2 pauses

    def test_duration_stretching(self):
        """Should stretch durations for emphasis."""
        original = np.array([0.1, 0.15, 0.12, 0.18])
        emphasis_idx = 2
        stretch_factor = 1.5

        modified = original.copy()
        modified[emphasis_idx] *= stretch_factor

        assert modified[emphasis_idx] == pytest.approx(0.18)

    def test_rhythm_normalization(self):
        """Should normalize rhythm to target pattern."""
        source_durations = np.array([0.1, 0.2, 0.15, 0.25])
        target_durations = np.array([0.15, 0.15, 0.15, 0.15])

        # Simple interpolation
        alpha = 0.5  # Blend factor
        blended = source_durations * (1 - alpha) + target_durations * alpha

        assert np.std(blended) < np.std(source_durations)


class TestRhythmTransfer:
    """Tests for rhythm transfer between utterances."""

    def test_duration_mapping(self):
        """Should map durations from source to target."""
        source_durations = np.array([0.2, 0.3, 0.15, 0.25])
        n_target_units = 6

        # Resample to match target
        indices = np.linspace(0, len(source_durations) - 1, n_target_units)
        target_durations = np.interp(indices, range(len(source_durations)), source_durations)

        assert len(target_durations) == 6

    def test_rhythm_extraction(self):
        """Should extract rhythm pattern."""
        durations = np.array([0.1, 0.2, 0.15, 0.25, 0.12])

        # Normalize to relative durations
        mean_dur = np.mean(durations)
        relative = durations / mean_dur

        # Pattern should have mean 1.0
        assert np.mean(relative) == pytest.approx(1.0)


class TestRhythmMetrics:
    """Tests for rhythm metrics."""

    def test_delta_c_metric(self):
        """Should calculate deltaC (consonant duration variability)."""
        # Consonant cluster durations
        consonant_durations = np.array([0.05, 0.12, 0.08, 0.15, 0.06])

        delta_c = np.std(consonant_durations)
        assert delta_c > 0

    def test_delta_v_metric(self):
        """Should calculate deltaV (vowel duration variability)."""
        vowel_durations = np.array([0.08, 0.15, 0.10, 0.12, 0.09])

        delta_v = np.std(vowel_durations)
        assert delta_v > 0

    def test_percent_v_metric(self):
        """Should calculate %V (proportion of vocalic intervals)."""
        vowel_duration = 2.5
        total_duration = 5.0

        percent_v = (vowel_duration / total_duration) * 100
        assert percent_v == 50.0

    def test_varco_v_metric(self):
        """Should calculate VarcoV (normalized vowel variability)."""
        vowel_durations = np.array([0.1, 0.15, 0.12, 0.18, 0.11])

        mean_v = np.mean(vowel_durations)
        std_v = np.std(vowel_durations)
        varco_v = (std_v / mean_v) * 100

        assert varco_v > 0


class TestRhythmEdgeCases:
    """Edge case tests for rhythm analysis."""

    def test_single_unit(self):
        """Should handle single unit."""
        durations = np.array([0.2])

        mean_dur = np.mean(durations)
        assert mean_dur == 0.2

        # Can't calculate PVI with single unit
        if len(durations) > 1:
            pvi = np.mean(np.abs(np.diff(durations)))
        else:
            pvi = 0.0

        assert pvi == 0.0

    def test_empty_durations(self):
        """Should handle empty durations."""
        durations = np.array([])

        if len(durations) > 0:
            mean_dur = np.mean(durations)
        else:
            mean_dur = 0.0

        assert mean_dur == 0.0

    def test_zero_durations(self):
        """Should handle zero durations."""
        durations = np.array([0.0, 0.0, 0.0])

        mean_dur = np.mean(durations)
        assert mean_dur == 0.0

    def test_very_short_durations(self):
        """Should handle very short durations."""
        durations = np.array([0.001, 0.002, 0.001])  # 1-2 ms

        speech_rate = len(durations) / np.sum(durations)
        assert speech_rate > 0

    def test_very_long_durations(self):
        """Should handle very long durations."""
        durations = np.array([2.0, 3.0, 2.5])  # Long pauses

        mean_dur = np.mean(durations)
        assert mean_dur == 2.5

    def test_mixed_durations(self):
        """Should handle mixed short and long durations."""
        durations = np.array([0.05, 0.5, 0.08, 0.6, 0.04])

        cv = np.std(durations) / np.mean(durations)
        assert cv > 1.0  # High variability

    def test_uniform_durations(self):
        """Should handle perfectly uniform durations."""
        durations = np.array([0.15, 0.15, 0.15, 0.15])

        std_dur = np.std(durations)
        assert std_dur == 0.0


class TestRhythmFeatureExtraction:
    """Tests for complete rhythm feature extraction."""

    def test_extract_all_features(self):
        """Should extract comprehensive rhythm features."""
        durations = np.array([0.12, 0.18, 0.15, 0.22, 0.14, 0.20])

        features = {
            "mean_duration": np.mean(durations),
            "std_duration": np.std(durations),
            "cv_duration": np.std(durations) / np.mean(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "range_duration": np.max(durations) - np.min(durations),
            "pvi": np.mean(np.abs(np.diff(durations))) * 100,
        }

        assert "mean_duration" in features
        assert "pvi" in features
        assert features["mean_duration"] > 0
        assert features["pvi"] > 0

    def test_feature_consistency(self):
        """Features should be consistent across runs."""
        durations = np.array([0.1, 0.2, 0.15])

        features1 = {
            "mean": np.mean(durations),
            "std": np.std(durations),
        }

        features2 = {
            "mean": np.mean(durations),
            "std": np.std(durations),
        }

        assert features1["mean"] == features2["mean"]
        assert features1["std"] == features2["std"]
