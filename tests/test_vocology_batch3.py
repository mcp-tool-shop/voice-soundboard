"""
Tests for Phase 9: Vocology Module - Batch 3
FormantFrequencies, FormantAnalysis, FormantAnalyzer

Tests cover:
- FormantFrequencies dataclass (TEST-FMT-01 to TEST-FMT-08)
- FormantAnalysis dataclass (TEST-FMT-09 to TEST-FMT-14)
- FormantAnalyzer class (TEST-FMT-15 to TEST-FMT-25)
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch


# =============================================================================
# TEST-FMT-01 to TEST-FMT-08: FormantFrequencies Dataclass Tests
# =============================================================================

from voice_soundboard.vocology.formants import (
    FormantFrequencies,
    FormantAnalysis,
    FormantAnalyzer,
    FormantShifter,
    analyze_formants,
    shift_formants,
)


class TestFormantFrequencies:
    """Tests for FormantFrequencies dataclass (TEST-FMT-01 to TEST-FMT-08)."""

    @pytest.fixture
    def typical_formants(self):
        """Create typical adult male formant frequencies."""
        return FormantFrequencies(
            f1=500.0,
            f2=1500.0,
            f3=2500.0,
            f4=3500.0,
            f5=4500.0,
            bandwidths=[50.0, 60.0, 70.0, 80.0, 90.0],
        )

    def test_fmt_01_has_f1_field(self, typical_formants):
        """TEST-FMT-01: FormantFrequencies has f1 field."""
        assert hasattr(typical_formants, 'f1')
        assert typical_formants.f1 == 500.0

    def test_fmt_02_has_f2_field(self, typical_formants):
        """TEST-FMT-02: FormantFrequencies has f2 field."""
        assert hasattr(typical_formants, 'f2')
        assert typical_formants.f2 == 1500.0

    def test_fmt_03_has_f3_field(self, typical_formants):
        """TEST-FMT-03: FormantFrequencies has f3 field."""
        assert hasattr(typical_formants, 'f3')
        assert typical_formants.f3 == 2500.0

    def test_fmt_04_has_f4_field(self, typical_formants):
        """TEST-FMT-04: FormantFrequencies has f4 field."""
        assert hasattr(typical_formants, 'f4')
        assert typical_formants.f4 == 3500.0

    def test_fmt_05_f5_optional(self):
        """TEST-FMT-05: FormantFrequencies f5 is optional (defaults to None)."""
        formants = FormantFrequencies(f1=500, f2=1500, f3=2500, f4=3500)
        assert formants.f5 is None

    def test_fmt_06_as_list_returns_list(self, typical_formants):
        """TEST-FMT-06: as_list property returns list of formant values."""
        result = typical_formants.as_list
        assert isinstance(result, list)
        assert result == [500.0, 1500.0, 2500.0, 3500.0, 4500.0]

    def test_fmt_07_as_list_without_f5(self):
        """TEST-FMT-07: as_list returns 4 elements when f5 is None."""
        formants = FormantFrequencies(f1=500, f2=1500, f3=2500, f4=3500)
        result = formants.as_list
        assert len(result) == 4
        assert result == [500, 1500, 2500, 3500]

    def test_fmt_08_singer_formant_detection(self):
        """TEST-FMT-08: singer_formant_present() detects clustered formants."""
        # Singer's formant: F3, F4, F5 cluster around 3000 Hz
        singer = FormantFrequencies(
            f1=500, f2=1500, f3=2900, f4=3000, f5=3100
        )
        assert singer.singer_formant_present() == True

        # Normal speech: formants spread out
        normal = FormantFrequencies(
            f1=500, f2=1500, f3=2500, f4=3500, f5=4500
        )
        assert normal.singer_formant_present() == False


# =============================================================================
# TEST-FMT-09 to TEST-FMT-14: FormantAnalysis Dataclass Tests
# =============================================================================

class TestFormantAnalysis:
    """Tests for FormantAnalysis dataclass (TEST-FMT-09 to TEST-FMT-14)."""

    @pytest.fixture
    def mock_analysis(self):
        """Create a mock FormantAnalysis."""
        # 10 frames, 5 formants each
        formants = np.array([
            [500, 1500, 2500, 3500, 4500],
            [510, 1510, 2510, 3510, 4510],
            [490, 1490, 2490, 3490, 4490],
            [500, 1500, 2500, 3500, 4500],
            [505, 1505, 2505, 3505, 4505],
            [495, 1495, 2495, 3495, 4495],
            [500, 1500, 2500, 3500, 4500],
            [500, 1500, 2500, 3500, 4500],
            [510, 1510, 2510, 3510, 4510],
            [490, 1490, 2490, 3490, 4490],
        ], dtype=np.float32)

        mean_formants = FormantFrequencies(
            f1=500.0, f2=1500.0, f3=2500.0, f4=3500.0, f5=4500.0
        )

        return FormantAnalysis(
            formants=formants,
            mean_formants=mean_formants,
            std_formants=[7.0, 7.0, 7.0, 7.0, 7.0],
            sample_rate=16000,
            hop_length=160,
        )

    def test_fmt_09_has_formants_array(self, mock_analysis):
        """TEST-FMT-09: FormantAnalysis has formants array."""
        assert hasattr(mock_analysis, 'formants')
        assert isinstance(mock_analysis.formants, np.ndarray)

    def test_fmt_10_has_mean_formants(self, mock_analysis):
        """TEST-FMT-10: FormantAnalysis has mean_formants."""
        assert hasattr(mock_analysis, 'mean_formants')
        assert isinstance(mock_analysis.mean_formants, FormantFrequencies)

    def test_fmt_11_n_frames_property(self, mock_analysis):
        """TEST-FMT-11: n_frames property returns correct frame count."""
        assert mock_analysis.n_frames == 10

    def test_fmt_12_n_formants_property(self, mock_analysis):
        """TEST-FMT-12: n_formants property returns correct formant count."""
        assert mock_analysis.n_formants == 5

    def test_fmt_13_get_frame_returns_frequencies(self, mock_analysis):
        """TEST-FMT-13: get_frame() returns FormantFrequencies for frame."""
        frame = mock_analysis.get_frame(0)
        assert isinstance(frame, FormantFrequencies)
        assert frame.f1 == 500.0

    def test_fmt_14_get_frame_different_frames(self, mock_analysis):
        """TEST-FMT-14: get_frame() returns different values for different frames."""
        frame0 = mock_analysis.get_frame(0)
        frame1 = mock_analysis.get_frame(1)
        assert frame0.f1 != frame1.f1  # 500 vs 510


# =============================================================================
# TEST-FMT-15 to TEST-FMT-25: FormantAnalyzer Class Tests
# =============================================================================

class TestFormantAnalyzer:
    """Tests for FormantAnalyzer class (TEST-FMT-15 to TEST-FMT-25)."""

    @pytest.fixture
    def mock_audio_array(self):
        """Create mock audio with formant-like spectral content."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))

        # Simulate voiced speech with formant-like structure
        f0 = 120  # Fundamental
        audio = np.zeros_like(t)

        # Add harmonics weighted by formant envelope
        formant_freqs = [500, 1500, 2500, 3500]
        formant_bw = 100

        for h in range(1, 30):
            harm_freq = f0 * h
            # Weight by proximity to formants
            weight = 0
            for ff in formant_freqs:
                weight += np.exp(-((harm_freq - ff) ** 2) / (2 * formant_bw ** 2))
            weight = max(weight, 0.01)
            audio += weight * np.sin(2 * np.pi * harm_freq * t) / h

        audio = audio / np.max(np.abs(audio))
        audio += 0.01 * np.random.randn(len(audio))

        return audio.astype(np.float32), sr

    def test_fmt_15_default_n_formants(self):
        """TEST-FMT-15: Default n_formants is 5."""
        analyzer = FormantAnalyzer()
        assert analyzer.n_formants == 5

    def test_fmt_16_default_lpc_order(self):
        """TEST-FMT-16: Default lpc_order is 2 * n_formants + 2."""
        analyzer = FormantAnalyzer()
        expected = 2 * 5 + 2  # 12
        assert analyzer.lpc_order == expected

    def test_fmt_17_custom_n_formants(self):
        """TEST-FMT-17: Custom n_formants is stored correctly."""
        analyzer = FormantAnalyzer(n_formants=4)
        assert analyzer.n_formants == 4

    def test_fmt_18_pre_emphasis_default(self):
        """TEST-FMT-18: Default pre_emphasis is 0.97."""
        analyzer = FormantAnalyzer()
        assert analyzer.pre_emphasis == 0.97

    def test_fmt_19_frame_length_default(self):
        """TEST-FMT-19: Default frame_length is 0.025 seconds."""
        analyzer = FormantAnalyzer()
        assert analyzer.frame_length == 0.025

    def test_fmt_20_hop_length_default(self):
        """TEST-FMT-20: Default hop_length is 0.010 seconds."""
        analyzer = FormantAnalyzer()
        assert analyzer.hop_length == 0.010

    def test_fmt_21_analyze_returns_analysis(self, mock_audio_array):
        """TEST-FMT-21: analyze() returns FormantAnalysis."""
        audio, sr = mock_audio_array
        analyzer = FormantAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert isinstance(result, FormantAnalysis)

    def test_fmt_22_analyze_requires_sample_rate(self):
        """TEST-FMT-22: analyze() raises ValueError without sample_rate for array."""
        audio = np.zeros(16000, dtype=np.float32)
        analyzer = FormantAnalyzer()
        with pytest.raises(ValueError, match="sample_rate"):
            analyzer.analyze(audio)

    def test_fmt_23_analyze_formants_positive(self, mock_audio_array):
        """TEST-FMT-23: analyze() returns positive formant frequencies."""
        audio, sr = mock_audio_array
        analyzer = FormantAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert result.mean_formants.f1 >= 0
        assert result.mean_formants.f2 >= 0

    def test_fmt_24_analyze_f1_less_than_f2(self, mock_audio_array):
        """TEST-FMT-24: F1 is less than F2 (normal ordering)."""
        audio, sr = mock_audio_array
        analyzer = FormantAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        # In well-ordered formants, F1 < F2 < F3 < F4
        if result.mean_formants.f1 > 0 and result.mean_formants.f2 > 0:
            assert result.mean_formants.f1 < result.mean_formants.f2

    def test_fmt_25_analyze_stores_sample_rate(self, mock_audio_array):
        """TEST-FMT-25: analyze() stores sample_rate in result."""
        audio, sr = mock_audio_array
        analyzer = FormantAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert result.sample_rate == sr
