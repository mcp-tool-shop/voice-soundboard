"""
Tests for Phase 9: Vocology Module - Batch 2
VoiceQualityAnalyzer class and convenience functions

Tests cover:
- VoiceQualityAnalyzer initialization (TEST-VQA-01 to TEST-VQA-05)
- VoiceQualityAnalyzer.analyze() method (TEST-VQA-06 to TEST-VQA-15)
- Convenience functions (TEST-VQA-16 to TEST-VQA-25)
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# TEST-VQA-01 to TEST-VQA-05: VoiceQualityAnalyzer Initialization Tests
# =============================================================================

from voice_soundboard.vocology.parameters import (
    VoiceQualityAnalyzer,
    VoiceQualityMetrics,
    JitterType,
    ShimmerType,
    analyze_voice_quality,
    get_jitter,
    get_shimmer,
    get_hnr,
    get_cpp,
)


class TestVoiceQualityAnalyzerInit:
    """Tests for VoiceQualityAnalyzer initialization (TEST-VQA-01 to TEST-VQA-05)."""

    def test_vqa_01_default_f0_min(self):
        """TEST-VQA-01: Default f0_min is 50.0 Hz."""
        analyzer = VoiceQualityAnalyzer()
        assert analyzer.f0_min == 50.0

    def test_vqa_02_default_f0_max(self):
        """TEST-VQA-02: Default f0_max is 500.0 Hz."""
        analyzer = VoiceQualityAnalyzer()
        assert analyzer.f0_max == 500.0

    def test_vqa_03_default_frame_length(self):
        """TEST-VQA-03: Default frame_length is 0.025 seconds."""
        analyzer = VoiceQualityAnalyzer()
        assert analyzer.frame_length == 0.025

    def test_vqa_04_default_hop_length(self):
        """TEST-VQA-04: Default hop_length is 0.010 seconds."""
        analyzer = VoiceQualityAnalyzer()
        assert analyzer.hop_length == 0.010

    def test_vqa_05_custom_parameters(self):
        """TEST-VQA-05: Custom parameters are stored correctly."""
        analyzer = VoiceQualityAnalyzer(
            f0_min=75.0,
            f0_max=400.0,
            frame_length=0.030,
            hop_length=0.015,
        )
        assert analyzer.f0_min == 75.0
        assert analyzer.f0_max == 400.0
        assert analyzer.frame_length == 0.030
        assert analyzer.hop_length == 0.015


# =============================================================================
# TEST-VQA-06 to TEST-VQA-15: VoiceQualityAnalyzer.analyze() Tests
# =============================================================================

class TestVoiceQualityAnalyzerAnalyze:
    """Tests for VoiceQualityAnalyzer.analyze() method (TEST-VQA-06 to TEST-VQA-15)."""

    @pytest.fixture
    def mock_audio_array(self):
        """Create a mock audio array (sine wave simulating voiced speech)."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        # Fundamental frequency around 120 Hz
        f0 = 120
        audio = 0.5 * np.sin(2 * np.pi * f0 * t)
        # Add some harmonics
        audio += 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        audio += 0.125 * np.sin(2 * np.pi * 3 * f0 * t)
        # Add slight noise
        audio += 0.02 * np.random.randn(len(audio))
        return audio.astype(np.float32), sr

    def test_vqa_06_analyze_returns_metrics(self, mock_audio_array):
        """TEST-VQA-06: analyze() returns VoiceQualityMetrics instance."""
        audio, sr = mock_audio_array
        analyzer = VoiceQualityAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert isinstance(result, VoiceQualityMetrics)

    def test_vqa_07_analyze_requires_sample_rate_for_array(self):
        """TEST-VQA-07: analyze() raises ValueError when sample_rate missing for array."""
        audio = np.zeros(16000, dtype=np.float32)
        analyzer = VoiceQualityAnalyzer()
        with pytest.raises(ValueError, match="sample_rate"):
            analyzer.analyze(audio)

    def test_vqa_08_analyze_f0_mean_reasonable(self, mock_audio_array):
        """TEST-VQA-08: analyze() returns reasonable f0_mean."""
        audio, sr = mock_audio_array
        analyzer = VoiceQualityAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        # F0 should be around 120 Hz (our test signal)
        assert 80 < result.f0_mean < 200

    def test_vqa_09_analyze_jitter_positive(self, mock_audio_array):
        """TEST-VQA-09: analyze() returns non-negative jitter values."""
        audio, sr = mock_audio_array
        analyzer = VoiceQualityAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert result.jitter_local >= 0
        assert result.jitter_rap >= 0
        assert result.jitter_ppq5 >= 0

    def test_vqa_10_analyze_shimmer_positive(self, mock_audio_array):
        """TEST-VQA-10: analyze() returns non-negative shimmer values."""
        audio, sr = mock_audio_array
        analyzer = VoiceQualityAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert result.shimmer_local >= 0
        assert result.shimmer_apq3 >= 0
        assert result.shimmer_apq5 >= 0

    def test_vqa_11_analyze_hnr_positive(self, mock_audio_array):
        """TEST-VQA-11: analyze() returns positive HNR for clean signal."""
        audio, sr = mock_audio_array
        analyzer = VoiceQualityAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert result.hnr > 0

    def test_vqa_12_analyze_cpp_positive(self, mock_audio_array):
        """TEST-VQA-12: analyze() returns positive CPP for periodic signal."""
        audio, sr = mock_audio_array
        analyzer = VoiceQualityAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert result.cpp > 0

    def test_vqa_13_analyze_duration_correct(self, mock_audio_array):
        """TEST-VQA-13: analyze() returns correct duration."""
        audio, sr = mock_audio_array
        analyzer = VoiceQualityAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        expected_duration = len(audio) / sr
        assert abs(result.duration - expected_duration) < 0.01

    def test_vqa_14_analyze_voiced_fraction_range(self, mock_audio_array):
        """TEST-VQA-14: analyze() returns voiced_fraction in [0, 1]."""
        audio, sr = mock_audio_array
        analyzer = VoiceQualityAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert 0 <= result.voiced_fraction <= 1

    def test_vqa_15_analyze_spectral_centroid_positive(self, mock_audio_array):
        """TEST-VQA-15: analyze() returns positive spectral_centroid."""
        audio, sr = mock_audio_array
        analyzer = VoiceQualityAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert result.spectral_centroid > 0


# =============================================================================
# TEST-VQA-16 to TEST-VQA-25: Convenience Functions Tests
# =============================================================================

class TestVoiceQualityConvenienceFunctions:
    """Tests for convenience functions (TEST-VQA-16 to TEST-VQA-25)."""

    @pytest.fixture
    def mock_audio_array(self):
        """Create a mock audio array."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 120 * t)
        audio += 0.02 * np.random.randn(len(audio))
        return audio.astype(np.float32), sr

    def test_vqa_16_analyze_voice_quality_returns_metrics(self, mock_audio_array):
        """TEST-VQA-16: analyze_voice_quality() returns VoiceQualityMetrics."""
        audio, sr = mock_audio_array
        result = analyze_voice_quality(audio, sample_rate=sr)
        assert isinstance(result, VoiceQualityMetrics)

    def test_vqa_17_get_jitter_returns_float(self, mock_audio_array):
        """TEST-VQA-17: get_jitter() returns float."""
        audio, sr = mock_audio_array
        result = get_jitter(audio, sample_rate=sr)
        assert isinstance(result, float)

    def test_vqa_18_get_jitter_local_type(self, mock_audio_array):
        """TEST-VQA-18: get_jitter() with LOCAL returns local jitter."""
        audio, sr = mock_audio_array
        result = get_jitter(audio, sample_rate=sr, jitter_type=JitterType.LOCAL)
        metrics = analyze_voice_quality(audio, sample_rate=sr)
        assert result == metrics.jitter_local

    def test_vqa_19_get_jitter_rap_type(self, mock_audio_array):
        """TEST-VQA-19: get_jitter() with RAP returns RAP jitter."""
        audio, sr = mock_audio_array
        result = get_jitter(audio, sample_rate=sr, jitter_type=JitterType.RAP)
        metrics = analyze_voice_quality(audio, sample_rate=sr)
        assert result == metrics.jitter_rap

    def test_vqa_20_get_shimmer_returns_float(self, mock_audio_array):
        """TEST-VQA-20: get_shimmer() returns float."""
        audio, sr = mock_audio_array
        result = get_shimmer(audio, sample_rate=sr)
        assert isinstance(result, float)

    def test_vqa_21_get_shimmer_local_type(self, mock_audio_array):
        """TEST-VQA-21: get_shimmer() with LOCAL returns local shimmer."""
        audio, sr = mock_audio_array
        result = get_shimmer(audio, sample_rate=sr, shimmer_type=ShimmerType.LOCAL)
        metrics = analyze_voice_quality(audio, sample_rate=sr)
        assert result == metrics.shimmer_local

    def test_vqa_22_get_shimmer_apq5_type(self, mock_audio_array):
        """TEST-VQA-22: get_shimmer() with APQ5 returns APQ5 shimmer."""
        audio, sr = mock_audio_array
        result = get_shimmer(audio, sample_rate=sr, shimmer_type=ShimmerType.APQ5)
        metrics = analyze_voice_quality(audio, sample_rate=sr)
        assert result == metrics.shimmer_apq5

    def test_vqa_23_get_hnr_returns_float(self, mock_audio_array):
        """TEST-VQA-23: get_hnr() returns float."""
        audio, sr = mock_audio_array
        result = get_hnr(audio, sample_rate=sr)
        assert isinstance(result, float)

    def test_vqa_24_get_cpp_returns_float(self, mock_audio_array):
        """TEST-VQA-24: get_cpp() returns float."""
        audio, sr = mock_audio_array
        result = get_cpp(audio, sample_rate=sr)
        assert isinstance(result, float)

    def test_vqa_25_get_hnr_matches_metrics(self, mock_audio_array):
        """TEST-VQA-25: get_hnr() matches metrics.hnr."""
        audio, sr = mock_audio_array
        result = get_hnr(audio, sample_rate=sr)
        metrics = analyze_voice_quality(audio, sample_rate=sr)
        assert result == metrics.hnr
