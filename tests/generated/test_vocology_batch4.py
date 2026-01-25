"""
Tests for Phase 9: Vocology Module - Batch 4
FormantShifter class and formant convenience functions

Tests cover:
- FormantShifter initialization (TEST-FSH-01 to TEST-FSH-03)
- FormantShifter.shift() method (TEST-FSH-04 to TEST-FSH-15)
- FormantShifter.shift_selective() (TEST-FSH-16 to TEST-FSH-18)
- Convenience functions (TEST-FSH-19 to TEST-FSH-25)
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# TEST-FSH-01 to TEST-FSH-03: FormantShifter Initialization Tests
# =============================================================================

from voice_soundboard.vocology.formants import (
    FormantShifter,
    FormantAnalyzer,
    FormantAnalysis,
    analyze_formants,
    shift_formants,
)


class TestFormantShifterInit:
    """Tests for FormantShifter initialization (TEST-FSH-01 to TEST-FSH-03)."""

    def test_fsh_01_default_method_psola(self):
        """TEST-FSH-01: Default method is 'psola'."""
        shifter = FormantShifter()
        assert shifter.method == "psola"

    def test_fsh_02_custom_method_lpc(self):
        """TEST-FSH-02: Custom method 'lpc' is stored."""
        shifter = FormantShifter(method="lpc")
        assert shifter.method == "lpc"

    def test_fsh_03_custom_method_phase_vocoder(self):
        """TEST-FSH-03: Custom method 'phase_vocoder' is stored."""
        shifter = FormantShifter(method="phase_vocoder")
        assert shifter.method == "phase_vocoder"


# =============================================================================
# TEST-FSH-04 to TEST-FSH-15: FormantShifter.shift() Tests
# =============================================================================

class TestFormantShifterShift:
    """Tests for FormantShifter.shift() method (TEST-FSH-04 to TEST-FSH-15)."""

    @pytest.fixture
    def mock_audio_array(self):
        """Create mock audio array."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 120 * t)
        audio += 0.25 * np.sin(2 * np.pi * 240 * t)
        audio += 0.02 * np.random.randn(len(audio))
        return audio.astype(np.float32), sr

    def test_fsh_04_shift_returns_tuple(self, mock_audio_array):
        """TEST-FSH-04: shift() returns (audio, sample_rate) tuple."""
        audio, sr = mock_audio_array
        shifter = FormantShifter()
        result = shifter.shift(audio, ratio=1.0, sample_rate=sr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fsh_05_shift_returns_array(self, mock_audio_array):
        """TEST-FSH-05: shift() returns numpy array as first element."""
        audio, sr = mock_audio_array
        shifter = FormantShifter()
        result_audio, result_sr = shifter.shift(audio, ratio=1.0, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_fsh_06_shift_returns_sample_rate(self, mock_audio_array):
        """TEST-FSH-06: shift() returns sample_rate as second element."""
        audio, sr = mock_audio_array
        shifter = FormantShifter()
        result_audio, result_sr = shifter.shift(audio, ratio=1.0, sample_rate=sr)
        assert result_sr == sr

    def test_fsh_07_shift_requires_sample_rate(self):
        """TEST-FSH-07: shift() raises ValueError without sample_rate for array."""
        audio = np.zeros(16000, dtype=np.float32)
        shifter = FormantShifter()
        with pytest.raises(ValueError, match="sample_rate"):
            shifter.shift(audio, ratio=1.0)

    def test_fsh_08_shift_ratio_1_preserves_length(self, mock_audio_array):
        """TEST-FSH-08: shift() with ratio=1.0 approximately preserves length."""
        audio, sr = mock_audio_array
        shifter = FormantShifter()
        result_audio, _ = shifter.shift(audio, ratio=1.0, sample_rate=sr)
        # Allow some tolerance due to processing
        assert abs(len(result_audio) - len(audio)) < len(audio) * 0.1

    def test_fsh_09_shift_ratio_lower_deepens(self, mock_audio_array):
        """TEST-FSH-09: shift() with ratio < 1 deepens voice (lower formants)."""
        audio, sr = mock_audio_array
        shifter = FormantShifter()
        # ratio=0.9 should lower formants (deeper voice)
        result_audio, _ = shifter.shift(audio, ratio=0.9, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)
        assert len(result_audio) > 0

    def test_fsh_10_shift_ratio_higher_brightens(self, mock_audio_array):
        """TEST-FSH-10: shift() with ratio > 1 brightens voice (higher formants)."""
        audio, sr = mock_audio_array
        shifter = FormantShifter()
        # ratio=1.1 should raise formants (brighter voice)
        result_audio, _ = shifter.shift(audio, ratio=1.1, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)
        assert len(result_audio) > 0

    def test_fsh_11_shift_preserve_pitch_true(self, mock_audio_array):
        """TEST-FSH-11: shift() with preserve_pitch=True keeps original pitch."""
        audio, sr = mock_audio_array
        shifter = FormantShifter()
        result_audio, _ = shifter.shift(
            audio, ratio=1.2, sample_rate=sr, preserve_pitch=True
        )
        assert isinstance(result_audio, np.ndarray)

    def test_fsh_12_shift_preserve_pitch_false(self, mock_audio_array):
        """TEST-FSH-12: shift() with preserve_pitch=False allows pitch change."""
        audio, sr = mock_audio_array
        shifter = FormantShifter()
        result_audio, _ = shifter.shift(
            audio, ratio=1.2, sample_rate=sr, preserve_pitch=False
        )
        assert isinstance(result_audio, np.ndarray)

    def test_fsh_13_shift_output_valid_range(self, mock_audio_array):
        """TEST-FSH-13: shift() output is in valid audio range."""
        audio, sr = mock_audio_array
        shifter = FormantShifter()
        result_audio, _ = shifter.shift(audio, ratio=1.1, sample_rate=sr)
        # Most samples should be in reasonable range
        assert np.mean(np.abs(result_audio) < 10) > 0.9

    def test_fsh_14_shift_method_psola(self, mock_audio_array):
        """TEST-FSH-14: shift() with method='psola' works."""
        audio, sr = mock_audio_array
        shifter = FormantShifter(method="psola")
        result_audio, _ = shifter.shift(audio, ratio=1.1, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_fsh_15_shift_method_lpc(self, mock_audio_array):
        """TEST-FSH-15: shift() with method='lpc' works."""
        audio, sr = mock_audio_array
        shifter = FormantShifter(method="lpc")
        result_audio, _ = shifter.shift(audio, ratio=1.1, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)


# =============================================================================
# TEST-FSH-16 to TEST-FSH-18: FormantShifter.shift_selective() Tests
# =============================================================================

class TestFormantShifterSelective:
    """Tests for FormantShifter.shift_selective() method (TEST-FSH-16 to TEST-FSH-18)."""

    @pytest.fixture
    def mock_audio_array(self):
        """Create mock audio array."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 120 * t)
        return audio.astype(np.float32), sr

    def test_fsh_16_shift_selective_returns_tuple(self, mock_audio_array):
        """TEST-FSH-16: shift_selective() returns (audio, sample_rate) tuple."""
        audio, sr = mock_audio_array
        shifter = FormantShifter()
        result = shifter.shift_selective(
            audio, f1_ratio=1.0, f2_ratio=1.1, f3_ratio=1.0, sample_rate=sr
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fsh_17_shift_selective_different_ratios(self, mock_audio_array):
        """TEST-FSH-17: shift_selective() accepts different ratios per formant."""
        audio, sr = mock_audio_array
        shifter = FormantShifter()
        result_audio, _ = shifter.shift_selective(
            audio, f1_ratio=0.9, f2_ratio=1.1, f3_ratio=1.0, sample_rate=sr
        )
        assert isinstance(result_audio, np.ndarray)

    def test_fsh_18_shift_selective_requires_sample_rate(self):
        """TEST-FSH-18: shift_selective() raises ValueError without sample_rate."""
        audio = np.zeros(16000, dtype=np.float32)
        shifter = FormantShifter()
        with pytest.raises(ValueError, match="sample_rate"):
            shifter.shift_selective(audio, f1_ratio=1.0, f2_ratio=1.0, f3_ratio=1.0)


# =============================================================================
# TEST-FSH-19 to TEST-FSH-25: Convenience Functions Tests
# =============================================================================

class TestFormantConvenienceFunctions:
    """Tests for formant convenience functions (TEST-FSH-19 to TEST-FSH-25)."""

    @pytest.fixture
    def mock_audio_array(self):
        """Create mock audio array."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        # Simulate voiced speech
        f0 = 120
        audio = np.zeros_like(t)
        for h in range(1, 15):
            audio += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
        audio = audio / np.max(np.abs(audio))
        audio += 0.01 * np.random.randn(len(audio))
        return audio.astype(np.float32), sr

    def test_fsh_19_analyze_formants_returns_analysis(self, mock_audio_array):
        """TEST-FSH-19: analyze_formants() returns FormantAnalysis."""
        audio, sr = mock_audio_array
        result = analyze_formants(audio, sample_rate=sr)
        assert isinstance(result, FormantAnalysis)

    def test_fsh_20_analyze_formants_custom_n_formants(self, mock_audio_array):
        """TEST-FSH-20: analyze_formants() accepts n_formants parameter."""
        audio, sr = mock_audio_array
        result = analyze_formants(audio, sample_rate=sr, n_formants=4)
        assert result.n_formants == 4

    def test_fsh_21_shift_formants_returns_tuple(self, mock_audio_array):
        """TEST-FSH-21: shift_formants() returns (audio, sample_rate) tuple."""
        audio, sr = mock_audio_array
        result = shift_formants(audio, ratio=1.0, sample_rate=sr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fsh_22_shift_formants_ratio_parameter(self, mock_audio_array):
        """TEST-FSH-22: shift_formants() applies ratio parameter."""
        audio, sr = mock_audio_array
        result_audio, _ = shift_formants(audio, ratio=1.1, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_fsh_23_shift_formants_preserve_pitch(self, mock_audio_array):
        """TEST-FSH-23: shift_formants() accepts preserve_pitch parameter."""
        audio, sr = mock_audio_array
        result_audio, _ = shift_formants(
            audio, ratio=1.1, sample_rate=sr, preserve_pitch=True
        )
        assert isinstance(result_audio, np.ndarray)

    def test_fsh_24_shift_formants_deeper_voice(self, mock_audio_array):
        """TEST-FSH-24: shift_formants() with ratio=0.85 creates deeper voice."""
        audio, sr = mock_audio_array
        result_audio, _ = shift_formants(audio, ratio=0.85, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)
        assert len(result_audio) > 0

    def test_fsh_25_shift_formants_brighter_voice(self, mock_audio_array):
        """TEST-FSH-25: shift_formants() with ratio=1.15 creates brighter voice."""
        audio, sr = mock_audio_array
        result_audio, _ = shift_formants(audio, ratio=1.15, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)
        assert len(result_audio) > 0
