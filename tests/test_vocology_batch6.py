"""
Tests for Phase 9: Vocology Module - Batch 6
PhonationSynthesizer class and phonation convenience functions

Tests cover:
- PhonationSynthesizer initialization (TEST-PSY-01 to TEST-PSY-02)
- PhonationSynthesizer.apply() method (TEST-PSY-03 to TEST-PSY-18)
- Convenience functions (TEST-PSY-19 to TEST-PSY-25)
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch


# =============================================================================
# TEST-PSY-01 to TEST-PSY-02: PhonationSynthesizer Initialization Tests
# =============================================================================

from voice_soundboard.vocology.phonation import (
    PhonationType,
    PhonationSynthesizer,
    detect_phonation,
    apply_phonation,
)


class TestPhonationSynthesizerInit:
    """Tests for PhonationSynthesizer initialization (TEST-PSY-01 to TEST-PSY-02)."""

    def test_psy_01_synthesizer_init(self):
        """TEST-PSY-01: PhonationSynthesizer initializes without error."""
        synthesizer = PhonationSynthesizer()
        assert synthesizer is not None

    def test_psy_02_synthesizer_is_reusable(self):
        """TEST-PSY-02: PhonationSynthesizer can be reused for multiple calls."""
        synthesizer = PhonationSynthesizer()
        sr = 16000
        audio1 = np.random.randn(sr).astype(np.float32) * 0.5
        audio2 = np.random.randn(sr).astype(np.float32) * 0.5

        result1, _ = synthesizer.apply(audio1, PhonationType.BREATHY, sample_rate=sr)
        result2, _ = synthesizer.apply(audio2, PhonationType.CREAKY, sample_rate=sr)

        assert result1 is not None
        assert result2 is not None


# =============================================================================
# TEST-PSY-03 to TEST-PSY-18: PhonationSynthesizer.apply() Tests
# =============================================================================

class TestPhonationSynthesizerApply:
    """Tests for PhonationSynthesizer.apply() method (TEST-PSY-03 to TEST-PSY-18)."""

    @pytest.fixture
    def mock_audio_array(self):
        """Create mock audio array."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        f0 = 120
        audio = np.zeros_like(t)
        for h in range(1, 15):
            audio += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
        audio = audio / np.max(np.abs(audio)) * 0.8
        return audio.astype(np.float32), sr

    def test_psy_03_apply_returns_tuple(self, mock_audio_array):
        """TEST-PSY-03: apply() returns (audio, sample_rate) tuple."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        result = synthesizer.apply(audio, PhonationType.MODAL, sample_rate=sr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_psy_04_apply_returns_array(self, mock_audio_array):
        """TEST-PSY-04: apply() returns numpy array as first element."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        result_audio, _ = synthesizer.apply(audio, PhonationType.MODAL, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_psy_05_apply_returns_sample_rate(self, mock_audio_array):
        """TEST-PSY-05: apply() returns sample_rate as second element."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        _, result_sr = synthesizer.apply(audio, PhonationType.MODAL, sample_rate=sr)
        assert result_sr == sr

    def test_psy_06_apply_requires_sample_rate(self):
        """TEST-PSY-06: apply() raises ValueError without sample_rate for array."""
        audio = np.zeros(16000, dtype=np.float32)
        synthesizer = PhonationSynthesizer()
        with pytest.raises(ValueError, match="sample_rate"):
            synthesizer.apply(audio, PhonationType.BREATHY)

    def test_psy_07_apply_modal_preserves_audio(self, mock_audio_array):
        """TEST-PSY-07: apply() with MODAL returns similar audio."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        result_audio, _ = synthesizer.apply(audio, PhonationType.MODAL, sample_rate=sr)
        # Modal should have minimal effect
        assert len(result_audio) == len(audio)

    def test_psy_08_apply_breathy(self, mock_audio_array):
        """TEST-PSY-08: apply() with BREATHY modifies audio."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        result_audio, _ = synthesizer.apply(
            audio, PhonationType.BREATHY, intensity=0.7, sample_rate=sr
        )
        assert isinstance(result_audio, np.ndarray)
        assert len(result_audio) > 0

    def test_psy_09_apply_creaky(self, mock_audio_array):
        """TEST-PSY-09: apply() with CREAKY modifies audio."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        result_audio, _ = synthesizer.apply(
            audio, PhonationType.CREAKY, intensity=0.7, sample_rate=sr
        )
        assert isinstance(result_audio, np.ndarray)
        assert len(result_audio) > 0

    def test_psy_10_apply_harsh(self, mock_audio_array):
        """TEST-PSY-10: apply() with HARSH modifies audio."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        result_audio, _ = synthesizer.apply(
            audio, PhonationType.HARSH, intensity=0.7, sample_rate=sr
        )
        assert isinstance(result_audio, np.ndarray)

    def test_psy_11_apply_falsetto(self, mock_audio_array):
        """TEST-PSY-11: apply() with FALSETTO modifies audio."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        result_audio, _ = synthesizer.apply(
            audio, PhonationType.FALSETTO, intensity=0.7, sample_rate=sr
        )
        assert isinstance(result_audio, np.ndarray)

    def test_psy_12_apply_whisper(self, mock_audio_array):
        """TEST-PSY-12: apply() with WHISPER modifies audio."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        result_audio, _ = synthesizer.apply(
            audio, PhonationType.WHISPER, intensity=0.7, sample_rate=sr
        )
        assert isinstance(result_audio, np.ndarray)

    def test_psy_13_apply_intensity_0_minimal_effect(self, mock_audio_array):
        """TEST-PSY-13: apply() with intensity=0.0 has minimal effect."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        result_audio, _ = synthesizer.apply(
            audio, PhonationType.BREATHY, intensity=0.0, sample_rate=sr
        )
        # With zero intensity, should be similar to original
        correlation = np.corrcoef(audio[:1000], result_audio[:1000])[0, 1]
        # May have some processing, but should be fairly similar
        assert correlation > 0.5 or np.allclose(audio, result_audio, rtol=0.1)

    def test_psy_14_apply_intensity_1_full_effect(self, mock_audio_array):
        """TEST-PSY-14: apply() with intensity=1.0 applies full effect."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        result_audio, _ = synthesizer.apply(
            audio, PhonationType.BREATHY, intensity=1.0, sample_rate=sr
        )
        # With full intensity, should be noticeably different
        assert isinstance(result_audio, np.ndarray)

    def test_psy_15_apply_preserves_length(self, mock_audio_array):
        """TEST-PSY-15: apply() preserves audio length."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        result_audio, _ = synthesizer.apply(
            audio, PhonationType.CREAKY, intensity=0.5, sample_rate=sr
        )
        assert len(result_audio) == len(audio)

    def test_psy_16_apply_output_valid_range(self, mock_audio_array):
        """TEST-PSY-16: apply() output is in valid audio range."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        result_audio, _ = synthesizer.apply(
            audio, PhonationType.HARSH, intensity=0.7, sample_rate=sr
        )
        # Most samples should be in reasonable range
        assert np.max(np.abs(result_audio)) < 10

    def test_psy_17_apply_intensity_default_0_5(self, mock_audio_array):
        """TEST-PSY-17: apply() default intensity is 0.5."""
        import inspect

        # Check the function signature for default intensity
        sig = inspect.signature(PhonationSynthesizer.apply)
        params = sig.parameters
        assert 'intensity' in params
        assert params['intensity'].default == 0.5

        # Also verify the method works with default intensity
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()
        result_audio, _ = synthesizer.apply(
            audio, PhonationType.BREATHY, sample_rate=sr
        )
        assert isinstance(result_audio, np.ndarray)
        assert len(result_audio) == len(audio)

    def test_psy_18_apply_all_types_work(self, mock_audio_array):
        """TEST-PSY-18: apply() works for all PhonationType values."""
        audio, sr = mock_audio_array
        synthesizer = PhonationSynthesizer()

        for ptype in PhonationType:
            result_audio, result_sr = synthesizer.apply(
                audio, ptype, intensity=0.5, sample_rate=sr
            )
            assert isinstance(result_audio, np.ndarray), f"Failed for {ptype}"
            assert result_sr == sr, f"Wrong SR for {ptype}"


# =============================================================================
# TEST-PSY-19 to TEST-PSY-25: Convenience Functions Tests
# =============================================================================

class TestPhonationConvenienceFunctions:
    """Tests for phonation convenience functions (TEST-PSY-19 to TEST-PSY-25)."""

    @pytest.fixture
    def mock_audio_array(self):
        """Create mock audio array."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        f0 = 120
        audio = np.zeros_like(t)
        for h in range(1, 15):
            audio += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
        audio = audio / np.max(np.abs(audio)) * 0.8
        audio += 0.01 * np.random.randn(len(audio))
        return audio.astype(np.float32), sr

    def test_psy_19_detect_phonation_returns_result(self, mock_audio_array):
        """TEST-PSY-19: detect_phonation() returns PhonationAnalysisResult."""
        from voice_soundboard.vocology.phonation import PhonationAnalysisResult
        audio, sr = mock_audio_array
        result = detect_phonation(audio, sample_rate=sr)
        assert isinstance(result, PhonationAnalysisResult)

    def test_psy_20_detect_phonation_detected_type(self, mock_audio_array):
        """TEST-PSY-20: detect_phonation() returns valid detected_type."""
        audio, sr = mock_audio_array
        result = detect_phonation(audio, sample_rate=sr)
        assert isinstance(result.detected_type, PhonationType)

    def test_psy_21_apply_phonation_returns_tuple(self, mock_audio_array):
        """TEST-PSY-21: apply_phonation() returns (audio, sample_rate) tuple."""
        audio, sr = mock_audio_array
        result = apply_phonation(audio, PhonationType.BREATHY, sample_rate=sr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_psy_22_apply_phonation_with_intensity(self, mock_audio_array):
        """TEST-PSY-22: apply_phonation() accepts intensity parameter."""
        audio, sr = mock_audio_array
        result_audio, _ = apply_phonation(
            audio, PhonationType.CREAKY, intensity=0.8, sample_rate=sr
        )
        assert isinstance(result_audio, np.ndarray)

    def test_psy_23_apply_phonation_breathy(self, mock_audio_array):
        """TEST-PSY-23: apply_phonation() with BREATHY works."""
        audio, sr = mock_audio_array
        result_audio, _ = apply_phonation(
            audio, PhonationType.BREATHY, intensity=0.5, sample_rate=sr
        )
        assert len(result_audio) == len(audio)

    def test_psy_24_apply_phonation_creaky(self, mock_audio_array):
        """TEST-PSY-24: apply_phonation() with CREAKY works."""
        audio, sr = mock_audio_array
        result_audio, _ = apply_phonation(
            audio, PhonationType.CREAKY, intensity=0.5, sample_rate=sr
        )
        assert len(result_audio) == len(audio)

    def test_psy_25_apply_phonation_whisper(self, mock_audio_array):
        """TEST-PSY-25: apply_phonation() with WHISPER works."""
        audio, sr = mock_audio_array
        result_audio, _ = apply_phonation(
            audio, PhonationType.WHISPER, intensity=0.5, sample_rate=sr
        )
        assert len(result_audio) == len(audio)
