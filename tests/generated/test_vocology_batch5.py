"""
Tests for Phase 9: Vocology Module - Batch 5
PhonationType enum, PhonationAnalyzer class, PHONATION_PARAMS constant

Tests cover:
- PhonationType enum values (TEST-PHN-01 to TEST-PHN-06)
- PHONATION_PARAMS constant (TEST-PHN-07 to TEST-PHN-12)
- PhonationAnalysisResult dataclass (TEST-PHN-13 to TEST-PHN-16)
- PhonationAnalyzer class (TEST-PHN-17 to TEST-PHN-25)
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch


# =============================================================================
# TEST-PHN-01 to TEST-PHN-06: PhonationType Enum Tests
# =============================================================================

from voice_soundboard.vocology.phonation import (
    PhonationType,
    PHONATION_PARAMS,
    PhonationAnalyzer,
    PhonationSynthesizer,
    PhonationAnalysisResult,
    detect_phonation,
    apply_phonation,
)


class TestPhonationTypeEnum:
    """Tests for PhonationType enum (TEST-PHN-01 to TEST-PHN-06)."""

    def test_phn_01_has_modal(self):
        """TEST-PHN-01: PhonationType.MODAL exists with value 'modal'."""
        assert hasattr(PhonationType, 'MODAL')
        assert PhonationType.MODAL.value == "modal"

    def test_phn_02_has_breathy(self):
        """TEST-PHN-02: PhonationType.BREATHY exists with value 'breathy'."""
        assert hasattr(PhonationType, 'BREATHY')
        assert PhonationType.BREATHY.value == "breathy"

    def test_phn_03_has_creaky(self):
        """TEST-PHN-03: PhonationType.CREAKY exists with value 'creaky'."""
        assert hasattr(PhonationType, 'CREAKY')
        assert PhonationType.CREAKY.value == "creaky"

    def test_phn_04_has_harsh(self):
        """TEST-PHN-04: PhonationType.HARSH exists with value 'harsh'."""
        assert hasattr(PhonationType, 'HARSH')
        assert PhonationType.HARSH.value == "harsh"

    def test_phn_05_has_falsetto(self):
        """TEST-PHN-05: PhonationType.FALSETTO exists with value 'falsetto'."""
        assert hasattr(PhonationType, 'FALSETTO')
        assert PhonationType.FALSETTO.value == "falsetto"

    def test_phn_06_has_whisper(self):
        """TEST-PHN-06: PhonationType.WHISPER exists with value 'whisper'."""
        assert hasattr(PhonationType, 'WHISPER')
        assert PhonationType.WHISPER.value == "whisper"


# =============================================================================
# TEST-PHN-07 to TEST-PHN-12: PHONATION_PARAMS Constant Tests
# =============================================================================

class TestPhonationParams:
    """Tests for PHONATION_PARAMS constant (TEST-PHN-07 to TEST-PHN-12)."""

    def test_phn_07_params_is_dict(self):
        """TEST-PHN-07: PHONATION_PARAMS is a dictionary."""
        assert isinstance(PHONATION_PARAMS, dict)

    def test_phn_08_params_has_all_types(self):
        """TEST-PHN-08: PHONATION_PARAMS has entry for each PhonationType."""
        for ptype in PhonationType:
            assert ptype in PHONATION_PARAMS, f"Missing params for {ptype}"

    def test_phn_09_modal_params_has_jitter_target(self):
        """TEST-PHN-09: MODAL params has jitter_target."""
        assert "jitter_target" in PHONATION_PARAMS[PhonationType.MODAL]

    def test_phn_10_modal_params_has_shimmer_target(self):
        """TEST-PHN-10: MODAL params has shimmer_target."""
        assert "shimmer_target" in PHONATION_PARAMS[PhonationType.MODAL]

    def test_phn_11_modal_params_has_hnr_target(self):
        """TEST-PHN-11: MODAL params has hnr_target."""
        assert "hnr_target" in PHONATION_PARAMS[PhonationType.MODAL]

    def test_phn_12_breathy_has_noise_level(self):
        """TEST-PHN-12: BREATHY params has noise_level > 0."""
        assert "noise_level" in PHONATION_PARAMS[PhonationType.BREATHY]
        assert PHONATION_PARAMS[PhonationType.BREATHY]["noise_level"] > 0


# =============================================================================
# TEST-PHN-13 to TEST-PHN-16: PhonationAnalysisResult Dataclass Tests
# =============================================================================

class TestPhonationAnalysisResult:
    """Tests for PhonationAnalysisResult dataclass (TEST-PHN-13 to TEST-PHN-16)."""

    @pytest.fixture
    def mock_result(self):
        """Create a mock PhonationAnalysisResult."""
        return PhonationAnalysisResult(
            detected_type=PhonationType.MODAL,
            confidence=0.85,
            type_scores={
                PhonationType.MODAL: 0.85,
                PhonationType.BREATHY: 0.08,
                PhonationType.CREAKY: 0.04,
                PhonationType.HARSH: 0.02,
                PhonationType.FALSETTO: 0.005,
                PhonationType.WHISPER: 0.005,
            },
            features={
                "jitter": 0.4,
                "shimmer": 1.8,
                "hnr": 21.0,
                "spectral_tilt": -12.0,
                "f0_variation": 0.08,
            },
        )

    def test_phn_13_has_detected_type(self, mock_result):
        """TEST-PHN-13: PhonationAnalysisResult has detected_type field."""
        assert hasattr(mock_result, 'detected_type')
        assert isinstance(mock_result.detected_type, PhonationType)

    def test_phn_14_has_confidence(self, mock_result):
        """TEST-PHN-14: PhonationAnalysisResult has confidence field."""
        assert hasattr(mock_result, 'confidence')
        assert 0 <= mock_result.confidence <= 1

    def test_phn_15_has_type_scores(self, mock_result):
        """TEST-PHN-15: PhonationAnalysisResult has type_scores dict."""
        assert hasattr(mock_result, 'type_scores')
        assert isinstance(mock_result.type_scores, dict)

    def test_phn_16_has_features(self, mock_result):
        """TEST-PHN-16: PhonationAnalysisResult has features dict."""
        assert hasattr(mock_result, 'features')
        assert isinstance(mock_result.features, dict)
        assert "jitter" in mock_result.features


# =============================================================================
# TEST-PHN-17 to TEST-PHN-25: PhonationAnalyzer Class Tests
# =============================================================================

class TestPhonationAnalyzer:
    """Tests for PhonationAnalyzer class (TEST-PHN-17 to TEST-PHN-25)."""

    @pytest.fixture
    def mock_audio_modal(self):
        """Create mock audio simulating modal (normal) phonation."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        # Clean voiced speech - regular periodicity, harmonics
        f0 = 120
        audio = np.zeros_like(t)
        for h in range(1, 20):
            audio += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
        audio = audio / np.max(np.abs(audio)) * 0.8
        # Very small noise for modal voice
        audio += 0.01 * np.random.randn(len(audio))
        return audio.astype(np.float32), sr

    @pytest.fixture
    def mock_audio_breathy(self):
        """Create mock audio simulating breathy phonation."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        f0 = 120
        audio = np.zeros_like(t)
        for h in range(1, 10):  # Fewer harmonics
            audio += (1.0 / (h * 2)) * np.sin(2 * np.pi * f0 * h * t)
        audio = audio / np.max(np.abs(audio)) * 0.5
        # More noise for breathy voice
        audio += 0.3 * np.random.randn(len(audio))
        return audio.astype(np.float32), sr

    def test_phn_17_analyzer_init(self):
        """TEST-PHN-17: PhonationAnalyzer initializes without error."""
        analyzer = PhonationAnalyzer()
        assert analyzer is not None

    def test_phn_18_analyze_returns_result(self, mock_audio_modal):
        """TEST-PHN-18: analyze() returns PhonationAnalysisResult."""
        audio, sr = mock_audio_modal
        analyzer = PhonationAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert isinstance(result, PhonationAnalysisResult)

    def test_phn_19_analyze_requires_sample_rate(self):
        """TEST-PHN-19: analyze() raises ValueError without sample_rate for array."""
        audio = np.zeros(16000, dtype=np.float32)
        analyzer = PhonationAnalyzer()
        with pytest.raises(ValueError, match="sample_rate"):
            analyzer.analyze(audio)

    def test_phn_20_analyze_detected_type_is_phonation(self, mock_audio_modal):
        """TEST-PHN-20: analyze() returns valid PhonationType."""
        audio, sr = mock_audio_modal
        analyzer = PhonationAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert isinstance(result.detected_type, PhonationType)

    def test_phn_21_analyze_confidence_in_range(self, mock_audio_modal):
        """TEST-PHN-21: analyze() returns confidence in [0, 1]."""
        audio, sr = mock_audio_modal
        analyzer = PhonationAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert 0 <= result.confidence <= 1

    def test_phn_22_analyze_type_scores_sum_to_one(self, mock_audio_modal):
        """TEST-PHN-22: analyze() type_scores approximately sum to 1.0."""
        audio, sr = mock_audio_modal
        analyzer = PhonationAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        total = sum(result.type_scores.values())
        assert abs(total - 1.0) < 0.01

    def test_phn_23_analyze_features_has_jitter(self, mock_audio_modal):
        """TEST-PHN-23: analyze() features includes jitter."""
        audio, sr = mock_audio_modal
        analyzer = PhonationAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert "jitter" in result.features

    def test_phn_24_analyze_features_has_shimmer(self, mock_audio_modal):
        """TEST-PHN-24: analyze() features includes shimmer."""
        audio, sr = mock_audio_modal
        analyzer = PhonationAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert "shimmer" in result.features

    def test_phn_25_analyze_features_has_hnr(self, mock_audio_modal):
        """TEST-PHN-25: analyze() features includes hnr."""
        audio, sr = mock_audio_modal
        analyzer = PhonationAnalyzer()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert "hnr" in result.features
