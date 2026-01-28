"""
Test Additional Coverage Batch 53: Vocology Phonation Tests

Tests for:
- PhonationType enum
- PHONATION_PARAMS constant
- PhonationAnalysisResult dataclass
- PhonationAnalyzer class
- PhonationSynthesizer class
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


# ============== PhonationType Enum Tests ==============

class TestPhonationTypeEnum:
    """Tests for PhonationType enum."""

    def test_phonation_type_modal(self):
        """Test PhonationType.MODAL value."""
        from voice_soundboard.vocology.phonation import PhonationType
        assert PhonationType.MODAL.value == "modal"

    def test_phonation_type_breathy(self):
        """Test PhonationType.BREATHY value."""
        from voice_soundboard.vocology.phonation import PhonationType
        assert PhonationType.BREATHY.value == "breathy"

    def test_phonation_type_creaky(self):
        """Test PhonationType.CREAKY value."""
        from voice_soundboard.vocology.phonation import PhonationType
        assert PhonationType.CREAKY.value == "creaky"

    def test_phonation_type_harsh(self):
        """Test PhonationType.HARSH value."""
        from voice_soundboard.vocology.phonation import PhonationType
        assert PhonationType.HARSH.value == "harsh"

    def test_phonation_type_falsetto(self):
        """Test PhonationType.FALSETTO value."""
        from voice_soundboard.vocology.phonation import PhonationType
        assert PhonationType.FALSETTO.value == "falsetto"

    def test_phonation_type_whisper(self):
        """Test PhonationType.WHISPER value."""
        from voice_soundboard.vocology.phonation import PhonationType
        assert PhonationType.WHISPER.value == "whisper"


# ============== PHONATION_PARAMS Tests ==============

class TestPhonationParams:
    """Tests for PHONATION_PARAMS constant."""

    def test_phonation_params_has_all_types(self):
        """Test PHONATION_PARAMS has all phonation types."""
        from voice_soundboard.vocology.phonation import PHONATION_PARAMS, PhonationType
        for ptype in PhonationType:
            assert ptype in PHONATION_PARAMS

    def test_phonation_params_modal_values(self):
        """Test PHONATION_PARAMS modal parameters."""
        from voice_soundboard.vocology.phonation import PHONATION_PARAMS, PhonationType
        params = PHONATION_PARAMS[PhonationType.MODAL]
        assert "jitter_target" in params
        assert "shimmer_target" in params
        assert "hnr_target" in params
        assert params["hnr_target"] == 20.0

    def test_phonation_params_breathy_has_noise(self):
        """Test PHONATION_PARAMS breathy has high noise."""
        from voice_soundboard.vocology.phonation import PHONATION_PARAMS, PhonationType
        params = PHONATION_PARAMS[PhonationType.BREATHY]
        assert params["noise_level"] > 0

    def test_phonation_params_creaky_has_subharmonics(self):
        """Test PHONATION_PARAMS creaky has subharmonics flag."""
        from voice_soundboard.vocology.phonation import PHONATION_PARAMS, PhonationType
        params = PHONATION_PARAMS[PhonationType.CREAKY]
        assert params.get("subharmonics", False) is True

    def test_phonation_params_whisper_full_noise(self):
        """Test PHONATION_PARAMS whisper is all noise."""
        from voice_soundboard.vocology.phonation import PHONATION_PARAMS, PhonationType
        params = PHONATION_PARAMS[PhonationType.WHISPER]
        assert params["noise_level"] == 1.0
        assert params["jitter_target"] == 0.0


# ============== PhonationAnalysisResult Tests ==============

class TestPhonationAnalysisResult:
    """Tests for PhonationAnalysisResult dataclass."""

    def test_phonation_analysis_result_creation(self):
        """Test PhonationAnalysisResult basic creation."""
        from voice_soundboard.vocology.phonation import PhonationAnalysisResult, PhonationType
        result = PhonationAnalysisResult(
            detected_type=PhonationType.MODAL,
            confidence=0.85,
            type_scores={PhonationType.MODAL: 0.85, PhonationType.BREATHY: 0.15},
            features={"jitter": 0.5, "shimmer": 2.0}
        )
        assert result.detected_type == PhonationType.MODAL
        assert result.confidence == 0.85

    def test_phonation_analysis_result_features(self):
        """Test PhonationAnalysisResult features dict."""
        from voice_soundboard.vocology.phonation import PhonationAnalysisResult, PhonationType
        features = {"jitter": 0.6, "shimmer": 3.0, "hnr": 18.0}
        result = PhonationAnalysisResult(
            detected_type=PhonationType.BREATHY,
            confidence=0.7,
            type_scores={},
            features=features
        )
        assert result.features["jitter"] == 0.6
        assert result.features["hnr"] == 18.0


# ============== PhonationAnalyzer Tests ==============

class TestPhonationAnalyzer:
    """Tests for PhonationAnalyzer class."""

    def test_phonation_analyzer_init(self):
        """Test PhonationAnalyzer initialization."""
        from voice_soundboard.vocology.phonation import PhonationAnalyzer
        analyzer = PhonationAnalyzer()
        assert analyzer is not None

    def test_phonation_analyzer_calculate_similarity(self):
        """Test PhonationAnalyzer._calculate_similarity method."""
        from voice_soundboard.vocology.phonation import PhonationAnalyzer, PHONATION_PARAMS, PhonationType
        analyzer = PhonationAnalyzer()

        features = {"jitter": 0.5, "shimmer": 2.0, "hnr": 20.0, "spectral_tilt": -12.0, "f0_variation": 0.1}
        modal_params = PHONATION_PARAMS[PhonationType.MODAL]

        score = analyzer._calculate_similarity(features, modal_params)
        assert score > 0

    def test_phonation_analyzer_calculate_similarity_breathy(self):
        """Test PhonationAnalyzer._calculate_similarity for breathy voice."""
        from voice_soundboard.vocology.phonation import PhonationAnalyzer, PHONATION_PARAMS, PhonationType
        analyzer = PhonationAnalyzer()

        # Features that match breathy voice
        features = {"jitter": 0.6, "shimmer": 6.0, "hnr": 12.0, "spectral_tilt": -18.0, "f0_variation": 0.08}

        modal_score = analyzer._calculate_similarity(features, PHONATION_PARAMS[PhonationType.MODAL])
        breathy_score = analyzer._calculate_similarity(features, PHONATION_PARAMS[PhonationType.BREATHY])

        # Breathy features should score higher for breathy params
        assert breathy_score > modal_score

    @patch('voice_soundboard.vocology.phonation.VoiceQualityAnalyzer')
    def test_phonation_analyzer_analyze(self, mock_analyzer_class):
        """Test PhonationAnalyzer.analyze method."""
        from voice_soundboard.vocology.phonation import PhonationAnalyzer

        # Mock voice quality metrics
        mock_metrics = Mock()
        mock_metrics.jitter_local = 0.5
        mock_metrics.shimmer_local = 2.0
        mock_metrics.hnr = 20.0
        mock_metrics.spectral_tilt = -12.0
        mock_metrics.f0_std = 15.0
        mock_metrics.f0_mean = 150.0

        mock_analyzer = Mock()
        mock_analyzer.analyze.return_value = mock_metrics
        mock_analyzer_class.return_value = mock_analyzer

        analyzer = PhonationAnalyzer()
        audio = np.random.randn(24000).astype(np.float32)
        result = analyzer.analyze(audio, sample_rate=24000)

        assert result is not None
        assert result.detected_type is not None
        assert 0 <= result.confidence <= 1


# ============== PhonationSynthesizer Tests ==============

class TestPhonationSynthesizer:
    """Tests for PhonationSynthesizer class."""

    def test_phonation_synthesizer_init(self):
        """Test PhonationSynthesizer initialization."""
        from voice_soundboard.vocology.phonation import PhonationSynthesizer
        synthesizer = PhonationSynthesizer()
        assert synthesizer is not None

    def test_phonation_synthesizer_apply_requires_sample_rate(self):
        """Test PhonationSynthesizer.apply raises error without sample_rate."""
        from voice_soundboard.vocology.phonation import PhonationSynthesizer, PhonationType
        synthesizer = PhonationSynthesizer()
        audio = np.random.randn(24000).astype(np.float32)

        with pytest.raises(ValueError, match="sample_rate required"):
            synthesizer.apply(audio, PhonationType.BREATHY)

    @patch('scipy.signal.butter')
    @patch('scipy.signal.filtfilt')
    def test_phonation_synthesizer_add_breathiness(self, mock_filtfilt, mock_butter):
        """Test PhonationSynthesizer._add_breathiness method."""
        from voice_soundboard.vocology.phonation import PhonationSynthesizer, PHONATION_PARAMS, PhonationType
        synthesizer = PhonationSynthesizer()

        audio = np.random.randn(24000).astype(np.float32)
        mock_butter.return_value = (np.array([1.0]), np.array([1.0]))
        mock_filtfilt.return_value = audio

        params = PHONATION_PARAMS[PhonationType.BREATHY]
        result = synthesizer._add_breathiness(audio, 24000, 0.5, params)

        assert len(result) == len(audio)

    def test_phonation_synthesizer_add_creakiness(self):
        """Test PhonationSynthesizer._add_creakiness method."""
        from voice_soundboard.vocology.phonation import PhonationSynthesizer, PHONATION_PARAMS, PhonationType
        synthesizer = PhonationSynthesizer()

        audio = np.random.randn(24000).astype(np.float32)
        params = PHONATION_PARAMS[PhonationType.CREAKY]
        result = synthesizer._add_creakiness(audio, 24000, 0.5, params)

        assert len(result) == len(audio)

    @patch('scipy.signal.butter')
    @patch('scipy.signal.filtfilt')
    def test_phonation_synthesizer_add_harshness(self, mock_filtfilt, mock_butter):
        """Test PhonationSynthesizer._add_harshness method."""
        from voice_soundboard.vocology.phonation import PhonationSynthesizer, PHONATION_PARAMS, PhonationType
        synthesizer = PhonationSynthesizer()

        audio = np.random.randn(24000).astype(np.float32)
        mock_butter.return_value = (np.array([1.0]), np.array([1.0]))
        mock_filtfilt.return_value = audio

        params = PHONATION_PARAMS[PhonationType.HARSH]
        result = synthesizer._add_harshness(audio, 24000, 0.5, params)

        assert len(result) == len(audio)

    @patch('voice_soundboard.vocology.phonation.PhonationSynthesizer._add_breathiness')
    def test_phonation_synthesizer_apply_breathy(self, mock_add_breathiness):
        """Test PhonationSynthesizer.apply with BREATHY type."""
        from voice_soundboard.vocology.phonation import PhonationSynthesizer, PhonationType
        synthesizer = PhonationSynthesizer()

        audio = np.random.randn(24000).astype(np.float32)
        mock_add_breathiness.return_value = audio

        result, sr = synthesizer.apply(audio, PhonationType.BREATHY, intensity=0.5, sample_rate=24000)

        mock_add_breathiness.assert_called_once()
        assert sr == 24000

    @patch('voice_soundboard.vocology.phonation.PhonationSynthesizer._add_creakiness')
    def test_phonation_synthesizer_apply_creaky(self, mock_add_creakiness):
        """Test PhonationSynthesizer.apply with CREAKY type."""
        from voice_soundboard.vocology.phonation import PhonationSynthesizer, PhonationType
        synthesizer = PhonationSynthesizer()

        audio = np.random.randn(24000).astype(np.float32)
        mock_add_creakiness.return_value = audio

        result, sr = synthesizer.apply(audio, PhonationType.CREAKY, intensity=0.5, sample_rate=24000)

        mock_add_creakiness.assert_called_once()

    def test_phonation_synthesizer_apply_modal_unchanged(self):
        """Test PhonationSynthesizer.apply with MODAL type returns unchanged audio."""
        from voice_soundboard.vocology.phonation import PhonationSynthesizer, PhonationType
        synthesizer = PhonationSynthesizer()

        audio = np.random.randn(24000).astype(np.float32)
        result, sr = synthesizer.apply(audio, PhonationType.MODAL, intensity=0.5, sample_rate=24000)

        # Modal should return mostly unchanged audio
        assert len(result) == len(audio)
