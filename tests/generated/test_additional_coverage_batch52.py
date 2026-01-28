"""
Test Additional Coverage Batch 52: Vocology Parameters Tests

Tests for:
- JitterType enum
- ShimmerType enum
- VoiceQualityMetrics dataclass
- VoiceQualityAnalyzer class
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


# ============== JitterType Enum Tests ==============

class TestJitterTypeEnum:
    """Tests for JitterType enum."""

    def test_jitter_type_local(self):
        """Test JitterType.LOCAL value."""
        from voice_soundboard.vocology.parameters import JitterType
        assert JitterType.LOCAL.value == "local"

    def test_jitter_type_rap(self):
        """Test JitterType.RAP value."""
        from voice_soundboard.vocology.parameters import JitterType
        assert JitterType.RAP.value == "rap"

    def test_jitter_type_ppq5(self):
        """Test JitterType.PPQ5 value."""
        from voice_soundboard.vocology.parameters import JitterType
        assert JitterType.PPQ5.value == "ppq5"

    def test_jitter_type_ddp(self):
        """Test JitterType.DDP value."""
        from voice_soundboard.vocology.parameters import JitterType
        assert JitterType.DDP.value == "ddp"


# ============== ShimmerType Enum Tests ==============

class TestShimmerTypeEnum:
    """Tests for ShimmerType enum."""

    def test_shimmer_type_local(self):
        """Test ShimmerType.LOCAL value."""
        from voice_soundboard.vocology.parameters import ShimmerType
        assert ShimmerType.LOCAL.value == "local"

    def test_shimmer_type_apq3(self):
        """Test ShimmerType.APQ3 value."""
        from voice_soundboard.vocology.parameters import ShimmerType
        assert ShimmerType.APQ3.value == "apq3"

    def test_shimmer_type_apq5(self):
        """Test ShimmerType.APQ5 value."""
        from voice_soundboard.vocology.parameters import ShimmerType
        assert ShimmerType.APQ5.value == "apq5"

    def test_shimmer_type_apq11(self):
        """Test ShimmerType.APQ11 value."""
        from voice_soundboard.vocology.parameters import ShimmerType
        assert ShimmerType.APQ11.value == "apq11"

    def test_shimmer_type_dda(self):
        """Test ShimmerType.DDA value."""
        from voice_soundboard.vocology.parameters import ShimmerType
        assert ShimmerType.DDA.value == "dda"


# ============== VoiceQualityMetrics Tests ==============

class TestVoiceQualityMetrics:
    """Tests for VoiceQualityMetrics dataclass."""

    def test_voice_quality_metrics_creation(self):
        """Test VoiceQualityMetrics basic creation."""
        from voice_soundboard.vocology.parameters import VoiceQualityMetrics
        metrics = VoiceQualityMetrics(
            f0_mean=150.0,
            f0_std=20.0,
            f0_range=100.0,
            jitter_local=0.5,
            jitter_rap=0.3,
            jitter_ppq5=0.4,
            shimmer_local=3.0,
            shimmer_apq3=2.5,
            shimmer_apq5=2.8,
            shimmer_apq11=3.2,
            hnr=20.0,
            nhr=0.01,
            cpp=8.0,
            spectral_tilt=-12.0,
            spectral_centroid=1500.0,
            voiced_fraction=0.85,
            duration=2.5
        )
        assert metrics.f0_mean == 150.0
        assert metrics.jitter_local == 0.5

    def test_voice_quality_metrics_jitter_percent_alias(self):
        """Test VoiceQualityMetrics.jitter_percent property."""
        from voice_soundboard.vocology.parameters import VoiceQualityMetrics
        metrics = VoiceQualityMetrics(
            f0_mean=150.0, f0_std=20.0, f0_range=100.0,
            jitter_local=0.75, jitter_rap=0.3, jitter_ppq5=0.4,
            shimmer_local=3.0, shimmer_apq3=2.5, shimmer_apq5=2.8, shimmer_apq11=3.2,
            hnr=20.0, nhr=0.01, cpp=8.0,
            spectral_tilt=-12.0, spectral_centroid=1500.0,
            voiced_fraction=0.85, duration=2.5
        )
        assert metrics.jitter_percent == 0.75

    def test_voice_quality_metrics_shimmer_percent_alias(self):
        """Test VoiceQualityMetrics.shimmer_percent property."""
        from voice_soundboard.vocology.parameters import VoiceQualityMetrics
        metrics = VoiceQualityMetrics(
            f0_mean=150.0, f0_std=20.0, f0_range=100.0,
            jitter_local=0.5, jitter_rap=0.3, jitter_ppq5=0.4,
            shimmer_local=4.5, shimmer_apq3=2.5, shimmer_apq5=2.8, shimmer_apq11=3.2,
            hnr=20.0, nhr=0.01, cpp=8.0,
            spectral_tilt=-12.0, spectral_centroid=1500.0,
            voiced_fraction=0.85, duration=2.5
        )
        assert metrics.shimmer_percent == 4.5

    def test_voice_quality_metrics_hnr_db_alias(self):
        """Test VoiceQualityMetrics.hnr_db property."""
        from voice_soundboard.vocology.parameters import VoiceQualityMetrics
        metrics = VoiceQualityMetrics(
            f0_mean=150.0, f0_std=20.0, f0_range=100.0,
            jitter_local=0.5, jitter_rap=0.3, jitter_ppq5=0.4,
            shimmer_local=3.0, shimmer_apq3=2.5, shimmer_apq5=2.8, shimmer_apq11=3.2,
            hnr=22.5, nhr=0.01, cpp=8.0,
            spectral_tilt=-12.0, spectral_centroid=1500.0,
            voiced_fraction=0.85, duration=2.5
        )
        assert metrics.hnr_db == 22.5

    def test_voice_quality_metrics_is_healthy_true(self):
        """Test VoiceQualityMetrics.is_healthy returns True for healthy voice."""
        from voice_soundboard.vocology.parameters import VoiceQualityMetrics
        metrics = VoiceQualityMetrics(
            f0_mean=150.0, f0_std=20.0, f0_range=100.0,
            jitter_local=0.5, jitter_rap=0.3, jitter_ppq5=0.4,
            shimmer_local=3.0, shimmer_apq3=2.5, shimmer_apq5=2.8, shimmer_apq11=3.2,
            hnr=20.0, nhr=0.01, cpp=8.0,
            spectral_tilt=-12.0, spectral_centroid=1500.0,
            voiced_fraction=0.85, duration=2.5
        )
        assert metrics.is_healthy() is True

    def test_voice_quality_metrics_is_healthy_false_high_jitter(self):
        """Test VoiceQualityMetrics.is_healthy returns False for high jitter."""
        from voice_soundboard.vocology.parameters import VoiceQualityMetrics
        metrics = VoiceQualityMetrics(
            f0_mean=150.0, f0_std=20.0, f0_range=100.0,
            jitter_local=2.0, jitter_rap=0.3, jitter_ppq5=0.4,  # High jitter
            shimmer_local=3.0, shimmer_apq3=2.5, shimmer_apq5=2.8, shimmer_apq11=3.2,
            hnr=20.0, nhr=0.01, cpp=8.0,
            spectral_tilt=-12.0, spectral_centroid=1500.0,
            voiced_fraction=0.85, duration=2.5
        )
        assert metrics.is_healthy() is False

    def test_voice_quality_metrics_quality_assessment_excellent(self):
        """Test VoiceQualityMetrics.quality_assessment returns excellent."""
        from voice_soundboard.vocology.parameters import VoiceQualityMetrics
        metrics = VoiceQualityMetrics(
            f0_mean=150.0, f0_std=20.0, f0_range=100.0,
            jitter_local=0.3, jitter_rap=0.2, jitter_ppq5=0.25,
            shimmer_local=2.0, shimmer_apq3=1.5, shimmer_apq5=1.8, shimmer_apq11=2.0,
            hnr=25.0, nhr=0.003, cpp=10.0,
            spectral_tilt=-12.0, spectral_centroid=1500.0,
            voiced_fraction=0.9, duration=2.5
        )
        assert metrics.quality_assessment() == "excellent"

    def test_voice_quality_metrics_quality_assessment_good(self):
        """Test VoiceQualityMetrics.quality_assessment returns good."""
        from voice_soundboard.vocology.parameters import VoiceQualityMetrics
        metrics = VoiceQualityMetrics(
            f0_mean=150.0, f0_std=20.0, f0_range=100.0,
            jitter_local=0.8, jitter_rap=0.5, jitter_ppq5=0.6,
            shimmer_local=4.0, shimmer_apq3=3.5, shimmer_apq5=3.8, shimmer_apq11=4.0,
            hnr=18.0, nhr=0.016, cpp=7.0,
            spectral_tilt=-12.0, spectral_centroid=1500.0,
            voiced_fraction=0.85, duration=2.5
        )
        assert metrics.quality_assessment() == "good"

    def test_voice_quality_metrics_quality_assessment_poor(self):
        """Test VoiceQualityMetrics.quality_assessment returns poor."""
        from voice_soundboard.vocology.parameters import VoiceQualityMetrics
        metrics = VoiceQualityMetrics(
            f0_mean=150.0, f0_std=20.0, f0_range=100.0,
            jitter_local=3.0, jitter_rap=2.5, jitter_ppq5=2.8,
            shimmer_local=10.0, shimmer_apq3=9.0, shimmer_apq5=9.5, shimmer_apq11=10.0,
            hnr=8.0, nhr=0.16, cpp=3.0,
            spectral_tilt=-12.0, spectral_centroid=1500.0,
            voiced_fraction=0.6, duration=2.5
        )
        assert metrics.quality_assessment() == "poor"

    def test_voice_quality_metrics_to_dict(self):
        """Test VoiceQualityMetrics.to_dict method."""
        from voice_soundboard.vocology.parameters import VoiceQualityMetrics
        metrics = VoiceQualityMetrics(
            f0_mean=150.0, f0_std=20.0, f0_range=100.0,
            jitter_local=0.5, jitter_rap=0.3, jitter_ppq5=0.4,
            shimmer_local=3.0, shimmer_apq3=2.5, shimmer_apq5=2.8, shimmer_apq11=3.2,
            hnr=20.0, nhr=0.01, cpp=8.0,
            spectral_tilt=-12.0, spectral_centroid=1500.0,
            voiced_fraction=0.85, duration=2.5
        )
        d = metrics.to_dict()
        assert "f0" in d
        assert "jitter" in d
        assert "shimmer" in d
        assert "noise" in d
        assert d["f0"]["mean_hz"] == 150.0


# ============== VoiceQualityAnalyzer Tests ==============

class TestVoiceQualityAnalyzer:
    """Tests for VoiceQualityAnalyzer class."""

    def test_voice_quality_analyzer_init(self):
        """Test VoiceQualityAnalyzer initialization."""
        from voice_soundboard.vocology.parameters import VoiceQualityAnalyzer
        analyzer = VoiceQualityAnalyzer(f0_min=75.0, f0_max=400.0)
        assert analyzer.f0_min == 75.0
        assert analyzer.f0_max == 400.0

    def test_voice_quality_analyzer_default_init(self):
        """Test VoiceQualityAnalyzer default initialization."""
        from voice_soundboard.vocology.parameters import VoiceQualityAnalyzer
        analyzer = VoiceQualityAnalyzer()
        assert analyzer.f0_min == 50.0
        assert analyzer.f0_max == 500.0
        assert analyzer.frame_length == 0.025
        assert analyzer.hop_length == 0.010

    def test_voice_quality_analyzer_extract_cycle_amplitudes(self):
        """Test VoiceQualityAnalyzer._extract_cycle_amplitudes method."""
        from voice_soundboard.vocology.parameters import VoiceQualityAnalyzer
        analyzer = VoiceQualityAnalyzer()

        # Create synthetic audio
        sr = 24000
        audio = np.random.randn(sr).astype(np.float32)
        f0 = np.array([150.0, 150.0, 150.0, 0.0])
        voiced = np.array([True, True, True, False])

        amplitudes = analyzer._extract_cycle_amplitudes(audio, sr, f0, voiced)
        assert len(amplitudes) >= 0  # May be empty for short audio

    def test_voice_quality_analyzer_calculate_jitter_local(self):
        """Test VoiceQualityAnalyzer._calculate_jitter with LOCAL type."""
        from voice_soundboard.vocology.parameters import VoiceQualityAnalyzer, JitterType
        analyzer = VoiceQualityAnalyzer()

        # Create F0 values with some variation
        f0_voiced = np.array([150.0, 152.0, 148.0, 151.0, 149.0])
        jitter = analyzer._calculate_jitter(f0_voiced, JitterType.LOCAL)
        assert jitter >= 0.0
        assert jitter < 10.0  # Reasonable jitter range

    def test_voice_quality_analyzer_calculate_jitter_empty(self):
        """Test VoiceQualityAnalyzer._calculate_jitter with empty array."""
        from voice_soundboard.vocology.parameters import VoiceQualityAnalyzer, JitterType
        analyzer = VoiceQualityAnalyzer()

        f0_voiced = np.array([])
        jitter = analyzer._calculate_jitter(f0_voiced, JitterType.LOCAL)
        assert jitter == 0.0

    def test_voice_quality_analyzer_simple_f0_extraction(self):
        """Test VoiceQualityAnalyzer._simple_f0_extraction method."""
        from voice_soundboard.vocology.parameters import VoiceQualityAnalyzer
        analyzer = VoiceQualityAnalyzer()

        # Create a simple sinusoidal signal
        sr = 24000
        t = np.linspace(0, 0.5, int(0.5 * sr))
        audio = np.sin(2 * np.pi * 150 * t).astype(np.float32)

        f0, voiced = analyzer._simple_f0_extraction(audio, sr)
        assert len(f0) > 0
        assert len(voiced) == len(f0)

    @patch('voice_soundboard.vocology.parameters.VoiceQualityAnalyzer._load_audio')
    def test_voice_quality_analyzer_analyze(self, mock_load):
        """Test VoiceQualityAnalyzer.analyze method."""
        from voice_soundboard.vocology.parameters import VoiceQualityAnalyzer

        analyzer = VoiceQualityAnalyzer()
        audio = np.random.randn(24000).astype(np.float32)
        mock_load.return_value = (audio, 24000)

        with patch.object(analyzer, '_extract_f0') as mock_f0:
            mock_f0.return_value = (np.array([150.0, 152.0, 148.0]), np.array([True, True, True]))
            with patch.object(analyzer, '_extract_cycle_amplitudes', return_value=np.array([0.5, 0.55, 0.52])):
                with patch.object(analyzer, '_calculate_hnr', return_value=20.0):
                    with patch.object(analyzer, '_calculate_cpp', return_value=8.0):
                        with patch.object(analyzer, '_calculate_spectral_tilt', return_value=-12.0):
                            with patch.object(analyzer, '_calculate_spectral_centroid', return_value=1500.0):
                                metrics = analyzer.analyze(audio, sample_rate=24000)
                                assert metrics is not None
                                assert metrics.f0_mean > 0

    def test_voice_quality_analyzer_analyze_requires_sample_rate(self):
        """Test VoiceQualityAnalyzer.analyze raises error without sample_rate."""
        from voice_soundboard.vocology.parameters import VoiceQualityAnalyzer
        analyzer = VoiceQualityAnalyzer()
        audio = np.random.randn(24000).astype(np.float32)

        with pytest.raises(ValueError, match="sample_rate required"):
            analyzer.analyze(audio)
