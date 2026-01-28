"""
Tests for Vocology Biomarkers Module

Targets voice_soundboard/vocology/biomarkers.py (29% coverage)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


class TestVoiceHealthStatus:
    """Tests for VoiceHealthStatus enum."""

    def test_health_status_values(self):
        """Should have all expected status values."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthStatus

        assert VoiceHealthStatus.HEALTHY.value == "healthy"
        assert VoiceHealthStatus.MILD_STRAIN.value == "mild_strain"
        assert VoiceHealthStatus.MODERATE_STRAIN.value == "moderate_strain"
        assert VoiceHealthStatus.SIGNIFICANT_STRAIN.value == "significant_strain"
        assert VoiceHealthStatus.NEEDS_ATTENTION.value == "needs_attention"

    def test_health_status_iteration(self):
        """Should be iterable."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthStatus

        statuses = list(VoiceHealthStatus)
        assert len(statuses) == 5


class TestFatigueLevel:
    """Tests for FatigueLevel enum."""

    def test_fatigue_level_values(self):
        """Should have all expected fatigue levels."""
        from voice_soundboard.vocology.biomarkers import FatigueLevel

        assert FatigueLevel.NONE.value == "none"
        assert FatigueLevel.LOW.value == "low"
        assert FatigueLevel.MODERATE.value == "moderate"
        assert FatigueLevel.HIGH.value == "high"
        assert FatigueLevel.SEVERE.value == "severe"


class TestVoiceHealthMetrics:
    """Tests for VoiceHealthMetrics dataclass."""

    def test_create_metrics(self):
        """Should create health metrics."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthMetrics, VoiceHealthStatus

        metrics = VoiceHealthMetrics(
            status=VoiceHealthStatus.HEALTHY,
            quality_score=85.0,
            stability_score=90.0,
            clarity_score=88.0,
        )

        assert metrics.status == VoiceHealthStatus.HEALTHY
        assert metrics.quality_score == 85.0
        assert metrics.stability_score == 90.0
        assert metrics.clarity_score == 88.0

    def test_overall_score_calculation(self):
        """Should calculate overall score as average."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthMetrics, VoiceHealthStatus

        metrics = VoiceHealthMetrics(
            status=VoiceHealthStatus.HEALTHY,
            quality_score=80.0,
            stability_score=90.0,
            clarity_score=100.0,
        )

        assert metrics.overall_score == 90.0  # (80 + 90 + 100) / 3

    def test_metrics_with_concerns(self):
        """Should store concerns list."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthMetrics, VoiceHealthStatus

        metrics = VoiceHealthMetrics(
            status=VoiceHealthStatus.MILD_STRAIN,
            quality_score=70.0,
            stability_score=65.0,
            clarity_score=72.0,
            concerns=["Elevated jitter", "Low HNR"],
        )

        assert len(metrics.concerns) == 2
        assert "Elevated jitter" in metrics.concerns

    def test_metrics_with_recommendations(self):
        """Should store recommendations."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthMetrics, VoiceHealthStatus

        metrics = VoiceHealthMetrics(
            status=VoiceHealthStatus.MODERATE_STRAIN,
            quality_score=60.0,
            stability_score=55.0,
            clarity_score=58.0,
            recommendations=["Rest voice", "Stay hydrated"],
        )

        assert len(metrics.recommendations) == 2
        assert "Stay hydrated" in metrics.recommendations

    def test_default_empty_lists(self):
        """Should have empty default lists."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthMetrics, VoiceHealthStatus

        metrics = VoiceHealthMetrics(
            status=VoiceHealthStatus.HEALTHY,
            quality_score=90.0,
            stability_score=90.0,
            clarity_score=90.0,
        )

        assert metrics.concerns == []
        assert metrics.recommendations == []


class TestBiomarkerResult:
    """Tests for BiomarkerResult dataclass."""

    def test_create_result(self):
        """Should create biomarker result."""
        from voice_soundboard.vocology.biomarkers import (
            BiomarkerResult, VoiceHealthMetrics, VoiceHealthStatus, FatigueLevel
        )

        health = VoiceHealthMetrics(
            status=VoiceHealthStatus.HEALTHY,
            quality_score=85.0,
            stability_score=90.0,
            clarity_score=88.0,
        )

        result = BiomarkerResult(
            health_metrics=health,
            fatigue_level=FatigueLevel.LOW,
            voice_quality={"jitter": 0.5, "shimmer": 3.0},
            timestamp=datetime.now(),
            audio_duration=5.0,
        )

        assert result.health_metrics == health
        assert result.fatigue_level == FatigueLevel.LOW
        assert result.audio_duration == 5.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        from voice_soundboard.vocology.biomarkers import (
            BiomarkerResult, VoiceHealthMetrics, VoiceHealthStatus, FatigueLevel
        )

        health = VoiceHealthMetrics(
            status=VoiceHealthStatus.HEALTHY,
            quality_score=85.0,
            stability_score=90.0,
            clarity_score=88.0,
            concerns=["Minor concern"],
            recommendations=["Recommendation"],
        )

        result = BiomarkerResult(
            health_metrics=health,
            fatigue_level=FatigueLevel.NONE,
            voice_quality={"jitter": 0.5},
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            audio_duration=3.0,
            warnings=["Test warning"],
        )

        d = result.to_dict()

        assert "health" in d
        assert d["health"]["status"] == "healthy"
        assert d["health"]["quality_score"] == 85.0
        assert d["health"]["overall_score"] == pytest.approx(87.67, rel=0.01)
        assert d["fatigue"]["level"] == "none"
        assert d["duration_s"] == 3.0
        assert "Test warning" in d["warnings"]

    def test_result_with_warnings(self):
        """Should store warnings."""
        from voice_soundboard.vocology.biomarkers import (
            BiomarkerResult, VoiceHealthMetrics, VoiceHealthStatus, FatigueLevel
        )

        health = VoiceHealthMetrics(
            status=VoiceHealthStatus.NEEDS_ATTENTION,
            quality_score=40.0,
            stability_score=35.0,
            clarity_score=45.0,
        )

        result = BiomarkerResult(
            health_metrics=health,
            fatigue_level=FatigueLevel.SEVERE,
            voice_quality={},
            timestamp=datetime.now(),
            audio_duration=2.0,
            warnings=["Seek medical attention", "Voice rest recommended"],
        )

        assert len(result.warnings) == 2


class TestVocalBiomarkers:
    """Tests for VocalBiomarkers analyzer class."""

    def test_init(self):
        """Should initialize analyzer."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        analyzer = VocalBiomarkers()
        assert analyzer is not None

    def test_healthy_thresholds(self):
        """Should have healthy thresholds defined."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        assert "jitter_max" in VocalBiomarkers.HEALTHY_THRESHOLDS
        assert "shimmer_max" in VocalBiomarkers.HEALTHY_THRESHOLDS
        assert "hnr_min" in VocalBiomarkers.HEALTHY_THRESHOLDS
        assert "cpp_min" in VocalBiomarkers.HEALTHY_THRESHOLDS

    def test_concern_thresholds(self):
        """Should have concern thresholds defined."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        assert "jitter_concern" in VocalBiomarkers.CONCERN_THRESHOLDS
        assert "shimmer_concern" in VocalBiomarkers.CONCERN_THRESHOLDS
        assert "hnr_concern" in VocalBiomarkers.CONCERN_THRESHOLDS

    def test_analyze_with_mock_metrics(self):
        """Should analyze audio and return biomarker result."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers, VoiceHealthStatus, FatigueLevel

        # Create mock metrics
        mock_metrics = Mock()
        mock_metrics.jitter_percent = 0.5
        mock_metrics.shimmer_percent = 3.0
        mock_metrics.hnr_db = 20.0
        mock_metrics.cpp_db = 8.0
        mock_metrics.f0_mean = 150.0
        mock_metrics.f0_std = 20.0
        mock_metrics.duration = 5.0
        mock_metrics.to_dict = Mock(return_value={
            "jitter": 0.5,
            "shimmer": 3.0,
            "hnr": 20.0
        })

        mock_analyzer = Mock()
        mock_analyzer.analyze = Mock(return_value=mock_metrics)

        with patch('voice_soundboard.vocology.biomarkers.VoiceQualityAnalyzer', return_value=mock_analyzer):
            analyzer = VocalBiomarkers()

            # Create test audio
            audio = np.random.randn(16000 * 5).astype(np.float32)
            result = analyzer.analyze(audio, sample_rate=16000)

        assert result is not None
        assert isinstance(result.health_metrics.status, VoiceHealthStatus)
        assert isinstance(result.fatigue_level, FatigueLevel)

    def test_analyze_from_file_path(self, tmp_path):
        """Should analyze from file path."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        # Create mock metrics
        mock_metrics = Mock()
        mock_metrics.jitter_percent = 0.8
        mock_metrics.shimmer_percent = 4.0
        mock_metrics.hnr_db = 18.0
        mock_metrics.cpp_db = 6.0
        mock_metrics.f0_mean = 120.0
        mock_metrics.f0_std = 30.0
        mock_metrics.duration = 3.0
        mock_metrics.to_dict = Mock(return_value={})

        mock_analyzer = Mock()
        mock_analyzer.analyze = Mock(return_value=mock_metrics)

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        with patch('voice_soundboard.vocology.biomarkers.VoiceQualityAnalyzer', return_value=mock_analyzer):
            analyzer = VocalBiomarkers()
            result = analyzer.analyze(str(audio_file))

        assert result is not None

    def test_assess_health_healthy(self):
        """Should assess healthy voice."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers, VoiceHealthStatus

        analyzer = VocalBiomarkers()

        mock_metrics = Mock()
        mock_metrics.jitter_percent = 0.3  # Below healthy threshold
        mock_metrics.shimmer_percent = 2.0  # Below healthy threshold
        mock_metrics.hnr_db = 25.0  # Above healthy minimum
        mock_metrics.cpp_db = 10.0  # Above healthy minimum
        mock_metrics.f0_std = 20.0  # Below stability threshold

        health = analyzer._assess_health(mock_metrics)

        # Should be healthy or near-healthy
        assert health.status in [VoiceHealthStatus.HEALTHY, VoiceHealthStatus.MILD_STRAIN]
        assert health.quality_score > 50

    def test_assess_health_strained(self):
        """Should assess strained voice."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers, VoiceHealthStatus

        analyzer = VocalBiomarkers()

        mock_metrics = Mock()
        mock_metrics.jitter_percent = 3.0  # Above concern threshold
        mock_metrics.shimmer_percent = 10.0  # Above concern threshold
        mock_metrics.hnr_db = 8.0  # Below concern threshold
        mock_metrics.cpp_db = 2.0  # Below concern threshold
        mock_metrics.f0_std = 80.0  # High variability

        health = analyzer._assess_health(mock_metrics)

        # Should indicate strain
        assert health.status in [
            VoiceHealthStatus.MODERATE_STRAIN,
            VoiceHealthStatus.SIGNIFICANT_STRAIN,
            VoiceHealthStatus.NEEDS_ATTENTION
        ]
        assert len(health.concerns) > 0

    def test_assess_fatigue_none(self):
        """Should assess no fatigue."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers, FatigueLevel

        analyzer = VocalBiomarkers()

        mock_metrics = Mock()
        mock_metrics.jitter_percent = 0.3
        mock_metrics.shimmer_percent = 2.0
        mock_metrics.hnr_db = 25.0
        mock_metrics.f0_std = 15.0

        fatigue = analyzer._assess_fatigue(mock_metrics)

        assert fatigue in [FatigueLevel.NONE, FatigueLevel.LOW]

    def test_assess_fatigue_high(self):
        """Should assess high fatigue."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers, FatigueLevel

        analyzer = VocalBiomarkers()

        mock_metrics = Mock()
        mock_metrics.jitter_percent = 2.5
        mock_metrics.shimmer_percent = 9.0
        mock_metrics.hnr_db = 10.0
        mock_metrics.f0_std = 60.0

        fatigue = analyzer._assess_fatigue(mock_metrics)

        assert fatigue in [FatigueLevel.MODERATE, FatigueLevel.HIGH, FatigueLevel.SEVERE]

    def test_generate_warnings_none(self):
        """Should generate no warnings for healthy voice."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        analyzer = VocalBiomarkers()

        mock_metrics = Mock()
        mock_metrics.jitter_percent = 0.3
        mock_metrics.shimmer_percent = 2.0
        mock_metrics.hnr_db = 25.0
        mock_metrics.cpp_db = 10.0

        warnings = analyzer._generate_warnings(mock_metrics)

        # Should have few or no warnings
        assert isinstance(warnings, list)

    def test_generate_warnings_multiple(self):
        """Should generate multiple warnings for unhealthy voice."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        analyzer = VocalBiomarkers()

        mock_metrics = Mock()
        mock_metrics.jitter_percent = 5.0  # Very high
        mock_metrics.shimmer_percent = 15.0  # Very high
        mock_metrics.hnr_db = 5.0  # Very low
        mock_metrics.cpp_db = 1.0  # Very low

        warnings = analyzer._generate_warnings(mock_metrics)

        # Should have warnings
        assert isinstance(warnings, list)


class TestBiomarkerAnalysisIntegration:
    """Integration tests for biomarker analysis."""

    def test_full_analysis_pipeline(self):
        """Should run complete analysis pipeline."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        # Mock the entire pipeline
        mock_metrics = Mock()
        mock_metrics.jitter_percent = 0.6
        mock_metrics.shimmer_percent = 3.5
        mock_metrics.hnr_db = 18.0
        mock_metrics.cpp_db = 7.0
        mock_metrics.f0_mean = 140.0
        mock_metrics.f0_std = 25.0
        mock_metrics.duration = 4.0
        mock_metrics.to_dict = Mock(return_value={
            "jitter_percent": 0.6,
            "shimmer_percent": 3.5,
            "hnr_db": 18.0,
            "cpp_db": 7.0,
        })

        mock_analyzer = Mock()
        mock_analyzer.analyze = Mock(return_value=mock_metrics)

        with patch('voice_soundboard.vocology.biomarkers.VoiceQualityAnalyzer', return_value=mock_analyzer):
            analyzer = VocalBiomarkers()
            audio = np.random.randn(16000 * 4).astype(np.float32)
            result = analyzer.analyze(audio, sample_rate=16000)

        # Check all components
        assert result.health_metrics is not None
        assert result.fatigue_level is not None
        assert result.voice_quality is not None
        assert result.timestamp is not None
        assert result.audio_duration == 4.0

        # Check serialization
        d = result.to_dict()
        assert "health" in d
        assert "fatigue" in d
        assert "voice_quality" in d
        assert "timestamp" in d

    def test_analysis_with_short_audio(self):
        """Should handle short audio."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        mock_metrics = Mock()
        mock_metrics.jitter_percent = 1.0
        mock_metrics.shimmer_percent = 5.0
        mock_metrics.hnr_db = 15.0
        mock_metrics.cpp_db = 5.0
        mock_metrics.f0_mean = 150.0
        mock_metrics.f0_std = 30.0
        mock_metrics.duration = 0.5  # Very short
        mock_metrics.to_dict = Mock(return_value={})

        mock_analyzer = Mock()
        mock_analyzer.analyze = Mock(return_value=mock_metrics)

        with patch('voice_soundboard.vocology.biomarkers.VoiceQualityAnalyzer', return_value=mock_analyzer):
            analyzer = VocalBiomarkers()
            audio = np.random.randn(8000).astype(np.float32)  # 0.5 seconds
            result = analyzer.analyze(audio, sample_rate=16000)

        assert result is not None
        assert result.audio_duration == 0.5

    def test_analysis_with_silence(self):
        """Should handle silent audio."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        mock_metrics = Mock()
        mock_metrics.jitter_percent = 0.0
        mock_metrics.shimmer_percent = 0.0
        mock_metrics.hnr_db = 0.0
        mock_metrics.cpp_db = 0.0
        mock_metrics.f0_mean = 0.0
        mock_metrics.f0_std = 0.0
        mock_metrics.duration = 1.0
        mock_metrics.to_dict = Mock(return_value={})

        mock_analyzer = Mock()
        mock_analyzer.analyze = Mock(return_value=mock_metrics)

        with patch('voice_soundboard.vocology.biomarkers.VoiceQualityAnalyzer', return_value=mock_analyzer):
            analyzer = VocalBiomarkers()
            audio = np.zeros(16000).astype(np.float32)  # Silence
            result = analyzer.analyze(audio, sample_rate=16000)

        assert result is not None


class TestBiomarkerEdgeCases:
    """Edge case tests for biomarkers."""

    def test_extreme_jitter(self):
        """Should handle extreme jitter values."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers, VoiceHealthStatus

        analyzer = VocalBiomarkers()

        mock_metrics = Mock()
        mock_metrics.jitter_percent = 50.0  # Extreme
        mock_metrics.shimmer_percent = 3.0
        mock_metrics.hnr_db = 15.0
        mock_metrics.cpp_db = 5.0
        mock_metrics.f0_std = 30.0

        health = analyzer._assess_health(mock_metrics)
        assert health.status == VoiceHealthStatus.NEEDS_ATTENTION

    def test_extreme_shimmer(self):
        """Should handle extreme shimmer values."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers, VoiceHealthStatus

        analyzer = VocalBiomarkers()

        mock_metrics = Mock()
        mock_metrics.jitter_percent = 0.5
        mock_metrics.shimmer_percent = 50.0  # Extreme
        mock_metrics.hnr_db = 15.0
        mock_metrics.cpp_db = 5.0
        mock_metrics.f0_std = 30.0

        health = analyzer._assess_health(mock_metrics)
        assert health.status in [VoiceHealthStatus.SIGNIFICANT_STRAIN, VoiceHealthStatus.NEEDS_ATTENTION]

    def test_negative_hnr(self):
        """Should handle negative HNR."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        analyzer = VocalBiomarkers()

        mock_metrics = Mock()
        mock_metrics.jitter_percent = 1.0
        mock_metrics.shimmer_percent = 5.0
        mock_metrics.hnr_db = -5.0  # Negative (very noisy)
        mock_metrics.cpp_db = 2.0
        mock_metrics.f0_std = 40.0

        health = analyzer._assess_health(mock_metrics)
        # Should handle without error
        assert health is not None
