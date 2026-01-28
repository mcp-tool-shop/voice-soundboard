"""
Test Additional Coverage Batch 54: Vocology Biomarkers Tests

Tests for:
- VoiceHealthStatus enum
- FatigueLevel enum
- VoiceHealthMetrics dataclass
- BiomarkerResult dataclass
- VocalBiomarkers class
- VoiceFatigueMonitor class
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime


# ============== VoiceHealthStatus Enum Tests ==============

class TestVoiceHealthStatusEnum:
    """Tests for VoiceHealthStatus enum."""

    def test_voice_health_status_healthy(self):
        """Test VoiceHealthStatus.HEALTHY value."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthStatus
        assert VoiceHealthStatus.HEALTHY.value == "healthy"

    def test_voice_health_status_mild_strain(self):
        """Test VoiceHealthStatus.MILD_STRAIN value."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthStatus
        assert VoiceHealthStatus.MILD_STRAIN.value == "mild_strain"

    def test_voice_health_status_moderate_strain(self):
        """Test VoiceHealthStatus.MODERATE_STRAIN value."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthStatus
        assert VoiceHealthStatus.MODERATE_STRAIN.value == "moderate_strain"

    def test_voice_health_status_significant_strain(self):
        """Test VoiceHealthStatus.SIGNIFICANT_STRAIN value."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthStatus
        assert VoiceHealthStatus.SIGNIFICANT_STRAIN.value == "significant_strain"

    def test_voice_health_status_needs_attention(self):
        """Test VoiceHealthStatus.NEEDS_ATTENTION value."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthStatus
        assert VoiceHealthStatus.NEEDS_ATTENTION.value == "needs_attention"


# ============== FatigueLevel Enum Tests ==============

class TestFatigueLevelEnum:
    """Tests for FatigueLevel enum."""

    def test_fatigue_level_none(self):
        """Test FatigueLevel.NONE value."""
        from voice_soundboard.vocology.biomarkers import FatigueLevel
        assert FatigueLevel.NONE.value == "none"

    def test_fatigue_level_low(self):
        """Test FatigueLevel.LOW value."""
        from voice_soundboard.vocology.biomarkers import FatigueLevel
        assert FatigueLevel.LOW.value == "low"

    def test_fatigue_level_moderate(self):
        """Test FatigueLevel.MODERATE value."""
        from voice_soundboard.vocology.biomarkers import FatigueLevel
        assert FatigueLevel.MODERATE.value == "moderate"

    def test_fatigue_level_high(self):
        """Test FatigueLevel.HIGH value."""
        from voice_soundboard.vocology.biomarkers import FatigueLevel
        assert FatigueLevel.HIGH.value == "high"

    def test_fatigue_level_severe(self):
        """Test FatigueLevel.SEVERE value."""
        from voice_soundboard.vocology.biomarkers import FatigueLevel
        assert FatigueLevel.SEVERE.value == "severe"


# ============== VoiceHealthMetrics Tests ==============

class TestVoiceHealthMetrics:
    """Tests for VoiceHealthMetrics dataclass."""

    def test_voice_health_metrics_creation(self):
        """Test VoiceHealthMetrics basic creation."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthMetrics, VoiceHealthStatus
        metrics = VoiceHealthMetrics(
            status=VoiceHealthStatus.HEALTHY,
            quality_score=85.0,
            stability_score=90.0,
            clarity_score=88.0
        )
        assert metrics.status == VoiceHealthStatus.HEALTHY
        assert metrics.quality_score == 85.0

    def test_voice_health_metrics_overall_score(self):
        """Test VoiceHealthMetrics.overall_score property."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthMetrics, VoiceHealthStatus
        metrics = VoiceHealthMetrics(
            status=VoiceHealthStatus.HEALTHY,
            quality_score=90.0,
            stability_score=80.0,
            clarity_score=70.0
        )
        assert metrics.overall_score == 80.0  # (90 + 80 + 70) / 3

    def test_voice_health_metrics_with_concerns(self):
        """Test VoiceHealthMetrics with concerns list."""
        from voice_soundboard.vocology.biomarkers import VoiceHealthMetrics, VoiceHealthStatus
        concerns = ["Elevated jitter", "Low HNR"]
        recommendations = ["Rest voice", "Stay hydrated"]
        metrics = VoiceHealthMetrics(
            status=VoiceHealthStatus.MILD_STRAIN,
            quality_score=70.0,
            stability_score=75.0,
            clarity_score=72.0,
            concerns=concerns,
            recommendations=recommendations
        )
        assert len(metrics.concerns) == 2
        assert len(metrics.recommendations) == 2


# ============== BiomarkerResult Tests ==============

class TestBiomarkerResult:
    """Tests for BiomarkerResult dataclass."""

    def test_biomarker_result_creation(self):
        """Test BiomarkerResult basic creation."""
        from voice_soundboard.vocology.biomarkers import BiomarkerResult, VoiceHealthMetrics, VoiceHealthStatus, FatigueLevel

        health_metrics = VoiceHealthMetrics(
            status=VoiceHealthStatus.HEALTHY,
            quality_score=85.0,
            stability_score=90.0,
            clarity_score=88.0
        )
        result = BiomarkerResult(
            health_metrics=health_metrics,
            fatigue_level=FatigueLevel.NONE,
            voice_quality={"jitter": 0.5, "shimmer": 2.0},
            timestamp=datetime.now(),
            audio_duration=2.5
        )
        assert result.fatigue_level == FatigueLevel.NONE
        assert result.audio_duration == 2.5

    def test_biomarker_result_to_dict(self):
        """Test BiomarkerResult.to_dict method."""
        from voice_soundboard.vocology.biomarkers import BiomarkerResult, VoiceHealthMetrics, VoiceHealthStatus, FatigueLevel

        health_metrics = VoiceHealthMetrics(
            status=VoiceHealthStatus.HEALTHY,
            quality_score=85.0,
            stability_score=90.0,
            clarity_score=88.0
        )
        result = BiomarkerResult(
            health_metrics=health_metrics,
            fatigue_level=FatigueLevel.LOW,
            voice_quality={"jitter": 0.5},
            timestamp=datetime(2026, 1, 28, 12, 0, 0),
            audio_duration=3.0,
            warnings=["Test warning"]
        )
        d = result.to_dict()
        assert "health" in d
        assert "fatigue" in d
        assert d["health"]["status"] == "healthy"
        assert d["fatigue"]["level"] == "low"


# ============== VocalBiomarkers Tests ==============

class TestVocalBiomarkers:
    """Tests for VocalBiomarkers class."""

    def test_vocal_biomarkers_init(self):
        """Test VocalBiomarkers initialization."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers
        analyzer = VocalBiomarkers()
        assert analyzer is not None

    def test_vocal_biomarkers_healthy_thresholds(self):
        """Test VocalBiomarkers.HEALTHY_THRESHOLDS constant."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers
        assert VocalBiomarkers.HEALTHY_THRESHOLDS["jitter_max"] == 1.0
        assert VocalBiomarkers.HEALTHY_THRESHOLDS["shimmer_max"] == 5.0
        assert VocalBiomarkers.HEALTHY_THRESHOLDS["hnr_min"] == 15.0

    def test_vocal_biomarkers_calculate_quality_score(self):
        """Test VocalBiomarkers._calculate_quality_score method."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        analyzer = VocalBiomarkers()
        mock_metrics = Mock()
        mock_metrics.jitter_local = 0.5
        mock_metrics.shimmer_local = 2.0
        mock_metrics.hnr = 20.0

        score = analyzer._calculate_quality_score(mock_metrics)
        assert 0 <= score <= 100

    def test_vocal_biomarkers_calculate_stability_score(self):
        """Test VocalBiomarkers._calculate_stability_score method."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        analyzer = VocalBiomarkers()
        mock_metrics = Mock()
        mock_metrics.f0_std = 15.0
        mock_metrics.jitter_local = 0.5

        score = analyzer._calculate_stability_score(mock_metrics)
        assert 0 <= score <= 100

    def test_vocal_biomarkers_calculate_clarity_score(self):
        """Test VocalBiomarkers._calculate_clarity_score method."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        analyzer = VocalBiomarkers()
        mock_metrics = Mock()
        mock_metrics.hnr = 20.0
        mock_metrics.cpp = 8.0

        score = analyzer._calculate_clarity_score(mock_metrics)
        assert 0 <= score <= 100

    def test_vocal_biomarkers_assess_fatigue_none(self):
        """Test VocalBiomarkers._assess_fatigue returns NONE for healthy voice."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers, FatigueLevel

        analyzer = VocalBiomarkers()
        mock_metrics = Mock()
        mock_metrics.jitter_local = 0.5
        mock_metrics.shimmer_local = 2.0
        mock_metrics.hnr = 20.0
        mock_metrics.f0_range = 100.0
        mock_metrics.spectral_tilt = -12.0

        fatigue = analyzer._assess_fatigue(mock_metrics)
        assert fatigue == FatigueLevel.NONE

    def test_vocal_biomarkers_assess_fatigue_high(self):
        """Test VocalBiomarkers._assess_fatigue returns HIGH for strained voice."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers, FatigueLevel

        analyzer = VocalBiomarkers()
        mock_metrics = Mock()
        mock_metrics.jitter_local = 2.5  # High jitter
        mock_metrics.shimmer_local = 9.0  # High shimmer
        mock_metrics.hnr = 8.0  # Low HNR
        mock_metrics.f0_range = 25.0  # Reduced range
        mock_metrics.spectral_tilt = -18.0  # More breathy

        fatigue = analyzer._assess_fatigue(mock_metrics)
        assert fatigue in [FatigueLevel.HIGH, FatigueLevel.SEVERE]

    def test_vocal_biomarkers_generate_warnings_none(self):
        """Test VocalBiomarkers._generate_warnings returns empty for healthy voice."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        analyzer = VocalBiomarkers()
        mock_metrics = Mock()
        mock_metrics.jitter_local = 0.5
        mock_metrics.shimmer_local = 2.0
        mock_metrics.hnr = 20.0
        mock_metrics.cpp = 8.0

        warnings = analyzer._generate_warnings(mock_metrics)
        assert len(warnings) == 0

    def test_vocal_biomarkers_generate_warnings_critical(self):
        """Test VocalBiomarkers._generate_warnings for critical values."""
        from voice_soundboard.vocology.biomarkers import VocalBiomarkers

        analyzer = VocalBiomarkers()
        mock_metrics = Mock()
        mock_metrics.jitter_local = 4.0  # Very high
        mock_metrics.shimmer_local = 12.0  # Very high
        mock_metrics.hnr = 4.0  # Very low
        mock_metrics.cpp = 1.5  # Very low

        warnings = analyzer._generate_warnings(mock_metrics)
        assert len(warnings) > 0
        assert any("DISCLAIMER" in w for w in warnings)


# ============== VoiceFatigueMonitor Tests ==============

class TestVoiceFatigueMonitor:
    """Tests for VoiceFatigueMonitor class."""

    def test_voice_fatigue_monitor_init(self):
        """Test VoiceFatigueMonitor initialization."""
        from voice_soundboard.vocology.biomarkers import VoiceFatigueMonitor
        monitor = VoiceFatigueMonitor()
        assert monitor.samples == []
        assert monitor.analyzer is not None

    def test_voice_fatigue_monitor_empty_samples(self):
        """Test VoiceFatigueMonitor starts with empty samples."""
        from voice_soundboard.vocology.biomarkers import VoiceFatigueMonitor
        monitor = VoiceFatigueMonitor()
        assert len(monitor.samples) == 0
