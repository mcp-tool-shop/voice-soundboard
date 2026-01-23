"""
Tests for Phase 9: Vocology Module - Batch 7
VocalBiomarkers, BiomarkerResult, VoiceHealthMetrics

Tests cover:
- VoiceHealthStatus enum (TEST-BIO-01 to TEST-BIO-03)
- FatigueLevel enum (TEST-BIO-04 to TEST-BIO-07)
- VoiceHealthMetrics dataclass (TEST-BIO-08 to TEST-BIO-13)
- BiomarkerResult dataclass (TEST-BIO-14 to TEST-BIO-18)
- VocalBiomarkers class (TEST-BIO-19 to TEST-BIO-25)
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch


# =============================================================================
# TEST-BIO-01 to TEST-BIO-03: VoiceHealthStatus Enum Tests
# =============================================================================

from voice_soundboard.vocology.biomarkers import (
    VoiceHealthStatus,
    FatigueLevel,
    VoiceHealthMetrics,
    BiomarkerResult,
    VocalBiomarkers,
    VoiceFatigueMonitor,
    analyze_biomarkers,
    assess_vocal_fatigue,
)


class TestVoiceHealthStatusEnum:
    """Tests for VoiceHealthStatus enum (TEST-BIO-01 to TEST-BIO-03)."""

    def test_bio_01_has_healthy(self):
        """TEST-BIO-01: VoiceHealthStatus.HEALTHY exists."""
        assert hasattr(VoiceHealthStatus, 'HEALTHY')
        assert VoiceHealthStatus.HEALTHY.value == "healthy"

    def test_bio_02_has_mild_strain(self):
        """TEST-BIO-02: VoiceHealthStatus.MILD_STRAIN exists."""
        assert hasattr(VoiceHealthStatus, 'MILD_STRAIN')
        assert VoiceHealthStatus.MILD_STRAIN.value == "mild_strain"

    def test_bio_03_has_needs_attention(self):
        """TEST-BIO-03: VoiceHealthStatus.NEEDS_ATTENTION exists."""
        assert hasattr(VoiceHealthStatus, 'NEEDS_ATTENTION')
        assert VoiceHealthStatus.NEEDS_ATTENTION.value == "needs_attention"


# =============================================================================
# TEST-BIO-04 to TEST-BIO-07: FatigueLevel Enum Tests
# =============================================================================

class TestFatigueLevelEnum:
    """Tests for FatigueLevel enum (TEST-BIO-04 to TEST-BIO-07)."""

    def test_bio_04_has_none(self):
        """TEST-BIO-04: FatigueLevel.NONE exists."""
        assert hasattr(FatigueLevel, 'NONE')
        assert FatigueLevel.NONE.value == "none"

    def test_bio_05_has_low(self):
        """TEST-BIO-05: FatigueLevel.LOW exists."""
        assert hasattr(FatigueLevel, 'LOW')
        assert FatigueLevel.LOW.value == "low"

    def test_bio_06_has_moderate(self):
        """TEST-BIO-06: FatigueLevel.MODERATE exists."""
        assert hasattr(FatigueLevel, 'MODERATE')
        assert FatigueLevel.MODERATE.value == "moderate"

    def test_bio_07_has_severe(self):
        """TEST-BIO-07: FatigueLevel.SEVERE exists."""
        assert hasattr(FatigueLevel, 'SEVERE')
        assert FatigueLevel.SEVERE.value == "severe"


# =============================================================================
# TEST-BIO-08 to TEST-BIO-13: VoiceHealthMetrics Dataclass Tests
# =============================================================================

class TestVoiceHealthMetrics:
    """Tests for VoiceHealthMetrics dataclass (TEST-BIO-08 to TEST-BIO-13)."""

    @pytest.fixture
    def healthy_metrics(self):
        """Create healthy voice metrics."""
        return VoiceHealthMetrics(
            status=VoiceHealthStatus.HEALTHY,
            quality_score=85.0,
            stability_score=90.0,
            clarity_score=88.0,
            concerns=[],
            recommendations=["Voice quality is within normal range"],
        )

    def test_bio_08_has_status(self, healthy_metrics):
        """TEST-BIO-08: VoiceHealthMetrics has status field."""
        assert hasattr(healthy_metrics, 'status')
        assert isinstance(healthy_metrics.status, VoiceHealthStatus)

    def test_bio_09_has_quality_score(self, healthy_metrics):
        """TEST-BIO-09: VoiceHealthMetrics has quality_score field."""
        assert hasattr(healthy_metrics, 'quality_score')
        assert healthy_metrics.quality_score == 85.0

    def test_bio_10_has_stability_score(self, healthy_metrics):
        """TEST-BIO-10: VoiceHealthMetrics has stability_score field."""
        assert hasattr(healthy_metrics, 'stability_score')
        assert healthy_metrics.stability_score == 90.0

    def test_bio_11_has_clarity_score(self, healthy_metrics):
        """TEST-BIO-11: VoiceHealthMetrics has clarity_score field."""
        assert hasattr(healthy_metrics, 'clarity_score')
        assert healthy_metrics.clarity_score == 88.0

    def test_bio_12_has_concerns_list(self, healthy_metrics):
        """TEST-BIO-12: VoiceHealthMetrics has concerns list."""
        assert hasattr(healthy_metrics, 'concerns')
        assert isinstance(healthy_metrics.concerns, list)

    def test_bio_13_overall_score_property(self, healthy_metrics):
        """TEST-BIO-13: overall_score property calculates average."""
        expected = (85.0 + 90.0 + 88.0) / 3
        assert abs(healthy_metrics.overall_score - expected) < 0.01


# =============================================================================
# TEST-BIO-14 to TEST-BIO-18: BiomarkerResult Dataclass Tests
# =============================================================================

class TestBiomarkerResult:
    """Tests for BiomarkerResult dataclass (TEST-BIO-14 to TEST-BIO-18)."""

    @pytest.fixture
    def mock_result(self):
        """Create mock BiomarkerResult."""
        health_metrics = VoiceHealthMetrics(
            status=VoiceHealthStatus.HEALTHY,
            quality_score=85.0,
            stability_score=90.0,
            clarity_score=88.0,
        )
        return BiomarkerResult(
            health_metrics=health_metrics,
            fatigue_level=FatigueLevel.LOW,
            voice_quality={"jitter": 0.4, "shimmer": 2.0, "hnr": 20.0},
            timestamp=datetime.now(),
            audio_duration=3.5,
            warnings=[],
        )

    def test_bio_14_has_health_metrics(self, mock_result):
        """TEST-BIO-14: BiomarkerResult has health_metrics field."""
        assert hasattr(mock_result, 'health_metrics')
        assert isinstance(mock_result.health_metrics, VoiceHealthMetrics)

    def test_bio_15_has_fatigue_level(self, mock_result):
        """TEST-BIO-15: BiomarkerResult has fatigue_level field."""
        assert hasattr(mock_result, 'fatigue_level')
        assert isinstance(mock_result.fatigue_level, FatigueLevel)

    def test_bio_16_has_voice_quality(self, mock_result):
        """TEST-BIO-16: BiomarkerResult has voice_quality dict."""
        assert hasattr(mock_result, 'voice_quality')
        assert isinstance(mock_result.voice_quality, dict)

    def test_bio_17_has_timestamp(self, mock_result):
        """TEST-BIO-17: BiomarkerResult has timestamp field."""
        assert hasattr(mock_result, 'timestamp')
        assert isinstance(mock_result.timestamp, datetime)

    def test_bio_18_to_dict_returns_dict(self, mock_result):
        """TEST-BIO-18: to_dict() returns serializable dictionary."""
        result = mock_result.to_dict()
        assert isinstance(result, dict)
        assert "health" in result
        assert "fatigue" in result
        assert "voice_quality" in result


# =============================================================================
# TEST-BIO-19 to TEST-BIO-25: VocalBiomarkers Class Tests
# =============================================================================

class TestVocalBiomarkers:
    """Tests for VocalBiomarkers class (TEST-BIO-19 to TEST-BIO-25)."""

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

    def test_bio_19_init(self):
        """TEST-BIO-19: VocalBiomarkers initializes without error."""
        analyzer = VocalBiomarkers()
        assert analyzer is not None

    def test_bio_20_analyze_returns_result(self, mock_audio_array):
        """TEST-BIO-20: analyze() returns BiomarkerResult."""
        audio, sr = mock_audio_array
        analyzer = VocalBiomarkers()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert isinstance(result, BiomarkerResult)

    def test_bio_21_analyze_has_health_status(self, mock_audio_array):
        """TEST-BIO-21: analyze() result has valid health status."""
        audio, sr = mock_audio_array
        analyzer = VocalBiomarkers()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert isinstance(result.health_metrics.status, VoiceHealthStatus)

    def test_bio_22_analyze_has_fatigue_level(self, mock_audio_array):
        """TEST-BIO-22: analyze() result has valid fatigue level."""
        audio, sr = mock_audio_array
        analyzer = VocalBiomarkers()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert isinstance(result.fatigue_level, FatigueLevel)

    def test_bio_23_analyze_scores_in_range(self, mock_audio_array):
        """TEST-BIO-23: analyze() scores are in valid range (0-100)."""
        audio, sr = mock_audio_array
        analyzer = VocalBiomarkers()
        result = analyzer.analyze(audio, sample_rate=sr)
        assert 0 <= result.health_metrics.quality_score <= 100
        assert 0 <= result.health_metrics.stability_score <= 100
        assert 0 <= result.health_metrics.clarity_score <= 100

    def test_bio_24_analyze_has_audio_duration(self, mock_audio_array):
        """TEST-BIO-24: analyze() result has audio duration."""
        audio, sr = mock_audio_array
        analyzer = VocalBiomarkers()
        result = analyzer.analyze(audio, sample_rate=sr)
        expected = len(audio) / sr
        assert abs(result.audio_duration - expected) < 0.01

    def test_bio_25_healthy_thresholds_defined(self):
        """TEST-BIO-25: VocalBiomarkers has HEALTHY_THRESHOLDS defined."""
        assert hasattr(VocalBiomarkers, 'HEALTHY_THRESHOLDS')
        assert "jitter_max" in VocalBiomarkers.HEALTHY_THRESHOLDS
        assert "shimmer_max" in VocalBiomarkers.HEALTHY_THRESHOLDS
        assert "hnr_min" in VocalBiomarkers.HEALTHY_THRESHOLDS
