"""
Tests for Phase 9: Vocology Module - Batch 1
VoiceQualityMetrics dataclass, JitterType enum, ShimmerType enum

Tests cover:
- JitterType enum values and types (TEST-VQM-01 to TEST-VQM-04)
- ShimmerType enum values and types (TEST-VQM-05 to TEST-VQM-09)
- VoiceQualityMetrics dataclass fields (TEST-VQM-10 to TEST-VQM-25)
"""

import pytest
from dataclasses import fields


# =============================================================================
# TEST-VQM-01 to TEST-VQM-04: JitterType Enum Tests
# =============================================================================

from voice_soundboard.vocology.parameters import JitterType


class TestJitterTypeEnum:
    """Tests for JitterType enum (TEST-VQM-01 to TEST-VQM-04)."""

    def test_vqm_01_jitter_type_has_local(self):
        """TEST-VQM-01: JitterType.LOCAL exists with value 'local'."""
        assert hasattr(JitterType, 'LOCAL')
        assert JitterType.LOCAL.value == "local"

    def test_vqm_02_jitter_type_has_rap(self):
        """TEST-VQM-02: JitterType.RAP exists with value 'rap'."""
        assert hasattr(JitterType, 'RAP')
        assert JitterType.RAP.value == "rap"

    def test_vqm_03_jitter_type_has_ppq5(self):
        """TEST-VQM-03: JitterType.PPQ5 exists with value 'ppq5'."""
        assert hasattr(JitterType, 'PPQ5')
        assert JitterType.PPQ5.value == "ppq5"

    def test_vqm_04_jitter_type_has_ddp(self):
        """TEST-VQM-04: JitterType.DDP exists with value 'ddp'."""
        assert hasattr(JitterType, 'DDP')
        assert JitterType.DDP.value == "ddp"


# =============================================================================
# TEST-VQM-05 to TEST-VQM-09: ShimmerType Enum Tests
# =============================================================================

from voice_soundboard.vocology.parameters import ShimmerType


class TestShimmerTypeEnum:
    """Tests for ShimmerType enum (TEST-VQM-05 to TEST-VQM-09)."""

    def test_vqm_05_shimmer_type_has_local(self):
        """TEST-VQM-05: ShimmerType.LOCAL exists with value 'local'."""
        assert hasattr(ShimmerType, 'LOCAL')
        assert ShimmerType.LOCAL.value == "local"

    def test_vqm_06_shimmer_type_has_apq3(self):
        """TEST-VQM-06: ShimmerType.APQ3 exists with value 'apq3'."""
        assert hasattr(ShimmerType, 'APQ3')
        assert ShimmerType.APQ3.value == "apq3"

    def test_vqm_07_shimmer_type_has_apq5(self):
        """TEST-VQM-07: ShimmerType.APQ5 exists with value 'apq5'."""
        assert hasattr(ShimmerType, 'APQ5')
        assert ShimmerType.APQ5.value == "apq5"

    def test_vqm_08_shimmer_type_has_apq11(self):
        """TEST-VQM-08: ShimmerType.APQ11 exists with value 'apq11'."""
        assert hasattr(ShimmerType, 'APQ11')
        assert ShimmerType.APQ11.value == "apq11"

    def test_vqm_09_shimmer_type_has_dda(self):
        """TEST-VQM-09: ShimmerType.DDA exists with value 'dda'."""
        assert hasattr(ShimmerType, 'DDA')
        assert ShimmerType.DDA.value == "dda"


# =============================================================================
# TEST-VQM-10 to TEST-VQM-25: VoiceQualityMetrics Dataclass Tests
# =============================================================================

from voice_soundboard.vocology.parameters import VoiceQualityMetrics


class TestVoiceQualityMetricsDataclass:
    """Tests for VoiceQualityMetrics dataclass (TEST-VQM-10 to TEST-VQM-25)."""

    @pytest.fixture
    def healthy_metrics(self):
        """Create metrics within healthy ranges."""
        return VoiceQualityMetrics(
            f0_mean=120.0,
            f0_std=10.0,
            f0_range=50.0,
            jitter_local=0.3,
            jitter_rap=0.2,
            jitter_ppq5=0.25,
            shimmer_local=2.0,
            shimmer_apq3=1.5,
            shimmer_apq5=1.8,
            shimmer_apq11=2.2,
            hnr=22.0,
            nhr=0.006,
            cpp=8.0,
            spectral_tilt=-12.0,
            spectral_centroid=1500.0,
            voiced_fraction=0.85,
            duration=3.5,
        )

    @pytest.fixture
    def unhealthy_metrics(self):
        """Create metrics outside healthy ranges."""
        return VoiceQualityMetrics(
            f0_mean=120.0,
            f0_std=30.0,
            f0_range=100.0,
            jitter_local=2.5,
            jitter_rap=2.0,
            jitter_ppq5=2.2,
            shimmer_local=9.0,
            shimmer_apq3=8.0,
            shimmer_apq5=8.5,
            shimmer_apq11=9.5,
            hnr=8.0,
            nhr=0.16,
            cpp=3.0,
            spectral_tilt=-18.0,
            spectral_centroid=1200.0,
            voiced_fraction=0.6,
            duration=2.0,
        )

    def test_vqm_10_has_f0_mean_field(self, healthy_metrics):
        """TEST-VQM-10: VoiceQualityMetrics has f0_mean field."""
        assert hasattr(healthy_metrics, 'f0_mean')
        assert healthy_metrics.f0_mean == 120.0

    def test_vqm_11_has_f0_std_field(self, healthy_metrics):
        """TEST-VQM-11: VoiceQualityMetrics has f0_std field."""
        assert hasattr(healthy_metrics, 'f0_std')
        assert healthy_metrics.f0_std == 10.0

    def test_vqm_12_has_jitter_local_field(self, healthy_metrics):
        """TEST-VQM-12: VoiceQualityMetrics has jitter_local field."""
        assert hasattr(healthy_metrics, 'jitter_local')
        assert healthy_metrics.jitter_local == 0.3

    def test_vqm_13_has_shimmer_local_field(self, healthy_metrics):
        """TEST-VQM-13: VoiceQualityMetrics has shimmer_local field."""
        assert hasattr(healthy_metrics, 'shimmer_local')
        assert healthy_metrics.shimmer_local == 2.0

    def test_vqm_14_has_hnr_field(self, healthy_metrics):
        """TEST-VQM-14: VoiceQualityMetrics has hnr field."""
        assert hasattr(healthy_metrics, 'hnr')
        assert healthy_metrics.hnr == 22.0

    def test_vqm_15_has_cpp_field(self, healthy_metrics):
        """TEST-VQM-15: VoiceQualityMetrics has cpp field."""
        assert hasattr(healthy_metrics, 'cpp')
        assert healthy_metrics.cpp == 8.0

    def test_vqm_16_jitter_percent_property(self, healthy_metrics):
        """TEST-VQM-16: jitter_percent property returns jitter_local."""
        assert healthy_metrics.jitter_percent == healthy_metrics.jitter_local

    def test_vqm_17_shimmer_percent_property(self, healthy_metrics):
        """TEST-VQM-17: shimmer_percent property returns shimmer_local."""
        assert healthy_metrics.shimmer_percent == healthy_metrics.shimmer_local

    def test_vqm_18_hnr_db_property(self, healthy_metrics):
        """TEST-VQM-18: hnr_db property returns hnr."""
        assert healthy_metrics.hnr_db == healthy_metrics.hnr

    def test_vqm_19_is_healthy_returns_true_for_healthy(self, healthy_metrics):
        """TEST-VQM-19: is_healthy() returns True for healthy metrics."""
        assert healthy_metrics.is_healthy() is True

    def test_vqm_20_is_healthy_returns_false_for_unhealthy(self, unhealthy_metrics):
        """TEST-VQM-20: is_healthy() returns False for unhealthy metrics."""
        assert unhealthy_metrics.is_healthy() is False

    def test_vqm_21_quality_assessment_excellent(self, healthy_metrics):
        """TEST-VQM-21: quality_assessment() returns 'excellent' for healthy voice."""
        assert healthy_metrics.quality_assessment() == "excellent"

    def test_vqm_22_quality_assessment_poor(self, unhealthy_metrics):
        """TEST-VQM-22: quality_assessment() returns 'poor' for unhealthy voice."""
        assert unhealthy_metrics.quality_assessment() == "poor"

    def test_vqm_23_quality_assessment_good(self):
        """TEST-VQM-23: quality_assessment() returns 'good' for moderate metrics."""
        metrics = VoiceQualityMetrics(
            f0_mean=120.0, f0_std=15.0, f0_range=60.0,
            jitter_local=0.7, jitter_rap=0.5, jitter_ppq5=0.6,
            shimmer_local=4.0, shimmer_apq3=3.5, shimmer_apq5=3.8, shimmer_apq11=4.2,
            hnr=17.0, nhr=0.02, cpp=6.0,
            spectral_tilt=-14.0, spectral_centroid=1400.0,
            voiced_fraction=0.8, duration=3.0,
        )
        assert metrics.quality_assessment() == "good"

    def test_vqm_24_to_dict_returns_dict(self, healthy_metrics):
        """TEST-VQM-24: to_dict() returns a dictionary."""
        result = healthy_metrics.to_dict()
        assert isinstance(result, dict)

    def test_vqm_25_to_dict_has_expected_keys(self, healthy_metrics):
        """TEST-VQM-25: to_dict() has expected top-level keys."""
        result = healthy_metrics.to_dict()
        expected_keys = ["f0", "jitter", "shimmer", "noise", "spectral", "general"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
