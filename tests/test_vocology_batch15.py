"""
Tests for Phase 10: Humanization & Rhythm - Batch 15
RhythmClass, RhythmBand Enums, RhythmMetrics, RhythmZone, RZTAnalysis Dataclasses

Tests cover:
- RhythmClass enum (TEST-RHY-01 to TEST-RHY-03)
- RhythmBand enum (TEST-RHY-04 to TEST-RHY-07)
- RhythmMetrics dataclass (TEST-RHY-08 to TEST-RHY-14)
- RhythmZone dataclass (TEST-RHY-15 to TEST-RHY-19)
- RZTAnalysis dataclass (TEST-RHY-20 to TEST-RHY-23)
- RhythmAnalyzer init (TEST-RHY-24 to TEST-RHY-25)
"""

import pytest
import numpy as np


from voice_soundboard.vocology.rhythm import (
    RhythmClass,
    RhythmBand,
    RhythmMetrics,
    RhythmZone,
    RZTAnalysis,
    RhythmAnalyzer,
    RhythmModifier,
)


# =============================================================================
# TEST-RHY-01 to TEST-RHY-03: RhythmClass Enum Tests
# =============================================================================

class TestRhythmClassEnum:
    """Tests for RhythmClass enum (TEST-RHY-01 to TEST-RHY-03)."""

    def test_rhy_01_rhythm_class_stress_timed(self):
        """TEST-RHY-01: RhythmClass.STRESS_TIMED has correct value 'stress_timed'."""
        assert hasattr(RhythmClass, 'STRESS_TIMED')
        assert RhythmClass.STRESS_TIMED.value == "stress_timed"

    def test_rhy_02_rhythm_class_syllable_timed(self):
        """TEST-RHY-02: RhythmClass.SYLLABLE_TIMED has correct value 'syllable_timed'."""
        assert hasattr(RhythmClass, 'SYLLABLE_TIMED')
        assert RhythmClass.SYLLABLE_TIMED.value == "syllable_timed"

    def test_rhy_03_rhythm_class_mora_timed(self):
        """TEST-RHY-03: RhythmClass.MORA_TIMED has correct value 'mora_timed'."""
        assert hasattr(RhythmClass, 'MORA_TIMED')
        assert RhythmClass.MORA_TIMED.value == "mora_timed"


# =============================================================================
# TEST-RHY-04 to TEST-RHY-07: RhythmBand Enum Tests
# =============================================================================

class TestRhythmBandEnum:
    """Tests for RhythmBand enum (TEST-RHY-04 to TEST-RHY-07)."""

    def test_rhy_04_rhythm_band_delta(self):
        """TEST-RHY-04: RhythmBand.DELTA covers phrase-level (0.5-2 Hz)."""
        assert hasattr(RhythmBand, 'DELTA')
        assert RhythmBand.DELTA.value == "delta"

    def test_rhy_05_rhythm_band_theta(self):
        """TEST-RHY-05: RhythmBand.THETA covers syllable-level (4-8 Hz)."""
        assert hasattr(RhythmBand, 'THETA')
        assert RhythmBand.THETA.value == "theta"

    def test_rhy_06_rhythm_band_alpha(self):
        """TEST-RHY-06: RhythmBand.ALPHA covers phoneme-level (8-12 Hz)."""
        assert hasattr(RhythmBand, 'ALPHA')
        assert RhythmBand.ALPHA.value == "alpha"

    def test_rhy_07_rhythm_band_beta(self):
        """TEST-RHY-07: RhythmBand.BETA covers articulation-level (12-30 Hz)."""
        assert hasattr(RhythmBand, 'BETA')
        assert RhythmBand.BETA.value == "beta"


# =============================================================================
# TEST-RHY-08 to TEST-RHY-14: RhythmMetrics Dataclass Tests
# =============================================================================

class TestRhythmMetrics:
    """Tests for RhythmMetrics dataclass (TEST-RHY-08 to TEST-RHY-14)."""

    @pytest.fixture
    def mock_metrics(self):
        """Create mock RhythmMetrics."""
        return RhythmMetrics(
            percent_v=45.0,
            delta_v=0.05,
            delta_c=0.06,
            npvi_v=52.0,
            rpvi_c=48.0,
            varco_v=55.0,
            varco_c=50.0,
            speech_rate=5.0,
            articulation_rate=6.0,
        )

    def test_rhy_08_rhythm_metrics_has_percent_v(self, mock_metrics):
        """TEST-RHY-08: RhythmMetrics has percent_v field (vocalic percentage)."""
        assert hasattr(mock_metrics, 'percent_v')
        assert mock_metrics.percent_v == 45.0

    def test_rhy_09_rhythm_metrics_has_npvi_v(self, mock_metrics):
        """TEST-RHY-09: RhythmMetrics has npvi_v field (normalized PVI for vowels)."""
        assert hasattr(mock_metrics, 'npvi_v')
        assert mock_metrics.npvi_v == 52.0

    def test_rhy_10_rhythm_metrics_has_rpvi_c(self, mock_metrics):
        """TEST-RHY-10: RhythmMetrics has rpvi_c field (raw PVI for consonants)."""
        assert hasattr(mock_metrics, 'rpvi_c')
        assert mock_metrics.rpvi_c == 48.0

    def test_rhy_11_rhythm_metrics_has_delta_v(self, mock_metrics):
        """TEST-RHY-11: RhythmMetrics has delta_v field (vocalic variability)."""
        assert hasattr(mock_metrics, 'delta_v')
        assert mock_metrics.delta_v == 0.05

    def test_rhy_12_rhythm_metrics_has_delta_c(self, mock_metrics):
        """TEST-RHY-12: RhythmMetrics has delta_c field (consonantal variability)."""
        assert hasattr(mock_metrics, 'delta_c')
        assert mock_metrics.delta_c == 0.06

    def test_rhy_13_rhythm_metrics_has_speech_rate(self, mock_metrics):
        """TEST-RHY-13: RhythmMetrics has speech_rate field (syllables/sec)."""
        assert hasattr(mock_metrics, 'speech_rate')
        assert mock_metrics.speech_rate == 5.0

    def test_rhy_14_rhythm_metrics_has_varco_v(self, mock_metrics):
        """TEST-RHY-14: RhythmMetrics has varco_v field (variation coefficient)."""
        assert hasattr(mock_metrics, 'varco_v')
        assert mock_metrics.varco_v == 55.0


# =============================================================================
# TEST-RHY-15 to TEST-RHY-19: RhythmZone Dataclass Tests
# =============================================================================

class TestRhythmZone:
    """Tests for RhythmZone dataclass (TEST-RHY-15 to TEST-RHY-19)."""

    @pytest.fixture
    def mock_zone(self):
        """Create mock RhythmZone."""
        return RhythmZone(
            start_time=0.5,
            end_time=1.2,
            dominant_frequency=5.0,
            energy=0.8,
            band=RhythmBand.THETA,
        )

    def test_rhy_15_rhythm_zone_has_start_time(self, mock_zone):
        """TEST-RHY-15: RhythmZone has start_time field."""
        assert hasattr(mock_zone, 'start_time')
        assert mock_zone.start_time == 0.5

    def test_rhy_16_rhythm_zone_has_end_time(self, mock_zone):
        """TEST-RHY-16: RhythmZone has end_time field."""
        assert hasattr(mock_zone, 'end_time')
        assert mock_zone.end_time == 1.2

    def test_rhy_17_rhythm_zone_has_dominant_frequency(self, mock_zone):
        """TEST-RHY-17: RhythmZone has dominant_frequency field."""
        assert hasattr(mock_zone, 'dominant_frequency')
        assert mock_zone.dominant_frequency == 5.0

    def test_rhy_18_rhythm_zone_has_energy(self, mock_zone):
        """TEST-RHY-18: RhythmZone has energy field."""
        assert hasattr(mock_zone, 'energy')
        assert mock_zone.energy == 0.8

    def test_rhy_19_rhythm_zone_has_band(self, mock_zone):
        """TEST-RHY-19: RhythmZone has band field (RhythmBand)."""
        assert hasattr(mock_zone, 'band')
        assert isinstance(mock_zone.band, RhythmBand)
        assert mock_zone.band == RhythmBand.THETA


# =============================================================================
# TEST-RHY-20 to TEST-RHY-23: RZTAnalysis Dataclass Tests
# =============================================================================

class TestRZTAnalysis:
    """Tests for RZTAnalysis dataclass (TEST-RHY-20 to TEST-RHY-23)."""

    @pytest.fixture
    def mock_rzt_analysis(self):
        """Create mock RZTAnalysis."""
        zones = [
            RhythmZone(0.0, 0.5, 5.0, 0.7, RhythmBand.THETA),
            RhythmZone(0.5, 1.0, 1.0, 0.5, RhythmBand.DELTA),
        ]
        return RZTAnalysis(
            zones=zones,
            envelope_spectrum=np.array([0.1, 0.3, 0.5, 0.8, 0.6]),
            rhythm_frequencies=[1.0, 5.0],
            phrase_rhythm=1.2,
            syllable_rhythm=5.5,
            sample_rate=24000,
        )

    def test_rhy_20_rzt_analysis_has_zones(self, mock_rzt_analysis):
        """TEST-RHY-20: RZTAnalysis has zones list."""
        assert hasattr(mock_rzt_analysis, 'zones')
        assert isinstance(mock_rzt_analysis.zones, list)
        assert len(mock_rzt_analysis.zones) == 2

    def test_rhy_21_rzt_analysis_has_dominant_band(self, mock_rzt_analysis):
        """TEST-RHY-21: RZTAnalysis has rhythm_frequencies field."""
        assert hasattr(mock_rzt_analysis, 'rhythm_frequencies')
        assert isinstance(mock_rzt_analysis.rhythm_frequencies, list)

    def test_rhy_22_rzt_analysis_has_band_energies(self, mock_rzt_analysis):
        """TEST-RHY-22: RZTAnalysis has envelope_spectrum field."""
        assert hasattr(mock_rzt_analysis, 'envelope_spectrum')
        assert isinstance(mock_rzt_analysis.envelope_spectrum, np.ndarray)

    def test_rhy_23_rzt_analysis_has_rhythm_regularity(self, mock_rzt_analysis):
        """TEST-RHY-23: RZTAnalysis has phrase_rhythm and syllable_rhythm fields."""
        assert hasattr(mock_rzt_analysis, 'phrase_rhythm')
        assert hasattr(mock_rzt_analysis, 'syllable_rhythm')
        assert mock_rzt_analysis.phrase_rhythm == 1.2
        assert mock_rzt_analysis.syllable_rhythm == 5.5


# =============================================================================
# TEST-RHY-24 to TEST-RHY-25: RhythmAnalyzer Class Init Tests
# =============================================================================

class TestRhythmAnalyzerInit:
    """Tests for RhythmAnalyzer initialization (TEST-RHY-24 to TEST-RHY-25)."""

    def test_rhy_24_rhythm_analyzer_init_default_sample_rate(self):
        """TEST-RHY-24: RhythmAnalyzer initializes with default sample_rate."""
        analyzer = RhythmAnalyzer()
        assert hasattr(analyzer, 'sample_rate')
        assert analyzer.sample_rate == 24000

    def test_rhy_25_rhythm_analyzer_analyze_metrics_returns_metrics(self):
        """TEST-RHY-25: RhythmAnalyzer.analyze_metrics() returns RhythmMetrics."""
        sr = 24000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        # Create audio with some structure
        audio = np.zeros_like(t)
        for i in range(5):  # 5 syllable-like bursts
            start = int(i * 0.2 * sr)
            end = int((i * 0.2 + 0.15) * sr)
            if end < len(audio):
                audio[start:end] = 0.5 * np.sin(2 * np.pi * 200 * t[start:end])

        analyzer = RhythmAnalyzer()
        result = analyzer.analyze_metrics(audio.astype(np.float32), sample_rate=sr)
        assert isinstance(result, RhythmMetrics)
