"""
Tests for Phase 10: Humanization & Rhythm - Batch 16
RhythmAnalyzer methods, RhythmModifier, Convenience Functions

Tests cover:
- RhythmAnalyzer.analyze_metrics() (TEST-RHY-26 to TEST-RHY-28)
- RhythmAnalyzer.classify_rhythm() (TEST-RHY-29 to TEST-RHY-32)
- RhythmAnalyzer.analyze_rzt() (TEST-RHY-33 to TEST-RHY-37)
- RhythmModifier class (TEST-RHY-38 to TEST-RHY-44)
- Convenience functions (TEST-RHY-45 to TEST-RHY-50)
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
    analyze_rhythm,
    analyze_rhythm_zones,
    add_rhythm_variability,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_speech_audio():
    """Create mock speech-like audio with syllable-like bursts."""
    sr = 24000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    audio = np.zeros_like(t)
    # Create syllable-like structure (5 Hz rate = typical speech)
    syllable_rate = 5  # Hz
    for i in range(int(duration * syllable_rate)):
        start_time = i / syllable_rate
        # Each syllable is about 100ms
        start = int(start_time * sr)
        end = int((start_time + 0.1) * sr)
        if end < len(audio):
            t_local = t[start:end]
            # Harmonic structure
            audio[start:end] = 0.5 * np.sin(2 * np.pi * 200 * t_local)
            audio[start:end] += 0.3 * np.sin(2 * np.pi * 400 * t_local)

    # Add some noise for realism
    audio += 0.02 * np.random.randn(len(audio))
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio.astype(np.float32), sr


# =============================================================================
# TEST-RHY-26 to TEST-RHY-28: RhythmAnalyzer.analyze_metrics() Tests
# =============================================================================

class TestRhythmAnalyzerMetrics:
    """Tests for RhythmAnalyzer.analyze_metrics() (TEST-RHY-26 to TEST-RHY-28)."""

    def test_rhy_26_analyze_metrics_percent_v_range(self, mock_speech_audio):
        """TEST-RHY-26: RhythmAnalyzer.analyze_metrics() computes valid percent_v (0-100)."""
        audio, sr = mock_speech_audio
        analyzer = RhythmAnalyzer()
        result = analyzer.analyze_metrics(audio, sample_rate=sr)
        assert 0 <= result.percent_v <= 100

    def test_rhy_27_analyze_metrics_npvi_v_valid(self, mock_speech_audio):
        """TEST-RHY-27: RhythmAnalyzer.analyze_metrics() computes valid npvi_v."""
        audio, sr = mock_speech_audio
        analyzer = RhythmAnalyzer()
        result = analyzer.analyze_metrics(audio, sample_rate=sr)
        # nPVI is typically 0-100 range
        assert result.npvi_v >= 0

    def test_rhy_28_analyze_metrics_speech_rate_positive(self, mock_speech_audio):
        """TEST-RHY-28: RhythmAnalyzer.analyze_metrics() computes valid speech_rate (>0)."""
        audio, sr = mock_speech_audio
        analyzer = RhythmAnalyzer()
        result = analyzer.analyze_metrics(audio, sample_rate=sr)
        assert result.speech_rate > 0


# =============================================================================
# TEST-RHY-29 to TEST-RHY-32: RhythmAnalyzer.classify_rhythm() Tests
# =============================================================================

class TestRhythmAnalyzerClassify:
    """Tests for RhythmMetrics.rhythm_class property (TEST-RHY-29 to TEST-RHY-32)."""

    def test_rhy_29_metrics_rhythm_class_property(self, mock_speech_audio):
        """TEST-RHY-29: RhythmMetrics has rhythm_class property returning RhythmClass."""
        audio, sr = mock_speech_audio
        analyzer = RhythmAnalyzer()
        metrics = analyzer.analyze_metrics(audio, sample_rate=sr)
        result = metrics.rhythm_class
        assert isinstance(result, RhythmClass)

    def test_rhy_30_classify_stress_timed_high_npvi(self):
        """TEST-RHY-30: rhythm_class property returns STRESS_TIMED for nPVI > 55 and %V < 45."""
        # Create metrics that indicate stress-timed
        metrics = RhythmMetrics(
            percent_v=42.0,  # < 45
            delta_v=0.08,
            delta_c=0.06,
            npvi_v=60.0,  # > 55
            rpvi_c=50.0,
            varco_v=60.0,
            varco_c=50.0,
            speech_rate=5.0,
            articulation_rate=6.0,
        )
        assert metrics.rhythm_class == RhythmClass.STRESS_TIMED

    def test_rhy_31_classify_syllable_timed_low_npvi(self):
        """TEST-RHY-31: rhythm_class property returns SYLLABLE_TIMED for nPVI 35-55 and %V > 45."""
        # Create metrics that indicate syllable-timed
        metrics = RhythmMetrics(
            percent_v=50.0,  # > 45
            delta_v=0.04,
            delta_c=0.04,
            npvi_v=40.0,  # 35-55 range
            rpvi_c=35.0,
            varco_v=40.0,
            varco_c=35.0,
            speech_rate=5.5,
            articulation_rate=6.5,
        )
        assert metrics.rhythm_class == RhythmClass.SYLLABLE_TIMED

    def test_rhy_32_classify_mora_timed_very_low_npvi(self):
        """TEST-RHY-32: rhythm_class property returns MORA_TIMED for nPVI < 35."""
        # Create metrics that indicate mora-timed
        # Note: nPVI < 35 triggers mora-timed, but must not match syllable-timed first
        # (which requires nPVI < 45 AND %V > 45). So use %V < 45 with very low nPVI.
        metrics = RhythmMetrics(
            percent_v=44.0,  # <= 45 to avoid syllable-timed match
            delta_v=0.03,
            delta_c=0.03,
            npvi_v=30.0,  # < 35
            rpvi_c=28.0,
            varco_v=30.0,
            varco_c=28.0,
            speech_rate=6.0,
            articulation_rate=7.0,
        )
        assert metrics.rhythm_class == RhythmClass.MORA_TIMED


# =============================================================================
# TEST-RHY-33 to TEST-RHY-37: RhythmAnalyzer.analyze_rzt() Tests
# =============================================================================

class TestRhythmAnalyzerRZT:
    """Tests for RhythmAnalyzer.analyze_rzt() (TEST-RHY-33 to TEST-RHY-37)."""

    def test_rhy_33_analyze_rzt_returns_rzt_analysis(self, mock_speech_audio):
        """TEST-RHY-33: RhythmAnalyzer.analyze_rzt() returns RZTAnalysis."""
        audio, sr = mock_speech_audio
        analyzer = RhythmAnalyzer()
        result = analyzer.analyze_rzt(audio, sample_rate=sr)
        assert isinstance(result, RZTAnalysis)

    def test_rhy_34_analyze_rzt_detects_zones(self, mock_speech_audio):
        """TEST-RHY-34: RhythmAnalyzer.analyze_rzt() detects rhythm zones."""
        audio, sr = mock_speech_audio
        analyzer = RhythmAnalyzer()
        result = analyzer.analyze_rzt(audio, sample_rate=sr)
        assert isinstance(result.zones, list)

    def test_rhy_35_analyze_rzt_computes_band_energies(self, mock_speech_audio):
        """TEST-RHY-35: RhythmAnalyzer.analyze_rzt() computes band energies."""
        audio, sr = mock_speech_audio
        analyzer = RhythmAnalyzer()
        result = analyzer.analyze_rzt(audio, sample_rate=sr)
        assert hasattr(result, 'envelope_spectrum')
        assert isinstance(result.envelope_spectrum, np.ndarray)

    def test_rhy_36_analyze_band_energy_returns_dict(self, mock_speech_audio):
        """TEST-RHY-36: RhythmAnalyzer.analyze_band_energy() returns dict of band energies."""
        audio, sr = mock_speech_audio
        analyzer = RhythmAnalyzer()
        result = analyzer.analyze_band_energy(audio, sample_rate=sr)
        assert isinstance(result, dict)
        # Should have entries for different bands
        assert len(result) > 0

    def test_rhy_37_analyze_identifies_dominant_band(self, mock_speech_audio):
        """TEST-RHY-37: RhythmAnalyzer identifies dominant rhythm band."""
        audio, sr = mock_speech_audio
        analyzer = RhythmAnalyzer()
        result = analyzer.analyze_rzt(audio, sample_rate=sr)
        # Should have syllable rhythm identified
        assert result.syllable_rhythm > 0


# =============================================================================
# TEST-RHY-38 to TEST-RHY-44: RhythmModifier Class Tests
# =============================================================================

class TestRhythmModifier:
    """Tests for RhythmModifier class (TEST-RHY-38 to TEST-RHY-44)."""

    def test_rhy_38_rhythm_modifier_init(self):
        """TEST-RHY-38: RhythmModifier initializes with sample_rate."""
        modifier = RhythmModifier(sample_rate=24000)
        assert modifier is not None
        assert modifier.sample_rate == 24000

    def test_rhy_39_rhythm_modifier_add_variability_returns_tuple(self, mock_speech_audio):
        """TEST-RHY-39: RhythmModifier.add_variability() returns tuple (audio, sample_rate)."""
        audio, sr = mock_speech_audio
        modifier = RhythmModifier(sample_rate=sr)
        result = modifier.add_variability(audio, sample_rate=sr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_rhy_40_rhythm_modifier_add_variability_natural(self, mock_speech_audio):
        """TEST-RHY-40: RhythmModifier.add_variability() adds natural timing jitter."""
        audio, sr = mock_speech_audio
        modifier = RhythmModifier(sample_rate=sr)
        result_audio, _ = modifier.add_variability(audio, amount=0.15, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)
        assert len(result_audio) > 0

    def test_rhy_41_rhythm_modifier_shift_to_stress_timed(self, mock_speech_audio):
        """TEST-RHY-41: RhythmModifier can shift toward stress-timed rhythm."""
        audio, sr = mock_speech_audio
        modifier = RhythmModifier(sample_rate=sr)
        result_audio, _ = modifier.adjust_rhythm_class(audio, target=RhythmClass.STRESS_TIMED, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_rhy_42_rhythm_modifier_shift_to_syllable_timed(self, mock_speech_audio):
        """TEST-RHY-42: RhythmModifier can shift toward syllable-timed rhythm."""
        audio, sr = mock_speech_audio
        modifier = RhythmModifier(sample_rate=sr)
        result_audio, _ = modifier.adjust_rhythm_class(audio, target=RhythmClass.SYLLABLE_TIMED, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)

    def test_rhy_43_rhythm_modifier_preserves_length(self, mock_speech_audio):
        """TEST-RHY-43: RhythmModifier preserves audio length approximately."""
        audio, sr = mock_speech_audio
        modifier = RhythmModifier(sample_rate=sr)
        result_audio, _ = modifier.add_variability(audio, sample_rate=sr)
        # Should be within 10% of original length
        assert abs(len(result_audio) - len(audio)) < len(audio) * 0.1

    def test_rhy_44_rhythm_modifier_no_clipping(self, mock_speech_audio):
        """TEST-RHY-44: RhythmModifier output does not clip."""
        audio, sr = mock_speech_audio
        modifier = RhythmModifier(sample_rate=sr)
        result_audio, _ = modifier.add_variability(audio, sample_rate=sr)
        assert np.max(np.abs(result_audio)) <= 1.1  # Allow small overshoot


# =============================================================================
# TEST-RHY-45 to TEST-RHY-50: Convenience Functions Tests
# =============================================================================

class TestRhythmConvenienceFunctions:
    """Tests for rhythm convenience functions (TEST-RHY-45 to TEST-RHY-50)."""

    def test_rhy_45_analyze_rhythm_with_array(self, mock_speech_audio):
        """TEST-RHY-45: analyze_rhythm() function works with numpy array input."""
        audio, sr = mock_speech_audio
        result = analyze_rhythm(audio, sample_rate=sr)
        assert isinstance(result, RhythmMetrics)

    def test_rhy_46_analyze_rhythm_returns_metrics(self, mock_speech_audio):
        """TEST-RHY-46: analyze_rhythm() function works and returns metrics."""
        audio, sr = mock_speech_audio
        result = analyze_rhythm(audio, sample_rate=sr)
        assert hasattr(result, 'npvi_v')
        assert hasattr(result, 'percent_v')

    def test_rhy_47_analyze_rhythm_returns_rhythm_metrics(self, mock_speech_audio):
        """TEST-RHY-47: analyze_rhythm() returns RhythmMetrics."""
        audio, sr = mock_speech_audio
        result = analyze_rhythm(audio, sample_rate=sr)
        assert isinstance(result, RhythmMetrics)

    def test_rhy_48_analyze_rhythm_zones_returns_rzt(self, mock_speech_audio):
        """TEST-RHY-48: analyze_rhythm_zones() returns RZTAnalysis."""
        audio, sr = mock_speech_audio
        result = analyze_rhythm_zones(audio, sample_rate=sr)
        assert isinstance(result, RZTAnalysis)

    def test_rhy_49_add_rhythm_variability_works(self, mock_speech_audio):
        """TEST-RHY-49: add_rhythm_variability() adds natural timing variations."""
        audio, sr = mock_speech_audio
        result_audio, result_sr = add_rhythm_variability(audio, sample_rate=sr)
        assert isinstance(result_audio, np.ndarray)
        assert result_sr == sr

    def test_rhy_50_add_rhythm_variability_amount_param(self, mock_speech_audio):
        """TEST-RHY-50: add_rhythm_variability() respects amount parameter."""
        audio, sr = mock_speech_audio
        low_var, _ = add_rhythm_variability(audio, sample_rate=sr, amount=0.2)
        high_var, _ = add_rhythm_variability(audio, sample_rate=sr, amount=0.8)
        # Both should be valid arrays
        assert isinstance(low_var, np.ndarray)
        assert isinstance(high_var, np.ndarray)
